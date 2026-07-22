"""Shared utilities for model equivalence and performance tools."""

from contextlib import nullcontext
from math import prod

import numpy as np
import torch
import yaml
from torch.utils.flop_counter import FlopCounterMode

from model import build_model
from utils.compile_utils import model_inductor_config, with_inductor_options
from utils.cuda_utils import configure_cuda_memory_limit
from utils.misc_utils import set_performance_level
from utils.training_utils import weights_init


LINE_ENCODING_MODELS = {"linennuev1", "mix9svq"}


def _convolution_backward_flops(
    grad_out_shape,
    x_shape,
    w_shape,
    _bias,
    _stride,
    _padding,
    _dilation,
    transposed,
    _output_padding,
    _groups,
    output_mask,
    out_shape,
) -> int:
    """Count convolution gradients without dropping the group factor.

    PyTorch 2.8's built-in grad-weight formula rewrites the operation as a
    convolution but omits groups, overcounting depthwise grad-weight FLOPs by
    the number of channels. Each requested input or weight gradient has the
    same multiply-add count as the corresponding grouped forward convolution.
    """
    spatial_shape = x_shape[2:] if transposed else grad_out_shape[2:]
    one_gradient = 2 * x_shape[0] * prod(w_shape) * prod(spatial_shape)
    return one_gradient * (int(output_mask[0]) + int(output_mask[1]))


def make_flop_counter(*, display: bool = False) -> FlopCounterMode:
    """Create the shared FLOP counter with corrected grouped-conv backward."""
    return FlopCounterMode(
        display=display,
        custom_mapping={torch.ops.aten.convolution_backward: _convolution_backward_flops},
    )


def sparse_feature_dimensions(model) -> list[int] | None:
    """Collect the sparse pattern dimensions required by a model tree."""
    fields = (("p_dim", 0, 8), ("p4_dim", 8, 10), ("pcode_dim", 10, 12))
    dimensions = [1] * 12
    found = False
    for attribute, begin, end in fields:
        values = {
            int(getattr(module, attribute))
            for module in model.modules()
            if hasattr(module, attribute)
        }
        if not values:
            continue
        if len(values) != 1:
            raise ValueError(f"model has inconsistent {attribute} values: {sorted(values)}")
        dimensions[begin:end] = [values.pop()] * (end - begin)
        found = True
    return dimensions if found else None


def parse_model_args(value: str | None) -> dict:
    if value is None:
        return {}
    args = yaml.safe_load(value)
    if not isinstance(args, dict):
        raise ValueError("--model-args must decode to a mapping")
    return args


def build_initialized_model(model_type: str, model_args: dict, seed: int):
    torch.manual_seed(seed)
    model = build_model(model_type, **model_args)
    model.apply(weights_init({}))
    return model


def make_synthetic_data(
    model_type: str,
    model,
    batch_size: int,
    board_size: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Build deterministic, legal-shaped CPU inputs for a registered model."""
    rng = np.random.default_rng(seed)
    cells = rng.choice(3, size=(batch_size, board_size, board_size), p=(0.72, 0.14, 0.14))
    board_input = np.stack((cells == 1, cells == 2), axis=1).astype(np.int8)
    stm_input = rng.choice((-1.0, 1.0), size=batch_size).astype(np.float32)

    value_class = rng.integers(0, 3, size=batch_size)
    value_target = np.eye(3, dtype=np.float32)[value_class]
    policy_index = rng.integers(0, board_size * board_size, size=batch_size)
    policy_target = np.zeros((batch_size, board_size * board_size), dtype=np.float32)
    policy_target[np.arange(batch_size), policy_index] = 1
    policy_target = policy_target.reshape(batch_size, board_size, board_size)

    data = {
        "board_input": torch.from_numpy(board_input),
        "board_size": torch.full((batch_size, 2), board_size, dtype=torch.int8),
        "stm_input": torch.from_numpy(stm_input),
        "value_target": torch.from_numpy(value_target),
        "policy_target": torch.from_numpy(policy_target),
    }

    sparse_dimensions = sparse_feature_dimensions(model)
    if sparse_dimensions is not None:
        sparse_feature_input = np.zeros(
            (batch_size, len(sparse_dimensions), board_size, board_size), dtype=np.int32
        )
        for channel, dimension in enumerate(sparse_dimensions):
            sparse_feature_input[:, channel] = rng.integers(
                0,
                dimension,
                size=(batch_size, board_size, board_size),
                dtype=np.int32,
            )
        sparse_feature_dim = np.broadcast_to(
            np.asarray(sparse_dimensions, dtype=np.int32),
            (batch_size, len(sparse_dimensions)),
        ).copy()
        data["sparse_feature_input"] = torch.from_numpy(sparse_feature_input)
        data["sparse_feature_dim"] = torch.from_numpy(sparse_feature_dim)

    if model_type in LINE_ENCODING_MODELS:
        from line_encoding_cpp import get_total_num_encoding, transform_boards_to_line_encoding

        line_encoding = np.empty((batch_size, 4, board_size, board_size), dtype=np.int32)
        transform_boards_to_line_encoding(board_input, line_encoding, 11, raw_code=False)
        total_num = get_total_num_encoding(11)
        data["line_encoding"] = torch.from_numpy(line_encoding)
        data["line_encoding_total_num"] = torch.full((batch_size,), total_num, dtype=torch.int64)
        if model_type == "linennuev1" and total_num != model.line_encoding_total_num:
            raise RuntimeError(
                f"LineNNUE expects {model.line_encoding_total_num} encodings, extension reports {total_num}"
            )

    return data


def move_data(data: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in data.items()}


def precision_dtype(precision: str):
    return {
        "fp32": None,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[precision]


def autocast_context(device: torch.device, precision: str):
    dtype = precision_dtype(precision)
    if device.type != "cuda" or dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def configure_torch_performance(performance_level: int, allow_tf32: bool | None = None) -> None:
    """Mirror the trainer's backend policy for reproducible performance measurements."""
    set_performance_level(performance_level)
    if allow_tf32 is not None:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32


def torch_performance_metadata(performance_level: int) -> dict[str, bool | int | str]:
    """Return the resolved backend policy recorded by performance artifacts."""
    return {
        "performance_level": performance_level,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "tf32": torch.backends.cuda.matmul.allow_tf32,
    }


def configure_model_compilation(
    model,
    *,
    enabled: bool,
    backend: str,
    mode: str,
    fullgraph: bool = False,
    dynamic: bool | None = None,
):
    """Apply the trainer-compatible compile hook and return compile_fn."""
    inductor_config = model_inductor_config(model)

    def compile_fn(fn, **overrides):
        if not enabled:
            return fn
        kwargs = {
            "backend": backend,
            "mode": mode,
            "fullgraph": fullgraph,
        }
        if dynamic is not None:
            kwargs["dynamic"] = dynamic
        kwargs.update(overrides)
        kwargs = with_inductor_options(kwargs, inductor_config)
        return torch.compile(fn, **kwargs)

    configure = getattr(model, "configure_compilation", None)
    if configure is not None:
        configure(compile_fn)
    return compile_fn


def benchmark_loss(results: dict) -> torch.Tensor:
    """A target-independent scalar that exercises all standard train outputs."""
    loss = results["value"].float().square().mean()
    loss = loss + results["policy"].float().square().mean()
    for aux_loss in (results.get("aux_losses") or {}).values():
        if isinstance(aux_loss, torch.Tensor):
            loss = loss + aux_loss.float()
    return loss


def clone_state_dict(model) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def collect_results(results: dict, model) -> dict:
    outputs = {
        "value": results["value"].detach().cpu().clone(),
        "policy": results["policy"].detach().cpu().clone(),
    }
    for name, aux_loss in (results.get("aux_losses") or {}).items():
        if isinstance(aux_loss, torch.Tensor):
            outputs[f"aux_losses.{name}"] = aux_loss.detach().cpu().clone()
    grads = {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return {"outputs": outputs, "grads": grads}
