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
from utils.file_utils import load_torch_ckpt
from utils.misc_utils import set_performance_level
from utils.training_utils import build_optimizer


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


# Matches torch.optim.AdamW's defaults, which is what these tools used before the
# optimizer became selectable, so the arithmetic per step is unchanged.
BENCHMARK_OPTIMIZER_LR = 1e-3
BENCHMARK_OPTIMIZER_WEIGHT_DECAY = 1e-2


def build_benchmark_optimizer(model, optimizer_type: str, optimizer_args: dict):
    """Build the benchmarked optimizer through the registry, with its metadata.

    Routing through :func:`build_optimizer` is what makes a registry-level
    optimizer change visible to these tools.  Pass ``fused: true`` in
    *optimizer_args* to reproduce the hardcoded ``torch.optim.AdamW(fused=True)``
    these tools used previously.
    """
    optimizer = build_optimizer(
        optimizer_type,
        model,
        lr=BENCHMARK_OPTIMIZER_LR,
        weight_decay=BENCHMARK_OPTIMIZER_WEIGHT_DECAY,
        **optimizer_args,
    )
    metadata = {
        "type": optimizer_type,
        "args": optimizer_args,
        "implementation": type(optimizer).__name__,
        "lr": BENCHMARK_OPTIMIZER_LR,
        "weight_decay": BENCHMARK_OPTIMIZER_WEIGHT_DECAY,
    }
    return optimizer, metadata


def build_initialized_model(model_type: str, model_args: dict, seed: int):
    torch.manual_seed(seed)
    return build_model(model_type, **model_args)


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


def prepare_model_workload(
    *,
    model_type: str,
    encoded_model_args: str | None,
    checkpoint: str | None,
    batch_size: int,
    board_size: int,
    seed: int,
    device_name: str,
    performance_level: int,
    allow_tf32: bool | None,
    max_memory_fraction: float,
):
    """Build the common deterministic model workload used by GPU tools."""
    configure_torch_performance(performance_level, allow_tf32)
    device = torch.device(device_name)
    allocator_limit_bytes = configure_cuda_memory_limit(device, max_memory_fraction)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model_args = parse_model_args(encoded_model_args)
    model = build_initialized_model(model_type, model_args, seed)
    if checkpoint:
        state_dict, _, _ = load_torch_ckpt(checkpoint)
        model.load_state_dict(state_dict)
    model = model.to(device).train()
    data = make_synthetic_data(model_type, model, batch_size, board_size, seed + 1)
    return device, allocator_limit_bytes, model_args, model, move_data(data, device)


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


class ModelTrainingStep:
    """Callable training step with optional phase-timing event capture."""

    def __init__(self, model, data, device, precision, forward_loss, optimizer, scaler):
        self.model = model
        self.data = data
        self.device = device
        self.precision = precision
        self.forward_loss = forward_loss
        self.optimizer = optimizer
        self.scaler = scaler

    def __call__(self) -> None:
        torch.compiler.cudagraph_mark_step_begin()
        self.model.zero_grad(set_to_none=True)
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)
        with autocast_context(self.device, self.precision):
            loss = self.forward_loss(self.data)
        self.scaler.scale(loss).backward()
        if self.optimizer is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()

    def record_phases(self):
        """Run one step while recording the historical CUDA phase boundaries."""
        start = torch.cuda.Event(enable_timing=True)
        forward_end = torch.cuda.Event(enable_timing=True)
        backward_end = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        torch.compiler.cudagraph_mark_step_begin()
        start.record()
        self.model.zero_grad(set_to_none=True)
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)
        with autocast_context(self.device, self.precision):
            loss = self.forward_loss(self.data)
        forward_end.record()
        self.scaler.scale(loss).backward()
        backward_end.record()
        if self.optimizer is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        end.record()
        return start, forward_end, backward_end, end


def make_training_step(
    model,
    data: dict[str, torch.Tensor],
    device: torch.device,
    *,
    precision: str,
    compile_enabled: bool,
    backend: str,
    mode: str,
    fullgraph: bool,
    dynamic: bool | None,
    optimizer_enabled: bool,
    optimizer_type: str,
    encoded_optimizer_args: str | None,
):
    """Create the shared benchmark loss step without changing timing boundaries."""
    compile_fn = configure_model_compilation(
        model,
        enabled=compile_enabled,
        backend=backend,
        mode=mode,
        fullgraph=fullgraph,
        dynamic=dynamic,
    )

    def forward_loss(batch):
        return benchmark_loss(model(batch))

    forward_loss = compile_fn(forward_loss)
    optimizer = None
    optimizer_metadata = None
    if optimizer_enabled:
        optimizer, optimizer_metadata = build_benchmark_optimizer(
            model, optimizer_type, parse_model_args(encoded_optimizer_args)
        )
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")

    step = ModelTrainingStep(model, data, device, precision, forward_loss, optimizer, scaler)
    return step, optimizer_metadata


def vq_initialization(model) -> dict[str, bool]:
    """Return initialization state for every VQ codebook in a model tree."""
    return {
        name: bool(buffer.item())
        for name, buffer in model.named_buffers()
        if name == "inited" or name.endswith(".inited")
    }


def validate_vq_initialization(model, *, action: str) -> dict[str, bool]:
    """Require every discovered VQ codebook to be initialized after warmup."""
    initialized = vq_initialization(model)
    if initialized and not all(initialized.values()):
        raise RuntimeError(
            "VQ codebooks are not initialized after warmup; pass an initialized checkpoint "
            f"or {action} with kmeans_init: false"
        )
    return initialized


def percentile(values: list[float], quantile: float) -> float:
    """Interpolate a quantile using the tools' historical percentile rule."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _aux_loss_tensors(aux_losses: dict | None):
    """Yield (name, tensor, is_raw_output) for every tensor in *aux_losses*.

    Tensor-form entries are already-reduced scalar losses. Tuple-form entries
    ``(loss_type, inputs)`` carry raw head outputs that training turns into
    weighted losses (trainer/loss/supervised.py); they must be exercised too,
    or the compiler can DCE the corresponding heads and equivalence checks
    never compare them.
    """
    for name, aux_loss in (aux_losses or {}).items():
        if isinstance(aux_loss, torch.Tensor):
            yield name, aux_loss, False
        elif isinstance(aux_loss, tuple) and len(aux_loss) == 2:
            aux_input = aux_loss[1]
            inputs = aux_input if isinstance(aux_input, tuple) else (aux_input,)
            for index, tensor in enumerate(inputs):
                if isinstance(tensor, torch.Tensor):
                    key = name if len(inputs) == 1 else f"{name}.{index}"
                    yield key, tensor, True


def benchmark_loss(results: dict) -> torch.Tensor:
    """A target-independent scalar that exercises all standard train outputs."""
    loss = results["value"].float().square().mean()
    loss = loss + results["policy"].float().square().mean()
    for _, aux_loss, is_raw_output in _aux_loss_tensors(results.get("aux_losses")):
        if is_raw_output:
            loss = loss + aux_loss.float().square().mean()
        else:
            loss = loss + aux_loss.float()
    return loss


def clone_state_dict(model) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def collect_results(results: dict, model) -> dict:
    outputs = {
        "value": results["value"].detach().cpu().clone(),
        "policy": results["policy"].detach().cpu().clone(),
    }
    for name, aux_loss, _ in _aux_loss_tensors(results.get("aux_losses")):
        outputs[f"aux_losses.{name}"] = aux_loss.detach().cpu().clone()
    grads = {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return {"outputs": outputs, "grads": grads}
