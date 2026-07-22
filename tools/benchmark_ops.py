"""Check and benchmark performance-sensitive model operators on one GPU."""

import argparse
import json
import platform

import torch
import torch.nn.functional as F

from model.layers.normalization import masked_batch_norm
from model.ops import (
    dynamic_pointwise_conv2d,
    pixelwise_quantized_depthwise_conv2d,
    quantized_sum_3x3_regions,
)
from tools.model_perf import (
    autocast_context,
    configure_cuda_memory_limit,
    configure_torch_performance,
    make_flop_counter,
    torch_performance_metadata,
)
from utils.quant_utils import fake_quant


def dynamic_pointwise_reference(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    batch_size, in_channels, height, width = input.shape
    weight = weight.reshape(-1, in_channels, 1, 1)
    output = F.conv2d(input.reshape(1, batch_size * in_channels, height, width), weight, groups=batch_size)
    return output.reshape(batch_size, -1, height, width)


def quantized_region_sum_reference(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    height, width = x.shape[-2:]
    h1 = (height // 3) + (height % 3 == 2)
    w1 = (width // 3) + (width % 3 == 2)
    h2 = (height // 3) * 2 + (height % 3 > 0)
    w2 = (width // 3) * 2 + (width % 3 > 0)
    regions = (
        x[:, :, 0:h1, 0:w1],
        x[:, :, 0:h1, w1:w2],
        x[:, :, 0:h1, w2:width],
        x[:, :, h1:h2, 0:w1],
        x[:, :, h1:h2, w1:w2],
        x[:, :, h1:h2, w2:width],
        x[:, :, h2:height, 0:w1],
        x[:, :, h2:height, w1:w2],
        x[:, :, h2:height, w2:width],
    )
    return tuple(
        fake_quant(region.sum(dim=(2, 3)) / 32, scale=128, num_bits=32, floor=True)
        for region in regions
    )


def pixelwise_quantized_depthwise(
    input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    input = fake_quant(input, scale=128, num_bits=16)
    weight = fake_quant(weight, scale=65536, num_bits=16)
    bias = fake_quant(bias, scale=128, num_bits=16)
    return pixelwise_quantized_depthwise_conv2d(
        input,
        weight,
        bias,
        (1, 1),
        (1, 1),
        (1, 1),
        input.shape[1],
        128,
        16,
        True,
    )


def pixelwise_quantized_depthwise_reference(
    input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    input = fake_quant(input, scale=128, num_bits=16)
    weight = fake_quant(weight, scale=65536, num_bits=16)
    bias = fake_quant(bias, scale=128, num_bits=16)
    batch_size, channels, height, width = input.shape
    patches = F.unfold(input, (3, 3), padding=(1, 1))
    products = fake_quant(
        patches * weight.reshape(-1)[None, :, None], scale=128, num_bits=16, floor=True
    )
    output = products.reshape(batch_size, channels, 9, height * width).sum(2)
    return output.reshape(batch_size, channels, height, width) + bias[None, :, None, None]


def directional_conv(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return F.conv2d(input, weight.unsqueeze(2), bias, padding=(0, 1))


def directional_conv_reference(
    input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    zero = torch.zeros_like(weight[..., 0])
    dense_weight = torch.stack(
        (zero, zero, zero, weight[..., 0], weight[..., 1], weight[..., 2], zero, zero, zero),
        dim=2,
    ).reshape(weight.shape[0], weight.shape[1], 3, 3)
    return F.conv2d(input, dense_weight, bias, padding=1)


def masked_batch_norm_candidate(
    input: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    output, _, _ = masked_batch_norm(input, mask, weight, bias, None, 1e-4)
    return output


def masked_batch_norm_reference(
    input: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    mask = mask.to(torch.float32)
    count = torch.sum(mask, dim=(0, 2, 3))
    mean = torch.sum(input * mask, dim=(0, 2, 3)) / count
    centered = input - mean.view(1, -1, 1, 1)
    var = torch.sum((centered * mask).square(), dim=(0, 2, 3)) / count
    output = centered / torch.sqrt(var + 1e-4).view(1, -1, 1, 1)
    output = output * weight.view(1, -1, 1, 1) + bias.view(1, -1, 1, 1)
    return (output * mask).to(input.dtype)


def make_case(args, device):
    generator = torch.Generator(device=device).manual_seed(args.seed)
    common = {
        "device": device,
        "dtype": torch.float32,
        "generator": generator,
    }
    if args.operator == "dynamic_pointwise":
        inputs = (
            torch.randn(args.batch_size, args.channels, args.board_size, args.board_size, **common),
            torch.randn(args.batch_size, args.out_channels, args.channels, **common),
        )
        return dynamic_pointwise_conv2d, dynamic_pointwise_reference, inputs
    if args.operator == "quantized_region_sum":
        inputs = (
            torch.randn(args.batch_size, args.channels, args.board_size, args.board_size, **common),
        )
        return quantized_sum_3x3_regions, quantized_region_sum_reference, inputs
    if args.operator == "pixelwise_quantized_depthwise":
        activation_dtype = (
            {
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
            }[args.precision]
            if args.mixed_dtypes
            else torch.float32
        )
        inputs = (
            torch.randn(
                args.batch_size,
                args.channels,
                args.board_size,
                args.board_size,
                device=device,
                dtype=activation_dtype,
                generator=generator,
            ),
            torch.randn(args.channels, 1, 3, 3, **common) / 8,
            torch.randn(args.channels, **common),
        )
        return (
            pixelwise_quantized_depthwise,
            pixelwise_quantized_depthwise_reference,
            inputs,
        )
    if args.operator == "directional_conv":
        inputs = (
            torch.randn(args.batch_size, args.channels, args.board_size, args.board_size, **common),
            torch.randn(args.out_channels, args.channels, 3, **common),
            torch.randn(args.out_channels, **common),
        )
        return directional_conv, directional_conv_reference, inputs
    if args.operator == "masked_batch_norm":
        activation_dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[args.precision]
        input = torch.randn(
            args.batch_size,
            args.channels,
            args.board_size,
            args.board_size,
            device=device,
            dtype=activation_dtype,
            generator=generator,
        )
        min_size = max(1, args.board_size // 2)
        heights = torch.randint(
            min_size,
            args.board_size + 1,
            (args.batch_size,),
            device=device,
            generator=generator,
        )
        widths = torch.randint(
            min_size,
            args.board_size + 1,
            (args.batch_size,),
            device=device,
            generator=generator,
        )
        rows = torch.arange(args.board_size, device=device).view(1, args.board_size, 1)
        cols = torch.arange(args.board_size, device=device).view(1, 1, args.board_size)
        mask = (rows < heights.view(-1, 1, 1)) & (cols < widths.view(-1, 1, 1))
        inputs = (
            input,
            mask.unsqueeze(1),
            torch.randn(args.channels, **common),
            torch.randn(args.channels, **common),
        )
        return masked_batch_norm_candidate, masked_batch_norm_reference, inputs
    raise ValueError(f"Unknown operator {args.operator}")


def clone_inputs(inputs) -> tuple[torch.Tensor, ...]:
    return tuple(
        value.detach().clone().requires_grad_(value.is_floating_point()) for value in inputs
    )


def output_tuple(output) -> tuple[torch.Tensor, ...]:
    return output if isinstance(output, tuple) else (output,)


def output_loss(output) -> torch.Tensor:
    return sum(value.float().square().mean() for value in output_tuple(output))


def compile_operator(fn, args):
    if not args.compile:
        return fn
    kwargs = {
        "backend": args.backend,
        "mode": args.mode,
        "fullgraph": args.fullgraph,
    }
    if args.dynamic is not None:
        kwargs["dynamic"] = args.dynamic
    return torch.compile(fn, **kwargs)


def run_once(fn, inputs, device, precision):
    actual_inputs = clone_inputs(inputs)
    with autocast_context(device, precision):
        output = fn(*actual_inputs)
    output_loss(output).backward()
    outputs = tuple(value.detach().float().cpu() for value in output_tuple(output))
    grads = tuple(
        value.grad.detach().float().cpu() for value in actual_inputs if value.requires_grad
    )
    return outputs, grads


def compare_tensors(name, actual, expected, rtol, atol) -> dict[str, float]:
    if len(actual) != len(expected):
        raise AssertionError(f"{name} tensor count differs: {len(actual)} != {len(expected)}")
    max_abs = 0.0
    max_mean_abs = 0.0
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=rtol, atol=atol)
        difference = (actual_tensor - expected_tensor).abs()
        max_abs = max(max_abs, difference.max().item())
        max_mean_abs = max(max_mean_abs, difference.mean().item())
    return {"max_abs": max_abs, "max_mean_abs": max_mean_abs}


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def benchmark(fn, inputs, args, device) -> dict[str, float]:
    actual_inputs = clone_inputs(inputs)

    def step():
        for value in actual_inputs:
            value.grad = None
        with autocast_context(device, args.precision):
            output = fn(*actual_inputs)
        output_loss(output).backward()

    for _ in range(args.warmup_steps):
        torch.compiler.cudagraph_mark_step_begin()
        step()
    torch.cuda.synchronize(device)

    events = []
    for _ in range(args.steps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.compiler.cudagraph_mark_step_begin()
        start.record()
        step()
        end.record()
        events.append((start, end))
    torch.cuda.synchronize(device)
    elapsed_ms = [start.elapsed_time(end) for start, end in events]
    median_ms = percentile(elapsed_ms, 0.5)
    return {
        "mean_ms": sum(elapsed_ms) / len(elapsed_ms),
        "p10_ms": percentile(elapsed_ms, 0.1),
        "median_ms": median_ms,
        "p90_ms": percentile(elapsed_ms, 0.9),
        "samples_per_second": args.batch_size * 1000 / median_ms,
    }


def estimate_flops(fn, inputs, args, device) -> int | None:
    actual_inputs = clone_inputs(inputs)
    counter = make_flop_counter()
    try:
        with counter, autocast_context(device, args.precision):
            output_loss(fn(*actual_inputs)).backward()
        return counter.get_total_flops() // args.batch_size
    except Exception as error:
        print(f"FLOP estimation unavailable: {type(error).__name__}: {error}")
        return None


def run(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA device")
    if args.warmup_steps < 1 or args.steps < 1:
        raise ValueError("--warmup-steps and --steps must both be positive")
    if args.mixed_dtypes and args.operator != "pixelwise_quantized_depthwise":
        raise ValueError("--mixed-dtypes is only valid for pixelwise_quantized_depthwise")
    if args.mixed_dtypes and args.precision == "fp32":
        raise ValueError("--mixed-dtypes requires --precision fp16 or bf16")

    configure_torch_performance(args.performance_level, args.allow_tf32)
    device = torch.device(args.device)
    allocator_limit_bytes = configure_cuda_memory_limit(device, args.max_memory_fraction)
    candidate, reference, inputs = make_case(args, device)
    training_flops_per_sample = estimate_flops(candidate, inputs, args, device)
    candidate = compile_operator(candidate, args)
    reference = compile_operator(reference, args)

    candidate_outputs, candidate_grads = run_once(candidate, inputs, device, args.precision)
    reference_outputs, reference_grads = run_once(reference, inputs, device, args.precision)
    correctness = {
        "outputs": compare_tensors(
            "outputs", candidate_outputs, reference_outputs, args.rtol, args.atol
        ),
        "gradients": compare_tensors(
            "gradients", candidate_grads, reference_grads, args.rtol, args.atol
        ),
    }

    candidate_timing = benchmark(candidate, inputs, args, device)
    reference_timing = benchmark(reference, inputs, args, device)
    achieved_tflops = None
    mfu = None
    if training_flops_per_sample:
        achieved_tflops = (
            training_flops_per_sample * candidate_timing["samples_per_second"] / 1e12
        )
        if args.peak_tflops is not None:
            mfu = achieved_tflops / args.peak_tflops

    properties = torch.cuda.get_device_properties(device)
    result = {
        "schema_version": 1,
        "operator": args.operator,
        "shape": {
            "batch_size": args.batch_size,
            "channels": args.channels,
            "out_channels": args.out_channels,
            "board_size": args.board_size,
        },
        "precision": args.precision,
        "input_dtypes": [str(value.dtype) for value in inputs],
        "compile": {
            "enabled": args.compile,
            "backend": args.backend if args.compile else None,
            "mode": args.mode if args.compile else None,
            "fullgraph": args.fullgraph if args.compile else None,
            "dynamic": args.dynamic if args.compile else None,
        },
        "correctness": correctness,
        "candidate": candidate_timing,
        "reference": reference_timing,
        "speedup": reference_timing["median_ms"] / candidate_timing["median_ms"],
        "roofline": {
            "training_flops_per_sample": training_flops_per_sample,
            "flop_source": "torch.utils.flop_counter" if training_flops_per_sample else None,
            "achieved_tflops": achieved_tflops,
            "peak_tflops": args.peak_tflops,
            "mfu": mfu,
        },
        "environment": {
            "gpu": properties.name,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "python": platform.python_version(),
            **torch_performance_metadata(args.performance_level),
        },
        "memory": {
            "allocator_limit_bytes": allocator_limit_bytes,
            "allocator_limit_fraction": args.max_memory_fraction,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operator",
        choices=(
            "dynamic_pointwise",
            "quantized_region_sum",
            "directional_conv",
            "masked_batch_norm",
            "pixelwise_quantized_depthwise",
        ),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--out-channels", type=int, default=32)
    parser.add_argument("--board-size", type=int, default=15)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument(
        "--mixed-dtypes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "use low-precision activations with FP32 parameters for the pixelwise "
            "quantized depthwise case, matching MixNet autocast training"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--backend", default="inductor")
    parser.add_argument("--mode", default="max-autotune")
    parser.add_argument("--fullgraph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dynamic", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=2e-3)
    parser.add_argument("--peak-tflops", type=float)
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--performance-level", type=int, choices=range(4), default=2)
    parser.add_argument(
        "--max-memory-fraction",
        type=float,
        help="cap the CUDA caching allocator to this fraction of total device memory",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
