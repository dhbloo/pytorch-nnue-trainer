"""Benchmark a deterministic, steady-state model training step on one GPU."""

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from tools.model_perf import (
    autocast_context,
    benchmark_loss,
    make_training_step,
    make_flop_counter,
    percentile,
    prepare_model_workload,
    torch_performance_metadata,
    validate_vq_initialization,
)
from utils.compile_utils import model_inductor_config


def one_sample(data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value[:1] if value.ndim > 0 else value for key, value in data.items()}


def estimate_training_flops(model, data, device, precision) -> int | None:
    """Count one-sample forward and backward FLOPs with PyTorch's operator mappings."""
    model.eval()
    model.zero_grad(set_to_none=True)
    counter = make_flop_counter()
    try:
        with counter, autocast_context(device, precision):
            loss = benchmark_loss(model(one_sample(data)))
            loss.backward()
        return counter.get_total_flops()
    except Exception as error:
        print(f"FLOP estimation unavailable: {type(error).__name__}: {error}")
        return None
    finally:
        model.zero_grad(set_to_none=True)
        model.train()


def timing_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": sum(values) / len(values),
        "p10_ms": percentile(values, 0.1),
        "median_ms": percentile(values, 0.5),
        "p90_ms": percentile(values, 0.9),
    }


def run(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA device")
    if args.warmup_steps < 1 or args.steps < 1:
        raise ValueError("--warmup-steps and --steps must both be positive")

    device, allocator_limit_bytes, model_args, model, data = prepare_model_workload(
        model_type=args.model_type,
        encoded_model_args=args.model_args,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        board_size=args.board_size,
        seed=args.seed,
        device_name=args.device,
        performance_level=args.performance_level,
        allow_tf32=args.allow_tf32,
        max_memory_fraction=args.max_memory_fraction,
    )

    if args.training_gflops_per_sample is not None:
        training_flops_per_sample = int(args.training_gflops_per_sample * 1e9)
        flop_source = "cli"
    elif args.estimate_flops:
        training_flops_per_sample = estimate_training_flops(model, data, device, args.precision)
        flop_source = "torch.utils.flop_counter" if training_flops_per_sample else None
    else:
        training_flops_per_sample = None
        flop_source = None

    step, optimizer_metadata = make_training_step(
        model,
        data,
        device,
        precision=args.precision,
        compile_enabled=args.compile,
        backend=args.backend,
        mode=args.mode,
        fullgraph=args.fullgraph,
        dynamic=args.dynamic,
        optimizer_enabled=args.optimizer,
        optimizer_type=args.optimizer_type,
        encoded_optimizer_args=args.optimizer_args,
    )

    warmup_started = time.perf_counter()
    for _ in range(args.warmup_steps):
        step()
    torch.cuda.synchronize(device)
    warmup_seconds = time.perf_counter() - warmup_started

    vq_initialized = validate_vq_initialization(model, action="benchmark")

    torch.cuda.reset_peak_memory_stats(device)
    recorded = []
    for _ in range(args.steps):
        recorded.append(step.record_phases())

    torch.cuda.synchronize(device)
    forward_ms = [start.elapsed_time(fwd) for start, fwd, _, _ in recorded]
    backward_ms = [fwd.elapsed_time(bwd) for _, fwd, bwd, _ in recorded]
    optimizer_ms = [bwd.elapsed_time(end) for _, _, bwd, end in recorded]
    total_ms = [start.elapsed_time(end) for start, _, _, end in recorded]
    median_step_ms = percentile(total_ms, 0.5)
    samples_per_second = args.batch_size * 1000 / median_step_ms

    achieved_tflops = None
    mfu = None
    if training_flops_per_sample is not None:
        achieved_tflops = training_flops_per_sample * samples_per_second / 1e12
        if args.peak_tflops is not None:
            mfu = achieved_tflops / args.peak_tflops

    device_properties = torch.cuda.get_device_properties(device)
    result = {
        "schema_version": 1,
        "model": {
            "type": args.model_type,
            "args": model_args,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "checkpoint": args.checkpoint,
        },
        "workload": {
            "batch_size": args.batch_size,
            "board_size": args.board_size,
            "precision": args.precision,
            "optimizer": optimizer_metadata,
            "warmup_steps": args.warmup_steps,
            "measured_steps": args.steps,
            "seed": args.seed,
        },
        "compile": {
            "enabled": args.compile,
            "backend": args.backend if args.compile else None,
            "mode": args.mode if args.compile else None,
            "fullgraph": args.fullgraph if args.compile else None,
            "dynamic": args.dynamic if args.compile else None,
            "inductor_config": model_inductor_config(model) if args.compile else {},
            "warmup_seconds": warmup_seconds,
        },
        "timing": {
            "forward": timing_summary(forward_ms),
            "backward": timing_summary(backward_ms),
            "optimizer": timing_summary(optimizer_ms),
            "step": timing_summary(total_ms),
            "samples_per_second": samples_per_second,
        },
        "roofline": {
            "training_flops_per_sample": training_flops_per_sample,
            "flop_source": flop_source,
            "achieved_tflops": achieved_tflops,
            "peak_tflops": args.peak_tflops,
            "mfu": mfu,
        },
        "memory": {
            "allocator_limit_bytes": allocator_limit_bytes,
            "allocator_limit_fraction": args.max_memory_fraction,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "environment": {
            "gpu": device_properties.name,
            "gpu_total_memory_bytes": device_properties.total_memory,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "python": platform.python_version(),
            **torch_performance_metadata(args.performance_level),
        },
        "vq_initialized": vq_initialized or None,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-type", required=True)
    parser.add_argument("--model-args")
    parser.add_argument("--checkpoint")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--board-size", type=int, default=15)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--backend", default="inductor")
    parser.add_argument("--mode", default="max-autotune")
    parser.add_argument("--fullgraph", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dynamic", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--optimizer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optimizer-type", default="adamw")
    parser.add_argument("--optimizer-args")
    parser.add_argument("--estimate-flops", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--training-gflops-per-sample", type=float)
    parser.add_argument("--peak-tflops", type=float)
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--performance-level", type=int, choices=range(4), default=2)
    parser.add_argument(
        "--max-memory-fraction",
        type=float,
        help=(
            "cap the CUDA caching allocator to this fraction of total device memory; "
            "oversized runs fail with CUDA OOM before exhausting VRAM"
        ),
    )
    parser.add_argument("--output")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
