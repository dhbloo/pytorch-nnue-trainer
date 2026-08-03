"""Profile steady-state model training steps with a bounded CUDA allocator."""

import argparse
import json
import platform
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from tools.model_perf import (
    make_training_step,
    prepare_model_workload,
    torch_performance_metadata,
    validate_vq_initialization,
)
from utils.compile_utils import model_inductor_config


def event_summary(events, row_limit: int) -> tuple[list[dict], float]:
    total_device_time = sum(max(0.0, event.self_device_time_total) for event in events)
    sorted_events = sorted(events, key=lambda event: event.self_device_time_total, reverse=True)
    top_events = []
    for event in sorted_events[:row_limit]:
        self_device_time = event.self_device_time_total
        top_events.append(
            {
                "name": event.key,
                "calls": event.count,
                "self_device_time_us": self_device_time,
                "self_device_time_percent": (
                    100 * self_device_time / total_device_time if total_device_time else None
                ),
                "device_time_total_us": event.device_time_total,
                "self_cpu_time_us": event.self_cpu_time_total,
                "cpu_time_total_us": event.cpu_time_total,
                "self_device_memory_bytes": event.self_device_memory_usage,
            }
        )
    return top_events, total_device_time


def run(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This profiler requires a CUDA device")
    if args.warmup_steps < 1 or args.steps < 1:
        raise ValueError("--warmup-steps and --steps must both be positive")
    if args.row_limit < 1:
        raise ValueError("--row-limit must be positive")

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

    for _ in range(args.warmup_steps):
        step()
    torch.cuda.synchronize(device)

    vq_initialized = validate_vq_initialization(model, action="profile")

    torch.cuda.reset_peak_memory_stats(device)
    with profile(
        activities=(ProfilerActivity.CPU, ProfilerActivity.CUDA),
        record_shapes=args.record_shapes,
        profile_memory=args.profile_memory,
        with_stack=args.with_stack,
    ) as profiler:
        for _ in range(args.steps):
            step()
    torch.cuda.synchronize(device)

    if args.trace:
        trace = Path(args.trace)
        trace.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(trace))

    events = profiler.key_averages(group_by_input_shape=args.record_shapes)
    kernel_events = [
        event
        for event in events
        if event.device_type == torch.autograd.DeviceType.CUDA
        and not event.key.startswith(("aten::", "autograd::", "Optimizer.step#"))
    ]
    operator_events = [
        event
        for event in events
        if event.device_type == torch.autograd.DeviceType.CPU and event.key.startswith("aten::")
    ]
    top_kernels, total_kernel_time = event_summary(kernel_events, args.row_limit)
    top_operators, total_operator_time = event_summary(operator_events, args.row_limit)
    properties = torch.cuda.get_device_properties(device)
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
            "profiled_steps": args.steps,
            "seed": args.seed,
        },
        "compile": {
            "enabled": args.compile,
            "backend": args.backend if args.compile else None,
            "mode": args.mode if args.compile else None,
            "fullgraph": args.fullgraph if args.compile else None,
            "dynamic": args.dynamic if args.compile else None,
            "inductor_config": model_inductor_config(model) if args.compile else {},
        },
        "profile": {
            "total_kernel_self_device_time_us": total_kernel_time,
            "total_operator_self_device_time_us": total_operator_time,
            "top_kernels": top_kernels,
            "top_operators": top_operators,
            "record_shapes": args.record_shapes,
            "profile_memory": args.profile_memory,
            "with_stack": args.with_stack,
            "trace": args.trace,
        },
        "memory": {
            "allocator_limit_bytes": allocator_limit_bytes,
            "allocator_limit_fraction": args.max_memory_fraction,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "environment": {
            "gpu": properties.name,
            "gpu_total_memory_bytes": properties.total_memory,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "python": platform.python_version(),
            **torch_performance_metadata(args.performance_level),
        },
        "vq_initialized": vq_initialized or None,
    }

    if args.table:
        print(events.table(sort_by="self_device_time_total", row_limit=args.row_limit))
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
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--backend", default="inductor")
    parser.add_argument("--mode", default="max-autotune")
    parser.add_argument("--fullgraph", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dynamic", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--optimizer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optimizer-type", default="adamw")
    parser.add_argument("--optimizer-args")
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--performance-level", type=int, choices=range(4), default=2)
    parser.add_argument("--max-memory-fraction", type=float)
    parser.add_argument("--row-limit", type=int, default=30)
    parser.add_argument("--record-shapes", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--profile-memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--with-stack", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--table", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trace")
    parser.add_argument("--output")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
