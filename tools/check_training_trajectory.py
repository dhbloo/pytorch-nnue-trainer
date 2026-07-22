"""Create and compare deterministic multi-step training snapshots."""

import argparse
from pathlib import Path

import torch

from tools.check_model_equivalence import add_compile_args, compare_mapping
from tools.model_perf import (
    autocast_context,
    benchmark_loss,
    build_initialized_model,
    clone_state_dict,
    configure_cuda_memory_limit,
    configure_model_compilation,
    configure_torch_performance,
    make_synthetic_data,
    move_data,
    parse_model_args,
)
from utils.file_utils import load_torch_ckpt


def run_trajectory(model, data, device, precision, compile_args, steps, learning_rate):
    model = model.to(device).train()
    data = move_data(data, device)
    compile_fn = configure_model_compilation(model, **compile_args)

    def forward_loss(batch):
        return benchmark_loss(model(batch))

    forward_loss = compile_fn(forward_loss)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        fused=device.type == "cuda",
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == "fp16")
    losses = []
    for _ in range(steps):
        if device.type == "cuda":
            torch.compiler.cudagraph_mark_step_begin()
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, precision):
            loss = forward_loss(data)
        losses.append(loss.detach().float().cpu())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return torch.stack(losses), clone_state_dict(model)


def resolved_compile_args(args, stored=None):
    stored = stored or {}
    return {
        "enabled": args.compile if args.compile is not None else stored.get("enabled", False),
        "backend": args.backend or stored.get("backend", "inductor"),
        "mode": args.mode or stored.get("mode", "max-autotune"),
        "fullgraph": args.fullgraph if args.fullgraph is not None else stored.get("fullgraph", False),
        "dynamic": args.dynamic if args.dynamic is not None else stored.get("dynamic"),
    }


def snapshot(args):
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    device = torch.device(args.device)
    configure_cuda_memory_limit(device, args.max_memory_fraction)
    configure_torch_performance(args.performance_level, args.allow_tf32)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model_args = parse_model_args(args.model_args)
    model = build_initialized_model(args.model_type, model_args, args.seed)
    if args.checkpoint:
        state_dict, _, _ = load_torch_ckpt(args.checkpoint)
        model.load_state_dict(state_dict)
    data = make_synthetic_data(args.model_type, model, args.batch_size, args.board_size, args.seed + 1)
    initial_state = clone_state_dict(model)
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    compile_args = resolved_compile_args(args)
    losses, final_state = run_trajectory(
        model,
        data,
        device,
        args.precision,
        compile_args,
        args.steps,
        args.learning_rate,
    )
    artifact = {
        "model_type": args.model_type,
        "model_args": model_args,
        "checkpoint": args.checkpoint,
        "batch_size": args.batch_size,
        "board_size": args.board_size,
        "seed": args.seed,
        "precision": args.precision,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "performance_level": args.performance_level,
        "allow_tf32": args.allow_tf32,
        "compile": compile_args,
        "initial_state": initial_state,
        "state_layout": [(key, tuple(value.shape)) for key, value in initial_state.items()],
        "data": {key: value.cpu() for key, value in data.items()},
        "cpu_rng_state": cpu_rng_state,
        "cuda_rng_state": cuda_rng_state,
        "losses": losses,
        "final_state": final_state,
    }
    output = Path(args.artifact)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    print(f"Saved {args.steps}-step training trajectory: {output}")


def compare(args):
    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    if artifact["steps"] < 1:
        raise ValueError("artifact steps must be positive")
    device = torch.device(args.device)
    configure_cuda_memory_limit(device, args.max_memory_fraction)
    performance_level = (
        args.performance_level
        if args.performance_level is not None
        else artifact.get("performance_level", 2)
    )
    allow_tf32 = args.allow_tf32 if args.allow_tf32 is not None else artifact.get("allow_tf32")
    configure_torch_performance(performance_level, allow_tf32)

    model = build_initialized_model(
        artifact["model_type"], artifact["model_args"], artifact["seed"]
    )
    layout = [(key, tuple(value.shape)) for key, value in model.state_dict().items()]
    if layout != artifact["state_layout"]:
        raise AssertionError("state-dict keys or tensor shapes changed")
    model.load_state_dict(artifact["initial_state"])
    torch.set_rng_state(artifact["cpu_rng_state"])
    if device.type == "cuda" and artifact.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state(artifact["cuda_rng_state"], device)

    compile_args = resolved_compile_args(args, artifact.get("compile"))
    losses, final_state = run_trajectory(
        model,
        artifact["data"],
        device,
        artifact["precision"],
        compile_args,
        artifact["steps"],
        artifact["learning_rate"],
    )
    compare_mapping(
        "loss trajectory",
        {"loss": losses},
        {"loss": artifact["losses"]},
        args.rtol,
        args.atol,
    )
    compare_mapping(
        "final state",
        final_state,
        artifact["final_state"],
        args.state_rtol,
        args.state_atol,
    )
    print("Training loss trajectory and final state are equivalent.")


def add_common_args(parser):
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-memory-fraction", type=float)
    parser.add_argument(
        "--performance-level",
        type=int,
        choices=range(4),
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=None)
    add_compile_args(parser)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot_parser = subparsers.add_parser("snapshot", help="Create a reference trajectory")
    snapshot_parser.add_argument("artifact")
    snapshot_parser.add_argument("--model-type", required=True)
    snapshot_parser.add_argument("--model-args")
    snapshot_parser.add_argument("--checkpoint")
    snapshot_parser.add_argument("--batch-size", type=int, default=2)
    snapshot_parser.add_argument("--board-size", type=int, default=15)
    snapshot_parser.add_argument("--seed", type=int, default=42)
    snapshot_parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    snapshot_parser.add_argument("--steps", type=int, default=10)
    snapshot_parser.add_argument("--learning-rate", type=float, default=1e-3)
    add_common_args(snapshot_parser)
    # A reference trajectory must reproduce across processes. Production
    # level 2 deliberately permits nondeterministic cuDNN kernels, so keep the
    # strict gate at level 0 unless the caller opts into statistical tolerances.
    snapshot_parser.set_defaults(func=snapshot, performance_level=0)

    compare_parser = subparsers.add_parser("compare", help="Replay and compare a trajectory")
    compare_parser.add_argument("artifact")
    compare_parser.add_argument("--rtol", type=float, default=1e-5)
    compare_parser.add_argument("--atol", type=float, default=1e-6)
    compare_parser.add_argument("--state-rtol", type=float, default=1e-5)
    compare_parser.add_argument("--state-atol", type=float, default=1e-6)
    add_common_args(compare_parser)
    compare_parser.set_defaults(
        func=compare,
        compile=None,
        backend=None,
        mode=None,
        fullgraph=None,
        dynamic=None,
        performance_level=None,
    )
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    parsed.func(parsed)
