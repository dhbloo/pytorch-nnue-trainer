"""Create and compare deterministic model output/gradient snapshots."""

import argparse
from pathlib import Path

import torch

from tools.model_perf import (
    autocast_context,
    benchmark_loss,
    build_initialized_model,
    clone_state_dict,
    collect_results,
    configure_cuda_memory_limit,
    configure_model_compilation,
    make_synthetic_data,
    move_data,
    parse_model_args,
)
from utils.file_utils import load_torch_ckpt


def add_compile_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--backend", default="inductor")
    parser.add_argument("--mode", default="max-autotune")
    parser.add_argument("--fullgraph", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dynamic", action=argparse.BooleanOptionalAction, default=None)


def run_model(model, data, device, precision, compile_args):
    model = model.to(device).train()
    data = move_data(data, device)
    compile_fn = configure_model_compilation(model, **compile_args)

    def forward_loss(batch):
        results = model(batch)
        return benchmark_loss(results), results

    forward_loss = compile_fn(forward_loss)
    with autocast_context(device, precision):
        loss, results = forward_loss(data)
    loss.backward()
    actual = collect_results(results, model)
    actual["state_after"] = clone_state_dict(model)
    return actual


def snapshot(args) -> None:
    device = torch.device(args.device)
    configure_cuda_memory_limit(device, args.max_memory_fraction)
    model_args = parse_model_args(args.model_args)
    model = build_initialized_model(args.model_type, model_args, args.seed)
    if args.checkpoint:
        state_dict, _, _ = load_torch_ckpt(args.checkpoint)
        model.load_state_dict(state_dict)
    data = make_synthetic_data(args.model_type, model, args.batch_size, args.board_size, args.seed + 1)
    state_dict = clone_state_dict(model)
    compile_args = {
        "enabled": args.compile,
        "backend": args.backend,
        "mode": args.mode,
        "fullgraph": args.fullgraph,
        "dynamic": args.dynamic,
    }
    actual = run_model(model, data, device, args.precision, compile_args)
    artifact = {
        "model_type": args.model_type,
        "model_args": model_args,
        "checkpoint": args.checkpoint,
        "batch_size": args.batch_size,
        "board_size": args.board_size,
        "seed": args.seed,
        "precision": args.precision,
        "state_dict": state_dict,
        "state_layout": [(key, tuple(value.shape)) for key, value in state_dict.items()],
        "data": {key: value.cpu() for key, value in data.items()},
        "compile": compile_args,
        **actual,
    }
    output = Path(args.artifact)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    print(f"Saved equivalence snapshot: {output}")
    print(f"Outputs: {', '.join(actual['outputs'])}; gradients: {len(actual['grads'])}")


def max_error(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    return diff.max().item(), diff.mean().item()


def compare_mapping(name, actual, expected, rtol, atol) -> None:
    if actual.keys() != expected.keys():
        missing = sorted(expected.keys() - actual.keys())
        unexpected = sorted(actual.keys() - expected.keys())
        raise AssertionError(f"{name} keys differ; missing={missing}, unexpected={unexpected}")
    worst_max = 0.0
    worst_mean = 0.0
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], rtol=rtol, atol=atol)
        item_max, item_mean = max_error(actual[key], expected[key])
        worst_max = max(worst_max, item_max)
        worst_mean = max(worst_mean, item_mean)
    print(f"{name}: {len(actual)} tensors, max_abs={worst_max:.6g}, max_mean_abs={worst_mean:.6g}")


def compare(args) -> None:
    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    device = torch.device(args.device)
    configure_cuda_memory_limit(device, args.max_memory_fraction)
    model = build_initialized_model(artifact["model_type"], artifact["model_args"], artifact["seed"])
    current_layout = [(key, tuple(value.shape)) for key, value in model.state_dict().items()]
    if current_layout != artifact["state_layout"]:
        raise AssertionError("state-dict keys or tensor shapes changed")
    model.load_state_dict(artifact["state_dict"])

    stored_compile = artifact.get("compile", {})
    compile_args = {
        "enabled": args.compile if args.compile is not None else stored_compile.get("enabled", False),
        "backend": args.backend or stored_compile.get("backend", "inductor"),
        "mode": args.mode or stored_compile.get("mode", "max-autotune"),
        "fullgraph": (
            args.fullgraph if args.fullgraph is not None else stored_compile.get("fullgraph", False)
        ),
        "dynamic": args.dynamic if args.dynamic is not None else stored_compile.get("dynamic"),
    }
    actual = run_model(model, artifact["data"], device, artifact["precision"], compile_args)
    compare_mapping("outputs", actual["outputs"], artifact["outputs"], args.rtol, args.atol)
    compare_mapping("gradients", actual["grads"], artifact["grads"], args.rtol, args.atol)
    compare_mapping(
        "state after forward", actual["state_after"], artifact["state_after"], args.rtol, args.atol
    )
    print("State layout, post-forward state, outputs, and gradients are equivalent.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot_parser = subparsers.add_parser("snapshot", help="Create a reference artifact")
    snapshot_parser.add_argument("artifact")
    snapshot_parser.add_argument("--model-type", required=True)
    snapshot_parser.add_argument("--model-args")
    snapshot_parser.add_argument("--checkpoint")
    snapshot_parser.add_argument("--batch-size", type=int, default=2)
    snapshot_parser.add_argument("--board-size", type=int, default=15)
    snapshot_parser.add_argument("--seed", type=int, default=42)
    snapshot_parser.add_argument("--device", default="cpu")
    snapshot_parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    snapshot_parser.add_argument("--max-memory-fraction", type=float)
    add_compile_args(snapshot_parser)
    snapshot_parser.set_defaults(func=snapshot)

    compare_parser = subparsers.add_parser("compare", help="Compare current code to an artifact")
    compare_parser.add_argument("artifact")
    compare_parser.add_argument("--device", default="cpu")
    compare_parser.add_argument("--max-memory-fraction", type=float)
    compare_parser.add_argument("--rtol", type=float, default=1e-5)
    compare_parser.add_argument("--atol", type=float, default=1e-6)
    add_compile_args(compare_parser)
    compare_parser.set_defaults(compile=None, backend=None, mode=None, fullgraph=None, dynamic=None)
    compare_parser.set_defaults(func=compare)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    parsed.func(parsed)
