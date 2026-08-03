"""Verify MultiTensorAdamW against torch.optim.AdamW.

Covers numerical equivalence over multi-step trajectories, parameter groups,
layouts the flat pass cannot express, state_dict interchange in both
directions, checkpoint resume, registry wiring, and the step time the
replacement exists for. Requires CUDA.

    python -m tools.check_optimizer_equivalence
"""

import argparse
import io
import sys

import torch

from tools.model_perf import (
    benchmark_loss,
    build_initialized_model,
    make_synthetic_data,
    move_data,
)
from utils.fused_adamw import MultiTensorAdamW
from utils.training_utils import build_optimizer

DEVICE = "cuda"
FAILURES = []
MODEL_ARGS = dict(
    dim_middle=128,
    dim_feature=64,
    dim_policy=32,
    dim_value=64,
    dim_dwconv=32,
)


def check(name, ours, reference, tol=1e-5):
    error = max((a - b).abs().max().item() for a, b in zip(ours, reference))
    scale = max(b.abs().max().item() for b in reference) or 1.0
    ok = error / scale < tol
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<52} rel={error / scale:.3e}")
    if not ok:
        FAILURES.append(name)
    return ok


def report(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<52} {detail}")
    if not ok:
        FAILURES.append(name)


def took_fast_path(optimizer, expected: int, label: str) -> None:
    """Assert how many parameters the single-launch kernel actually updated.

    Without this every numerical check below passes whether or not the code
    being merged runs at all: the reference fallback is equivalent to
    torch.optim.AdamW by construction, so stubbing the fast path out leaves the
    whole suite green. That blind spot has already let one regression through,
    where 15 of 86 parameters silently took the reference path.
    """
    on_fast = sum(len(plan.fast) for plan in optimizer._launch_cache.values())
    report(label, on_fast == expected, f"fast={on_fast} expected={expected}")


def assert_finite_parameters(model) -> tuple[int, int]:
    """Reject an incompletely initialized fixture before it reaches an optimizer."""
    tensor_count = 0
    parameter_count = 0
    offenders = []
    for name, parameter in model.named_parameters():
        tensor_count += 1
        parameter_count += parameter.numel()
        nonfinite_count = torch.count_nonzero(
            ~torch.isfinite(parameter.detach())
        ).item()
        if nonfinite_count:
            offenders.append(
                f"{name}: {nonfinite_count} / {parameter.numel()} values"
            )

    if offenders:
        raise RuntimeError(
            "real Mix9s fixture has non-finite initialized parameters: "
            + "; ".join(offenders)
        )
    return tensor_count, parameter_count


def build_real_mix9s_model():
    """Build, validate, and transfer the initialized model used by real cases."""
    model = build_initialized_model("mix9s", MODEL_ARGS, seed=5)
    tensor_count, parameter_count = assert_finite_parameters(model)
    report(
        "all real-Mix9s fixture parameters finite before stepping",
        True,
        f"tensors={tensor_count} params={parameter_count}",
    )
    return model.to(DEVICE)


def build_real_mix9s_optimizer_fixture():
    """Create the registry fixture with validation ahead of optimizer setup."""
    model = build_real_mix9s_model()
    optimizer = build_optimizer("adamw", model, lr=1e-3)
    return model, optimizer


def trajectory(make_optimizer, shapes, steps=25, seed=0, group_split=None, skip_grad=None):
    """Run `steps` updates from a fixed seed and return the final parameters."""
    torch.manual_seed(seed)
    params = [torch.randn(*s, device=DEVICE) for s in shapes]
    for p in params:
        p.requires_grad_(True)
    optimizer = make_optimizer(params, group_split)
    generator = torch.Generator(device=DEVICE).manual_seed(seed + 1)
    for step in range(steps):
        for index, p in enumerate(params):
            if skip_grad is not None and skip_grad(step, index):
                p.grad = None
                continue
            p.grad = torch.randn(p.shape, device=DEVICE, generator=generator)
        optimizer.step()
    return [p.detach() for p in params], optimizer


def make_groups(params, group_split):
    if group_split is None:
        return [{"params": params}]
    return [
        {"params": params[: group_split[0]], "lr": 1e-3, "weight_decay": 1e-2},
        {"params": params[group_split[0]:], "lr": 5e-4, "weight_decay": 0.0},
    ]


SHAPES = [(128, 128), (64,), (32, 3, 3), (49152,), (1,), (17, 5)]


def run(args) -> int:
    """Run every check and return a process exit status."""
    global DEVICE
    DEVICE = args.device
    if not torch.cuda.is_available():
        raise RuntimeError("these checks compare against the CUDA optimizer paths")

    print("=== 1. single group, default hyperparameters ===")
    ref, _ = trajectory(lambda p, g: torch.optim.AdamW(make_groups(p, g), lr=1e-3, weight_decay=1e-2), SHAPES)
    ours, ours_opt = trajectory(lambda p, g: MultiTensorAdamW(make_groups(p, g), lr=1e-3, weight_decay=1e-2), SHAPES)
    check("25 steps vs torch.optim.AdamW", ours, ref)
    took_fast_path(ours_opt, len(SHAPES), "all parameters used the single-launch kernel")

    print("\n=== 2. two param groups (different lr, one with weight_decay=0) ===")
    ref, _ = trajectory(lambda p, g: torch.optim.AdamW(make_groups(p, g), lr=1e-3, weight_decay=1e-2), SHAPES, group_split=(3,))
    ours, _ = trajectory(lambda p, g: MultiTensorAdamW(make_groups(p, g), lr=1e-3, weight_decay=1e-2), SHAPES, group_split=(3,))
    check("25 steps, per-group lr/weight_decay", ours, ref)

    print("\n=== 3. a parameter that intermittently has no gradient ===")
    skip = lambda step, index: index == 2 and step % 3 == 0
    ref, _ = trajectory(lambda p, g: torch.optim.AdamW(make_groups(p, g), lr=1e-3, weight_decay=1e-2), SHAPES, skip_grad=skip)
    ours, _ = trajectory(lambda p, g: MultiTensorAdamW(make_groups(p, g), lr=1e-3, weight_decay=1e-2), SHAPES, skip_grad=skip)
    check("divergent step counts stay correct", ours, ref)

    print("\n=== 4. channels-last parameters (ResNet layout) ===")
    torch.manual_seed(3)
    base = [torch.randn(16, 8, 3, 3, device=DEVICE).to(memory_format=torch.channels_last) for _ in range(3)]
    for variant, factory in (("torch", torch.optim.AdamW), ("ours", MultiTensorAdamW)):
        params = [b.clone().to(memory_format=torch.channels_last).requires_grad_(True) for b in base]
        optimizer = factory(params, lr=1e-3, weight_decay=1e-2)
        generator = torch.Generator(device=DEVICE).manual_seed(7)
        for _ in range(15):
            for p in params:
                p.grad = torch.randn(p.shape, device=DEVICE, generator=generator).to(memory_format=torch.channels_last)
            optimizer.step()
        if variant == "torch":
            ref = [p.detach() for p in params]
        else:
            ours = [p.detach() for p in params]
    check("channels-last params and grads", ours, ref)
    took_fast_path(optimizer, 3, "channels-last stayed on the single-launch kernel")

    print("\n=== 5. amsgrad and maximize fall back to the reference path ===")
    ref, _ = trajectory(lambda p, g: torch.optim.AdamW(make_groups(p, g), lr=1e-3, weight_decay=1e-2, amsgrad=True), SHAPES)
    ours, _ = trajectory(lambda p, g: MultiTensorAdamW(make_groups(p, g), lr=1e-3, weight_decay=1e-2, amsgrad=True), SHAPES)
    check("amsgrad", ours, ref)
    ref, _ = trajectory(lambda p, g: torch.optim.AdamW(make_groups(p, g), lr=1e-3, weight_decay=1e-2, maximize=True), SHAPES)
    ours, _ = trajectory(lambda p, g: MultiTensorAdamW(make_groups(p, g), lr=1e-3, weight_decay=1e-2, maximize=True), SHAPES)
    check("maximize", ours, ref)

    print("\n=== 5b. a transposed gradient must not reach the single-launch kernel ===")
    # A square parameter whose gradient is column-major walks the same bytes in
    # a different order. Accepting it would train the transpose of the gradient,
    # silently and with no fallback, so it must land on the reference path.
    torch.manual_seed(13)
    square = [torch.randn(64, 64, device=DEVICE, requires_grad=True) for _ in range(2)]
    mirror_square = [q.detach().clone().requires_grad_(True) for q in square]
    opt_t = MultiTensorAdamW(square, lr=1e-3, weight_decay=1e-2)
    opt_m = torch.optim.AdamW(mirror_square, lr=1e-3, weight_decay=1e-2)
    generator = torch.Generator(device=DEVICE).manual_seed(14)
    for _ in range(5):
        for q, r in zip(square, mirror_square):
            column_major = torch.randn(64, 64, device=DEVICE, generator=generator).t()
            q.grad, r.grad = column_major, column_major.clone()
        opt_t.step()
        opt_m.step()
    took_fast_path(opt_t, 0, "column-major gradients fell back to the reference path")
    check("transposed gradients still match torch.optim.AdamW",
          [q.detach() for q in square], [r.detach() for r in mirror_square])

    print("\n=== 6. state_dict interchangeability with torch.optim.AdamW ===")
    torch.manual_seed(11)
    params_a = [torch.randn(*s, device=DEVICE, requires_grad=True) for s in SHAPES]
    params_b = [p.detach().clone().requires_grad_(True) for p in params_a]
    opt_a = MultiTensorAdamW(make_groups(params_a, (3,)), lr=1e-3, weight_decay=1e-2)
    opt_b = torch.optim.AdamW(make_groups(params_b, (3,)), lr=1e-3, weight_decay=1e-2, fused=True)
    generator = torch.Generator(device=DEVICE).manual_seed(12)


    def feed(steps=10):
        for _ in range(steps):
            for p_a, p_b in zip(params_a, params_b):
                grad = torch.randn(p_a.shape, device=DEVICE, generator=generator)
                p_a.grad, p_b.grad = grad, grad.clone()
            opt_a.step()
            opt_b.step()


    feed()
    report(
        "group key sets match torch.optim.AdamW",
        set(opt_a.state_dict()["param_groups"][0]) == set(opt_b.state_dict()["param_groups"][0]),
    )
    report(
        "step is a scalar tensor, as torch stores it",
        all(
            isinstance(entry["step"], torch.Tensor) and entry["step"].shape == ()
            for entry in opt_a.state_dict()["state"].values()
        ),
    )
    # Cross-load through a real checkpoint round-trip: torch's load_state_dict
    # hands back the source tensors unchanged when dtype and device already match,
    # so loading a live state_dict would leave both optimizers sharing one moment
    # buffer. Checkpoints always go through torch.save/torch.load.


    def roundtrip(state_dict):
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        buffer.seek(0)
        return torch.load(buffer, map_location="cpu", weights_only=True)


    opt_b.load_state_dict(roundtrip(opt_a.state_dict()))
    opt_a.load_state_dict(roundtrip(opt_b.state_dict()))
    feed()
    check("cross-loaded state_dict then 10 more steps", [p.detach() for p in params_a], [p.detach() for p in params_b])
    report(
        "state_dict() did not corrupt the live step counters",
        all(isinstance(opt_a.state[p]["step"], int) for p in params_a),
    )

    print("\n=== 7. checkpoint save / restore round-trip (trainer's torch.load path) ===")
    buffer = io.BytesIO()
    torch.save(opt_a.state_dict(), buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, map_location="cpu", weights_only=True)
    restored = MultiTensorAdamW(make_groups(params_a, (3,)), lr=1e-3, weight_decay=1e-2)
    restored.load_state_dict(loaded)
    report(
        "state restored exactly through weights_only=True",
        all(
            torch.equal(restored.state[p]["exp_avg"], opt_a.state[p]["exp_avg"])
            and torch.equal(restored.state[p]["exp_avg_sq"], opt_a.state[p]["exp_avg_sq"])
            and restored.state[p]["step"] == opt_a.state[p]["step"]
            and restored.state[p]["exp_avg"].device == p.device
            for p in params_a
        ),
    )
    report(
        "per-group hyperparameters restored",
        [(g["lr"], g["weight_decay"]) for g in restored.param_groups]
        == [(g["lr"], g["weight_decay"]) for g in opt_a.param_groups],
    )

    print("\n=== 8. gradient reallocation between steps (set_to_none) ===")
    torch.manual_seed(21)
    params_c = [torch.randn(*s, device=DEVICE, requires_grad=True) for s in SHAPES]
    params_d = [p.detach().clone().requires_grad_(True) for p in params_c]
    opt_c = MultiTensorAdamW(params_c, lr=1e-3, weight_decay=1e-2)
    opt_d = torch.optim.AdamW(params_d, lr=1e-3, weight_decay=1e-2)
    generator = torch.Generator(device=DEVICE).manual_seed(22)
    scratch = []
    for _ in range(12):
        for p_c, p_d in zip(params_c, params_d):
            grad = torch.randn(p_c.shape, device=DEVICE, generator=generator)
            p_c.grad, p_d.grad = grad.clone(), grad.clone()
        opt_c.step()
        opt_d.step()
        for p in params_c + params_d:
            p.grad = None
        # Hold on to freshly allocated blocks so the next gradients land at new
        # addresses; this is what forces the launch tables to be rebuilt.
        scratch.append(torch.empty(1 << 16, device=DEVICE))
    check("grad storage changes every step", [p.detach() for p in params_c], [p.detach() for p in params_d])
    took_fast_path(opt_c, len(SHAPES), "reallocated grads stayed on the single-launch kernel")
    del scratch

    print("\n=== 9. registry wiring and opt-out ===")
    model, registered_optimizer = build_real_mix9s_optimizer_fixture()
    report("build_optimizer('adamw') selects MultiTensorAdamW",
           type(registered_optimizer).__name__ == "MultiTensorAdamW")
    report("optim_args {fused: false} opts out to torch.optim.AdamW",
           type(build_optimizer("adamw", model, lr=1e-3, fused=False)).__name__ == "AdamW")
    report("optim_args {fused: true} opts out to torch.optim.AdamW",
           type(build_optimizer("adamw", model, lr=1e-3, fused=True)).__name__ == "AdamW")
    chained = build_optimizer("muon-adamw", model, lr=1e-3, weight_decay=1e-2)
    report("muon-adamw AdamW leg uses MultiTensorAdamW",
           any(type(o).__name__ == "MultiTensorAdamW" for o in chained.optimizers),
           f"legs={[type(o).__name__ for o in chained.optimizers]}")

    print("\n=== 10. real Mix9s parameter set, 20 steps vs fused AdamW ===")
    params_e = list(model.parameters())
    params_f = [p.detach().clone().requires_grad_(True) for p in params_e]
    opt_e = MultiTensorAdamW(params_e, lr=1e-3, weight_decay=1e-2)
    opt_f = torch.optim.AdamW(params_f, lr=1e-3, weight_decay=1e-2, fused=True, capturable=True)
    generator = torch.Generator(device=DEVICE).manual_seed(33)
    for _ in range(20):
        for p_e, p_f in zip(params_e, params_f):
            grad = torch.randn(p_e.shape, device=DEVICE, generator=generator)
            p_e.grad, p_f.grad = grad, grad.clone()
        opt_e.step()
        opt_f.step()
    check(f"{len(params_e)} tensors / {sum(p.numel() for p in params_e)} params",
          [p.detach() for p in params_e], [p.detach() for p in params_f])
    took_fast_path(opt_e, len(params_e), "every Mix9s parameter used the single-launch kernel")

    print("\n=== 11. GPU time on the Mix9s parameter set ===")


    def bench(fn, iters=50, warm=5):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters * 1000


    def graph_replay_us(optimizer):
        """CUDA-graph replay time: GPU time only, free of WSL dispatch overhead."""
        for _ in range(5):
            optimizer.step()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            optimizer.step()
        torch.cuda.synchronize()
        return bench(graph.replay)


    print(f"  eager stream   torch fused AdamW   {bench(opt_f.step):7.1f} us")
    print(f"  eager stream   MultiTensorAdamW    {bench(opt_e.step):7.1f} us")
    print(f"  graph replay   torch fused AdamW   {graph_replay_us(opt_f):7.1f} us")
    print(f"  graph replay   MultiTensorAdamW    {graph_replay_us(opt_e):7.1f} us")

    print("\n=== 12. end-to-end training step, real backward, resume from checkpoint ===")
    def fresh_model():
        return build_real_mix9s_model().train()


    model_g = fresh_model()
    data = move_data(make_synthetic_data("mix9s", model_g, 32, 15, 6), torch.device(DEVICE))
    opt_g = build_optimizer("adamw", model_g, lr=1e-3, weight_decay=1e-2)
    # Mirror parameters stepped by torch's fused AdamW on the *same* gradients. Two
    # independent backward passes would not do: cuDNN's backward is not bitwise
    # reproducible, and at step 1 Adam's update is lr*sign(g), so a 1e-7 difference
    # in a gradient near zero flips a sign and moves that weight by 2*lr.
    mirror = [p.detach().clone().requires_grad_(True) for p in model_g.parameters()]
    opt_h = torch.optim.AdamW(mirror, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, fused=True)
    rebuilds = [0]
    original_partition = opt_g._partition


    def counting_partition(group):
        rebuilds[0] += 1
        return original_partition(group)


    opt_g._partition = counting_partition


    def train_steps(model, optimizer, count, mirror_optimizer=None):
        for _ in range(count):
            optimizer.zero_grad(set_to_none=True)
            benchmark_loss(model(data)).backward()
            if mirror_optimizer is not None:
                for target, source in zip(mirror, model.parameters()):
                    target.grad = source.grad.clone()
                mirror_optimizer.step()
            optimizer.step()


    train_steps(model_g, opt_g, 10, opt_h)
    check("10 real training steps vs fused AdamW, same gradients",
          [p.detach() for p in model_g.parameters()], mirror)
    report("launch tables rebuilt once, not per step", rebuilds[0] == 1, f"rebuilds={rebuilds[0]} over 10 steps")

    buffer = io.BytesIO()
    torch.save({"model": model_g.state_dict(), "optimizer": opt_g.state_dict()}, buffer)


    def replay_fixed_gradients(model, optimizer, count, seed=44):
        """Drive `count` updates from a seeded gradient sequence.

        Real forward/backward passes cannot be used here: cuDNN's backward is not
        bitwise reproducible, so the resumed and uninterrupted runs would see
        different gradients and Adam would amplify that into a visible parameter
        difference. A fixed gradient sequence isolates what this check is about,
        which is whether the optimizer state survives the checkpoint intact.
        """
        replay = torch.Generator(device=DEVICE).manual_seed(seed)
        for _ in range(count):
            for p in model.parameters():
                p.grad = torch.randn(p.shape, device=DEVICE, generator=replay)
            optimizer.step()


    replay_fixed_gradients(model_g, opt_g, 5)

    buffer.seek(0)
    saved = torch.load(buffer, map_location="cpu", weights_only=True)
    model_r = fresh_model()
    model_r.load_state_dict(saved["model"])
    opt_r = build_optimizer("adamw", model_r, lr=1e-3, weight_decay=1e-2)
    opt_r.load_state_dict(saved["optimizer"])
    replay_fixed_gradients(model_r, opt_r, 5)
    identical = all(
        torch.equal(a.detach(), b.detach())
        for a, b in zip(model_r.parameters(), model_g.parameters())
    )
    report("resume from checkpoint matches uninterrupted run", identical, "bitwise identical")

    print("\n=== 13. gradients produced by the compiled backward ===")
    # Every other check feeds freshly allocated contiguous gradients. Inductor
    # picks its own output layouts, and those are the ones the optimizer meets
    # in training, so assert the fast path survives them instead of assuming it.
    compiled_model = fresh_model()
    compiled_optimizer = build_optimizer("adamw", compiled_model, lr=1e-3, weight_decay=1e-2)
    forward_loss = torch.compile(lambda batch: benchmark_loss(compiled_model(batch)))
    for _ in range(3):
        torch.compiler.cudagraph_mark_step_begin()
        compiled_optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", torch.bfloat16):
            compiled_loss = forward_loss(data)
        compiled_loss.backward()
        compiled_optimizer.step()
    trainable = sum(1 for parameter in compiled_model.parameters() if parameter.requires_grad)
    took_fast_path(compiled_optimizer, trainable, "compiled-backward gradients stay on the fast path")
    report(
        "compiled training leaves every parameter finite",
        all(torch.isfinite(parameter).all().item() for parameter in compiled_model.parameters()),
    )

    print("\n" + ("ALL CHECKS PASSED" if not FAILURES else f"FAILURES: {FAILURES}"))
    return 1 if FAILURES else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    return parser


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
