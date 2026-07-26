"""A single-pass multi-tensor AdamW for CUDA training.

`torch.optim.AdamW(fused=True)` is the wrong default on Ada consumer parts.
Its `FusedAdamMathFunctor` takes the hyperparameters as `double` and recomputes
`pow(beta, step)` in double precision inside every thread. sm_89 runs FP64 at
1/64 rate with two FP64 units per SM, so that recomputation, not the parameter
traffic, sets the kernel time. GPU time of one step on an RTX 4080 SUPER over
the Mix9s parameter set (757,604 parameters in 86 tensors), CUDA-graph replay:

    torch.optim.AdamW(fused=True)      512.5 us
    torch.optim.AdamW(foreach=True)    284.7 us
    this implementation                 12.0 us

The cost is also nearly invariant to parameter count and tensor count, because
each fused launch is capped at 36 tensors and every tiny bias tensor still gets
a full block paying the same two double-precision `pow` calls. For the small
board-game networks trained here that turns the optimizer into a fixed ~0.48 ms
tax on every step regardless of model size.

The update itself is ordinary AdamW in FP32 with the bias correction, the decay
factor and the step size evaluated once per step on the host, so no FP64
arithmetic ever reaches the GPU. Parameters keep their own storage and the
optimizer state stays per-parameter, so the `state_dict` layout is
interchangeable with `torch.optim.AdamW` in both directions and existing
checkpoints load unchanged.

The host side of a step costs ~175 us on this machine, most of it Triton's
launch path; that is WSL dispatch overhead, and it stays hidden behind the
milliseconds of GPU work a real training step queues ahead of it.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.optim import Optimizer

try:
    import triton
    import triton.language as tl
except ImportError:  # Triton is not shipped with every PyTorch backend.
    triton = None
    tl = None


# Swept over the Mix9s parameter set, graph replay: BLOCK 512-2048 with enough
# warps all land within noise of 12 us, BLOCK <= 256 costs ~18 us (more chunks,
# more pointer indirection) and BLOCK >= 2048 with 1-2 warps collapses on
# register pressure. 1024 keeps the chunk map small for large models.
_BLOCK = 1024
_NUM_WARPS = 4

# Group keys torch.optim.AdamW reads that only select one of its own
# implementations. They are carried so param_groups round-trip through
# torch.optim.AdamW's state_dict in both directions; this class ignores them.
_TORCH_IMPL_KEYS = {"foreach": None, "capturable": False, "fused": None}

# Slots in the gradient pointer ring. Two would be enough to keep the host off
# the compute stream; three leaves margin for a deeper queue.
_STAGING_SLOTS = 3


if triton is not None:

    @triton.jit
    def _multi_tensor_adamw_kernel(
        param_ptrs,
        grad_ptrs,
        exp_avg_ptrs,
        exp_avg_sq_ptrs,
        chunk_tensor,
        chunk_offset,
        numels,
        decay,
        beta2,
        one_minus_beta1,
        one_minus_beta2,
        eps,
        step_size,
        rsqrt_bias_correction2,
        BLOCK: tl.constexpr,
    ):
        chunk = tl.program_id(0)
        tensor = tl.load(chunk_tensor + chunk)
        numel = tl.load(numels + tensor)
        param_ptr = tl.load(param_ptrs + tensor).to(tl.pointer_type(tl.float32))
        grad_ptr = tl.load(grad_ptrs + tensor).to(tl.pointer_type(tl.float32))
        exp_avg_ptr = tl.load(exp_avg_ptrs + tensor).to(tl.pointer_type(tl.float32))
        exp_avg_sq_ptr = tl.load(exp_avg_sq_ptrs + tensor).to(tl.pointer_type(tl.float32))

        offsets = tl.load(chunk_offset + chunk) + tl.arange(0, BLOCK)
        mask = offsets < numel
        p = tl.load(param_ptr + offsets, mask=mask, other=0.0)
        g = tl.load(grad_ptr + offsets, mask=mask, other=0.0)
        m = tl.load(exp_avg_ptr + offsets, mask=mask, other=0.0)
        v = tl.load(exp_avg_sq_ptr + offsets, mask=mask, other=0.0)

        # Mirrors _single_tensor_adamw: decoupled decay, a lerp on the first
        # moment, a scaled squared-gradient accumulation on the second.
        p = p * decay
        m = m + (g - m) * one_minus_beta1
        v = v * beta2 + (g * g) * one_minus_beta2
        tl.store(exp_avg_ptr + offsets, m, mask=mask)
        tl.store(exp_avg_sq_ptr + offsets, v, mask=mask)

        denominator = tl.sqrt(v) * rsqrt_bias_correction2 + eps
        tl.store(param_ptr + offsets, p - step_size * (m / denominator), mask=mask)


def _flat_layout(tensor: Tensor) -> tuple | None:
    """Describe how linear indexing from ``data_ptr()`` traverses a tensor.

    Returns the ``(size, stride)`` pairs of the dimensions that vary, or
    ``None`` if walking ``numel`` consecutive elements would not visit exactly
    this tensor. Any non-overlapping dense layout qualifies, not just contiguous
    ones: a channels-last parameter and a gradient that is a slice of a DDP
    reduction bucket both do, since ``data_ptr()`` already carries the storage
    offset.

    The result doubles as a layout key: two tensors of the same shape walk the
    same logical elements in the same order exactly when their pairs match, in
    dimension order. Comparing the pairs *sorted* would not be sufficient --
    sorting drops which dimension owns which stride, so it cannot tell a square
    tensor from its transpose, and a `[64, 64]` parameter would accept a
    column-major gradient and train the transpose of it.

    Size-1 dimensions carry an arbitrary stride and are excluded: Inductor hands
    back gradients for `[C, C, 1, 1]` convolution weights with the trailing
    strides set to C rather than 1, which describes the identical traversal.
    """
    extents = tuple(
        (size, stride) for size, stride in zip(tensor.shape, tensor.stride()) if size != 1
    )
    covered = 1
    for stride, size in sorted((stride, size) for size, stride in extents):
        if stride != covered:
            return None
        covered *= size
    return extents


def _supports_fast_path(param: Tensor, grad: Tensor, state: dict) -> bool:
    """Whether one flat elementwise pass is valid for this parameter.

    The kernel walks all four buffers with the same linear index, so they must
    share a traversal order as well as a dtype and device.
    """
    layout = _flat_layout(param)
    return layout is not None and all(
        buffer.is_cuda
        and buffer.dtype == torch.float32
        and not buffer.is_sparse
        and buffer.device == param.device
        and _flat_layout(buffer) == layout
        for buffer in (grad, state["exp_avg"], state["exp_avg_sq"])
    )


def _grad_layout_unchanged(param: Tensor, layout: tuple) -> bool:
    """Whether a freshly allocated gradient still matches the cached plan.

    Only the gradient can have been replaced, and the kernel only cares that it
    walks the same elements in the same order, so nothing else is re-checked.
    """
    grad = param.grad
    return (
        grad.dtype == torch.float32
        and grad.is_cuda
        and not grad.is_sparse
        and _flat_layout(grad) == layout
    )


class _LaunchPlan:
    """Cached partition and kernel argument tables for one parameter group."""

    __slots__ = (
        "param_addresses",
        "grad_addresses",
        "fast",
        "slow",
        "arguments",
        "num_chunks",
        "fast_layouts",
        "_grad_staging",
        "_grad_staging_view",
        "_staging_slot",
    )

    def __init__(self, param_addresses, grad_addresses, fast, slow, arguments, num_chunks):
        self.param_addresses = param_addresses
        self.grad_addresses = grad_addresses
        self.fast = fast
        self.slow = slow
        self.arguments = arguments
        self.num_chunks = num_chunks
        self.fast_layouts = [_flat_layout(p) for p in fast]
        self._staging_slot = 0
        # The gradient pointer table is the one kernel argument that changes
        # between steps, and uploading it must never make the host wait on the
        # compute stream: that stream holds the whole forward and backward, so
        # anything ordered behind it costs milliseconds. Both a plain
        # `torch.tensor(..., device=...)` and a pinned copy followed by a wait on
        # an event recorded in the compute stream measured ~96 ms of host time
        # with a realistic queue pending, against ~0.5 ms for the entire step.
        #
        # So the table is a ring of host/device buffer pairs. A step writes the
        # next slot and enqueues an asynchronous copy; the slot's event is
        # recorded after the kernel that reads it, so waiting on that event
        # before reusing the slot waits for work queued _STAGING_SLOTS steps ago,
        # which has long finished. Nothing in the steady state blocks.
        self._grad_staging = None
        self._grad_staging_view = None
        if fast:
            self._grad_staging = [
                (
                    torch.empty(len(fast), dtype=torch.int64, pin_memory=True),
                    torch.empty(len(fast), dtype=torch.int64, device=fast[0].device),
                    torch.cuda.Event(),
                )
                for _ in range(_STAGING_SLOTS)
            ]
            # Filled through NumPy: a bulk assignment from a Python list costs a
            # few microseconds, where one Tensor.__setitem__ per parameter costs
            # ~250 over the Mix9s parameter set.
            self._grad_staging_view = [host.numpy() for host, _, _ in self._grad_staging]

    def refresh_grad_pointers(self) -> None:
        """Point the launch at a freshly uploaded gradient address table."""
        self._staging_slot = slot = (self._staging_slot + 1) % _STAGING_SLOTS
        host, device, consumed = self._grad_staging[slot]
        consumed.synchronize()
        self._grad_staging_view[slot][:] = [param.grad.data_ptr() for param in self.fast]
        device.copy_(host, non_blocking=True)
        self.arguments[1] = device

    def mark_launched(self) -> None:
        """Record that the kernel reading the current slot has been enqueued."""
        self._grad_staging[self._staging_slot][2].record()


class MultiTensorAdamW(Optimizer):
    """AdamW whose CUDA step is a single elementwise pass over all parameters.

    Falls back to ``torch.optim.AdamW``'s reference behaviour, per parameter,
    for anything the flat pass cannot express (non-CUDA, non-FP32, sparse or
    mismatched-layout gradients, ``amsgrad``, ``maximize``).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        maximize: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
            differentiable=False,
            decoupled_weight_decay=True,
            **_TORCH_IMPL_KEYS,
        )
        # One launch plan per param-group index, revalidated every step against
        # the parameter and gradient addresses it was built from. State buffers
        # are only ever replaced by load_state_dict / add_param_group, which
        # drop the cache explicitly, so they need no per-step address check.
        # Created before super().__init__, whose add_param_group clears it.
        self._launch_cache: dict[int, _LaunchPlan] = {}
        super().__init__(params, defaults)

    # ── State dict ────────────────────────────────────────────────
    #
    # The step counter lives on the host as a plain int: reading it back from a
    # device tensor every step would serialize on the GPU, and the bias
    # correction is evaluated on the host anyway. It is converted to and from
    # the scalar tensor torch.optim.AdamW uses at the checkpoint boundary, so
    # both directions of a state_dict exchange keep working.

    def state_dict(self) -> dict:
        packed = super().state_dict()
        # super() reuses the per-parameter state dicts verbatim; rebuild them so
        # materializing the step tensor does not write back into self.state.
        packed["state"] = {
            index: (
                {**entry, "step": torch.tensor(float(entry["step"]))}
                if isinstance(entry.get("step"), int)
                else entry
            )
            for index, entry in packed["state"].items()
        }
        return packed

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        for entry in self.state.values():
            step = entry.get("step")
            if isinstance(step, Tensor):
                entry["step"] = int(step.item())
        self._launch_cache.clear()

    def add_param_group(self, param_group: dict) -> None:
        super().add_param_group(param_group)
        self._launch_cache.clear()

    # ── Update ────────────────────────────────────────────────────

    def _reference_update(self, group: dict, params: list[Tensor]) -> None:
        """Run the reference AdamW update for parameters off the fast path."""
        beta1, beta2 = group["betas"]
        for param in params:
            grad = param.grad
            if grad.is_sparse:
                raise RuntimeError("MultiTensorAdamW does not support sparse gradients")
            if group["maximize"]:
                grad = -grad
            state = self.state[param]
            step = state["step"] = state["step"] + 1
            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            param.mul_(1.0 - group["lr"] * group["weight_decay"])
            exp_avg.lerp_(grad, 1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
            if group["amsgrad"]:
                max_exp_avg_sq = state["max_exp_avg_sq"]
                torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                exp_avg_sq = max_exp_avg_sq
            denominator = (exp_avg_sq.sqrt() / bias_correction2**0.5).add_(group["eps"])
            param.addcdiv_(exp_avg, denominator, value=-group["lr"] / bias_correction1)

    def _partition(self, group: dict) -> tuple[list[Tensor], list[Tensor]]:
        """Split a group's parameters into the fast-path and reference lists.

        A parameter that missed an update (no gradient that step) falls behind
        on the step counter, and one launch carries a single bias correction, so
        any minority step count is demoted to the reference path rather than
        silently getting the wrong correction.
        """
        fast: list[Tensor] = []
        slow: list[Tensor] = []
        flat_pass_usable = triton is not None and not group["amsgrad"] and not group["maximize"]
        for param in group["params"]:
            if param.grad is None:
                continue
            state = self.state[param]
            if "exp_avg" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(param)
                state["exp_avg_sq"] = torch.zeros_like(param)
                if group["amsgrad"]:
                    state["max_exp_avg_sq"] = torch.zeros_like(param)
            usable = flat_pass_usable and _supports_fast_path(param, param.grad, state)
            (fast if usable else slow).append(param)

        if fast:
            steps = [self.state[p]["step"] for p in fast]
            if len(set(steps)) > 1:
                majority = max(set(steps), key=steps.count)
                slow += [p for p, s in zip(fast, steps) if s != majority]
                fast = [p for p, s in zip(fast, steps) if s == majority]
        return fast, slow

    def _launch_plan(self, index: int, group: dict) -> _LaunchPlan:
        """Build, refresh, or reuse the launch plan for one group.

        Steady state costs one ``data_ptr()`` sweep and one tuple comparison.
        A gradient that lands at a new address -- which
        ``zero_grad(set_to_none=True)`` plus a fresh backward can do at any step
        -- only replaces the gradient pointer table. Everything else (the
        partition and the chunk map, which is the expensive part to build) is
        rebuilt only when a parameter is re-allocated or joins or leaves the set
        that has gradients.
        """
        params = group["params"]
        grad_addresses = tuple(0 if p.grad is None else p.grad.data_ptr() for p in params)
        plan = self._launch_cache.get(index)
        if (
            plan is not None
            and plan.grad_addresses == grad_addresses
            # Equal addresses do not by themselves mean equal layouts: the
            # caching allocator routinely hands a freed block straight back, so
            # a recompile or a different autotune winner can put a differently
            # strided gradient at the same address. The layouts are already
            # cached, so re-checking them costs a stride tuple per parameter.
            and all(
                _grad_layout_unchanged(param, layout)
                for param, layout in zip(plan.fast, plan.fast_layouts)
            )
        ):
            return plan

        # A new gradient buffer only keeps the partition valid if the same
        # parameters have gradients and each one is still laid out the way the
        # kernel assumes; re-verify rather than trust that a graph never changes.
        #
        # This runs whenever `zero_grad(set_to_none=True)` plus a fresh backward
        # moves a gradient, which for the real training loop is every step, so it
        # sits directly in the critical path. The full `_supports_fast_path`
        # sweep costs 571 us over the Mix9s parameter set -- more than the whole
        # optimizer budget -- because `untyped_storage()` is expensive. Only the
        # gradient can have changed here, and only its layout matters, so check
        # that alone: 32 us for the same guarantee.
        param_addresses = tuple(p.data_ptr() for p in params)
        if (
            plan is not None
            and plan.param_addresses == param_addresses
            and all(
                (before == 0) == (after == 0)
                for before, after in zip(plan.grad_addresses, grad_addresses)
            )
            and all(
                _grad_layout_unchanged(q, layout)
                for q, layout in zip(plan.fast, plan.fast_layouts)
            )
        ):
            plan.grad_addresses = grad_addresses
            if plan.fast:
                plan.refresh_grad_pointers()
            return plan

        fast, slow = self._partition(group)
        arguments = None
        num_chunks = 0
        if fast:
            device = fast[0].device
            states = [self.state[p] for p in fast]
            chunk_tensor: list[int] = []
            chunk_offset: list[int] = []
            for position, param in enumerate(fast):
                for offset in range(0, param.numel(), _BLOCK):
                    chunk_tensor.append(position)
                    chunk_offset.append(offset)
            num_chunks = len(chunk_tensor)

            def table(values: list[int]) -> Tensor:
                return torch.tensor(values, dtype=torch.int64, device=device)

            arguments = [
                table([p.data_ptr() for p in fast]),
                table([p.grad.data_ptr() for p in fast]),
                table([s["exp_avg"].data_ptr() for s in states]),
                table([s["exp_avg_sq"].data_ptr() for s in states]),
                table(chunk_tensor),
                table(chunk_offset),
                table([p.numel() for p in fast]),
            ]
        plan = _LaunchPlan(param_addresses, grad_addresses, fast, slow, arguments, num_chunks)
        self._launch_cache[index] = plan
        return plan

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for index, group in enumerate(self.param_groups):
            if not group.get("decoupled_weight_decay", True):
                raise ValueError(
                    "MultiTensorAdamW implements decoupled weight decay only; "
                    "use torch.optim.Adam for the coupled form"
                )
            plan = self._launch_plan(index, group)
            fast = plan.fast
            if not fast:
                if plan.slow:
                    self._reference_update(group, plan.slow)
                continue
            # The reference update runs first only for its own parameters, but
            # it can raise, and `_launch_plan` has already written this step's
            # staging slot. Keeping it inside the same try as the launch means
            # the slot's event is recorded on every path out of here.
            try:
                if plan.slow:
                    self._reference_update(group, plan.slow)

                # Every parameter on the fast path shares this group's schedule
                # and step count, so one host-side bias correction serves the
                # whole launch. The counters stay per-parameter in `state` to
                # keep the state_dict layout interchangeable with
                # torch.optim.AdamW.
                beta1, beta2 = group["betas"]
                lr = group["lr"]
                step = self.state[fast[0]]["step"] + 1
                for param in fast:
                    self.state[param]["step"] = step
                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step

                _multi_tensor_adamw_kernel[(plan.num_chunks,)](
                    *plan.arguments,
                    1.0 - lr * group["weight_decay"],
                    beta2,
                    1.0 - beta1,
                    1.0 - beta2,
                    group["eps"],
                    lr / bias_correction1,
                    bias_correction2**-0.5,
                    BLOCK=_BLOCK,
                    num_warps=_NUM_WARPS,
                )
            finally:
                # The staging slot was already written and its copy enqueued, so
                # its event must be recorded even if the launch raised and the
                # caller swallows it. Otherwise the slot's event still describes
                # a launch from several steps ago, and reusing it would let the
                # host overwrite a buffer whose copy is still in flight.
                plan.mark_launched()

        return loss


def multi_tensor_adamw_available() -> bool:
    """Whether the single-pass CUDA step can be used on this build."""
    return triton is not None and torch.cuda.is_available()


__all__ = ["MultiTensorAdamW", "multi_tensor_adamw_available"]
