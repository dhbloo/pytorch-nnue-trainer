import fnmatch
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding, Module, Parameter
from torch.optim.optimizer import required


def get_bf16_support_set():
    """Get the set of devices that supports bf16."""
    bf16_support_set = set()

    if not torch.cuda.is_available():
        return bf16_support_set

    device_count = torch.cuda.device_count()
    if device_count == 0:
        return bf16_support_set

    for i in range(device_count):
        device = torch.device(f"cuda:{i}")
        major, minor = torch.cuda.get_device_capability(device)
        if major >= 8:
            bf16_support_set.add(device)

    return bf16_support_set


def zeropower_via_newtonschulz5(G: Tensor, steps: int, use_baddbmm: bool, use_bf16: bool) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() if use_bf16 else G.float()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = F.normalize(X, p=2.0, dim=(-2, -1), eps=1e-7)

    # Perform the NS iterations
    if use_baddbmm:
        for _ in range(steps):
            A = X @ X.mT
            B = torch.baddbmm(A, A, A, beta=b, alpha=c)
            X = torch.baddbmm(X, B, X, beta=a, alpha=1)
    else:
        for _ in range(steps):
            A = X @ X.mT
            # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
            B = b * A + c * A @ A
            X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X.to(G)


_GRAPH_CAPTURE_WARMUP_ITERS = 3


class _CapturedNewtonSchulz:
    """A replayable CUDA graph for one Newton-Schulz input shape.

    Muon reruns the same kernel sequence on the same shapes every step and the
    iteration has no data-dependent control flow, so it captures cleanly. Only
    the iteration is captured: the learning rate enters afterwards, in the
    parameter update, and capturing that would freeze it at its current value.
    """

    def __init__(self, orthogonalize, stacked: Tensor):
        self.static_input = torch.empty_like(stacked)
        self.static_input.copy_(stacked)
        # Autotuning, allocator growth and library workspace setup have to
        # finish before capture, otherwise the graph records that work.
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(_GRAPH_CAPTURE_WARMUP_ITERS):
                orthogonalize(self.static_input)
        torch.cuda.current_stream().wait_stream(side_stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_output = orthogonalize(self.static_input)

    def __call__(self, stacked: Tensor) -> Tensor:
        self.static_input.copy_(stacked)
        self.graph.replay()
        # Reused every step; ``Muon.step`` consumes it before the next replay.
        return self.static_output


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - Do not share a global grad-norm clip with another optimizer. Newton-Schulz
    discards ||G||, so a clip coefficient derived from a joint norm only shrinks
    the other optimizer's step. ``ChainedOptimizer`` can opt a child out.

    Every argument below lives in the param groups, so it can be overridden per
    group. Values are read with a default so that optimizer states saved before
    an argument existed still load: ``load_state_dict`` replaces param groups
    wholesale rather than merging them.

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
        ns_backend: ``gram`` (default Gram-NS on the smaller side, with an
            optional Triton path) or ``newtonschulz`` (quintic NS).
        reset_iterations: Gram-NS steps that refresh the residual from X.
            Ignored by the ``newtonschulz`` backend.
        use_fused_kernels: Use the Triton Gram-NS kernels. Requires
            ``ns_backend='gram'`` and CUDA.
        use_cuda_graph: Capture the Newton-Schulz iteration into a CUDA graph,
            one per input shape, and replay it. On by default; ignored on CPU.
            Set ``False`` to opt out.
        use_baddbmm: Use torch.baddbmm() for speeding up the ``newtonschulz`` path.
        use_bf16: Run the ``newtonschulz`` path in bf16 on devices that support it.
    """

    def __init__(
        self,
        params,
        lr=required,
        weight_decay=0.01,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        ns_backend="gram",
        reset_iterations=None,
        use_fused_kernels=False,
        use_cuda_graph=True,
        use_baddbmm=True,
        use_bf16=False,
    ):
        if ns_backend not in ("newtonschulz", "gram"):
            raise ValueError(f"ns_backend must be 'newtonschulz' or 'gram', got {ns_backend!r}")
        if use_fused_kernels and ns_backend != "gram":
            raise ValueError("use_fused_kernels requires ns_backend='gram'")
        if reset_iterations is None:
            reset_iterations = (2,) if ns_steps > 2 else ()
        reset_iterations = tuple(reset_iterations)
        # A reset at step 0, or at or past the last step, is silently a no-op.
        # Reject those so a mistyped schedule fails instead of doing nothing.
        out_of_range = [i for i in reset_iterations if not 1 <= i < ns_steps]
        if ns_backend == "gram" and out_of_range:
            raise ValueError(
                "reset_iterations entries must be in [1, ns_steps - 1] "
                f"(ns_steps={ns_steps}), got {out_of_range}"
            )
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            ns_backend=ns_backend,
            reset_iterations=reset_iterations,
            use_fused_kernels=use_fused_kernels,
            use_cuda_graph=use_cuda_graph,
            use_baddbmm=use_baddbmm,
            use_bf16=use_bf16,
        )
        super().__init__(params, defaults)
        self._bf16_devices = None
        self._ns_graphs: dict[tuple, _CapturedNewtonSchulz] = {}

    def load_state_dict(self, state_dict):
        # Legacy checkpoints did not save these flags; retain the configured
        # values for them while letting explicit checkpoint values take priority.
        legacy_flags = [
            {key: group.get(key, self.defaults[key]) for key in ("use_bf16", "use_baddbmm")}
            for group in self.param_groups
        ]
        super().load_state_dict(state_dict)
        for group, flags in zip(self.param_groups, legacy_flags):
            for key, value in flags.items():
                group.setdefault(key, value)

    def _use_bf16_for(self, stacked: Tensor, group) -> bool:
        if not group.get("use_bf16", False):
            return False
        if self._bf16_devices is None:
            # Deferred: querying device capability initializes a CUDA context.
            self._bf16_devices = get_bf16_support_set()
        return stacked.device in self._bf16_devices

    @staticmethod
    def _ns_graph_key(stacked: Tensor, group) -> tuple:
        """Everything that fixes the captured kernel sequence."""
        return (
            tuple(stacked.shape),
            stacked.dtype,
            stacked.device,
            group.get("ns_backend", "newtonschulz"),
            group["ns_steps"],
            group.get("reset_iterations", (2,)),
            bool(group.get("use_fused_kernels")),
            bool(group.get("use_baddbmm", True)),
            bool(group.get("use_bf16", False)),
        )

    def _orthogonalize(self, stacked: Tensor, group) -> Tensor:
        if not (group.get("use_cuda_graph") and stacked.is_cuda):
            return self._run_newton_schulz(stacked, group)
        key = self._ns_graph_key(stacked, group)
        captured = self._ns_graphs.get(key)
        if captured is None:
            captured = _CapturedNewtonSchulz(
                lambda buffer: self._run_newton_schulz(buffer, group), stacked
            )
            self._ns_graphs[key] = captured
        return captured(stacked)

    def _run_newton_schulz(self, stacked: Tensor, group) -> Tensor:
        # Keep the backend precision independent of the training autocast context.
        with torch.autocast(device_type=stacked.device.type, enabled=False):
            backend = group.get("ns_backend", "newtonschulz")
            steps = group["ns_steps"]
            if backend == "gram":
                from utils.gram_ns import gram_newton_schulz, gram_ns_triton

                reset = group.get("reset_iterations", (2,))
                if group.get("use_fused_kernels") and stacked.is_cuda:
                    return gram_ns_triton(stacked, steps=steps, reset_iterations=reset)
                return gram_newton_schulz(stacked, steps=steps, reset_iterations=reset)
            return zeropower_via_newtonschulz5(
                stacked,
                steps=steps,
                use_baddbmm=group.get("use_baddbmm", True),
                use_bf16=self._use_bf16_for(stacked, group),
            )

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            shape_groups = {}
            for p in filter(lambda p: p.grad is not None, group["params"]):
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf: Tensor = state["momentum_buffer"]
                key = (p.shape, p.device, p.dtype)
                if key not in shape_groups:
                    shape_groups[key] = {"params": [], "grads": [], "momentum_buffers": []}
                shape_groups[key]["params"].append(p)
                shape_groups[key]["grads"].append(g)
                shape_groups[key]["momentum_buffers"].append(buf)
            for key in shape_groups:
                group_data = shape_groups[key]
                g = torch.stack(group_data["grads"])
                torch._foreach_lerp_(
                    group_data["momentum_buffers"],
                    group_data["grads"],
                    1 - group["momentum"],
                )
                m = torch.stack(group_data["momentum_buffers"])
                g = g.lerp_(m, group["momentum"]) if group["nesterov"] else m
                if g.ndim >= 4:  # for the case of 1d/2d conv filters
                    # Compiled convolutions may produce channels-last or other
                    # non-contiguous gradient layouts.  ``reshape`` preserves
                    # the flattening semantics while materializing a contiguous
                    # tensor when ``view`` cannot represent that layout.
                    g = g.reshape(g.size(0), g.size(1), -1)
                g = self._orthogonalize(g, group)
                params = group_data["params"]
                updates = [update.reshape_as(p) for update, p in zip(g.unbind(), params)]
                if group["weight_decay"] > 0:
                    torch._foreach_mul_(params, 1 - group["lr"] * group["weight_decay"])
                # A semi-orthogonal m x n matrix has entry RMS 1/sqrt(max(m, n)),
                # so sqrt(max(m, n)) puts the per-entry update RMS at lr. This
                # deliberately takes (m, n) from the weight's leading axes rather
                # than from the flattened matrix Newton-Schulz orthogonalized,
                # which leaves a k x k conv's update a factor k smaller.
                out_dim, in_dim = params[0].shape[:2]
                torch._foreach_add_(
                    params,
                    updates,
                    alpha=-group["lr"] * max(out_dim, in_dim) ** 0.5,
                )

        return loss


def _normalize_exclude_keys(exclude_keys: str | Iterable[str] | None) -> list[str]:
    keys = [exclude_keys] if isinstance(exclude_keys, str) else list(exclude_keys or ())
    for key in keys:
        if not isinstance(key, str) or not key:
            raise ValueError(f"exclude_keys entries must be non-empty strings, got {key!r}")
    return keys


def _strip_distributed_prefix(name: str) -> str:
    while name.startswith("module."):
        name = name[len("module.") :]
    return name


def _key_matches(name: str, pattern: str) -> bool:
    return fnmatch.fnmatchcase(
        _strip_distributed_prefix(name),
        _strip_distributed_prefix(pattern),
    )


def _is_muon_eligible_param(module: Module, local_name: str, param: Parameter) -> bool:
    if not param.requires_grad:
        return False
    return (
        param.dim() >= 2
        and not isinstance(module, Embedding)
        and not local_name.endswith(("emb", "embed", "embedding"))
    )


def _iter_muon_eligible(model: Module):
    """Yield ``(named_parameters() key, parameter)`` for Muon-eligible tensors."""
    seen: set[int] = set()
    for module_name, module in model.named_modules():
        for local_name, param in module.named_parameters(recurse=False):
            param_id = id(param)
            if param_id in seen or not _is_muon_eligible_param(module, local_name, param):
                continue
            seen.add(param_id)
            yield f"{module_name}.{local_name}" if module_name else local_name, param


def partition_params_for_muon(
    models: Module | Iterable[Module],
    exclude_keys: str | Iterable[str] | None = None,
) -> tuple[list[Parameter], list[str]]:
    """Split Muon-eligible parameters, honoring ``exclude_keys`` globs.

    Trainable >=2D non-embedding tensors are eligible. ``exclude_keys`` are
    fnmatch patterns against ``named_parameters()`` keys (a leading ``module.``
    wrapper prefix is ignored on both sides) and are subtracted from that set;
    the returned names are the excluded ones, for the caller to route elsewhere.
    Each pattern must match at least one eligible parameter across *models*,
    otherwise this raises ``ValueError``.
    """
    if isinstance(models, Module):
        models = (models,)
    patterns = _normalize_exclude_keys(exclude_keys)
    named_eligible = [
        named_param
        for model in models
        for named_param in _iter_muon_eligible(model)
    ]
    matched_patterns: set[str] = set()
    muon_params = []
    excluded_names = []
    for name, param in named_eligible:
        # Credit every pattern that matches, not just the first: overlapping
        # patterns must not make the later ones look unused.
        hits = [pattern for pattern in patterns if _key_matches(name, pattern)]
        if not hits:
            muon_params.append(param)
            continue
        matched_patterns.update(hits)
        excluded_names.append(name)
    unmatched = [pattern for pattern in patterns if pattern not in matched_patterns]
    if unmatched:
        available = sorted(
            {_strip_distributed_prefix(name) for name, _ in named_eligible}
        )
        raise ValueError(
            f"exclude_keys matched no Muon-eligible parameter: {unmatched}. "
            f"Muon-eligible keys: {available}"
        )
    return muon_params, excluded_names
