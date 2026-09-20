"""Gram Newton-Schulz orthogonalization and optional Triton kernels.

Reference Gram-NS and the fused Triton path live in this one file so Muon can
select them through ``ns_backend='gram'`` / ``use_fused_kernels=True``.  Both
backends share :func:`_gram_ns_iterate`, so they differ only in which matmul
implementation they pass in.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


# Quintic Newton-Schulz coefficients, selected to maximize the slope at zero.
_NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
# Both backends iterate in fp16 with fp32 accumulation inside the matmuls.
_NS_DTYPE = torch.float16


def _gram_ns_iterate(
    X: Tensor,
    steps: int,
    reset_iterations: Sequence[int],
    bmm: Callable[[Tensor, Tensor], Tensor],
    baddbmm: Callable[..., Tensor],
) -> Tensor:
    """Run the Gram-NS iteration on a spectrally normalized ``X``.

    ``baddbmm(D, A, B, alpha=..., beta=...)`` must compute
    ``beta * D + alpha * (A @ B)``, matching ``torch.baddbmm``.
    """
    a, b, c = _NS_COEFFICIENTS

    if X.size(-2) == X.size(-1):
        for _ in range(steps):
            A = bmm(X, X.mT)
            B = baddbmm(A, A, A, alpha=c, beta=b)
            X = baddbmm(X, B, X, alpha=1.0, beta=a)
        return X

    # Rectangular case: iterate the cubic residual R on the smaller side and
    # accumulate the polynomial in Q, touching the full X only on a reset step
    # and once at the end.
    tall = X.size(-2) < X.size(-1)

    def gram(X: Tensor) -> Tensor:
        return bmm(X, X.mT) if tall else bmm(X.mT, X)

    def apply_q(Q: Tensor, X: Tensor) -> Tensor:
        return bmm(Q, X) if tall else bmm(X, Q)

    R = gram(X)
    Q = None
    for i in range(steps):
        if i != 0 and i in reset_iterations:
            X = apply_q(Q, X)
            R = gram(X)
            Q = None
        Z = baddbmm(R, R, R, alpha=c, beta=b)
        if Q is None:
            Q = Z.clone()
            Q.diagonal(dim1=-2, dim2=-1).add_(a)
        else:
            Q = baddbmm(Q, Q, Z, alpha=1.0, beta=a)
        if i < steps - 1 and (i + 1) not in reset_iterations:
            RZ = baddbmm(R, R, Z, alpha=1.0, beta=a)
            R = baddbmm(RZ, Z, RZ, alpha=1.0, beta=a)
    return apply_q(Q, X)


def _torch_baddbmm(D: Tensor, A: Tensor, B: Tensor, *, alpha: float, beta: float) -> Tensor:
    return torch.baddbmm(D, A, B, beta=beta, alpha=alpha)


def gram_newton_schulz(
    G: Tensor, steps: int, reset_iterations: Sequence[int]
) -> Tensor:
    """Orthogonalize ``G`` by iterating on the smaller Gram matrix.

    Mathematically the same family as quintic Newton-Schulz, but the cubic
    residual lives on ``min(M, N)^2`` instead of the full ``M x N`` factor.
    """
    assert G.ndim == 3
    X = F.normalize(G.float(), p=2.0, dim=(-2, -1))
    X = _gram_ns_iterate(
        X.to(_NS_DTYPE), steps, reset_iterations, torch.bmm, _torch_baddbmm
    )
    return X.to(G)


def gram_ns_shape_groups(params: Iterable[Tensor]) -> list[tuple[int, int, int]]:
    """Collapse Muon parameters into the ``(B, M, N)`` shapes Gram-NS sees.

    Approximates ``Muon.step`` grouping: ``B`` is the number of parameters that
    share a flattened shape; conv filters ``(out, in, ...)`` flatten to
    ``(out, in*...)``.  Used to warm Triton kernels for exactly those shapes at
    startup; autotuning keys on ``(M, N, K)`` only, so grouping parameters that
    ``Muon.step`` would split by device or dtype is harmless here.
    """
    groups = Counter()
    for param in params:
        if param.ndim >= 3:
            rows, cols = param.shape[0], param.numel() // param.shape[0]
        else:
            rows, cols = param.shape
        groups[(rows, cols)] += 1
    return [(count, rows, cols) for (rows, cols), count in sorted(groups.items())]


if triton is not None:

    _MM_CONFIGS = [
        triton.Config(
            {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k},
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for block_m, block_n, block_k, num_warps, num_stages in (
            (64, 64, 32, 4, 3),
            (64, 64, 64, 4, 3),
            (64, 64, 64, 4, 4),
            (128, 64, 32, 4, 3),
            (128, 64, 64, 4, 4),
            (128, 64, 64, 8, 3),
            (64, 128, 32, 4, 3),
            (64, 128, 64, 4, 4),
            (64, 128, 64, 8, 3),
            (128, 128, 32, 8, 3),
        )
    ]

    @triton.autotune(configs=_MM_CONFIGS, key=["M", "N", "K"])
    @triton.jit
    def _bmm_kernel(
        A,
        B,
        C,
        stride_ab,
        stride_am,
        stride_ak,
        stride_bb,
        stride_bk,
        stride_bn,
        stride_cb,
        stride_cm,
        stride_cn,
        M,
        N,
        K,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_n = tl.program_id(2)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < M
        mask_n = offs_n < N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K
            a_ptrs = A + pid_b * stride_ab + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            b_ptrs = B + pid_b * stride_bb + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
            a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            acc += tl.dot(a, b)
        c_ptrs = C + pid_b * stride_cb + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])

    @triton.autotune(configs=_MM_CONFIGS, key=["M", "N", "K"])
    @triton.jit
    def _baddbmm_kernel(
        D,
        A,
        B,
        C,
        stride_db,
        stride_dm,
        stride_dn,
        stride_ab,
        stride_am,
        stride_ak,
        stride_bb,
        stride_bk,
        stride_bn,
        stride_cb,
        stride_cm,
        stride_cn,
        M,
        N,
        K,
        ALPHA,
        BETA,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_n = tl.program_id(2)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < M
        mask_n = offs_n < N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K
            a_ptrs = A + pid_b * stride_ab + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            b_ptrs = B + pid_b * stride_bb + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
            a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            acc += tl.dot(a, b)
        d_ptrs = D + pid_b * stride_db + offs_m[:, None] * stride_dm + offs_n[None, :] * stride_dn
        d = tl.load(d_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        acc = ALPHA * acc + BETA * d
        c_ptrs = C + pid_b * stride_cb + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])

    def _grid(batch, M, N):
        return lambda META: (
            batch,
            triton.cdiv(M, META["BLOCK_M"]),
            triton.cdiv(N, META["BLOCK_N"]),
        )

    def _bmm(A: Tensor, B: Tensor) -> Tensor:
        batch, M, K = A.shape
        _, K2, N = B.shape
        assert K == K2
        C = torch.empty(batch, M, N, device=A.device, dtype=A.dtype)
        _bmm_kernel[_grid(batch, M, N)](
            A,
            B,
            C,
            A.stride(0),
            A.stride(1),
            A.stride(2),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            C.stride(0),
            C.stride(1),
            C.stride(2),
            M,
            N,
            K,
        )
        return C

    def _baddbmm(D: Tensor, A: Tensor, B: Tensor, *, alpha: float, beta: float) -> Tensor:
        batch, M, K = A.shape
        _, K2, N = B.shape
        assert K == K2 and D.shape == (batch, M, N)
        C = torch.empty(batch, M, N, device=A.device, dtype=A.dtype)
        _baddbmm_kernel[_grid(batch, M, N)](
            D,
            A,
            B,
            C,
            D.stride(0),
            D.stride(1),
            D.stride(2),
            A.stride(0),
            A.stride(1),
            A.stride(2),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            C.stride(0),
            C.stride(1),
            C.stride(2),
            M,
            N,
            K,
            alpha,
            beta,
        )
        return C

    def gram_ns_triton(
        G: Tensor, steps: int = 5, reset_iterations: Sequence[int] = (2,)
    ) -> Tensor:
        """Gram-NS on Triton matmuls: fp16 storage, fp32 accumulation."""
        assert G.ndim == 3
        X = F.normalize(G.float(), p=2.0, dim=(-2, -1))
        X = _gram_ns_iterate(X.to(_NS_DTYPE), steps, reset_iterations, _bmm, _baddbmm)
        return X.to(G)

    def warmup_gram_ns(
        shapes: Sequence[tuple[int, int, int]],
        device: torch.device,
        steps: int = 5,
        reset_iterations: Sequence[int] = (2,),
    ) -> None:
        """Run each ``(B, M, N)`` once so Triton autotune is not on the first step."""
        if device.type != "cuda":
            raise ValueError(f"warmup_gram_ns requires a CUDA device, got {device}")
        for batch, rows, cols in shapes:
            gram_ns_triton(
                torch.randn(batch, rows, cols, device=device, dtype=torch.float32),
                steps=steps,
                reset_iterations=reset_iterations,
            )
        if shapes:
            torch.cuda.synchronize(device)

else:

    def gram_ns_triton(
        G: Tensor, steps: int = 5, reset_iterations: Sequence[int] = (2,)
    ) -> Tensor:
        raise ImportError("triton is required for fused Gram-NS")

    def warmup_gram_ns(
        shapes: Sequence[tuple[int, int, int]],
        device: torch.device,
        steps: int = 5,
        reset_iterations: Sequence[int] = (2,),
    ) -> None:
        raise ImportError("triton is required for fused Gram-NS")
