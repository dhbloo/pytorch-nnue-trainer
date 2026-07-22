"""Embedding primitives with compiler-controlled gradient implementations."""

import torch
import torch.nn.functional as F
from torch import Tensor


def _small_table_embedding_backward_impl(
    grad_output: Tensor,
    indices: Tensor,
    num_weights: int,
) -> Tensor:
    """Keep the native dense embedding backward opaque to Inductor."""
    return torch.ops.aten.embedding_dense_backward.default(
        grad_output,
        indices,
        num_weights,
        -1,
        False,
    )


def _small_table_embedding_backward_fake(
    grad_output: Tensor,
    indices: Tensor,
    num_weights: int,
) -> Tensor:
    return grad_output.new_empty((num_weights, grad_output.shape[-1]))


# Use the low-level registration API available in the project's minimum
# supported PyTorch version (2.3).  A dedicated CUDA/CPU dispatch entry keeps
# this call opaque to Inductor; the Meta implementation supplies shape
# propagation for FakeTensor/AOTAutograd tracing.
_EMBEDDING_LIBRARY = torch.library.Library("nnue", "FRAGMENT")
_EMBEDDING_LIBRARY.define(
    "small_table_embedding_backward(Tensor grad_output, Tensor indices, int num_weights) -> Tensor"
)
_EMBEDDING_LIBRARY.impl(
    "small_table_embedding_backward",
    _small_table_embedding_backward_impl,
    "CPU",
)
_EMBEDDING_LIBRARY.impl(
    "small_table_embedding_backward",
    _small_table_embedding_backward_impl,
    "CUDA",
)
_EMBEDDING_LIBRARY.impl(
    "small_table_embedding_backward",
    _small_table_embedding_backward_fake,
    "Meta",
)
_small_table_embedding_backward = torch.ops.nnue.small_table_embedding_backward.default


class _SmallTableEmbeddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices: Tensor, weight: Tensor) -> Tensor:
        ctx.save_for_backward(indices)
        ctx.num_weights = weight.shape[0]
        return F.embedding(indices, weight)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[None, Tensor]:
        (indices,) = ctx.saved_tensors
        grad_weight = _small_table_embedding_backward(
            grad_output,
            indices,
            ctx.num_weights,
        )
        return None, grad_weight


def small_table_embedding(indices: Tensor, weight: Tensor) -> Tensor:
    """Compile-friendly embedding for small tables with highly repeated indices.

    Inductor normally lowers the dense weight gradient to direct atomic adds.
    PyTorch's native kernel first groups repeated indices, which is substantially
    faster for pattern-code tables but slower for large, sparsely reused tables.
    Keeping only the backward operator opaque preserves forward compilation and
    makes the intended workload constraint explicit at each call site. Eager
    execution keeps the ordinary embedding path, which already uses the native
    backward kernel.
    """
    if torch.compiler.is_compiling():
        return _SmallTableEmbeddingFunction.apply(indices, weight)
    return F.embedding(indices, weight)


__all__ = ["small_table_embedding"]
