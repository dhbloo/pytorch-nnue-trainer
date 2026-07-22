"""Per-sample dynamic linear operators."""

import torch
from torch import Tensor


def dynamic_pointwise_conv2d(input: Tensor, weight: Tensor) -> Tensor:
    """Apply per-sample 1x1 convolution weights with one batched GEMM.

    ``weight`` may have shape ``[B, O, C]`` or any shape that flattens to
    ``[B, O * C]``.  Expressing this operation as ``bmm`` avoids the very
    inefficient ``groups=B`` convolution traditionally used for dynamic
    pointwise layers, especially its backward pass at large batch sizes.

    This is the backend-independent reference and current optimized PyTorch
    implementation.  Keeping it outside the model definitions provides a
    stable target for operator benchmarks and future Triton/CUDA variants.
    """
    batch_size, in_channels, height, width = input.shape
    weight = weight.reshape(batch_size, -1, in_channels)
    output = torch.bmm(weight, input.flatten(2))
    return output.reshape(batch_size, -1, height, width)
