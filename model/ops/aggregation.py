"""Spatial aggregation primitives shared by NNUE-style value heads."""

import torch
from torch import Tensor

from utils.quant_utils import fake_quant


def split_3x3_regions(x: Tensor) -> tuple[Tensor, ...]:
    """Split ``[B, C, H, W]`` into the models' canonical nine regions.

    Remainder rows and columns are assigned exactly as in the original model
    implementations.  Views are returned instead of a stacked tensor so each
    consumer can preserve its reduction and quantization order.
    """
    height, width = x.shape[-2:]
    h1 = (height // 3) + (height % 3 == 2)
    w1 = (width // 3) + (width % 3 == 2)
    h2 = (height // 3) * 2 + (height % 3 > 0)
    w2 = (width // 3) * 2 + (width % 3 > 0)
    return (
        x[:, :, 0:h1, 0:w1],
        x[:, :, 0:h1, w1:w2],
        x[:, :, 0:h1, w2:width],
        x[:, :, h1:h2, 0:w1],
        x[:, :, h1:h2, w1:w2],
        x[:, :, h1:h2, w2:width],
        x[:, :, h2:height, 0:w1],
        x[:, :, h2:height, w1:w2],
        x[:, :, h2:height, w2:width],
    )


def sum_3x3_regions(x: Tensor) -> tuple[Tensor, ...]:
    """Sum each canonical region independently over its spatial dimensions."""
    return tuple(torch.sum(region, dim=(2, 3)) for region in split_3x3_regions(x))


def mean_3x3_regions(x: Tensor) -> tuple[Tensor, ...]:
    """Average each canonical region independently over its spatial dimensions."""
    return tuple(torch.mean(region, dim=(2, 3)) for region in split_3x3_regions(x))


def quantized_sum_3x3_regions(x: Tensor) -> tuple[Tensor, ...]:
    """Apply the Mix9/LineNNUE srai-5 quantization to nine region sums."""
    return tuple(
        fake_quant(region_sum / 32, scale=128, num_bits=32, floor=True)
        for region_sum in sum_3x3_regions(x)
    )


def quantized_avg4(a: Tensor, b: Tensor, c: Tensor, d: Tensor) -> Tensor:
    """Quantized pairwise average used to combine four neighboring regions."""
    a = fake_quant(a, floor=True)
    b = fake_quant(b, floor=True)
    c = fake_quant(c, floor=True)
    d = fake_quant(d, floor=True)
    ab = fake_quant((a + b + 1 / 128) / 2, floor=True)
    cd = fake_quant((c + d + 1 / 128) / 2, floor=True)
    return fake_quant((ab + cd + 1 / 128) / 2, floor=True)


__all__ = [
    "mean_3x3_regions",
    "quantized_avg4",
    "quantized_sum_3x3_regions",
    "split_3x3_regions",
    "sum_3x3_regions",
]
