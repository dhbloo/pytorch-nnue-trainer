"""Quantized star-shaped feed-forward blocks."""

import torch.nn as nn

from .linear import LinearBlock
from ..trace import TraceableModule, model_trace, print_tensor_trace
from utils.quant_utils import fake_quant


class StarBlock(TraceableModule, nn.Module):
    def __init__(self, dim_in, dim_out, expand=1):
        super().__init__()
        self.up1 = LinearBlock(dim_in, dim_out * 2 * expand, activation="relu", quant=True)
        self.up2 = LinearBlock(dim_in, dim_out * 2 * expand, activation="none", quant=True)
        self.down = LinearBlock(dim_out * expand, dim_out, activation="relu", quant=True)

    def forward(self, x):
        x1 = self.up1(x)
        x2 = self.up2(x)
        x1 = fake_quant(x1, scale=128, num_bits=8, floor=True)
        x2 = fake_quant(x2, scale=128, num_bits=8, floor=True)
        self._trace("up1", x1)
        self._trace("up2", x2)
        # i32 dot product of two adjacent pairs of u8 and i8
        x = (x1 * x2).view(*x.shape[:-1], -1, 2).sum(-1)
        self._trace("product", x)
        # clamp to int8, scale=128, [-1,1]
        x = fake_quant(x, scale=128, num_bits=8, floor=True)
        x = self.down(x)  # int8 linear layer with signed input
        self._trace("down", x)
        return x

    def forward_debug_print(self, x, name="starblock"):
        callback = lambda event, tensor: print_tensor_trace(f"{name}.{event}", tensor)
        with model_trace(self, callback):
            return self(x)
