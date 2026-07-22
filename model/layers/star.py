"""Quantized star-shaped feed-forward blocks."""

import torch.nn as nn

from .linear import LinearBlock
from utils.quant_utils import fake_quant


class StarBlock(nn.Module):
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
        # i32 dot product of two adjacent pairs of u8 and i8
        x = (x1 * x2).view(*x.shape[:-1], -1, 2).sum(-1)
        # clamp to int8, scale=128, [-1,1]
        x = fake_quant(x, scale=128, num_bits=8, floor=True)
        return self.down(x)  # int8 linear layer with signed input

    def forward_debug_print(self, x, name="starblock"):
        x1 = self.up1(x)
        x2 = self.up2(x)
        x1 = fake_quant(x1, scale=128, num_bits=8, floor=True)
        x2 = fake_quant(x2, scale=128, num_bits=8, floor=True)
        print(f"{name} up1: \n{(x1*128).int()}")
        print(f"{name} up2: \n{(x2*128).int()}")
        # i32 dot product of two adjacent pairs of u8 and i8
        x = (x1 * x2).view(*x.shape[:-1], -1, 2).sum(-1)
        x = fake_quant(x, scale=128, num_bits=8, floor=True)
        print(f"{name} dot2 product: \n{(x*128).int()}")
        x = self.down(x)  # int8 linear layer with signed input
        print(f"{name} down: \n{(x*128).int()}")
        return x
