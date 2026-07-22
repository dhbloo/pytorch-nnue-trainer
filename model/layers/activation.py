"""Activation layers and their configuration factory."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utils.quant_utils import fake_quant


def build_activation_layer(activation, inplace=False):
    """Build an activation layer from a string identifier.

    Options:
        "relu" - Rectified linear unit
        "lrelu" - Leaky ReLU with negative slope 0.1
        "lrelu/{N}" - Leaky ReLU with negative slope 1/N (e.g., "lrelu/3" gives slope 1/3)
        "crelu" - Clipped ReLU, clamped to [0, 1]
        "tanh" - Hyperbolic tangent
        "sigmoid" - Sigmoid function
        "gelu" - Gaussian error linear unit
        "silu" - Sigmoid linear unit (Swish)
        "mish" - Mish activation
        "none" - No activation (returns None)
    """
    if activation == "relu":
        return nn.ReLU(inplace=inplace)
    elif activation == "lrelu":
        return nn.LeakyReLU(0.1, inplace=inplace)
    elif activation.startswith("lrelu/"):  # custom slope
        neg_slope = 1.0 / int(activation[6:])
        return nn.LeakyReLU(neg_slope, inplace=inplace)
    elif activation == "crelu":
        return ClippedReLU(inplace=inplace)
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "silu":
        return nn.SiLU(inplace=inplace)
    elif activation == "mish":
        return nn.Mish(inplace=inplace)
    elif activation == "none":
        return None
    else:
        raise ValueError(f"Unsupported activation: {activation}")


class ClippedReLU(nn.Module):
    """ReLU clamped to [0, max]."""
    def __init__(self, inplace=False, max=1):
        super().__init__()
        self.inplace = inplace
        self.max = max

    def forward(self, x: Tensor):
        if self.inplace:
            return x.clamp_(0, self.max)
        else:
            return x.clamp(0, self.max)


class ChannelWiseLeakyReLU(nn.Module):
    """LeakyReLU with per-channel learnable negative slopes and optional bias."""
    def __init__(self, dim, bias=True, bound=0):
        super().__init__()
        self.neg_slope = nn.Parameter(torch.ones(dim) * 0.5)
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
        self.bound = bound

    def forward(self, x):
        assert x.ndim >= 2
        shape = [1, -1] + [1] * (x.ndim - 2)

        slope = self.neg_slope.view(shape)
        # limit slope to range [-bound, bound]
        if self.bound != 0:
            slope = torch.tanh(slope / self.bound) * self.bound

        x += -torch.relu(-x) * (slope - 1)
        if self.bias is not None:
            x += self.bias.view(shape)
        return x


class QuantPReLU(nn.PReLU):
    """PReLU with fake quantization on both input and weight."""
    def __init__(
        self,
        num_parameters: int = 1,
        init: float = 0.25,
        input_quant_scale=128,
        input_quant_bits=8,
        weight_quant_scale=128,
        weight_quant_bits=8,
        weight_signed=True,
    ):
        super().__init__(num_parameters, init)
        self.input_quant_scale = input_quant_scale
        self.input_quant_bits = input_quant_bits
        self.weight_quant_scale = weight_quant_scale
        self.weight_quant_bits = weight_quant_bits
        self.weight_signed = weight_signed

    def forward(self, input: Tensor) -> Tensor:
        input = fake_quant(
            input,
            self.input_quant_scale,
            num_bits=self.input_quant_bits,
        )
        weight = fake_quant(
            self.weight,
            self.weight_quant_scale,
            num_bits=self.weight_quant_bits,
            signed=self.weight_signed,
        )
        return F.prelu(input, weight)


class SwitchPReLU(nn.Module):
    """PReLU with per-expert learnable negative slopes for MoE networks.

    Args:
        num_parameters: Number of learnable slope parameters (typically number of channels)
        num_experts: Number of experts in the MoE routing
        init: Initial value for all negative slopes (default: 0.25)
    """
    def __init__(
        self, num_parameters: int, num_experts: int, init: float = 0.25, device=None, dtype=None
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_parameters = num_parameters
        self.weight = nn.Parameter(torch.zeros((num_experts, num_parameters), **factory_kwargs))
        self.weight_fact = nn.Parameter(torch.full((1, num_parameters), init, **factory_kwargs))

    def get_weight(self, route_index: Tensor) -> Tensor:
        return F.embedding(route_index, self.weight) + self.weight_fact

    def get_weight_at_idx(self, index: int) -> Tensor:
        return self.weight[index] + self.weight_fact[0]

    def forward(self, input: Tensor, route_index: Tensor) -> Tensor:
        neg_slope = self.get_weight(route_index)
        neg_slope = neg_slope.view(
            neg_slope.shape[0], neg_slope.shape[1], *([1] * (input.ndim - neg_slope.ndim))
        )
        return torch.where(input >= 0, input, neg_slope * input)
