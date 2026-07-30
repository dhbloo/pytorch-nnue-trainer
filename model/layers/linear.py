"""Reusable linear blocks."""

import torch.nn as nn
import torch.nn.functional as F

from .activation import build_activation_layer
from utils.quant_utils import fake_quant


class Linear(nn.Linear):
    """Project-default linear layer with complete local initialization."""

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.weight, a=0, mode="fan_in")
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class LinearBlock(nn.Module):
    """
    Linear layer followed by an activation function, with optional fake quantization.

    Args:
        in_dim: Input feature dimension
        out_dim: Output feature dimension
        activation: Activation function type, e.g., "relu", "lrelu", "none" (default: "relu")
        bias: If True, adds a learnable bias to the linear layer (default: True)
        norm: Optional feature normalization, one of "none", "bn", or "in" (default: "none")
        quant: If True, enables fake quantization for weights, inputs, and biases (default: False)
        input_quant_scale: Quantization scale for input (default: 128)
        input_quant_bits: Quantization bit width for input (default: 8)
        weight_quant_scale: Quantization scale for weights (default: 128)
        weight_quant_bits: Quantization bit width for weights (default: 8)
        bias_quant_bits: Quantization bit width for bias (default: 32)
    """
    def __init__(
        self,
        in_dim,
        out_dim,
        activation="relu",
        bias=True,
        norm="none",
        quant=False,
        input_quant_scale=128,
        input_quant_bits=8,
        weight_quant_scale=128,
        weight_quant_bits=8,
        bias_quant_bits=32,
    ):
        super(LinearBlock, self).__init__()
        self.fc = Linear(in_dim, out_dim, bias)
        if norm == "bn":
            self.norm = nn.BatchNorm1d(out_dim)
        elif norm == "in":
            self.norm = nn.InstanceNorm1d(out_dim)
        elif norm == "none":
            self.norm = None
        else:
            raise ValueError(f"Unsupported linear normalization: {norm}")
        self.quant = quant
        if quant:
            self.input_quant_scale = input_quant_scale
            self.input_quant_bits = input_quant_bits
            self.weight_quant_scale = weight_quant_scale
            self.weight_quant_bits = weight_quant_bits
            self.bias_quant_bits = bias_quant_bits

        self.activation = build_activation_layer(activation)

    def forward(self, x):
        if self.quant:
            # Using floor for inputs leads to closer results to the actual inference code
            x = fake_quant(x, self.input_quant_scale, num_bits=self.input_quant_bits, floor=True)
            w = fake_quant(self.fc.weight, self.weight_quant_scale, num_bits=self.weight_quant_bits)
            if self.fc.bias is not None:
                b = fake_quant(
                    self.fc.bias,
                    self.weight_quant_scale * self.input_quant_scale,
                    num_bits=self.bias_quant_bits,
                )
                out = F.linear(x, w, b)
            else:
                out = F.linear(x, w)
        else:
            out = self.fc(x)

        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out
