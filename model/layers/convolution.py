"""Reusable convolution blocks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..ops.quantized import (
    pixelwise_quantized_depthwise_conv2d,
    product_quantization_has_linear_gradient,
)
from .activation import build_activation_layer
from .normalization import build_norm2d_layer
from utils.quant_utils import fake_quant


def _conv1d_out_size(in_size: int, kernel_size: int, stride: int, padding: int, dilation: int) -> int:
    """Compute the output size of a 1d convolution."""
    return 1 + (in_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride


def _convnd_out_size(
    in_size: tuple[int, ...],
    kernel_size: tuple[int, ...],
    stride: None | tuple[int, ...] = None,
    padding: None | tuple[int, ...] = None,
    dilation: None | tuple[int, ...] = None,
) -> tuple[int, ...]:
    """Compute the output size of a Nd convolution."""
    return tuple(
        _conv1d_out_size(
            in_size[i],
            kernel_size[i],
            1 if stride is None else stride[i],
            0 if padding is None else padding[i],
            1 if dilation is None else dilation[i],
        )
        for i in range(len(in_size))
    )


def pointwise_matmul(x: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    """Apply a 1x1 convolution as a matmul over a channel-last view of ``x``.

    ``aten.convolution_backward`` is an unconditional Inductor fallback, so a 1x1
    ``conv2d`` gets a cuDNN data gradient that no Triton epilogue can attach to: every
    activation backward next to it stays a separate memory-bound pass, and the bias
    gradient is stuck with the fallback's fixed input strides.  Emitting ``mm`` instead
    lets Inductor autotune a Triton template and fuse the activation backward into it.
    Measured on the Mix9s ``Mapping`` trunk at batch 128: 14 of 22 SiLU backwards became
    template epilogues and total kernel time dropped 15%.

    ``weight`` is the 2D ``[out_channels, in_channels]`` view of a 1x1 conv weight.
    """
    n, _, h, w = x.shape
    flat = x.permute(0, 2, 3, 1).reshape(n * h * w, weight.shape[1])
    out = F.linear(flat, weight, bias)
    return out.view(n, h, w, -1).permute(0, 3, 1, 2)


class Conv2d(nn.Conv2d):
    """Project-default convolution with complete local initialization."""

    def __init__(self, *args, weight_scale: float = 1.0, **kwargs):
        object.__setattr__(self, "weight_scale", weight_scale)
        super().__init__(*args, **kwargs)

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.weight, a=0, mode="fan_in")
        with torch.no_grad():
            self.weight.mul_(self.weight_scale)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class PointwiseConv2d(Conv2d):
    """1x1 ``nn.Conv2d`` whose forward is traced as a matmul.

    Parameters, buffers and ``state_dict`` are exactly those of
    ``nn.Conv2d(in_dim, out_dim, kernel_size=1)``, so checkpoints and the export path
    are unaffected; only the traced operator changes.  See ``pointwise_matmul``.
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__(in_dim, out_dim, kernel_size=1, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return pointwise_matmul(x, self.weight.flatten(1), self.bias)


class Conv2dBlock(nn.Module):
    """
    2D convolution block with configurable normalization, activation, and activation order.
    Supports optional fake quantization and mask propagation for variable-sized inputs.

    Args:
        in_dim: Input channel dimension
        out_dim: Output channel dimension
        ks: Kernel size for convolution
        st: Stride for convolution
        padding: Padding size (default: 0)
        norm: Normalization type, e.g., "bn", "gn", "ln", "mask", "maskbn", "none" (default: "none")
        activation: Activation function type, e.g., "relu", "lrelu", "none" (default: "relu")
        pad_type: Padding type, one of "zeros", "reflect", "replicate", "circular" (default: "zeros")
        bias: If True, adds a learnable bias to the conv layer (default: True)
        dilation: Dilation rate for convolution (default: 1)
        groups: Number of groups for grouped convolution (default: 1)
        activation_first: If True, applies norm→activation→conv (pre-activation).
                          If False, applies conv→norm→activation (post-activation) (default: False)
        quant: If True, enables fake quantization. Can also be "pixel-dwconv" or "pixel-dwconv-floor"
               for pixel-wise quantization in depthwise convolutions (default: False)
        input_quant_scale: Quantization scale for input (default: 128)
        input_quant_bits: Quantization bit width for input (default: 8)
        weight_quant_scale: Quantization scale for weights (default: 128)
        weight_quant_bits: Quantization bit width for weights (default: 8)
        bias_quant_scale: Quantization scale for bias. If None, uses input_quant_scale * weight_quant_scale
        bias_quant_bits: Quantization bit width for bias (default: 32)
    """
    def __init__(
        self,
        in_dim,
        out_dim,
        ks,
        st,
        padding=0,
        norm="none",
        activation="relu",
        pad_type="zeros",
        bias=True,
        dilation=1,
        groups=1,
        activation_first=False,
        quant=False,
        input_quant_scale=128,
        input_quant_bits=8,
        weight_quant_scale=128,
        weight_quant_bits=8,
        bias_quant_scale=None,
        bias_quant_bits=32,
        weight_scale=1.0,
    ):
        super(Conv2dBlock, self).__init__()
        assert pad_type in [
            "zeros",
            "reflect",
            "replicate",
            "circular",
        ], f"Unsupported padding mode: {pad_type}"
        self.activation_first = activation_first
        self.norm = build_norm2d_layer(norm, out_dim)
        self.activation = build_activation_layer(activation)
        self.conv = Conv2d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=ks,
            stride=st,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=pad_type,
            weight_scale=weight_scale,
        )

        self.quant = quant
        if quant:
            assert self.conv.padding_mode == "zeros", "quant conv requires zero padding"
            self.input_quant_scale = input_quant_scale
            self.input_quant_bits = input_quant_bits
            self.weight_quant_scale = weight_quant_scale
            self.weight_quant_bits = weight_quant_bits
            self.bias_quant_scale = bias_quant_scale or (input_quant_scale * weight_quant_scale)
            self.bias_quant_bits = bias_quant_bits
            self._product_quantization_has_linear_gradient = (
                product_quantization_has_linear_gradient(
                    input_quant_scale,
                    input_quant_bits,
                    weight_quant_scale,
                    weight_quant_bits,
                    self.bias_quant_scale,
                    bias_quant_bits,
                )
            )

    def _erode_mask(self, mask: Tensor) -> Tensor:
        _, _, h_in, w_in = mask.shape
        p, d, k, s = self.conv.padding, self.conv.dilation, self.conv.kernel_size, self.conv.stride
        h_out, w_out = _convnd_out_size((h_in, w_in), k, s, p, d)
        h_off, w_off = h_in - h_out, w_in - w_out
        if h_off > 0 or w_off > 0:
            return mask[:, :, h_off:, w_off:]
        else:
            return mask

    def forward(self, x: Tensor, mask: None | Tensor = None) -> Tensor | tuple[Tensor, Tensor]:
        if self.activation_first:
            if self.norm:
                x = self.norm(x, mask=mask)
            if self.activation:
                x = self.activation(x)
        if self.quant:
            x = fake_quant(x, self.input_quant_scale, num_bits=self.input_quant_bits)
            w = fake_quant(self.conv.weight, self.weight_quant_scale, num_bits=self.weight_quant_bits)
            b = self.conv.bias
            if b is not None:
                b = fake_quant(b, self.bias_quant_scale, num_bits=self.bias_quant_bits)
            if (
                self.quant == "pixel-dwconv" or self.quant == "pixel-dwconv-floor"
            ):  # pixel-wise quantization in depthwise conv
                assert self.conv.groups == x.size(1), "must be dwconv in pixel-dwconv quant mode!"
                use_linear_gradient = (
                    self.training
                    and torch.is_grad_enabled()
                    and self._product_quantization_has_linear_gradient
                    and not torch.jit.is_tracing()
                    and not torch.jit.is_scripting()
                )
                if use_linear_gradient:
                    x = pixelwise_quantized_depthwise_conv2d(
                        x,
                        w,
                        b,
                        self.conv.stride,
                        self.conv.padding,
                        self.conv.dilation,
                        self.conv.groups,
                        self.bias_quant_scale,
                        self.bias_quant_bits,
                        self.quant == "pixel-dwconv-floor",
                    )
                else:
                    assert isinstance(self.conv.padding, tuple)
                    batch_size, _, h_in, w_in = x.shape
                    p, d, k, s = (
                        self.conv.padding,
                        self.conv.dilation,
                        self.conv.kernel_size,
                        self.conv.stride,
                    )
                    h_out, w_out = _convnd_out_size((h_in, w_in), k, s, p, d)

                    x = F.unfold(x, k, d, p, s)
                    x = fake_quant(
                        x * w.view(-1)[None, :, None],
                        self.bias_quant_scale,
                        num_bits=self.bias_quant_bits,
                        floor=(self.quant == "pixel-dwconv-floor"),
                    )
                    x = x.reshape(batch_size, self.conv.out_channels, -1, h_out * w_out).sum(2)
                    x = F.fold(x, (h_out, w_out), (1, 1))
                    if b is not None:
                        x = x + b[None, :, None, None]
            else:
                x = F.conv2d(
                    x, w, b, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups
                )
        else:
            x = self.conv(x)
        if mask is not None:
            mask = self._erode_mask(mask)
        if not self.activation_first:
            if self.norm:
                x = self.norm(x, mask=mask)
            if self.activation:
                x = self.activation(x)
        if mask is not None:
            return x, mask
        else:
            return x
