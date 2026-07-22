"""Training operators for quantization-aware convolutions."""

import torch
import torch.nn.functional as F
from torch import Tensor

from utils.quant_utils import fake_quant


def product_quantization_has_linear_gradient(
    input_scale: float,
    input_bits: int,
    weight_scale: float,
    weight_bits: int,
    product_scale: float,
    product_bits: int,
) -> bool:
    """Whether every quantized input/weight product fits the product range.

    In that case the product fake-quantizer's STE derivative is always one,
    so its backward is exactly an ordinary convolution backward.
    """
    input_bounds = (
        -(2 ** (input_bits - 1)) / input_scale,
        (2 ** (input_bits - 1) - 1) / input_scale,
    )
    weight_bounds = (
        -(2 ** (weight_bits - 1)) / weight_scale,
        (2 ** (weight_bits - 1) - 1) / weight_scale,
    )
    products = tuple(x * w for x in input_bounds for w in weight_bounds)
    product_bounds = (
        -(2 ** (product_bits - 1)) / product_scale,
        (2 ** (product_bits - 1) - 1) / product_scale,
    )
    return min(products) >= product_bounds[0] and max(products) <= product_bounds[1]


class _LinearGradPixelwiseQuantizedDepthwiseConv2d(torch.autograd.Function):
    """Keep the exact product-quantized forward and use its linear STE gradient."""

    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        stride: tuple[int, int],
        padding: tuple[int, int],
        dilation: tuple[int, int],
        groups: int,
        product_scale: float,
        product_bits: int,
        floor: bool,
    ) -> Tensor:
        batch_size, in_channels, height, width = input.shape
        out_channels, _, kernel_height, kernel_width = weight.shape
        assert groups == in_channels == out_channels

        out_height = (
            1
            + (height + 2 * padding[0] - dilation[0] * (kernel_height - 1) - 1)
            // stride[0]
        )
        out_width = (
            1
            + (width + 2 * padding[1] - dilation[1] * (kernel_width - 1) - 1)
            // stride[1]
        )
        patches = F.unfold(input, weight.shape[-2:], dilation, padding, stride)
        products = fake_quant(
            patches * weight.reshape(-1)[None, :, None],
            product_scale,
            num_bits=product_bits,
            floor=floor,
        )
        output = products.reshape(batch_size, out_channels, -1, out_height * out_width).sum(2)
        output = output.reshape(batch_size, out_channels, out_height, out_width)
        if bias is not None:
            output = output + bias[None, :, None, None]

        ctx.save_for_backward(input, weight)
        ctx.input_size = input.shape
        ctx.weight_size = weight.shape
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.has_bias = bias is not None
        ctx.bias_dtype = None if bias is None else bias.dtype
        ctx.compute_dtype = output.dtype
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        input, weight = ctx.saved_tensors
        compute_grad_output = grad_output.to(ctx.compute_dtype)
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(
                ctx.input_size,
                weight.to(ctx.compute_dtype),
                compute_grad_output,
                ctx.stride,
                ctx.padding,
                ctx.dilation,
                ctx.groups,
            )
            grad_input = grad_input.to(input.dtype)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(
                input.to(ctx.compute_dtype),
                ctx.weight_size,
                compute_grad_output,
                ctx.stride,
                ctx.padding,
                ctx.dilation,
                ctx.groups,
            )
            grad_weight = grad_weight.to(weight.dtype)
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = compute_grad_output.sum(dim=(0, 2, 3)).to(ctx.bias_dtype)
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None


def pixelwise_quantized_depthwise_conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    groups: int,
    product_scale: float,
    product_bits: int,
    floor: bool,
) -> Tensor:
    """Depthwise convolution with independently fake-quantized products.

    The caller must prove that product quantization cannot saturate. The
    forward retains the deployment-oriented per-product rounding exactly,
    while backward avoids materializing im2col/col2im gradients.
    """
    return _LinearGradPixelwiseQuantizedDepthwiseConv2d.apply(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        product_scale,
        product_bits,
        floor,
    )


__all__ = [
    "pixelwise_quantized_depthwise_conv2d",
    "product_quantization_has_linear_gradient",
]
