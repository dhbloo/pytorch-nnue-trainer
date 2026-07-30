"""Normalization layers and their configuration factory."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BatchNorm(nn.BatchNorm2d):
    def forward(self, input: Tensor, mask=None) -> Tensor:
        # Pytorch's BatchNorm2d does not support mask, so we just call the parent class method
        return super().forward(input)


class GroupNorm(nn.GroupNorm):
    def forward(self, input: Tensor, mask=None) -> Tensor:
        # Pytorch's GroupNorm does not support mask, so we just call the parent class method
        return super().forward(input)


class LocalLayerNorm(nn.Module):
    """
    Local LayerNorm for inputs with varying spatial dimensions.

    Args:
        num_features (int): Number of channels (C).
        eps (float): A small value added for numerical stability.
        channelwise_affine (bool): If True, learnable affine parameters (scale and bias) are used.
        bias (bool): If True, includes learnable bias.
        data_format (str): The data format of the input feature tensor. Either "channels_first" or "channels_last".
    """

    def __init__(
        self, num_features, eps=1e-5, channelwise_affine=True, bias=True, data_format="channels_first"
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.channelwise_affine = channelwise_affine
        self.data_format = data_format
        assert self.data_format in [
            "channels_last",
            "channels_first",
        ], f"Unsupported data format: {self.data_format}"
        if channelwise_affine:
            self.weight = nn.Parameter(torch.empty(num_features))
            if bias:
                self.bias = nn.Parameter(torch.empty(num_features))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.channelwise_affine:
            nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, mask=None) -> Tensor:
        """
        Args:
            x: input feature tensor (B, C, *) if channels_first, or (B, *, C) if channels_last.
        """
        if self.data_format == "channels_last":
            return F.layer_norm(x, (self.num_features,), self.weight, self.bias, self.eps)
        else:
            shape = [self.num_features] + [1] * (x.ndim - 2)
            mean = x.mean(dim=1, keepdim=True)
            var = x.var(dim=1, unbiased=False, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            if self.channelwise_affine:
                x = x * self.weight.view(shape) + self.bias.view(shape)
        return x


class _MaskedBatchNormFunction(torch.autograd.Function):
    """Masked normalization with a compact, compiler-friendly backward formula."""

    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        mask: None | Tensor,
        weight: None | Tensor,
        bias: None | Tensor,
        scale: None | Tensor,
        epsilon: float,
    ):
        low_precision = x.dtype in (torch.float16, torch.bfloat16)
        compute_dtype = torch.float32 if low_precision else x.dtype
        stats_x = x.to(compute_dtype)
        if mask is not None:
            stats_mask = mask.to(compute_dtype)
            moments_x = stats_x * stats_mask
            inv_count = torch.sum(stats_mask, dim=(0, 2, 3)).reciprocal()
        else:
            stats_mask = x.new_empty(0, dtype=compute_dtype)
            moments_x = stats_x
            inv_count = stats_x.new_full((), 1.0 / (x.shape[0] * x.shape[2] * x.shape[3]))

        mean = torch.sum(moments_x, dim=(0, 2, 3)) * inv_count
        second_moment = torch.sum(moments_x.square(), dim=(0, 2, 3)) * inv_count
        var = torch.clamp(second_moment - mean.square(), min=0)
        inv_std = torch.rsqrt(var + epsilon)
        normalized = (stats_x - mean.view(1, -1, 1, 1)) * inv_std.view(1, -1, 1, 1)

        stats_scale = stats_x.new_ones(()) if scale is None else scale.to(compute_dtype)
        output = normalized * stats_scale
        if weight is not None:
            stats_weight = weight.to(compute_dtype)
            gain = stats_weight * stats_scale
            output = output * stats_weight.view(1, -1, 1, 1)
            output = output + bias.to(compute_dtype).view(1, -1, 1, 1)
        else:
            gain = stats_scale.expand(mean.shape)

        if mask is not None:
            output = output * stats_mask

        ctx.save_for_backward(x, stats_mask, mean, inv_std, inv_count, gain, stats_scale)
        ctx.has_affine = weight is not None
        ctx.has_mask = mask is not None
        ctx.mark_non_differentiable(mean, var)
        return output.to(x.dtype), mean, var

    @staticmethod
    def backward(
        ctx,
        grad_output: Tensor,
        _grad_mean: None | Tensor,
        _grad_var: None | Tensor,
    ):
        x, mask, mean, inv_std, inv_count, gain, scale = ctx.saved_tensors
        grad = grad_output.to(mean.dtype)
        normalized = (x.to(mean.dtype) - mean.view(1, -1, 1, 1)) * inv_std.view(1, -1, 1, 1)
        if ctx.has_mask:
            grad = grad * mask

        grad_sum = torch.sum(grad, dim=(0, 2, 3))
        grad_dot = torch.sum(grad * normalized, dim=(0, 2, 3))
        grad_x = grad * gain.view(1, -1, 1, 1)
        grad_x = grad_x - (grad_sum * gain).view(1, -1, 1, 1) * inv_count
        grad_x = grad_x - normalized * (grad_dot * gain).view(1, -1, 1, 1) * inv_count
        grad_x = grad_x * inv_std.view(1, -1, 1, 1)
        if ctx.has_mask:
            grad_x = grad_x * mask

        grad_weight = grad_dot * scale if ctx.has_affine else None
        grad_bias = grad_sum if ctx.has_affine else None
        return grad_x.to(x.dtype), None, grad_weight, grad_bias, None, None


def masked_batch_norm(
    x: Tensor,
    mask: None | Tensor,
    weight: None | Tensor,
    bias: None | Tensor,
    scale: None | Tensor,
    epsilon: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Apply training-mode masked batch normalization and return output, mean, and variance."""
    return _MaskedBatchNormFunction.apply(x, mask, weight, bias, scale, epsilon)


class MaskNorm(nn.Module):
    """
    Various kinds of normalization with masked input.
    This class is simplified from original implementation of Katago:
    https://github.com/lightvector/KataGo/blob/master/python/model_pytorch.py

    Available norm types:
    bnorm - batch norm
    fixup - fixup initialization https://arxiv.org/abs/1901.09321
    """

    def __init__(
        self,
        num_features: int,
        norm_type: str,
        affine: bool = False,
        bnorm_epsilon: float = 1e-4,
        bnorm_running_avg_momentum: float = 1e-3,
    ):
        super().__init__()
        assert norm_type in ["bnorm", "fixup"], f"Invalid norm type {norm_type}"
        self.num_features = num_features
        self.norm_type = norm_type
        self.affine = affine
        self.epsilon = bnorm_epsilon
        self.running_avg_momentum = bnorm_running_avg_momentum

        if affine:
            self.weight = nn.Parameter(torch.empty(num_features))
            self.bias = nn.Parameter(torch.empty(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if norm_type == "bnorm":
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)

        self.reset_parameters()

    def reset_running_stats(self):
        if self.norm_type == "bnorm":
            self.running_mean.zero_()
            self.running_var.fill_(1.0)

    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
        self.reset_running_stats()

    def _apply_affine_transform(self, x: Tensor) -> Tensor:
        if self.affine:
            x = x * self.weight.view(1, self.num_features, 1, 1) + self.bias.view(1, self.num_features, 1, 1)
        return x

    def forward(self, x: Tensor, mask: None | Tensor) -> Tensor:
        if self.norm_type == "bnorm":
            assert x.ndim == 4 and x.shape[1] == self.num_features

            if self.training:
                x, mean, var = masked_batch_norm(
                    x,
                    mask,
                    self.weight,
                    self.bias,
                    None,
                    self.epsilon,
                )
                with torch.no_grad():
                    self.running_mean += self.running_avg_momentum * (mean.detach() - self.running_mean)
                    self.running_var += self.running_avg_momentum * (var.detach() - self.running_var)
                return x
            else:
                mean, var = self.running_mean, self.running_var
                zeromean_x = x - mean.view(1, self.num_features, 1, 1)

            if not self.training and torch.onnx.is_in_onnx_export():
                weight = self.weight if self.affine else None

                x = F.batch_norm(
                    input=x,
                    running_mean=mean,
                    running_var=var,
                    weight=weight,
                    bias=self.bias if self.affine else None,
                    training=False,
                    momentum=self.running_avg_momentum,
                    eps=self.epsilon,
                )
            else:
                std = torch.sqrt(var + self.epsilon).view(1, self.num_features, 1, 1)
                x = zeromean_x / std
                x = self._apply_affine_transform(x)
        else:
            x = self._apply_affine_transform(x)
        if mask is not None:
            x = x * mask
        return x


def build_norm2d_layer(norm: str, num_features=None, norm_groups=None):
    """Build a 2D normalization layer from a string identifier.

    Options:
        "bn" - Batch normalization (BatchNorm2d)
        "gn" - Group normalization, requires norm_groups parameter
        "gn-{N}" - Group normalization with N groups parsed from the string (e.g., "gn-8")
        "ln" - Local layer normalization (channel-wise)
        "mask" - Mask-aware fixup normalization without affine parameters
        "mask-affine" - Mask-aware fixup normalization with affine parameters
        "maskbn" - Mask-aware batch normalization with affine parameters
        "maskbn-noaffine" - Mask-aware batch normalization without affine parameters
        "none" - No normalization (returns None)
    """
    assert isinstance(num_features, int)
    if norm == "bn":
        return BatchNorm(num_features)
    elif norm == "gn":
        assert isinstance(norm_groups, int)
        return GroupNorm(norm_groups, num_features)
    elif norm.startswith("gn-"):
        norm_groups = int(norm[3:])
        return GroupNorm(norm_groups, num_features)
    elif norm == "ln":
        return LocalLayerNorm(num_features)
    elif norm == "mask":
        return MaskNorm(num_features, "fixup", affine=False)
    elif norm == "mask-affine":
        return MaskNorm(num_features, "fixup", affine=True)
    elif norm == "maskbn":
        return MaskNorm(num_features, "bnorm", affine=True)
    elif norm == "maskbn-noaffine":
        return MaskNorm(num_features, "bnorm", affine=False)
    elif norm == "none":
        return None
    else:
        raise ValueError(f"Unsupported normalization: {norm}")
