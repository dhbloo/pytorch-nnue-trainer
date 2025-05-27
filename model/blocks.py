import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from utils.quant_utils import fake_quant

# ---------------------------------------------------------


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


# ---------------------------------------------------------
# Normalization Layers


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


class MaskNorm(nn.Module):
    """
    Various kinds of normalization with masked input.
    This class is simplified from original implementation of Katago:
    https://github.com/lightvector/KataGo/blob/master/python/model_pytorch.py

    Available norm types:
    bnorm - batch norm
    fixup - fixup initialization https://arxiv.org/abs/1901.09321
    fixscale - fixed scaling initialization. Normalization layers simply multiply a constant scalar according
        to what batchnorm *would* do if all inputs were unit variance and all linear layers or convolutions
        preserved variance.
    fixscaleonenorm - fixed scaling normalization PLUS only have one batch norm layer in the entire net, at the end of the residual trunk.
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

        self.register_parameter("scale", None)
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
        self.scale = None
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
        self.reset_running_stats()

    def set_scale(self, scale: None | float):
        if scale is not None:
            self.scale = nn.Parameter(torch.full((), scale), requires_grad=False)
        else:
            self.scale = None

    def _compute_bnorm_stats(self, x: Tensor, mask: None | Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if mask is not None:
            # Compute mean and variance over the areas in the mask
            mask_sum = torch.sum(mask, dim=(0, 2, 3))
            mean = torch.sum(x * mask, dim=(0, 2, 3)) / mask_sum
            zeromean_x = x - mean.view(1, self.num_features, 1, 1)
            var = torch.sum((zeromean_x * mask).square(), dim=(0, 2, 3)) / mask_sum
        else:
            mean = torch.mean(x, dim=(0, 2, 3))
            zeromean_x = x - mean.view(1, self.num_features, 1, 1)
            var = torch.mean(zeromean_x.square(), dim=(0, 2, 3))
        return zeromean_x, mean, var

    def _apply_affine_transform(self, x: Tensor) -> Tensor:
        if self.scale is not None:
            x *= self.scale
        if self.affine:
            x = x * self.weight.view(1, self.num_features, 1, 1) + self.bias.view(1, self.num_features, 1, 1)
        return x

    def forward(self, x: Tensor, mask: None | Tensor) -> Tensor:
        if self.norm_type == "bnorm":
            assert x.ndim == 4 and x.shape[1] == self.num_features

            if self.training:
                zeromean_x, mean, var = self._compute_bnorm_stats(x, mask)
                with torch.no_grad():
                    self.running_mean += self.running_avg_momentum * (mean.detach() - self.running_mean)
                    self.running_var += self.running_avg_momentum * (var.detach() - self.running_var)
            else:
                mean, var = self.running_mean, self.running_var
                zeromean_x = x - mean.view(1, self.num_features, 1, 1)
            
            if not self.training and torch.onnx.is_in_onnx_export():
                if self.scale is not None:
                    if self.affine:
                        weight = self.scale * self.weight
                    else:
                        weight = self.scale.view(1).expand(self.num_features)
                else:
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
                x = self._apply_affine_transform(zeromean_x / std)
        else:
            x = self._apply_affine_transform(x)
        if mask is not None:
            x = x * mask
        return x
                

def build_norm2d_layer(norm: str, num_features=None, norm_groups=None):
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


# ---------------------------------------------------------
# Activation Layers


def build_activation_layer(activation, inplace=False):
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


# ---------------------------------------------------------
# Neural Network Layers


class LinearBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        activation="relu",
        bias=True,
        quant=False,
        input_quant_scale=128,
        input_quant_bits=8,
        weight_quant_scale=128,
        weight_quant_bits=8,
        bias_quant_bits=32,
    ):
        super(LinearBlock, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias)
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

        if self.activation:
            out = self.activation(out)
        return out


class Conv2dBlock(nn.Module):
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
        use_spectral_norm=False,
        quant=False,
        input_quant_scale=128,
        input_quant_bits=8,
        weight_quant_scale=128,
        weight_quant_bits=8,
        bias_quant_scale=None,
        bias_quant_bits=32,
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
        self.conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=ks,
            stride=st,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=pad_type,
        )

        if use_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

        self.quant = quant
        if quant:
            assert self.conv.padding_mode == "zeros", "quant conv requires zero padding"
            self.input_quant_scale = input_quant_scale
            self.input_quant_bits = input_quant_bits
            self.weight_quant_scale = weight_quant_scale
            self.weight_quant_bits = weight_quant_bits
            self.bias_quant_scale = bias_quant_scale or (input_quant_scale * weight_quant_scale)
            self.bias_quant_bits = bias_quant_bits

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
                assert isinstance(self.conv.padding, tuple)
                batch_size, _, h_in, w_in = x.shape
                p, d, k, s = self.conv.padding, self.conv.dilation, self.conv.kernel_size, self.conv.stride
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


class Conv1dLine4Block(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        ks,
        st,
        padding=0,
        norm="none",
        activation="relu",
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
    ):
        super().__init__()
        assert in_dim % groups == 0, f"in_dim({in_dim}) should be divisible by groups({groups})"
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = ks
        self.stride = st
        self.padding = padding  # zeros padding
        self.dilation = dilation
        self.groups = groups
        self.activation_first = activation_first
        self.norm = build_norm2d_layer(norm, out_dim)
        self.activation = build_activation_layer(activation)
        self.weight = nn.Parameter(torch.empty((4 * ks - 3, out_dim, in_dim // groups)))
        self.bias = nn.Parameter(torch.zeros((out_dim,))) if bias else None
        nn.init.kaiming_normal_(self.weight)

        self.quant = quant
        if quant:
            self.input_quant_scale = input_quant_scale
            self.input_quant_bits = input_quant_bits
            self.weight_quant_scale = weight_quant_scale
            self.weight_quant_bits = weight_quant_bits
            self.bias_quant_scale = bias_quant_scale or (input_quant_scale * weight_quant_scale)
            self.bias_quant_bits = bias_quant_bits

    def make_kernel(self):
        kernel = []
        weight_index = 0
        zero = torch.zeros_like(self.weight[0])
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                if (
                    i == j
                    or i + j == self.kernel_size - 1
                    or i == self.kernel_size // 2
                    or j == self.kernel_size // 2
                ):
                    kernel.append(self.weight[weight_index])
                    weight_index += 1
                else:
                    kernel.append(zero)

        assert weight_index == self.weight.size(0), f"{weight_index} != {self.weight.size(0)}"
        kernel = torch.stack(kernel, dim=2)
        kernel = kernel.reshape(self.out_dim, self.in_dim // self.groups, self.kernel_size, self.kernel_size)
        return kernel

    def conv(self, x):
        w = self.make_kernel()
        b = self.bias

        if self.quant:
            x = fake_quant(x, self.input_quant_scale, num_bits=self.input_quant_bits)
            w = fake_quant(w, self.weight_quant_scale, num_bits=self.weight_quant_bits)
            if b is not None:
                b = fake_quant(b, self.bias_quant_scale, num_bits=self.bias_quant_bits)

        return F.conv2d(x, w, b, self.stride, self.padding, self.dilation, self.groups)

    def forward(self, x):
        if self.activation_first:
            if self.norm:
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)
        x = self.conv(x)
        if not self.activation_first:
            if self.norm:
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)
        return x


class Conv2dSymBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        ks: int,
        st: int,
        padding=0,
        norm="none",
        activation="relu",
        bias=True,
        groups=1,
        activation_first=False,
        quant=False,
        input_quant_scale=128,
        input_quant_bits=8,
        weight_quant_scale=128,
        weight_quant_bits=8,
        bias_quant_scale=None,
        bias_quant_bits=32,
    ):
        super().__init__()
        self.activation_first = activation_first
        self.norm = build_norm2d_layer(norm, out_dim)
        self.activation = build_activation_layer(activation)

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.kernel_size = ks
        self.stride = st
        self.padding = padding  # zero padding
        self.groups = groups

        # Compute num_cells = 1 + 2 + ... + ks
        self.num_cells = ks * (ks + 1) // 2
        self.weight = nn.Parameter(torch.empty((self.num_cells, out_dim, in_dim // groups)))

        half_ks = (self.kernel_size + 1) // 2
        self.weight_index = []
        for y in range(self.kernel_size):
            for x in range(self.kernel_size):
                i, j = y, x
                if i >= half_ks:
                    i = self.kernel_size - i - 1
                if j >= half_ks:
                    j = self.kernel_size - j - 1
                if i > j:
                    i, j = j, i
                self.weight_index.append(i * half_ks + j - i * (i + 1) // 2)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_dim))
        else:
            self.register_parameter("bias", None)

        self.quant = quant
        if quant:
            self.input_quant_scale = input_quant_scale
            self.input_quant_bits = input_quant_bits
            self.weight_quant_scale = weight_quant_scale
            self.weight_quant_bits = weight_quant_bits
            self.bias_quant_scale = bias_quant_scale or (input_quant_scale * weight_quant_scale)
            self.bias_quant_bits = bias_quant_bits

    def reset_parameters(self):
        init_weight = torch.empty(
            (self.out_channels, self.in_channels // self.groups, self.kernel_size, self.kernel_size)
        )
        nn.init.kaiming_uniform_(init_weight, a=math.sqrt(5))

        # fill the weight tensor with the upper triangular part of the kernel
        half_ks = (self.kernel_size + 1) // 2
        for i in range(half_ks):
            for j in range(i, half_ks):
                idx = i * half_ks + j - i * (i + 1) // 2
                self.weight.data[idx, :, :] = init_weight[:, :, i, j]

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(init_weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def get_kernel_weight(self):
        kernel = [self.weight[i] for i in self.weight_index]
        kernel = torch.stack(kernel, dim=2)
        kernel = kernel.reshape(
            self.out_channels, self.in_channels // self.groups, self.kernel_size, self.kernel_size
        )
        return kernel

    def forward(self, x):
        if self.activation_first:
            if self.norm:
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)

        w = self.get_kernel_weight()
        b = self.bias

        if self.quant:
            x = fake_quant(x, self.input_quant_scale, num_bits=self.input_quant_bits)
            w = fake_quant(w, self.weight_quant_scale, num_bits=self.weight_quant_bits)
            if b is not None:
                b = fake_quant(b, self.bias_quant_scale, num_bits=self.bias_quant_bits)
            if (
                self.quant == "pixel-dwconv" or self.quant == "pixel-dwconv-floor"
            ):  # pixel-wise quantization in depthwise conv
                assert self.groups == x.size(1), "must be dwconv in pixel-dwconv quant mode!"
                batch_size, _, h_in, w_in = x.shape
                h_out, w_out = _convnd_out_size(
                    (h_in, w_in),
                    (self.kernel_size, self.kernel_size),
                    (self.stride, self.stride),
                    (self.padding, self.padding),
                    None,
                )
                x = F.unfold(x, self.kernel_size, 1, self.padding, self.stride)
                x = fake_quant(
                    x * w.view(-1)[None, :, None],
                    self.bias_quant_scale,
                    num_bits=self.bias_quant_bits,
                    floor=(self.quant == "pixel-dwconv-floor"),
                )
                x = x.reshape(batch_size, self.out_channels, -1, h_out * w_out).sum(2)
                x = F.fold(x, (h_out, w_out), (1, 1))
                if b is not None:
                    x = x + b[None, :, None, None]
            else:
                x = F.conv2d(x, w, b, self.stride, self.padding, 1, self.groups)
        else:
            x = F.conv2d(x, w, b, self.stride, self.padding, 1, self.groups)

        if not self.activation_first:
            if self.norm:
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)
        return x


class HashLayer(nn.Module):
    """Maps an input space of size=(input_level**input_size) to a hashed feature space of size=(2**hash_logsize)."""

    def __init__(
        self,
        input_size,
        input_level=2,
        hash_logsize=20,
        dim_feature=32,
        quant_int8=True,
        sub_features=1,
        sub_divisor=2,
        scale_grad_by_freq=False,
    ):
        super().__init__()
        self.input_size = input_size
        self.input_level = input_level
        self.hash_logsize = hash_logsize
        self.dim_feature = dim_feature
        self.quant_int8 = quant_int8
        self.sub_features = sub_features
        self.sub_divisor = sub_divisor

        self.perfect_hash = 2**hash_logsize >= input_level**input_size
        if self.perfect_hash:  # Do perfect hashing
            n_features = input_level**input_size
            self.register_buffer("idx_stride", input_level ** torch.arange(input_size, dtype=torch.int64))
        else:
            n_features = 2**hash_logsize
            ii32 = torch.iinfo(torch.int32)
            self.hashs = nn.Parameter(
                torch.randint(ii32.min, ii32.max, size=(input_size, input_level), dtype=torch.int32),
                requires_grad=False,
            )

        n_features = [math.ceil(n_features / sub_divisor**i) for i in range(sub_features)]
        self.offsets = [sum(n_features[:i]) for i in range(sub_features + 1)]
        level_dim = max(dim_feature // sub_features, 1)
        self.features = nn.Embedding(sum(n_features), level_dim, scale_grad_by_freq=scale_grad_by_freq)
        if self.quant_int8:
            nn.init.trunc_normal_(self.features.weight.data, std=0.5, a=-1, b=127 / 128)
        if self.sub_features > 1:
            self.feature_mapping = nn.Linear(level_dim * sub_features, dim_feature)

    def forward(self, x, x_long=None):
        """
        Args:
            x: float tensor of (batch_size, input_size), in range [0, 1].
            x_long: long tensor of (batch_size, input_size), in range [0, input_level-1].
                If not None, x_long will be used and x will be ignored.
        Returns:
            x_features: float tensor of (batch_size, dim_feature).
        """
        # Quantize input to [0, input_level-1] level
        if x_long is None:
            assert torch.all((x >= 0) & (x <= 1)), f"Input x should be in range [0, 1], but got {x}"
            x_long = torch.round((self.input_level - 1) * x).long()  # (batch_size, input_size)

        if self.perfect_hash:
            x_indices = torch.sum(x_long * self.idx_stride, dim=1)  # (batch_size,)
        else:
            x_onthot = torch.zeros(
                (x_long.shape[0], self.input_size, self.input_level), dtype=torch.bool, device=x_long.device
            )
            x_onthot.scatter_(2, x_long.unsqueeze(-1), 1)  # (batch_size, input_size, input_level)
            x_hash = x_onthot * self.hashs  # (batch_size, input_size, input_level)
            x_hash = torch.sum(x_hash, dim=(1, 2))  # (batch_size,)

            x_indices = x_hash % (2**self.hash_logsize)  # (batch_size,)

        if self.sub_features > 1:
            x_features = []
            for i in range(self.sub_features):
                assert torch.all(
                    x_indices < self.offsets[i + 1] - self.offsets[i]
                ), f"indices overflow: {i}, {(x_indices.min(), x_indices.max())}, {(self.offsets[i], self.offsets[i+1])}"
                x_features.append(self.features(x_indices + self.offsets[i]))
                x_indices = torch.floor_divide(x_indices, self.sub_divisor)
            x_features = torch.cat(x_features, dim=1)  # (batch_size, level_dim * sub_features)
            x_features = self.feature_mapping(x_features)  # (batch_size, dim_feature)
        else:
            x_features = self.features(x_indices)  # (batch_size, dim_feature)

        if self.quant_int8:
            x_features = fake_quant(torch.clamp(x_features, min=-1, max=127 / 128), scale=128)
        return x_features


# ---------------------------------------------------------
# Switch Variant of Networks


class SwitchGate(nn.Module):
    """Switch Gating for MoE networks"""

    def __init__(self, num_experts: int, jitter_eps=0.0, no_scaling=False) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.jitter_eps = jitter_eps
        self.no_scaling = no_scaling

    def forward(self, route_logits: Tensor) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        """
        Apply switch gating to routing logits.
        Args:
            route_logits: float tensor of (batch_size, num_experts).
        Returns:
            route_idx: routing index, long tensor of (batch_size,).
            route_multiplier: multipler for expert outputs, float tensor of (batch_size,).
            load_balancing_loss: load balancing loss, float tensor of ().
            aux_outputs: auxiliary outputs, dict.
        """
        # add random jittering when training
        if self.training and self.jitter_eps > 0:
            noise = torch.rand_like(route_logits)
            noise = noise * 2 * self.jitter_eps + 1 - self.jitter_eps
            route_logits = route_logits * noise

        # get routing probabilities and index
        route_probs = torch.softmax(route_logits, dim=1)  # [B, num_experts]
        route_prob_max, route_idx = torch.max(route_probs, dim=1)  # [B]

        # calc load balancing loss
        inv_batch_size = 1.0 / route_logits.shape[0]
        route_frac = torch.tensor(
            [(route_idx == i).sum() * inv_batch_size for i in range(self.num_experts)],
            dtype=route_probs.dtype,
            device=route_probs.device,
        )  # [num_experts]
        route_prob_mean = route_probs.mean(0)  # [num_experts]
        load_balancing_loss = self.num_experts * torch.dot(route_frac, route_prob_mean)
        load_balancing_loss = load_balancing_loss - 1.0

        if self.no_scaling:
            route_multiplier = route_prob_max / route_prob_max.detach()
        else:
            route_multiplier = route_prob_max

        aux_outputs = {
            "route_prob_max": route_prob_max,
            "route_frac_min": route_frac.min(),
            "route_frac_max": route_frac.max(),
            "route_frac_std": route_frac.std(),
        }
        return route_idx, route_multiplier, load_balancing_loss, aux_outputs


class SwitchLinear(nn.Module):
    """Switchable linear layer for MoE networks"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        assert num_experts > 0, "Number of experts must be at least 1"
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.weight = nn.Parameter(torch.empty((num_experts, out_features * in_features), **factory_kwargs))
        self.weight_fact = nn.Parameter(torch.empty((1, out_features * in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty((num_experts, out_features), **factory_kwargs))
            self.bias_fact = nn.Parameter(torch.empty((1, out_features), **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # We initialize the weights and biases to be uniform on all experts
        weight_fact = self.weight_fact.view(self.out_features, self.in_features)
        nn.init.kaiming_uniform_(weight_fact, a=math.sqrt(5))
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight_fact)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_fact, -bound, bound)
            nn.init.zeros_(self.bias)

    def get_weight_and_bias(self, route_index: Tensor) -> tuple[Tensor, None | Tensor]:
        batch_size = route_index.shape[0]
        expert_weight = F.embedding(route_index, self.weight) + self.weight_fact
        expert_weight = expert_weight.view(batch_size, self.out_features, self.in_features)
        if self.bias is not None:
            expert_bias = F.embedding(route_index, self.bias) + self.bias_fact
        else:
            expert_bias = None
        return expert_weight, expert_bias

    def get_weight_and_bias_at_idx(self, index: int) -> tuple[Tensor, None | Tensor]:
        expert_weight = self.weight[index] + self.weight_fact[0]
        expert_weight = expert_weight.view(self.out_features, self.in_features)
        if self.bias is not None:
            expert_bias = self.bias[index] + self.bias_fact[0]
        else:
            expert_bias = None
        return expert_weight, expert_bias

    def forward(self, input: Tensor, route_index: Tensor) -> Tensor:
        expert_weight, expert_bias = self.get_weight_and_bias(route_index)
        output = torch.einsum("bmn,bn->bm", expert_weight, input)
        if expert_bias is not None:
            output = output + expert_bias
        return output


class SwitchLinearBlock(nn.Module):
    """LinearBlock with switchable linear layer for MoE networks"""

    def __init__(
        self,
        in_dim,
        out_dim,
        num_experts,
        activation="relu",
        bias=True,
        quant=False,
        input_quant_scale=128,
        input_quant_bits=8,
        weight_quant_scale=128,
        weight_quant_bits=8,
        bias_quant_bits=32,
    ) -> None:
        super().__init__()
        self.fc = SwitchLinear(in_dim, out_dim, num_experts, bias)
        self.quant = quant
        if quant:
            self.input_quant_scale = input_quant_scale
            self.input_quant_bits = input_quant_bits
            self.weight_quant_scale = weight_quant_scale
            self.weight_quant_bits = weight_quant_bits
            self.bias_quant_bits = bias_quant_bits

        # initialize activation
        self.activation = build_activation_layer(activation)

    def forward(self, x: Tensor, route_index: Tensor):
        if self.quant:
            weight, bias = self.fc.get_weight_and_bias(route_index)
            x = fake_quant(x, self.input_quant_scale, num_bits=self.input_quant_bits)
            w = fake_quant(weight, self.weight_quant_scale, num_bits=self.weight_quant_bits)
            if bias is not None:
                b = fake_quant(
                    bias, self.weight_quant_scale * self.input_quant_scale, num_bits=self.bias_quant_bits
                )
                out = F.linear(x, w, b)
            else:
                out = F.linear(x, w)
        else:
            out = self.fc(x, route_index)

        if self.activation:
            out = self.activation(out)
        return out


class SwitchConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_experts: int,
        kernel_size: int,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        from torch.nn.modules.utils import _pair, _reverse_repeat_tuple

        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_experts = num_experts
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.padding_mode = padding_mode
        self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)

        self.weight_shape = (out_channels, in_channels // groups, *self.kernel_size)
        num_weight_params = math.prod(self.weight_shape)
        self.weight = nn.Parameter(torch.empty((num_experts, num_weight_params), **factory_kwargs))
        self.weight_fact = nn.Parameter(torch.empty((1, num_weight_params), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty((num_experts, out_channels), **factory_kwargs))
            self.bias_fact = nn.Parameter(torch.empty((1, out_channels), **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # We initialize the weights and biases to be uniform on all experts
        weight_fact = self.weight_fact.view(self.weight_shape)
        nn.init.kaiming_uniform_(weight_fact, a=math.sqrt(5))
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight_fact)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_fact, -bound, bound)
            nn.init.zeros_(self.bias)

    def get_weight_and_bias(self, route_index: Tensor) -> tuple[Tensor, None | Tensor]:
        batch_size = route_index.shape[0]
        expert_weight = F.embedding(route_index, self.weight) + self.weight_fact
        expert_weight = expert_weight.view(batch_size, *self.weight_shape)
        if self.bias is not None:
            expert_bias = F.embedding(route_index, self.bias) + self.bias_fact
        else:
            expert_bias = None
        return expert_weight, expert_bias

    def get_weight_and_bias_at_idx(self, index: int) -> tuple[Tensor, None | Tensor]:
        expert_weight = (self.weight[index] + self.weight_fact[0]).view(self.weight_shape)
        if self.bias is not None:
            expert_bias = self.bias[index] + self.bias_fact[0]
        else:
            expert_bias = None
        return expert_weight, expert_bias

    def forward(self, input: Tensor, route_index: Tensor):
        if self.padding_mode != "zeros":
            input = F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            padding = (0, 0)
        else:
            padding = self.padding

        batch_size = route_index.shape[0]
        expert_weight, expert_bias = self.get_weight_and_bias(route_index)
        expert_weight = expert_weight.reshape(-1, *self.weight_shape[1:])
        if expert_bias is not None:
            expert_bias = expert_bias.view(-1)

        input = input.view(1, -1, input.size(-2), input.size(-1))
        output = F.conv2d(
            input, expert_weight, expert_bias, self.stride, padding, self.dilation, self.groups * batch_size
        )
        output = output.view(batch_size, self.weight_shape[0], output.size(-2), output.size(-1))
        return output


class SwitchConv2dBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        num_experts,
        ks,
        st=1,
        padding=0,
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
    ):
        super().__init__()
        assert pad_type in [
            "zeros",
            "reflect",
            "replicate",
            "circular",
        ], f"Unsupported padding mode: {pad_type}"
        self.activation_first = activation_first
        self.activation = build_activation_layer(activation)
        self.conv = SwitchConv2d(
            in_channels=in_dim,
            out_channels=out_dim,
            num_experts=num_experts,
            kernel_size=ks,
            stride=st,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=pad_type,
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

    def forward(self, x: Tensor, route_index: Tensor):
        if self.activation and self.activation_first:
            x = self.activation(x)
        if self.quant:
            batch_size = x.shape[0]
            weight, bias = self.conv.get_weight_and_bias(route_index)
            x = fake_quant(x, self.input_quant_scale, num_bits=self.input_quant_bits)
            x = x.view(1, -1, *x.shape[2:])
            w = fake_quant(weight, self.weight_quant_scale, num_bits=self.weight_quant_bits)
            w = w.view(-1, *w.shape[2:])
            if bias is not None:
                bias = fake_quant(bias, self.bias_quant_scale, num_bits=self.bias_quant_bits)
                bias = bias.reshape(-1)
            x = F.conv2d(
                x,
                w,
                bias,
                self.conv.stride,
                self.conv.padding,
                self.conv.dilation,
                self.conv.groups * batch_size,
            )
            x = x.view(batch_size, -1, x.size(-2), x.size(-1))
        else:
            x = self.conv(x, route_index)
        if not self.activation_first and self.activation:
            x = self.activation(x)
        return x


# ---------------------------------------------------------
# Custom Containers


class SequentialWithExtraArguments(nn.Sequential):
    def forward(self, x, *args, **kwargs):
        for module in self:
            x = module(x, *args, **kwargs)
        return x
