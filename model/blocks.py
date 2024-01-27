import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.quant_utils import fake_quant


def build_activation_layer(activation):
    if activation == 'relu':
        return nn.ReLU(inplace=True)
    elif activation == 'lrelu':
        return nn.LeakyReLU(0.1, inplace=True)
    elif activation.startswith('lrelu/'):  # custom slope
        neg_slope = 1.0 / int(activation[6:])
        return nn.LeakyReLU(neg_slope, inplace=True)
    elif activation == 'crelu':
        return ClippedReLU(inplace=True)
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation == 'silu':
        return nn.SiLU(inplace=True)
    elif activation == 'mish':
        return nn.Mish(inplace=True)
    elif activation == 'none':
        return None
    else:
        assert 0, f"Unsupported activation: {activation}"


def build_norm2d_layer(norm, norm_dim=None):
    if norm == 'bn':
        assert norm_dim is not None
        return nn.BatchNorm2d(norm_dim)
    elif norm == 'in':
        assert norm_dim is not None
        return nn.InstanceNorm2d(norm_dim)
    elif norm == 'none':
        return None
    else:
        assert 0, f"Unsupported normalization: {norm}"


class ClippedReLU(nn.Module):
    def __init__(self, inplace=False, max=1):
        super().__init__()
        self.inplace = inplace
        self.max = max

    def forward(self, x: torch.Tensor):
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
    def __init__(self,
                 num_parameters: int = 1,
                 init: float = 0.25,
                 input_quant_scale=128,
                 input_quant_bits=8,
                 weight_quant_scale=128,
                 weight_quant_bits=8,
                 weight_signed=True):
        super().__init__(num_parameters, init)
        self.input_quant_scale = input_quant_scale
        self.input_quant_bits = input_quant_bits
        self.weight_quant_scale = weight_quant_scale
        self.weight_quant_bits = weight_quant_bits
        self.weight_signed = weight_signed

    def forward(self, input: torch.Tensor) -> torch.Tensor:
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


class LinearBlock(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 norm='none',
                 activation='relu',
                 bias=True,
                 quant=False,
                 input_quant_scale=128,
                 input_quant_bits=8,
                 weight_quant_scale=128,
                 weight_quant_bits=8,
                 bias_quant_bits=32):
        super(LinearBlock, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias)
        self.quant = quant
        if quant:
            self.input_quant_scale = input_quant_scale
            self.input_quant_bits = input_quant_bits
            self.weight_quant_scale = weight_quant_scale
            self.weight_quant_bits = weight_quant_bits
            self.bias_quant_bits = bias_quant_bits

        # initialize normalization
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, f"Unsupported normalization: {norm}"

        # initialize activation
        self.activation = build_activation_layer(activation)

    def forward(self, x):
        if self.quant:
            # Using floor for inputs leads to closer results to the actual inference code
            x = fake_quant(x, self.input_quant_scale, num_bits=self.input_quant_bits, floor=True)
            w = fake_quant(self.fc.weight, self.weight_quant_scale, num_bits=self.weight_quant_bits)
            if self.fc.bias is not None:
                b = fake_quant(self.fc.bias,
                               self.weight_quant_scale * self.input_quant_scale,
                               num_bits=self.bias_quant_bits)
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


class Conv2dBlock(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 ks,
                 st,
                 padding=0,
                 norm='none',
                 activation='relu',
                 pad_type='zeros',
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
                 bias_quant_bits=32):
        super(Conv2dBlock, self).__init__()
        assert pad_type in ['zeros', 'reflect', 'replicate',
                            'circular'], f"Unsupported padding mode: {pad_type}"
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
            assert self.conv.padding_mode == 'zeros', "quant conv requires zero padding"
            self.input_quant_scale = input_quant_scale
            self.input_quant_bits = input_quant_bits
            self.weight_quant_scale = weight_quant_scale
            self.weight_quant_bits = weight_quant_bits
            self.bias_quant_scale = bias_quant_scale or (input_quant_scale * weight_quant_scale)
            self.bias_quant_bits = bias_quant_bits

    def forward(self, x):
        if self.activation_first:
            if self.norm:
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)
        if self.quant:
            x = fake_quant(x, self.input_quant_scale, num_bits=self.input_quant_bits)
            w = fake_quant(self.conv.weight,
                           self.weight_quant_scale,
                           num_bits=self.weight_quant_bits)
            b = self.conv.bias
            if b is not None:
                b = fake_quant(b, self.bias_quant_scale, num_bits=self.bias_quant_bits)
            x = F.conv2d(x, w, b, self.conv.stride, self.conv.padding, self.conv.dilation,
                         self.conv.groups)
        else:
            x = self.conv(x)
        if not self.activation_first:
            if self.norm:
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)
        return x


class Conv1dLine4Block(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 ks,
                 st,
                 padding=0,
                 norm='none',
                 activation='relu',
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
                 bias_quant_bits=32):
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
        self.bias = nn.Parameter(torch.zeros((out_dim, ))) if bias else None
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
                if i == j or i + j == self.kernel_size - 1 or i == self.kernel_size // 2 or j == self.kernel_size // 2:
                    kernel.append(self.weight[weight_index])
                    weight_index += 1
                else:
                    kernel.append(zero)

        assert weight_index == self.weight.size(0), f"{weight_index} != {self.weight.size(0)}"
        kernel = torch.stack(kernel, dim=2)
        kernel = kernel.reshape(self.out_dim, self.in_dim // self.groups, self.kernel_size,
                                self.kernel_size)
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


class ResBlock(nn.Module):
    def __init__(self,
                 dim_in,
                 ks=3,
                 st=1,
                 padding=1,
                 norm='none',
                 activation='relu',
                 pad_type='zeros',
                 dim_out=None,
                 dim_hidden=None,
                 activation_first=False):
        super(ResBlock, self).__init__()
        dim_out = dim_out or dim_in
        dim_hidden = dim_hidden or min(dim_in, dim_out)

        self.learned_shortcut = dim_in != dim_out
        self.activation_first = activation_first
        self.activation = build_activation_layer(activation)
        self.conv = nn.Sequential(
            Conv2dBlock(dim_in,
                        dim_hidden,
                        ks,
                        st,
                        padding,
                        norm,
                        activation,
                        pad_type,
                        activation_first=activation_first),
            Conv2dBlock(dim_hidden,
                        dim_out,
                        ks,
                        st,
                        padding,
                        norm,
                        activation if activation_first else 'none',
                        pad_type,
                        activation_first=activation_first),
        )
        if self.learned_shortcut:
            self.conv_shortcut = Conv2dBlock(dim_in,
                                             dim_out,
                                             1,
                                             1,
                                             norm=norm,
                                             activation=activation,
                                             activation_first=activation_first)

    def forward(self, x):
        residual = self.conv_shortcut(x) if self.learned_shortcut else x
        out = self.conv(x)
        out += residual
        if not self.activation_first and self.activation:
            out = self.activation(out)
        return out


class HashLayer(nn.Module):
    """Maps an input space of size=(input_level**input_size) to a hashed feature space of size=(2**hash_logsize)."""
    def __init__(self,
                 input_size,
                 input_level=2,
                 hash_logsize=20,
                 dim_feature=32,
                 quant_int8=True,
                 sub_features=1,
                 sub_divisor=2,
                 scale_grad_by_freq=False):
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
            self.register_buffer('idx_stride',
                                 input_level**torch.arange(input_size, dtype=torch.int64))
        else:
            n_features = 2**hash_logsize
            ii32 = torch.iinfo(torch.int32)
            self.hashs = nn.Parameter(torch.randint(ii32.min,
                                                    ii32.max,
                                                    size=(input_size, input_level),
                                                    dtype=torch.int32),
                                      requires_grad=False)

        n_features = [math.ceil(n_features / sub_divisor**i) for i in range(sub_features)]
        self.offsets = [sum(n_features[:i]) for i in range(sub_features + 1)]
        level_dim = max(dim_feature // sub_features, 1)
        self.features = nn.Embedding(sum(n_features),
                                     level_dim,
                                     scale_grad_by_freq=scale_grad_by_freq)
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
            x_onthot = torch.zeros((x_long.shape[0], self.input_size, self.input_level),
                                   dtype=torch.bool,
                                   device=x_long.device)
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
