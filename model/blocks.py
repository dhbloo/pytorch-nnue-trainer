import torch
import torch.nn as nn
import torch.nn.functional as F


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


class LinearBlock(nn.Module):

    def __init__(self, in_dim, out_dim, norm='none', activation='relu', bias=True):
        super(LinearBlock, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias)

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
                 use_spectral_norm=False):
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

    def forward(self, x):
        if self.activation and self.activation_first:
            x = self.activation(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation and not self.activation_first:
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
                 activation_first=False):
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

    def make_kernel(self):
        kernel = []
        weight_index = 0
        zero = torch.zeros_like(self.weight[0])
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                if i == j or i + j == self.kernel_size - 1 or \
                    i == self.kernel_size // 2 or j == self.kernel_size // 2:
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
        return F.conv2d(x, self.make_kernel(), self.bias, self.stride, self.padding, self.dilation,
                        self.groups)

    def forward(self, x):
        if self.activation and self.activation_first:
            x = self.activation(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation and not self.activation_first:
            x = self.activation(x)
        return x


class ResBlock(nn.Module):

    def __init__(self,
                 dim,
                 ks=3,
                 st=1,
                 padding=1,
                 norm='none',
                 activation='relu',
                 pad_type='zeros',
                 dim_hidden=None):
        super(ResBlock, self).__init__()
        dim_hidden = dim_hidden or dim

        self.activation = build_activation_layer(activation)
        self.conv = nn.Sequential(
            Conv2dBlock(dim, dim_hidden, ks, st, padding, norm, activation, pad_type),
            Conv2dBlock(dim_hidden, dim, ks, st, padding, norm, 'none', pad_type),
        )

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out += residual
        if self.activation:
            out = self.activation(out)
        return out


class ActFirstResBlock(nn.Module):

    def __init__(self,
                 dim_in,
                 dim_out,
                 dim_hidden=None,
                 ks=3,
                 st=1,
                 padding=1,
                 norm='none',
                 activation='relu',
                 pad_type='zeros'):
        super(ActFirstResBlock, self).__init__()
        dim_hidden = dim_hidden or min(dim_in, dim_out)

        self.learned_shortcut = dim_in != dim_out
        self.conv = nn.Sequential(
            Conv2dBlock(dim_in,
                        dim_hidden,
                        ks,
                        st,
                        padding,
                        norm,
                        activation,
                        pad_type,
                        activation_first=True),
            Conv2dBlock(dim_hidden,
                        dim_out,
                        ks,
                        st,
                        padding,
                        norm,
                        activation,
                        pad_type,
                        activation_first=True),
        )
        if self.learned_shortcut:
            self.conv_shortcut = Conv2dBlock(dim_in,
                                             dim_out,
                                             1,
                                             1,
                                             norm=norm,
                                             activation=activation,
                                             activation_first=True)

    def forward(self, x):
        xs = self.learned_shortcut(x) if self.learned_shortcut else x
        return self.conv(x) + xs
