import torch
import torch.nn as nn


def build_activation_layer(activation):
    if activation == 'relu':
        return nn.ReLU(inplace=True)
    elif activation == 'lrelu':
        return nn.LeakyReLU(0.1, inplace=True)
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
    elif activation == 'none':
        return None
    else:
        assert 0, f"Unsupported activation: {activation}"


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

        # initialize normalization
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, f"Unsupported normalization: {norm}"

        # initialize activation
        self.activation = build_activation_layer(activation)

        self.conv = nn.Conv2d(
            in_dim,
            out_dim,
            ks,
            st,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=pad_type,
        )

        if use_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

    def forward(self, x):
        if self.activation_first:
            if self.activation:
                x = self.activation(x)
            x = self.conv(x)
            if self.norm:
                x = self.norm(x)
        else:
            x = self.conv(x)
            if self.norm:
                x = self.norm(x)
            if self.activation:
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
