"""Reusable structural components shared by MixNet-family models."""

import torch
import torch.nn as nn

from .layers.activation import build_activation_layer
from .layers.container import SequentialWithExtraArguments
from .layers.directional import DirectionalConvLayer
from .layers.normalization import build_norm2d_layer


def _tuple_op(f, x):
    return tuple((f(xi) if xi is not None else None) for xi in x)


def _add_op(t):
    return None if (t[0] is None and t[1] is None) else t[0] + t[1]


class DirectionalConvResBlock(nn.Module):
    def __init__(self, dim, act, norm, **dirconv_kwargs):
        super().__init__()
        self.d_conv = DirectionalConvLayer(dim, dim, **dirconv_kwargs)
        self.norm = build_norm2d_layer(norm, dim)
        self.conv1x1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.activation = build_activation_layer(act)

    def forward(self, x, mask=None):
        residual = x
        x = self.d_conv(x)
        if self.norm is not None:
            x = _tuple_op(lambda t: self.norm(t) if mask is None else self.norm(t, mask=mask), x)
        x = _tuple_op(self.activation, x)
        x = _tuple_op(self.conv1x1, x)
        x = _tuple_op(self.activation, x)
        x = _tuple_op(_add_op, zip(x, residual))
        return x


class Conv0dResBlock(nn.Module):
    def __init__(self, dim, activation, middle_dim_multipler=1):
        super().__init__()
        middle_dim = middle_dim_multipler * dim
        self.conv1 = nn.Conv2d(dim, middle_dim, kernel_size=1)
        self.conv2 = nn.Conv2d(middle_dim, dim, kernel_size=1)
        self.activation = build_activation_layer(activation)

    def forward(self, x, mask=None):
        residual = x
        x = _tuple_op(self.conv1, x)
        x = _tuple_op(self.activation, x)
        x = _tuple_op(self.conv2, x)
        x = _tuple_op(self.activation, x)
        x = _tuple_op(_add_op, zip(x, residual))
        return x


class Mapping(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_middle,
        dim_out,
        use_channel_last=True,
        fix_direction_order=False,
        activation="silu",
        normalization="none",
        line_length=11,
    ):
        super().__init__()
        self.d_conv = DirectionalConvLayer(
            dim_in,
            dim_middle,
            use_nonzero_padding=False,
            use_channel_last=use_channel_last,
            fix_direction_order=fix_direction_order,
        )
        self.norm = build_norm2d_layer(normalization, dim_middle)
        self.convs = SequentialWithExtraArguments(
            *[
                DirectionalConvResBlock(
                    dim_middle,
                    activation,
                    normalization,
                    use_nonzero_padding=False,
                    use_channel_last=use_channel_last,
                    fix_direction_order=fix_direction_order,
                )
                for _ in range((line_length // 2) - 1)
            ],
            Conv0dResBlock(dim_middle, activation),
        )
        self.final_conv = nn.Conv2d(dim_middle, dim_out, kernel_size=1)
        self.activation = build_activation_layer(activation)

    def forward(self, x, mask=None, dirs=[0, 1, 2, 3]):
        x = tuple((x if i in dirs else None) for i in range(4))
        x = self.d_conv(x)
        if self.norm is not None:
            x = _tuple_op(lambda t: self.norm(t) if mask is None else self.norm(t, mask=mask), x)
        x = _tuple_op(self.activation, x)
        x = self.convs(x, mask=mask)
        x = _tuple_op(self.final_conv, x)
        x = tuple(xi for xi in x if xi is not None)
        return torch.stack(x, dim=1)  # [B, <=4, dim_out, H, W]


__all__ = [
    "Conv0dResBlock",
    "DirectionalConvResBlock",
    "Mapping",
]
