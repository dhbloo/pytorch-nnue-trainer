import torch.nn as nn

from . import MODELS
from .blocks import Conv2dBlock, build_activation_layer
from .input import build_input_plane
from .head import build_head


class ResBlock(nn.Module):
    def __init__(
        self,
        dim_in,
        ks=3,
        st=1,
        padding=1,
        pad_type="zeros",
        conv1_norm="none",
        conv2_norm="none",
        activation="relu",
        activation_first=False,
        dim_out=None,
        dim_hidden=None,
    ):
        super().__init__()
        dim_out = dim_out or dim_in
        dim_hidden = dim_hidden or min(dim_in, dim_out)

        self.learned_shortcut = dim_in != dim_out
        self.activation_first = activation_first
        self.activation = build_activation_layer(activation)
        self.conv = nn.Sequential(
            Conv2dBlock(
                dim_in,
                dim_hidden,
                ks,
                st,
                padding,
                conv1_norm,
                activation,
                pad_type,
                activation_first=activation_first,
            ),
            Conv2dBlock(
                dim_hidden,
                dim_out,
                ks,
                st,
                padding,
                conv2_norm,
                activation if activation_first else "none",
                pad_type,
                activation_first=activation_first,
            ),
        )
        if self.learned_shortcut:
            self.conv_shortcut = Conv2dBlock(
                dim_in,
                dim_out,
                ks=1,
                st=1,
                activation="none",
            )

    def forward(self, x, mask=None):
        shortcut = self.conv_shortcut(x) if self.learned_shortcut else x
        if mask is not None:
            x, mask = self.conv[0](x, mask)
            x, mask = self.conv[1](x, mask)
        else:
            x = self.conv[0](x)
            x = self.conv[1](x)
        x = x + shortcut
        if not self.activation_first and self.activation:
            x = self.activation(x)
        if mask is not None:
            return x, mask
        else:
            return x


@MODELS.register("resnet")
class ResNet(nn.Module):
    def __init__(
        self,
        num_blocks,
        dim_feature,
        head_type="v0",
        input_type="basic",
        input_kernel_size=3,
        input_stride=1,
        input_padding=1,
        input_activation="lrelu",
        trunk_kernel_size=3,
        trunk_stride=1,
        trunk_padding=1,
        trunk_norm="bn",
        trunk_activation="relu",
    ):
        super().__init__()
        self.model_size = (num_blocks, dim_feature)
        self.head_type = head_type
        self.input_type = input_type

        self.input_plane = build_input_plane(input_type)
        self.conv_input = Conv2dBlock(
            self.input_plane.dim_plane,
            dim_feature,
            ks=input_kernel_size,
            st=input_stride,
            padding=input_padding,
            activation=input_activation,
        )
        conv_trunk = []
        for _ in range(num_blocks):
            block = ResBlock(
                dim_feature,
                ks=trunk_kernel_size,
                st=trunk_stride,
                padding=trunk_padding,
                conv1_norm=trunk_norm,
                conv2_norm=trunk_norm,
                activation=trunk_activation,
            )
            conv_trunk.append(block)
        self.conv_trunk = nn.Sequential(*conv_trunk)
        self.output_head = build_head(head_type, dim_feature)

    def forward(self, data):
        input_plane = self.input_plane(data)
        feature = self.conv_input(input_plane)
        feature = self.conv_trunk(feature)
        return self.output_head(feature)

    @property
    def name(self):
        b, f = self.model_size
        return f"resnet_{self.input_type}_{b}b{f}f{self.head_type}"


@MODELS.register("resnetv2")
class ResNetv2(nn.Module):
    def __init__(
        self,
        num_blocks,
        dim_feature,
        head_type="v0",
        input_type="basic",
        input_kernel_size=5,
        input_stride=1,
        input_padding=2,
        trunk_kernel_size=3,
        trunk_stride=1,
        trunk_padding=1,
        trunk_norm1="bn",
        trunk_norm2="bn",
        trunk_activation="relu",
    ):
        super().__init__()
        self.model_size = (num_blocks, dim_feature)
        self.head_type = head_type
        self.input_type = input_type

        self.input_plane = build_input_plane(input_type)
        self.conv_input = Conv2dBlock(
            self.input_plane.dim_plane,
            dim_feature,
            ks=input_kernel_size,
            st=input_stride,
            padding=input_padding,
            activation="none",
        )
        conv_trunk = []
        for _ in range(num_blocks):
            block = ResBlock(
                dim_feature,
                ks=trunk_kernel_size,
                st=trunk_stride,
                padding=trunk_padding,
                conv1_norm=trunk_norm1,
                conv2_norm=trunk_norm2,
                activation=trunk_activation,
                activation_first=True,
            )
            conv_trunk.append(block)
        self.conv_trunk = nn.Sequential(*conv_trunk)
        self.output_head = build_head(head_type, dim_feature)

    def forward(self, data):
        input_plane = self.input_plane(data)
        feature = self.conv_input(input_plane)
        feature = self.conv_trunk(feature)
        return self.output_head(feature)

    @property
    def name(self):
        b, f = self.model_size
        return f"resnetv2_{self.input_type}_{b}b{f}f{self.head_type}"


@MODELS.register("resnetv3")
class ResNetv3(nn.Module):
    def __init__(
        self,
        num_blocks,
        dim_feature,
        head_type="v0",
        input_type="mask",
        input_kernel_size=5,
        input_stride=1,
        input_padding=2,
        trunk_kernel_size=3,
        trunk_stride=1,
        trunk_padding=1,
        trunk_norm1="maskbn",
        trunk_norm2="maskbn",
        trunk_activation="relu",
    ):
        super().__init__()
        self.model_size = (num_blocks, dim_feature)
        self.head_type = head_type
        self.input_type = input_type

        self.input_plane = build_input_plane(input_type)
        self.conv_input = Conv2dBlock(
            self.input_plane.dim_plane,
            dim_feature,
            ks=input_kernel_size,
            st=input_stride,
            padding=input_padding,
            norm="mask",
            activation="none",
        )
        self.conv_trunk = nn.ModuleList()
        for _ in range(num_blocks):
            block = ResBlock(
                dim_feature,
                ks=trunk_kernel_size,
                st=trunk_stride,
                padding=trunk_padding,
                conv1_norm=trunk_norm1 + "-nogamma",
                conv2_norm=trunk_norm2,
                activation=trunk_activation,
                activation_first=True,
            )
            self.conv_trunk.append(block)
        self.output_head = build_head(head_type + "-mask", dim_feature)

    def forward(self, data):
        input_plane, mask_plane = self.input_plane(data)
        x, mask = self.conv_input(input_plane, mask_plane)
        for conv_block in self.conv_trunk:
            x, mask = conv_block(x, mask)
        x = self.output_head(x, mask)
        return x

    @property
    def name(self):
        b, f = self.model_size
        return f"resnetv3_{self.input_type}_{b}b{f}f{self.head_type}"