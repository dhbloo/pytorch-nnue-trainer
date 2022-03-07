import torch.nn as nn

from .blocks import Conv2dBlock
from .input import build_input_plane
from .resnet import build_head


def depthwise_conv_block(in_dim, out_dim):
    return nn.Sequential(
        Conv2dBlock(
            in_dim,
            in_dim,
            ks=3,
            st=1,
            padding=1,
            bias=False,
            groups=in_dim,
            norm='bn',
            activation='relu',
        ),
        Conv2dBlock(
            in_dim,
            out_dim,
            ks=1,
            st=1,
            padding=0,
            bias=False,
            norm='bn',
            activation='relu',
        ),
    )


class InvertedResidual(nn.Module):
    def __init__(self, in_dim, out_dim, expand_ratio):
        super().__init__()
        hidden_dim = int(in_dim * expand_ratio)
        self.use_res_connect = in_dim == out_dim

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                Conv2dBlock(in_dim,
                            in_dim,
                            ks=3,
                            st=1,
                            padding=1,
                            groups=in_dim,
                            bias=False,
                            norm='bn',
                            activation='relu'),  # dw
                Conv2dBlock(in_dim, out_dim, ks=1, st=1, padding=0, bias=False,
                            norm='bn'),  # pw-linear
            )
        else:
            self.conv = nn.Sequential(
                Conv2dBlock(in_dim,
                            hidden_dim,
                            ks=1,
                            st=1,
                            padding=0,
                            bias=False,
                            norm='bn',
                            activation='relu'),  # pw
                Conv2dBlock(hidden_dim,
                            hidden_dim,
                            ks=3,
                            st=1,
                            padding=1,
                            groups=hidden_dim,
                            bias=False,
                            norm='bn',
                            activation='relu'),  # dw
                Conv2dBlock(hidden_dim, out_dim, ks=1, st=1, padding=0, bias=False,
                            norm='bn'),  # pw-linear
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV1(nn.Module):
    def __init__(self, num_blocks, dim_feature, head_type='v0', input_type='basic'):
        super().__init__()
        self.model_size = (num_blocks, dim_feature)
        self.head_type = head_type
        self.input_type = input_type

        self.input_plane = build_input_plane(input_type)
        self.conv_input = Conv2dBlock(self.input_plane.dim_plane,
                                      dim_feature,
                                      ks=3,
                                      st=1,
                                      padding=1,
                                      activation='lrelu')
        self.conv_trunk = nn.Sequential(
            *[depthwise_conv_block(dim_feature, dim_feature) for i in range(num_blocks)])
        self.output_head = build_head(head_type, dim_feature)

    def forward(self, data):
        input_plane = self.input_plane(data)
        feature = self.conv_input(input_plane)
        feature = self.conv_trunk(feature)
        return self.output_head(feature)

    @property
    def name(self):
        b, f = self.model_size
        return f"mobilenetv1_{self.input_type}_{b}b{f}f{self.head_type}"


class MobileNetV2(nn.Module):
    def __init__(self,
                 num_blocks,
                 dim_feature,
                 expand_ratio=6,
                 head_type='v0',
                 input_type='basic'):
        super().__init__()
        self.model_size = (num_blocks, dim_feature, expand_ratio)
        self.head_type = head_type
        self.input_type = input_type

        self.input_plane = build_input_plane(input_type)
        self.conv_input = Conv2dBlock(self.input_plane.dim_plane,
                                      dim_feature,
                                      ks=3,
                                      st=1,
                                      padding=1,
                                      activation='lrelu')
        self.conv_trunk = nn.Sequential(
            *[InvertedResidual(dim_feature, dim_feature, expand_ratio) for i in range(num_blocks)])
        self.output_head = build_head(head_type, dim_feature)

    def forward(self, data):
        input_plane = self.input_plane(data)
        feature = self.conv_input(input_plane)
        feature = self.conv_trunk(feature)
        return self.output_head(feature)

    @property
    def name(self):
        b, f, expand_ratio = self.model_size
        return f"mobilenetv2_{self.input_type}_{b}b{f}f{expand_ratio}t{self.head_type}"
