import torch.nn as nn

from . import MODELS
from .blocks import Conv2dBlock, ResBlock
from .input import build_input_plane
from .head import build_head


@MODELS.register('resnet')
class ResNet(nn.Module):
    def __init__(self,
                 num_blocks,
                 dim_feature,
                 head_type='v0',
                 input_type='basic',
                 activation_first=False):
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
        self.conv_trunk = nn.Sequential(*[
            ResBlock(dim_feature, norm='bn', activation_first=activation_first)
            for i in range(num_blocks)
        ])
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
