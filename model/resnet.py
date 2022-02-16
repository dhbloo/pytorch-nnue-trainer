import torch
import torch.nn as nn
import torch.functional as F

from .blocks import Conv2dBlock, LinearBlock, ResBlock


def build_input_plane(input_type):
    if input_type == 'basic':
        return BasicInputPlane(with_stm=True)
    elif input_type == 'basic_no_stm':
        return BasicInputPlane(with_stm=False)
    elif input_type == 'raw':
        return lambda x: x  # identity transform
    else:
        assert 0, f"Unsupported input: {input_type}"


def build_head(head_type, dim_feature, dim_value=3, dim_policy=1):
    if head_type == 'v0':
        return OutputHeadV0(dim_feature, dim_value, dim_policy)
    else:
        assert 0, f"Unsupported head: {head_type}"


class BasicInputPlane(nn.Module):
    def __init__(self, with_stm=True):
        super().__init__()
        self.with_stm = with_stm

    def forward(self, data):
        board_input = data['board_input']
        stm_input = data['stm_input']

        if self.with_stm:
            B, C, H, W = board_input.shape
            stm_input = stm_input.reshape(B, 1, 1, 1).expand(B, 1, H, W)
            input_plane = torch.cat([board_input, stm_input], dim=1)
        else:
            input_plane = board_input

        return input_plane

    @property
    def dim_plane(self):
        return 2 + self.with_stm


class OutputHeadV0(nn.Module):
    def __init__(self, dim_feature, dim_value=3, dim_policy=1):
        super().__init__()
        self.value_head = LinearBlock(dim_feature, dim_value)
        self.policy_head = Conv2dBlock(dim_feature, dim_policy, ks=1, st=1)

    def forward(self, feature):
        # value head
        value = torch.mean(feature, dim=(2, 3))
        value = self.value_head(value)

        # policy head
        policy = self.policy_head(feature)
        policy = torch.squeeze(policy, dim=1)

        return value, policy


class ResNet(nn.Module):
    def __init__(self, num_blocks, dim_feature, head_type='v0', input_type='basic', **kwargs):
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
            *[ResBlock(dim_feature, norm='bn') for i in range(num_blocks)])
        self.output_head = build_head(head_type, dim_feature)

    def forward(self, data):
        input_plane = self.input_plane(data)
        feature = self.conv_input(input_plane)
        feature = self.conv_trunk(feature)
        return self.output_head(feature)

    @property
    def name(self):
        b, f = self.model_size
        return f"Resnet_{self.input_type}_{b}b{f}f{self.head_type}"
