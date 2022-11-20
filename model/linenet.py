import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from . import MODELS
from .blocks import Conv2dBlock, LinearBlock, ChannelWiseLeakyReLU


def get_total_num_line_encoding(length):
    half = length // 2
    code = 2 * 3**length
    for i in range(0, half + 1):
        code += 2 * 3**i
    for i in range(half + 2, length):
        code += 1 * 3**i
    return code + 1


@MODELS.register('line_nnue_v1')
class LineNNUEv1(nn.Module):
    def __init__(self, dim_policy=16, dim_value=32, map_max=32, line_length=11):
        super().__init__()
        self.model_size = (dim_policy, dim_value)
        self.map_max = map_max
        self.line_length = line_length
        self.line_encoding_total_num = get_total_num_line_encoding(line_length)
        dim_embed = dim_policy + dim_value

        self.mapping = nn.Embedding(self.line_encoding_total_num,
                                    dim_embed,
                                    scale_grad_by_freq=True)
        self.mapping_activation = ChannelWiseLeakyReLU(dim_embed, bound=6)

        # policy nets
        self.policy_conv = Conv2dBlock(dim_policy,
                                       dim_policy,
                                       ks=3,
                                       st=1,
                                       padding=1,
                                       groups=dim_policy)
        self.policy_linear = Conv2dBlock(dim_policy,
                                         1,
                                         ks=1,
                                         st=1,
                                         padding=0,
                                         activation='none',
                                         bias=False)
        self.policy_activation = ChannelWiseLeakyReLU(1, bias=False)

        # value nets
        self.value_activation = ChannelWiseLeakyReLU(dim_value, bias=False)
        self.value_linear = nn.Sequential(LinearBlock(dim_value, dim_value),
                                          LinearBlock(dim_value, dim_value))
        self.value_linear_final = LinearBlock(dim_value, 3, activation='none')

    def forward(self, data):
        dim_policy, _ = self.model_size
        assert torch.all(data['line_encoding_total_num'] == self.line_encoding_total_num)

        feature_index = data['line_encoding']  # [B, 4, H, W]
        feature = self.mapping(feature_index)  # [B, 4, H, W, PC+VC]
        feature = torch.permute(feature, (0, 1, 4, 2, 3))  # [B, 4, PC+VC, H, W]
        # resize feature to range [-map_max, map_max]
        if self.map_max != 0:
            feature = self.map_max * torch.tanh(feature / self.map_max)

        # average feature across four directions
        feature = torch.mean(feature, dim=1)  # [B, PC+VC, H, W]
        feature = self.mapping_activation(feature)

        # policy head
        policy = feature[:, :dim_policy]
        policy = self.policy_conv(policy)
        policy = self.policy_linear(policy)
        policy = self.policy_activation(policy)

        # value head
        value = torch.mean(feature[:, dim_policy:], dim=(2, 3))
        value = self.value_activation(value)
        value = self.value_linear(value)
        value = self.value_linear_final(value)

        return value, policy

    @property
    def name(self):
        p, v = self.model_size
        return f"linennuev1_len{self.line_length}_{p}p{v}v" + (f"-{self.map_max}mm"
                                                               if self.map_max != 0 else "")


@MODELS.register('line_nnue_v2')
class LineNNUEv2(nn.Module):
    LINE_LENGTH = 11

    def __init__(self,
                 dim_policy=16,
                 dim_value=32,
                 dim_dwconv=None,
                 dwconv_kernel_size=3,
                 map_max=32):
        super().__init__()
        self.model_size = (dim_policy, dim_value)
        self.map_max = map_max
        self.dwconv_kernel_size = dwconv_kernel_size
        dim_embed = max(dim_policy, dim_value)
        self.dim_dwconv = dim_embed if dim_dwconv is None else dim_dwconv
        self.line_encoding_total_num = get_total_num_line_encoding(self.LINE_LENGTH)

        self.mapping = nn.Embedding(self.line_encoding_total_num,
                                    dim_embed,
                                    scale_grad_by_freq=True)
        self.mapping_activation = nn.PReLU(dim_embed)

        # feature depth-wise conv
        self.feature_dwconv = Conv2dBlock(self.dim_dwconv,
                                          self.dim_dwconv,
                                          ks=dwconv_kernel_size,
                                          st=1,
                                          padding=dwconv_kernel_size // 2,
                                          groups=self.dim_dwconv)

        # policy head (point-wise conv)
        self.policy_pwconv = Conv2dBlock(dim_policy,
                                         1,
                                         ks=1,
                                         st=1,
                                         padding=0,
                                         activation='none',
                                         bias=False)
        self.policy_activation = nn.PReLU(1)

        # value head
        self.value_linear = nn.Sequential(LinearBlock(dim_value, dim_value),
                                          LinearBlock(dim_value, dim_value),
                                          LinearBlock(dim_value, 3, activation='none'))

    def forward(self, data):
        dim_policy, dim_value = self.model_size
        assert torch.all(data['line_encoding_total_num'] == self.line_encoding_total_num)

        feature_index = data['line_encoding']  # [B, 4, H, W]
        feature = self.mapping(feature_index)  # [B, 4, H, W, PC+VC]
        feature = torch.permute(feature, (0, 1, 4, 2, 3))  # [B, 4, PC+VC, H, W]
        # resize feature to range [-map_max, map_max]
        if self.map_max != 0:
            feature = self.map_max * torch.tanh(feature / self.map_max)

        # average feature across four directions
        feature = torch.mean(feature, dim=1)  # [B, PC+VC, H, W]
        feature = self.mapping_activation(feature)

        # feature conv
        feat_dwconv = self.feature_dwconv(feature[:, :self.dim_dwconv])  # [B, dwconv, H, W]
        feat_direct = feature[:, self.dim_dwconv:]  # [B, max(PC,VC)-dwconv, H, W]
        feature = torch.cat((feat_dwconv, feat_direct), dim=1)  # [B, max(PC,VC), H, W]

        # policy head
        policy = feature[:, :dim_policy]
        policy = self.policy_pwconv(policy)
        policy = self.policy_activation(policy)

        # value head
        value = feature[:, :dim_value]
        value = torch.mean(value, dim=(2, 3))
        value = self.value_linear(value)

        return value, policy

    @property
    def weight_clipping(self):
        # Clip prelu weight of mapping activation to [-1,1] to avoid overflow
        # In this range, prelu is the same as `max(x, ax)`.
        return [{
            'params': ['mapping_activation.weight'],
            'min_weight': -1.0,
            'max_weight': 1.0,
        }, {
            'params': ['feature_dwconv.conv.weight'],
            'min_weight': -2.0,
            'max_weight': 2.0,
        }]

    @property
    def name(self):
        p, v = self.model_size
        d, ks = self.dim_dwconv, self.dwconv_kernel_size
        return f"linennuev2_{p}p{v}v{d}d{ks}ks" + (f"-{self.map_max}mm"
                                                   if self.map_max != 0 else "")
