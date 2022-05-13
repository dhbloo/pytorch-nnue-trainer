import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from . import MODELS
from .blocks import Conv2dBlock, LinearBlock


def get_total_num_line_encoding(length):
    half = length // 2
    code = 2 * 3**length
    for i in range(0, half + 1):
        code += 2 * 3**i
    for i in range(half + 2, length):
        code += 1 * 3**i
    return code + 1


class ChannelWiseLeakyReLU(nn.Module):
    def __init__(self, dim, bias=True, bound=0):
        super().__init__()
        self.neg_slope = Parameter(torch.ones(dim) * 0.5)
        self.bias = Parameter(torch.zeros(dim)) if bias else None
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
