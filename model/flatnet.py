import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Iterable

from . import MODELS
from .blocks import LinearBlock, Conv2dBlock
from .input import build_input_plane


@MODELS.register('flat_nnue_v1')
class FlatNNUEv1(nn.Module):
    def __init__(self, flat_board_size, dim_policy=16, dim_value=32, input_type='basic-nostm'):
        super().__init__()
        if isinstance(flat_board_size, Iterable):
            assert len(flat_board_size) == 2, \
                f"flat_board_size should be a list of length 2, got {flat_board_size}"
            flat_board_size = flat_board_size[0] * flat_board_size[1]
        self.model_size = (dim_policy, dim_value)
        self.flat_board_size = flat_board_size
        dim_embed = dim_policy + dim_value

        self.input_plane = build_input_plane(input_type)
        self.embed = LinearBlock(flat_board_size * self.input_plane.dim_plane,
                                 dim_embed,
                                 activation="none")

        # policy head
        self.policy_key = nn.Sequential(
            LinearBlock(dim_policy, dim_policy, activation="relu"),
            LinearBlock(dim_policy, dim_policy, activation="none"),
        )
        self.policy_query = nn.Parameter(torch.randn(flat_board_size, dim_policy))

        # value head
        self.value = nn.Sequential(
            LinearBlock(dim_value, dim_value, activation="relu"),
            LinearBlock(dim_value, dim_value, activation="relu"),
            LinearBlock(dim_value, 3, activation="none"),
        )

    def forward(self, data):
        dim_policy, _ = self.model_size

        input_plane = self.input_plane(data)  # [B, C, H, W]
        input_plane = torch.permute(input_plane, (0, 2, 3, 1))  # [B, H, W, C]
        feature = self.embed(input_plane.flatten(start_dim=1))  # [B, dim_embed]

        # policy head
        policy_key = self.policy_key(feature[:, :dim_policy])  # [B, dim_policy]
        policy = torch.matmul(policy_key, self.policy_query.t())  # [B, flat_board_size]

        # value head
        value = self.value(feature[:, dim_policy:])  # [B, 3]

        return value, policy

    @property
    def name(self):
        p, v = self.model_size
        return f"flat_nnue_v1_{self.flat_board_size}fbs{p}p{v}v"


class LadderConvLayer(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((3, dim_out, dim_in)))
        self.bias = nn.Parameter(torch.zeros((dim_out, )))
        nn.init.kaiming_normal_(self.weight)

    def forward(self, x):
        w = self.weight
        _, dim_out, dim_in = w.shape
        zero = torch.zeros((dim_out, dim_in), dtype=w.dtype, device=w.device)
        weight = [w[0], zero, w[1], w[2]]
        weight = torch.stack(weight, dim=2)
        weight = weight.reshape(dim_out, dim_in, 2, 2)
        return F.conv2d(x, weight, self.bias, padding=0)


@MODELS.register('flat_ladder7x7_nnue_v1')
class FlatLadder7x7NNUEv1(nn.Module):
    def __init__(self, dim_middle=128, dim_policy=16, dim_value=32, input_type='basic-nostm'):
        super().__init__()
        self.model_size = (dim_middle, dim_policy, dim_value)
        dim_mapping = dim_policy + dim_value

        self.input_plane = build_input_plane(input_type)
        self.mapping = nn.Sequential(
            LadderConvLayer(self.input_plane.dim_plane, dim_middle),
            nn.Mish(inplace=True),
            Conv2dBlock(dim_middle, dim_middle, ks=1, st=1, norm="bn", activation="mish"),
            LadderConvLayer(dim_middle, dim_middle),
            nn.Mish(inplace=True),
            Conv2dBlock(dim_middle, dim_middle, ks=1, st=1, norm="bn", activation="mish"),
            LadderConvLayer(dim_middle, dim_mapping),
        )

        # policy head
        if dim_policy > 0:
            self.policy_key = nn.Sequential(
                LinearBlock(dim_policy, dim_policy, activation="relu"),
                LinearBlock(dim_policy, dim_policy, activation="none"),
            )
            self.policy_query = nn.Parameter(torch.randn(7 * 7, dim_policy))

        # value head
        self.value_linears = nn.ModuleList([
            LinearBlock(dim_value, dim_value, activation="none"),
            LinearBlock(dim_value, dim_value, activation="none"),
            LinearBlock(dim_value, 3, activation="none"),
        ])

    def get_feature_sum(self, data):
        input_plane = self.input_plane(data)  # [B, C, H, W]

        index = [
            [0, 4, 0, 4, 0],
            [0, 4, 0, 4, 2],
            [0, 4, 3, 7, 1],
            [0, 4, 3, 7, -1],
            [3, 7, 3, 7, 2],
            [3, 7, 3, 7, 0],
            [3, 7, 0, 4, -1],
            [3, 7, 0, 4, 1],
        ]
        features = []
        for y0, y1, x0, x1, k in index:
            chunk = torch.rot90(input_plane[:, :, y0:y1, x0:x1], k, (2, 3))
            feat = torch.clamp(self.mapping(chunk), min=-1, max=127 / 128)
            features.append(feat)

        return torch.sum(torch.stack(features), dim=0).squeeze(-1).squeeze(-1)

    def forward(self, data):
        _, dim_policy, _ = self.model_size

        # get feature sum from chunks
        feature = self.get_feature_sum(data)  # [B, dim_mapping]

        # policy head
        if dim_policy > 0:
            policy_key = self.policy_key(feature[:, :dim_policy])  # [B, dim_policy]
            policy = torch.matmul(policy_key, self.policy_query.t())  # [B, 7*7]
            policy = policy.view(-1, 7, 7)  # [B, 7, 7]
        else:
            policy = torch.zeros((feature.shape[0], 7, 7),
                                 dtype=feature.dtype,
                                 device=feature.device)

        # value head
        value = feature[:, dim_policy:]  # [B, dim_value]
        for layer in self.value_linears:
            value = torch.clamp(value, min=0, max=127 / 128)
            value = layer(value)

        return value, policy

    @property
    def weight_clipping(self):
        return [
            {
                'params': ['value.0.weight', 'value.1.weight', 'value.2.weight'],
                'min_weight': -128 / 64,
                'max_weight': 127 / 64
            },
        ]

    @property
    def name(self):
        m, p, v = self.model_size
        return f"flat_ladder7x7_nnue_v1_{m}m{p}p{v}v"