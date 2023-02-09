import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Iterable

from . import MODELS
from .blocks import LinearBlock
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