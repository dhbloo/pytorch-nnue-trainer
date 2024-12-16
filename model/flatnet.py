import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

from . import MODELS
from .blocks import LinearBlock, Conv2dBlock, HashLayer
from .input import build_input_plane
from .vq import VectorQuantize
from utils.quant_utils import fake_quant


@MODELS.register("flat_nnue_v1")
class FlatNNUEv1(nn.Module):
    def __init__(self, flat_board_size, dim_policy=16, dim_value=32, input_type="basic-nostm"):
        super().__init__()
        if isinstance(flat_board_size, Iterable):
            assert len(flat_board_size) == 2, f"flat_board_size should be a list of length 2, got {flat_board_size}"
            flat_board_size = flat_board_size[0] * flat_board_size[1]
        self.model_size = (dim_policy, dim_value)
        self.flat_board_size = flat_board_size
        dim_embed = dim_policy + dim_value

        self.input_plane = build_input_plane(input_type)
        self.embed = LinearBlock(flat_board_size * self.input_plane.dim_plane, dim_embed, activation="none")

        # policy head
        if dim_policy > 0:
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
        if dim_policy > 0:
            policy_key = self.policy_key(feature[:, :dim_policy])  # [B, dim_policy]
            policy = torch.matmul(policy_key, self.policy_query.t())  # [B, flat_board_size]
        else:
            policy = torch.zeros((feature.shape[0], self.flat_board_size), dtype=feature.dtype, device=feature.device)

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
        self.bias = nn.Parameter(torch.zeros((dim_out,)))
        nn.init.kaiming_normal_(self.weight)

    def forward(self, x):
        w = self.weight
        _, dim_out, dim_in = w.shape
        zero = torch.zeros((dim_out, dim_in), dtype=w.dtype, device=w.device)
        weight = [w[0], zero, w[1], w[2]]
        weight = torch.stack(weight, dim=2)
        weight = weight.reshape(dim_out, dim_in, 2, 2)
        return F.conv2d(x, weight, self.bias, padding=0)


@MODELS.register("flat_ladder7x7_nnue_v1")
class FlatLadder7x7NNUEv1(nn.Module):
    def __init__(self, dim_middle=128, dim_policy=16, dim_value=32, input_type="basic-nostm", value_no_draw=False):
        super().__init__()
        self.model_size = (dim_middle, dim_policy, dim_value)
        self.value_no_draw = value_no_draw
        dim_mapping = dim_policy + dim_value

        self.input_plane = build_input_plane(input_type)
        self.mapping = nn.Sequential(
            LadderConvLayer(self.input_plane.dim_plane, dim_middle),
            nn.Mish(inplace=True),
            Conv2dBlock(dim_middle, dim_middle, ks=1, st=1, norm="bn", activation="mish"),
            LadderConvLayer(dim_middle, dim_middle),
            nn.Mish(inplace=True),
            Conv2dBlock(dim_middle, dim_middle, ks=1, st=1, norm="bn", activation="mish"),
            LadderConvLayer(dim_middle, dim_middle),
            nn.Mish(inplace=True),
            Conv2dBlock(dim_middle, dim_middle, ks=1, st=1, norm="bn", activation="mish"),
            Conv2dBlock(dim_middle, dim_mapping, ks=1, st=1, norm="bn", activation="none"),
        )

        # policy head
        if dim_policy > 0:
            self.policy_key = nn.Sequential(
                LinearBlock(dim_policy, dim_policy, activation="relu"),
                LinearBlock(dim_policy, dim_policy, activation="none"),
            )
            self.policy_query = nn.Parameter(torch.randn(7 * 7, dim_policy))

        # value head
        dim_vhead = 1 if value_no_draw else 3
        self.value_linears = nn.ModuleList(
            [
                LinearBlock(dim_value, 32, activation="none", quant=True),
                LinearBlock(32, 32, activation="none", quant=True),
                LinearBlock(32, 32, activation="none", quant=True),
                LinearBlock(32, dim_vhead, activation="none", quant=True),
            ]
        )

    def get_feature_sum(self, data):
        input_plane = self.input_plane(data)  # [B, C, H, W]

        index = [
            [0, 4, 0, 4, 0, False],
            [0, 4, 0, 4, 0, True],
            [0, 4, 3, 7, 1, False],
            [0, 4, 3, 7, -1, True],
            [3, 7, 3, 7, 2, False],
            [3, 7, 3, 7, 2, True],
            [3, 7, 0, 4, -1, False],
            [3, 7, 0, 4, 1, True],
        ]
        features = []
        for y0, y1, x0, x1, k, t in index:
            chunk = input_plane[:, :, y0:y1, x0:x1]
            if t:
                chunk = torch.transpose(chunk, 2, 3)
            chunk = torch.rot90(chunk, k, (2, 3))
            feat = torch.clamp(self.mapping(chunk), min=-1, max=127 / 128)
            feat = fake_quant(feat, scale=128)
            features.append(feat.squeeze(-1).squeeze(-1))

        return torch.sum(torch.stack(features), dim=0)

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
            policy = torch.zeros((feature.shape[0], 7, 7), dtype=feature.dtype, device=feature.device)

        # value head
        value = feature[:, dim_policy:]  # [B, dim_value]
        for layer in self.value_linears:
            value = torch.clamp(value, min=0, max=127 / 128)
            value = layer(value)

        return value, policy

    def forward_debug_print(self, data):
        _, dim_policy, _ = self.model_size

        # get feature sum from chunks
        feature = self.get_feature_sum(data)  # [B, dim_mapping]
        print(f"feature sum: \n{(feature * 128).int()}")

        # policy head
        if dim_policy > 0:
            policy_key = self.policy_key(feature[:, :dim_policy])  # [B, dim_policy]
            policy = torch.matmul(policy_key, self.policy_query.t())  # [B, 7*7]
            policy = policy.view(-1, 7, 7)  # [B, 7, 7]
        else:
            policy = torch.zeros((feature.shape[0], 7, 7), dtype=feature.dtype, device=feature.device)

        # value head
        value = feature[:, dim_policy:]  # [B, dim_value]
        for i, layer in enumerate(self.value_linears):
            value = torch.clamp(value, min=0, max=127 / 128)
            print(f"value input{i+1}: \n{(value * 128).int()}")
            value = layer(value)
            print(f"value output{i+1}: \n{(value * 128).int()}")

        return value, policy

    @property
    def weight_clipping(self):
        return [
            {
                "params": [f"value_linears.{i}.fc.weight" for i in range(len(self.value_linears))],
                "min_weight": -128 / 128,
                "max_weight": 127 / 128,
            },
        ]

    @property
    def name(self):
        m, p, v = self.model_size
        return f"flat_ladder7x7_nnue_v1_{m}m{p}p{v}v"


@MODELS.register("flat_ladder7x7_nnue_v2")
class FlatLadder7x7NNUEv2(nn.Module):
    def __init__(self, dim_middle=128, dim_policy=16, dim_value=32, input_type="basic-nostm", value_no_draw=False):
        super().__init__()
        self.model_size = (dim_middle, dim_policy, dim_value)
        self.value_no_draw = value_no_draw
        self.input_plane = build_input_plane(input_type)
        self.mapping1 = self._make_ladder_mapping(dim_middle, dim_policy + dim_value)
        self.mapping2 = self._make_ladder_mapping(dim_middle, dim_policy + dim_value)

        # policy head
        if dim_policy > 0:
            self.policy_key = nn.Sequential(
                LinearBlock(dim_policy, dim_policy, activation="relu"),
                LinearBlock(dim_policy, dim_policy, activation="none"),
            )
            self.policy_query = nn.Parameter(torch.randn(7 * 7, dim_policy))

        # value head
        dim_vhead = 1 if value_no_draw else 3
        self.value_linears = nn.ModuleList(
            [
                LinearBlock(dim_value, 32, activation="none", quant=True),
                LinearBlock(32, 32, activation="none", quant=True),
                LinearBlock(32, 32, activation="none", quant=True),
                LinearBlock(32, dim_vhead, activation="none", quant=True),
            ]
        )

    def _make_ladder_mapping(self, dim_middle, dim_mapping):
        return nn.Sequential(
            LadderConvLayer(self.input_plane.dim_plane, dim_middle),
            nn.Mish(inplace=True),
            Conv2dBlock(dim_middle, dim_middle, ks=1, st=1, norm="bn", activation="mish"),
            LadderConvLayer(dim_middle, dim_middle),
            nn.Mish(inplace=True),
            Conv2dBlock(dim_middle, dim_middle, ks=1, st=1, norm="bn", activation="mish"),
            LadderConvLayer(dim_middle, dim_middle),
            nn.Mish(inplace=True),
            Conv2dBlock(dim_middle, dim_middle, ks=1, st=1, norm="bn", activation="mish"),
            Conv2dBlock(dim_middle, dim_mapping, ks=1, st=1, norm="bn", activation="none"),
        )

    def get_feature_sum(self, data):
        input_plane = self.input_plane(data)  # [B, C, H, W]
        features = []

        mapping1_index = [
            [0, 4, 0, 4, 0, False],
            [0, 4, 0, 4, 0, True],
            [0, 4, 3, 7, 1, False],
            [0, 4, 3, 7, -1, True],
            [3, 7, 3, 7, 2, False],
            [3, 7, 3, 7, 2, True],
            [3, 7, 0, 4, -1, False],
            [3, 7, 0, 4, 1, True],
        ]
        for y0, y1, x0, x1, k, t in mapping1_index:
            chunk = input_plane[:, :, y0:y1, x0:x1]
            if t:
                chunk = torch.transpose(chunk, 2, 3)
            chunk = torch.rot90(chunk, k, (2, 3))
            feat = torch.clamp(self.mapping1(chunk), min=-1, max=127 / 128)
            feat = fake_quant(feat, scale=128)
            features.append(feat.squeeze(-1).squeeze(-1))

        mapping2_index = [
            [0, 4, 0, 4, 1],
            [0, 4, 3, 7, 2],
            [3, 7, 3, 7, -1],
            [3, 7, 0, 4, 0],
        ]
        for y0, y1, x0, x1, k in mapping2_index:
            chunk = torch.rot90(input_plane[:, :, y0:y1, x0:x1], k, (2, 3))
            feat = torch.clamp(self.mapping2(chunk), min=-1, max=127 / 128)
            feat = fake_quant(feat, scale=128)
            features.append(feat.squeeze(-1).squeeze(-1))

        return torch.sum(torch.stack(features), dim=0)

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
            policy = torch.zeros((feature.shape[0], 7, 7), dtype=feature.dtype, device=feature.device)

        # value head
        value = feature[:, dim_policy:]  # [B, dim_value]
        for layer in self.value_linears:
            value = torch.clamp(value, min=0, max=127 / 128)
            value = layer(value)

        return value, policy

    def forward_debug_print(self, data):
        _, dim_policy, _ = self.model_size

        # get feature sum from chunks
        feature = self.get_feature_sum(data)  # [B, dim_mapping]
        print(f"feature sum: \n{(feature * 128).int()}")

        # policy head
        if dim_policy > 0:
            policy_key = self.policy_key(feature[:, :dim_policy])  # [B, dim_policy]
            policy = torch.matmul(policy_key, self.policy_query.t())  # [B, 7*7]
            policy = policy.view(-1, 7, 7)  # [B, 7, 7]
        else:
            policy = torch.zeros((feature.shape[0], 7, 7), dtype=feature.dtype, device=feature.device)

        # value head
        value = feature[:, dim_policy:]  # [B, dim_value]
        for i, layer in enumerate(self.value_linears):
            value = torch.clamp(value, min=0, max=127 / 128)
            print(f"value input{i+1}: \n{(value * 128).int()}")
            value = layer(value)
            print(f"value output{i+1}: \n{(value * 128).int()}")

        return value, policy

    @property
    def weight_clipping(self):
        return [
            {
                "params": [f"value_linears.{i}.fc.weight" for i in range(len(self.value_linears))],
                "min_weight": -128 / 128,
                "max_weight": 127 / 128,
            },
        ]

    @property
    def name(self):
        m, p, v = self.model_size
        return f"flat_ladder7x7_nnue_v2_{m}m{p}p{v}v"


@MODELS.register("flat_ladder7x7_nnue_v3")
class FlatLadder7x7NNUEv3(nn.Module):
    def __init__(
        self,
        dim_middle=128,
        dim_global_feature=128,
        dim_conv_feature=32,
        input_type="basic-nostm",
        value_no_draw=False,
    ):
        super().__init__()
        self.model_size = (dim_middle, dim_global_feature, dim_conv_feature)
        self.value_no_draw = value_no_draw
        self.input_plane = build_input_plane(input_type)
        self.mapping1 = self._make_ladder_mapping(dim_middle, dim_global_feature)
        self.mapping2 = self._make_ladder_mapping(dim_middle, dim_global_feature)
        self.convs = self._make_conv3x3_mappings(6, dim_middle, dim_conv_feature)

        # value head
        dim_vhead = 1 if value_no_draw else 3
        self.value_linears = nn.ModuleList(
            [
                LinearBlock(dim_global_feature + dim_conv_feature, 32, activation="none", quant=True),
                LinearBlock(32, 32, activation="none", quant=True),
                LinearBlock(32, dim_vhead, activation="none", quant=True),
            ]
        )

    def _make_ladder_mapping(self, dim_middle, dim_mapping):
        return nn.Sequential(
            LadderConvLayer(self.input_plane.dim_plane, dim_middle),
            nn.BatchNorm2d(dim_middle),
            nn.Mish(inplace=True),
            Conv2dBlock(dim_middle, dim_middle, ks=1, st=1, norm="bn", activation="mish"),
            LadderConvLayer(dim_middle, dim_middle),
            nn.BatchNorm2d(dim_middle),
            nn.Mish(inplace=True),
            Conv2dBlock(dim_middle, dim_middle, ks=1, st=1, norm="bn", activation="mish"),
            LadderConvLayer(dim_middle, dim_middle),
            nn.BatchNorm2d(dim_middle),
            nn.Mish(inplace=True),
            Conv2dBlock(dim_middle, dim_middle, ks=1, st=1, norm="bn", activation="mish"),
            Conv2dBlock(dim_middle, dim_mapping, ks=1, st=1, norm="bn", activation="none"),
        )

    def _make_conv3x3_mappings(self, num_conv, dim_middle, dim_mapping):
        conv_list = nn.ModuleList()
        for _ in range(num_conv):
            conv_list.append(
                nn.Sequential(
                    Conv2dBlock(self.input_plane.dim_plane, dim_middle, ks=2, st=1, norm="bn", activation="mish"),
                    Conv2dBlock(dim_middle, dim_middle, ks=2, st=1, norm="bn", activation="mish"),
                    Conv2dBlock(dim_middle, dim_middle, ks=1, st=1, norm="bn", activation="mish"),
                    Conv2dBlock(dim_middle, dim_mapping, ks=1, st=1, norm="bn", activation="none"),
                )
            )
        return conv_list

    def get_phase_plane(self, data):
        input_plane = self.input_plane(data)  # [B, C, H, W]
        empty_plane = (input_plane[:, 0] + input_plane[:, 1]) != 0
        empty_count = torch.sum(empty_plane, dim=(1, 2))  # [B]
        phase = torch.div(49 - empty_count, 49.0)  # [B]
        x = phase * 2 - 1  # scale [0,1] to [-1,1]
        return x[:, None, None, None].expand(-1, -1, 7, 7)  # [B, 1, 7, 7]

    def get_feature_sum(self, data):
        input_plane = self.input_plane(data)  # [B, C, H, W]
        features = []
        conv_features = []

        mapping1_index = [
            [0, 4, 0, 4, 0, False],
            [0, 4, 0, 4, 0, True],
            [0, 4, 3, 7, 1, False],
            [0, 4, 3, 7, -1, True],
            [3, 7, 3, 7, 2, False],
            [3, 7, 3, 7, 2, True],
            [3, 7, 0, 4, -1, False],
            [3, 7, 0, 4, 1, True],
        ]
        for y0, y1, x0, x1, k, t in mapping1_index:
            chunk = input_plane[:, :, y0:y1, x0:x1]
            if t:
                chunk = torch.transpose(chunk, 2, 3)
            chunk = torch.rot90(chunk, k, (2, 3))
            feat = torch.clamp(self.mapping1(chunk), min=-1, max=127 / 128)
            feat = fake_quant(feat, scale=128)
            features.append(feat.squeeze(-1).squeeze(-1))

        mapping2_index = [
            [0, 4, 0, 4, 1],
            [0, 4, 3, 7, 2],
            [3, 7, 3, 7, -1],
            [3, 7, 0, 4, 0],
        ]
        for y0, y1, x0, x1, k in mapping2_index:
            chunk = torch.rot90(input_plane[:, :, y0:y1, x0:x1], k, (2, 3))
            feat = torch.clamp(self.mapping2(chunk), min=-1, max=127 / 128)
            feat = fake_quant(feat, scale=128)
            features.append(feat.squeeze(-1).squeeze(-1))

        conv_index = [
            # line 1
            [0, 0, False],
            [1, 0, False],
            [2, 0, False],
            [1, -1, True],
            [0, -1, False],
            # line 2
            [1, 0, False],
            [3, 0, False],
            [4, 0, False],
            [3, -1, False],
            [1, -1, True],
            # line 3
            [2, -1, False],
            [4, -1, False],
            [5, 0, False],
            [4, 1, False],
            [2, 1, False],
            # line 4
            [1, -1, False],
            [3, -1, False],
            [4, 2, False],
            [3, 2, False],
            [1, 2, True],
            # line 5
            [0, -1, False],
            [1, 1, True],
            [2, 2, False],
            [1, 2, False],
            [0, 2, False],
        ]
        for (y, x), (conv_idx, k, t) in zip([(y, x) for y in range(0, 5) for x in range(0, 5)], conv_index):
            chunk = input_plane[:, :, y : y + 3, x : x + 3]
            if t:
                chunk = torch.transpose(chunk, 2, 3)
            chunk = torch.rot90(chunk, k, (2, 3))
            feat = torch.clamp(self.convs[conv_idx](chunk), min=-1, max=127 / 128)
            feat = fake_quant(feat, scale=128)
            conv_features.append(feat.squeeze(-1).squeeze(-1))

        global_feature = torch.sum(torch.stack(features), dim=0)
        conv_feature = torch.sum(torch.stack(conv_features), dim=0)
        return torch.cat([global_feature, conv_feature], dim=1)

    def forward(self, data):
        # get feature sum from chunks
        feature = self.get_feature_sum(data)

        # value head
        value = feature  # [B, dim_mapping]
        for i, layer in enumerate(self.value_linears):
            value = torch.clamp(value, min=(-1 if i == 0 else 0), max=127 / 128)
            value = layer(value)

        policy = torch.zeros((feature.shape[0], 7, 7), dtype=feature.dtype, device=feature.device)
        return value, policy

    def forward_debug_print(self, data):
        # get feature sum from chunks
        feature = self.get_feature_sum(data)
        print(f"feature sum: \n{(feature * 128).int()}")

        # value head
        value = feature
        for i, layer in enumerate(self.value_linears):
            value = torch.clamp(value, min=0, max=127 / 128)
            print(f"value input{i+1}: \n{(value * 128).int()}")
            value = layer(value)
            print(f"value output{i+1}: \n{(value * 128).int()}")

        policy = torch.zeros((feature.shape[0], 7, 7), dtype=feature.dtype, device=feature.device)
        return value, policy

    @property
    def weight_clipping(self):
        return [
            {
                "params": [f"value_linears.{i}.fc.weight" for i in range(len(self.value_linears))],
                "min_weight": -128 / 128,
                "max_weight": 127 / 128,
            },
        ]

    @property
    def name(self):
        m, gf, cf = self.model_size
        return f"flat_ladder7x7_nnue_v3_{m}m{gf}gf{cf}cf"


@MODELS.register("flat_posconv3x3_nnue")
class FlatConv3x3NNUE(nn.Module):
    def __init__(self, dim_middle=128, dim_feature=32, input_type="basic-nostm", value_no_draw=False):
        super().__init__()
        self.model_size = (dim_middle, dim_feature)
        self.value_no_draw = value_no_draw
        self.input_plane = build_input_plane(input_type)
        self.conv_mappings = self._make_conv3x3_mappings(25, dim_middle, dim_feature)

        # value head
        dim_vhead = 1 if value_no_draw else 3
        self.value_linears = nn.ModuleList(
            [
                LinearBlock(dim_feature, 32, activation="none", quant=True),
                LinearBlock(32, 32, activation="none", quant=True),
                LinearBlock(32, dim_vhead, activation="none", quant=True),
            ]
        )

    def _make_conv3x3_mappings(self, num_conv, dim_middle, dim_mapping):
        conv_list = nn.ModuleList()
        for _ in range(num_conv):
            conv_list.append(
                nn.Sequential(
                    Conv2dBlock(self.input_plane.dim_plane, dim_middle, ks=2, st=1, norm="bn", activation="mish"),
                    Conv2dBlock(dim_middle, dim_middle, ks=2, st=1, norm="bn", activation="mish"),
                    Conv2dBlock(dim_middle, dim_middle, ks=1, st=1, norm="bn", activation="mish"),
                    Conv2dBlock(dim_middle, dim_mapping, ks=1, st=1, norm="bn", activation="none"),
                )
            )
        return conv_list

    def get_feature_sum(self, input_plane):
        _, _, H, W = input_plane.shape
        features = []
        conv_coords = [(y, x) for y in range(0, H - 2) for x in range(0, W - 2)]
        for conv_idx, (y, x) in enumerate(conv_coords):
            chunk = input_plane[:, :, y : y + 3, x : x + 3]
            feat = torch.clamp(self.conv_mappings[conv_idx](chunk), min=-1, max=127 / 128)
            feat = fake_quant(feat, scale=128)
            features.append(feat.squeeze(-1).squeeze(-1))

        feature = torch.sum(torch.stack(features), dim=0)
        return feature

    def forward(self, data):
        input_plane = self.input_plane(data)  # [B, C, H, W]
        _, _, H, W = input_plane.shape

        # get feature sum from chunks
        feature = self.get_feature_sum(input_plane)

        # value head
        value = feature  # [B, dim_mapping]
        for i, layer in enumerate(self.value_linears):
            value = torch.clamp(value, min=(-1 if i == 0 else 0), max=127 / 128)
            value = layer(value)

        policy = torch.zeros((feature.shape[0], H, W), dtype=feature.dtype, device=feature.device)
        return value, policy

    @property
    def weight_clipping(self):
        return [
            {
                "params": [f"value_linears.{i}.fc.weight" for i in range(len(self.value_linears))],
                "min_weight": -128 / 128,
                "max_weight": 127 / 128,
            },
        ]

    @property
    def name(self):
        m, f = self.model_size
        return f"flat_posconv3x3_nnue_{m}m{f}f"


@MODELS.register("flat_posconv3x4_4x3_nnue")
class FlatConv3x44x3NNUE(nn.Module):
    def __init__(
        self,
        dim_middle=128,
        dim_feature=32,
        stride_3x4=(1, 1),
        stride_4x3=(1, 1),
        board_size=(7, 7),
        input_type="basic-nostm",
        value_no_draw=False,
    ):
        super().__init__()
        self.model_size = (dim_middle, dim_feature)
        self.value_no_draw = value_no_draw
        self.input_plane = build_input_plane(input_type)
        self.stride_3x4 = stride_3x4
        self.stride_4x3 = stride_4x3
        num_conv3x4 = sum(
            [1 for y in range(0, board_size[0] - 2, stride_3x4[0]) for x in range(0, board_size[1] - 3, stride_3x4[1])]
        )
        num_conv4x3 = sum(
            [1 for y in range(0, board_size[0] - 3, stride_4x3[0]) for x in range(0, board_size[1] - 2, stride_4x3[1])]
        )
        self.conv_mappings3x4 = self._make_conv3x4_mappings(num_conv3x4, dim_middle, dim_feature)
        self.conv_mappings4x3 = self._make_conv4x3_mappings(num_conv4x3, dim_middle, dim_feature)

        # value head
        dim_vhead = 1 if value_no_draw else 3
        self.value_linears = nn.ModuleList(
            [
                LinearBlock(dim_feature * 2, 32, activation="none", quant=True),
                LinearBlock(32, 32, activation="none", quant=True),
                LinearBlock(32, dim_vhead, activation="none", quant=True),
            ]
        )

    def _make_conv3x4_mappings(self, num_conv, dim_middle, dim_mapping):
        conv_list = nn.ModuleList()
        for _ in range(num_conv):
            conv_list.append(
                nn.Sequential(
                    Conv2dBlock(self.input_plane.dim_plane, dim_middle, ks=2, st=1, norm="bn", activation="mish"),
                    Conv2dBlock(dim_middle, dim_middle, ks=2, st=1, norm="bn", activation="mish"),
                    Conv2dBlock(dim_middle, dim_middle, ks=(1, 2), st=1, norm="bn", activation="mish"),
                    Conv2dBlock(dim_middle, dim_mapping, ks=1, st=1, norm="bn", activation="none"),
                )
            )
        return conv_list

    def _make_conv4x3_mappings(self, num_conv, dim_middle, dim_mapping):
        conv_list = nn.ModuleList()
        for _ in range(num_conv):
            conv_list.append(
                nn.Sequential(
                    Conv2dBlock(self.input_plane.dim_plane, dim_middle, ks=2, st=1, norm="bn", activation="mish"),
                    Conv2dBlock(dim_middle, dim_middle, ks=2, st=1, norm="bn", activation="mish"),
                    Conv2dBlock(dim_middle, dim_middle, ks=(2, 1), st=1, norm="bn", activation="mish"),
                    Conv2dBlock(dim_middle, dim_mapping, ks=1, st=1, norm="bn", activation="none"),
                )
            )
        return conv_list

    def get_feature_sum(self, input_plane):
        _, _, H, W = input_plane.shape
        features3x4 = []
        conv_coords = [
            (y, x) for y in range(0, H - 2, self.stride_3x4[0]) for x in range(0, W - 3, self.stride_3x4[1])
        ]
        for conv_idx, (y, x) in enumerate(conv_coords):
            chunk = input_plane[:, :, y : y + 3, x : x + 4]
            feat = torch.clamp(self.conv_mappings3x4[conv_idx](chunk), min=-1, max=127 / 128)
            feat = fake_quant(feat, scale=128)
            features3x4.append(feat.squeeze(-1).squeeze(-1))
        features3x4 = torch.sum(torch.stack(features3x4), dim=0)

        features4x3 = []
        conv_coords = [
            (y, x) for y in range(0, H - 3, self.stride_4x3[0]) for x in range(0, W - 2, self.stride_4x3[1])
        ]
        for conv_idx, (y, x) in enumerate(conv_coords):
            chunk = input_plane[:, :, y : y + 4, x : x + 3]
            feat = torch.clamp(self.conv_mappings4x3[conv_idx](chunk), min=-1, max=127 / 128)
            feat = fake_quant(feat, scale=128)
            features4x3.append(feat.squeeze(-1).squeeze(-1))
        features4x3 = torch.sum(torch.stack(features4x3), dim=0)

        return torch.cat([features3x4, features4x3], dim=1)

    def forward(self, data):
        input_plane = self.input_plane(data)  # [B, C, H, W]
        _, _, H, W = input_plane.shape

        # get feature sum from chunks
        feature = self.get_feature_sum(input_plane)

        # value head
        value = feature  # [B, dim_mapping]
        for i, layer in enumerate(self.value_linears):
            value = torch.clamp(value, min=(-1 if i == 0 else 0), max=127 / 128)
            value = layer(value)

        policy = torch.zeros((feature.shape[0], H, W), dtype=feature.dtype, device=feature.device)
        return value, policy

    @property
    def weight_clipping(self):
        return [
            {
                "params": [f"value_linears.{i}.fc.weight" for i in range(len(self.value_linears))],
                "min_weight": -128 / 128,
                "max_weight": 127 / 128,
            },
        ]

    @property
    def name(self):
        m, f = self.model_size
        return f"flat_posconv3x4_4x3_nnue_{m}m{f}f"


@MODELS.register("flat_hashconv_nnue")
class FlatHashConvNNUE(nn.Module):
    def __init__(self, kernel_size, hash_logsize, dim_feature=32, input_type="basic-nostm", value_no_draw=False):
        super().__init__()
        self.model_size = (kernel_size, hash_logsize, dim_feature)
        self.value_no_draw = value_no_draw
        self.input_plane = build_input_plane(input_type)

        self.hash_layer = HashLayer(
            self.input_plane.dim_plane * kernel_size * kernel_size,
            input_level=2,  # 1 bits per plane
            hash_logsize=hash_logsize,
            dim_feature=dim_feature,
        )

        # value head
        dim_vhead = 1 if value_no_draw else 3
        self.value_linears = nn.ModuleList(
            [
                LinearBlock(dim_feature, 32, activation="none", quant=True),
                LinearBlock(32, 32, activation="none", quant=True),
                LinearBlock(32, dim_vhead, activation="none", quant=True),
            ]
        )

    def get_feature_sum(self, input_plane):
        _, C, H, W = input_plane.shape
        k, _, _ = self.model_size
        h, w = H - k + 1, W - k + 1

        # chunk inputs by doing convolution of (k, k) kernel
        x = F.unfold(input_plane, kernel_size=k)  # (N, C*k*k, h*w)
        x = x.transpose(1, 2).reshape(-1, C * k * k)  # (N*h*w, C*k*k)

        # compute features of each chunk using the hash layer
        features = self.hash_layer(x)  # (N*h*w, dim_feature)

        # sum all features from chunks
        features = features.reshape(-1, h, w, features.shape[-1])  # (N, h, w, dim_feature)
        features = fake_quant(torch.clamp(features, min=-1, max=127 / 128), scale=128)
        feature = torch.sum(features, dim=(1, 2))  # (N, dim_feature)
        return feature

    def forward(self, data):
        input_plane = self.input_plane(data)  # [B, C, H, W]
        _, _, H, W = input_plane.shape

        # get feature sum from chunks
        feature = self.get_feature_sum(input_plane)

        # value head
        value = feature  # [B, dim_feature]
        for i, layer in enumerate(self.value_linears):
            value = torch.clamp(value, min=(-1 if i == 0 else 0), max=127 / 128)
            value = layer(value)

        policy = torch.zeros((feature.shape[0], H, W), dtype=feature.dtype, device=feature.device)
        return value, policy

    @property
    def weight_clipping(self):
        return [
            {
                "params": [f"value_linears.{i}.fc.weight" for i in range(len(self.value_linears))],
                "min_weight": -128 / 128,
                "max_weight": 127 / 128,
            },
            {"params": [f"hash_layer.features.weight"], "min_weight": -128 / 128, "max_weight": 127 / 128},
        ]

    @property
    def name(self):
        ks, hs, f = self.model_size
        return f"flat_hashconv_nnue_{ks}ks{hs}hs{f}f"


@MODELS.register("flat_hash7x7_nnue_v1")
class FlatHash7x7NNUEv1(nn.Module):
    def __init__(
        self,
        hash_logsize,
        dim_feature=32,
        sub_features=1,
        sub_divisor=2,
        input_type="basic-nostm",
        value_no_draw=False,
    ):
        super().__init__()
        self.model_size = (hash_logsize, dim_feature)
        self.value_no_draw = value_no_draw
        self.input_plane = build_input_plane(input_type)

        self.hash_layer_corner = HashLayer(
            4 * 4,
            input_level=3,
            hash_logsize=hash_logsize,
            dim_feature=dim_feature,
            sub_features=sub_features,
            sub_divisor=sub_divisor,
        )

        # value head
        dim_vhead = 1 if value_no_draw else 3
        self.value_linears = nn.ModuleList(
            [
                LinearBlock(dim_feature, 32, activation="none", quant=True),
                LinearBlock(32, 32, activation="none", quant=True),
                LinearBlock(32, dim_vhead, activation="none", quant=True),
            ]
        )

    def _make_4x4_mapping(self, dim_middle, dim_mapping):
        return nn.Sequential(
            Conv2dBlock(self.input_plane.dim_plane, dim_middle, ks=2, st=1, activation="mish"),
            Conv2dBlock(dim_middle, dim_middle, ks=1, st=1, norm="bn", activation="mish"),
            Conv2dBlock(dim_middle, dim_middle, ks=2, st=1, norm="bn", activation="mish"),
            Conv2dBlock(dim_middle, dim_middle, ks=1, st=1, norm="bn", activation="mish"),
            Conv2dBlock(dim_middle, dim_middle, ks=2, st=1, norm="bn", activation="mish"),
            Conv2dBlock(dim_middle, dim_middle, ks=1, st=1, norm="bn", activation="mish"),
            Conv2dBlock(dim_middle, dim_mapping, ks=1, st=1, norm="bn", activation="none"),
        )

    def get_feature_sum(self, input_plane):
        input_plane = input_plane[:, 0] / 2 + input_plane[:, 1]  # map {0,1,2} into [0.0, 1.0]
        assert torch.all(input_plane <= 1.0)
        features = []

        corner_index = [
            [0, 4, 0, 4, 0],
            [0, 4, 3, 7, 1],
            [3, 7, 3, 7, 2],
            [3, 7, 0, 4, 3],
        ]
        for y0, y1, x0, x1, k in corner_index:
            chunk = torch.rot90(input_plane[:, y0:y1, x0:x1], k, (1, 2))
            chunk = chunk.flatten(1, -1)
            feat = self.hash_layer_corner(chunk)
            # feat = fake_quant(torch.clamp(feat, min=-1, max=127 / 128), scale=128)
            # feat = feat.squeeze(-1).squeeze(-1)
            features.append(feat)

        return torch.sum(torch.stack(features), dim=0)

    def forward(self, data):
        input_plane = self.input_plane(data)  # [B, C, H, W]
        _, _, H, W = input_plane.shape

        # get feature sum from chunks
        feature = self.get_feature_sum(input_plane)

        # value head
        value = feature  # [B, dim_feature]
        for i, layer in enumerate(self.value_linears):
            value = torch.clamp(value, min=(-1 if i == 0 else 0), max=127 / 128)
            value = layer(value)

        policy = torch.zeros((feature.shape[0], H, W), dtype=feature.dtype, device=feature.device)
        return value, policy

    @property
    def weight_clipping(self):
        return [
            {
                "params": [f"value_linears.{i}.fc.weight" for i in range(len(self.value_linears))],
                "min_weight": -128 / 128,
                "max_weight": 127 / 128,
            },
            {"params": ["hash_layer_corner.features.weight"], "min_weight": -128 / 128, "max_weight": 127 / 128},
        ]

    @property
    def name(self):
        hs, f = self.model_size
        return f"flat_hash7x7_nnue_v1_{hs}hs{f}f"


@MODELS.register("flat_square7x7_nnue_v1")
class FlatSquare7x7NNUEv1(nn.Module):
    def __init__(self, dim_middle=128, dim_feature=8, quant_int4=False, input_type="basic-nostm", value_no_draw=False):
        super().__init__()
        self.model_size = (dim_middle, dim_feature)
        self.quant_int4 = quant_int4
        self.value_no_draw = value_no_draw
        self.input_plane = build_input_plane(input_type)
        self.mapping = self._make_4x4_mapping(dim_middle, dim_feature)

        # value head
        dim_vhead = 1 if value_no_draw else 3
        self.value_linears = nn.ModuleList(
            [
                LinearBlock(dim_feature * 4, 32, activation="none", quant=True),
                LinearBlock(32, 32, activation="none", quant=True),
                LinearBlock(32, dim_vhead, activation="none", quant=True),
            ]
        )

    def _make_4x4_mapping(self, dim_middle, dim_mapping):
        return nn.Sequential(
            Conv2dBlock(self.input_plane.dim_plane, dim_middle, ks=2, st=1, activation="mish"),
            Conv2dBlock(dim_middle, dim_middle, ks=1, st=1, norm="bn", activation="mish"),
            Conv2dBlock(dim_middle, dim_middle, ks=2, st=1, norm="bn", activation="mish"),
            Conv2dBlock(dim_middle, dim_middle, ks=1, st=1, norm="bn", activation="mish"),
            Conv2dBlock(dim_middle, dim_middle, ks=2, st=1, norm="bn", activation="mish"),
            Conv2dBlock(dim_middle, dim_middle, ks=1, st=1, norm="bn", activation="mish"),
            Conv2dBlock(dim_middle, dim_mapping, ks=1, st=1, norm="bn", activation="none"),
        )

    def get_feature_sum(self, input_plane):
        features = []

        corner_index = [
            [0, 4, 0, 4, 0],
            [0, 4, 3, 7, 1],
            [3, 7, 3, 7, 2],
            [3, 7, 0, 4, 3],
        ]
        for y0, y1, x0, x1, k in corner_index:
            chunk = torch.rot90(input_plane[:, :, y0:y1, x0:x1], k, (2, 3))
            feat = self.mapping(chunk)
            if self.quant_int4:  # int4 quant
                feat = fake_quant(torch.clamp(feat, min=-1, max=7 / 8), scale=8, num_bits=4)
            else:  # int8 quant
                feat = fake_quant(torch.clamp(feat, min=-1, max=127 / 128), scale=128)
            feat = feat.squeeze(-1).squeeze(-1)
            features.append(feat)

        # return torch.sum(torch.stack(features), dim=0)
        return torch.cat(features, dim=1)

    def forward(self, data):
        input_plane = self.input_plane(data)  # [B, C, H, W]
        _, _, H, W = input_plane.shape

        # get feature sum from chunks
        feature = self.get_feature_sum(input_plane)

        # value head
        value = feature  # [B, dim_feature]
        for i, layer in enumerate(self.value_linears):
            value = torch.clamp(value, min=(-1 if i == 0 else 0), max=127 / 128)
            value = layer(value)

        policy = torch.zeros((feature.shape[0], H, W), dtype=feature.dtype, device=feature.device)
        return value, policy

    @property
    def weight_clipping(self):
        return [
            {
                "params": [f"value_linears.{i}.fc.weight" for i in range(len(self.value_linears))],
                "min_weight": -128 / 128,
                "max_weight": 127 / 128,
            }
        ]

    @property
    def name(self):
        m, f = self.model_size
        return f"flat_square7x7_nnue_v1_{m}m{f}f"


@MODELS.register("flat_square7x7_nnue_v2")
class FlatSquare7x7NNUEv2(nn.Module):
    def __init__(
        self, dim_middle=128, dim_feature=32, quant_int4=False, input_type="basic-nostm", value_no_draw=False
    ):
        super().__init__()
        self.model_size = (dim_middle, dim_feature)
        self.quant_int4 = quant_int4
        self.value_no_draw = value_no_draw
        self.input_plane = build_input_plane(input_type)
        self.mapping4x4 = self._make_4x4_mapping(dim_middle, dim_feature)
        self.mapping4x3 = self._make_4x3_mapping(dim_middle, dim_feature)

        # value head
        dim_vhead = 1 if value_no_draw else 3
        self.value_linears = nn.ModuleList(
            [
                LinearBlock(dim_feature * 4, 32, activation="none", quant=True, weight_quant_scale=256),
                LinearBlock(32, 32, activation="none", quant=True),
                LinearBlock(32, dim_vhead, activation="none", quant=True),
            ]
        )

    def _make_4x4_mapping(self, dim_middle, dim_mapping):
        return nn.Sequential(
            Conv2dBlock(self.input_plane.dim_plane, dim_middle, ks=2, st=1, activation="mish"),
            Conv2dBlock(dim_middle, dim_middle, ks=1, st=1, norm="bn", activation="mish"),
            Conv2dBlock(dim_middle, dim_middle, ks=2, st=1, norm="bn", activation="mish"),
            Conv2dBlock(dim_middle, dim_middle, ks=1, st=1, norm="bn", activation="mish"),
            Conv2dBlock(dim_middle, dim_middle, ks=2, st=1, norm="bn", activation="mish"),
            Conv2dBlock(dim_middle, dim_middle, ks=1, st=1, norm="bn", activation="mish"),
            Conv2dBlock(dim_middle, dim_mapping, ks=1, st=1, norm="bn", activation="none"),
        )

    def _make_4x3_mapping(self, dim_middle, dim_mapping):
        return nn.Sequential(
            Conv2dBlock(self.input_plane.dim_plane, dim_middle, ks=(2, 2), st=1, activation="mish"),
            Conv2dBlock(dim_middle, dim_middle, ks=1, st=1, norm="bn", activation="mish"),
            Conv2dBlock(dim_middle, dim_middle, ks=(2, 1), st=1, norm="bn", activation="mish"),
            Conv2dBlock(dim_middle, dim_middle, ks=1, st=1, norm="bn", activation="mish"),
            Conv2dBlock(dim_middle, dim_middle, ks=(2, 2), st=1, norm="bn", activation="mish"),
            Conv2dBlock(dim_middle, dim_middle, ks=1, st=1, norm="bn", activation="mish"),
            Conv2dBlock(dim_middle, dim_mapping, ks=1, st=1, norm="bn", activation="none"),
        )

    def get_features(self, input_plane):
        corner_index = [
            [0, 4, 0, 4, 0],
            [0, 4, 3, 7, 1],
            [3, 7, 3, 7, 2],
            [3, 7, 0, 4, 3],
        ]
        features_corner = []
        for y0, y1, x0, x1, k in corner_index:
            chunk = torch.rot90(input_plane[:, :, y0:y1, x0:x1], k, (2, 3))
            feat = self.mapping4x4(chunk)
            if self.quant_int4:  # int4 quant
                feat = fake_quant(torch.clamp(feat, min=-1, max=7 / 8), scale=8, num_bits=4)
            else:  # int8 quant
                feat = fake_quant(torch.clamp(feat, min=-1, max=127 / 128), scale=128)
            feat = feat.squeeze(-1).squeeze(-1)
            features_corner.append(feat)

        middle_index = [
            [0, 4, 2, 5, 0],
            [2, 5, 3, 7, 1],
            [3, 7, 2, 5, 2],
            [2, 5, 0, 4, -1],
        ]
        features_middle = []
        for y0, y1, x0, x1, k in middle_index:
            chunk = torch.rot90(input_plane[:, :, y0:y1, x0:x1], k, (2, 3))
            feat = self.mapping4x3(chunk)
            feat = fake_quant(torch.clamp(feat, min=-1, max=127 / 128), scale=128)
            feat = feat.squeeze(-1).squeeze(-1)
            features_middle.append(feat)

        return features_corner, features_middle

    def forward(self, data):
        input_plane = self.input_plane(data)  # [B, C, H, W]
        _, _, H, W = input_plane.shape

        # get feature sum from chunks
        features_corner, features_middle = self.get_features(input_plane)
        feature_corner = torch.cat(features_corner, dim=1)
        feature_middle = torch.cat(features_middle, dim=1)
        feature = feature_corner + feature_middle

        # value head
        value = feature  # [B, dim_feature]
        for i, layer in enumerate(self.value_linears):
            value = torch.clamp(value, min=(-1 if i == 0 else 0), max=127 / 128)
            value = layer(value)

        policy = torch.zeros((feature.shape[0], H, W), dtype=feature.dtype, device=feature.device)
        return value, policy

    def forward_debug_print(self, data):
        input_plane = self.input_plane(data)  # [B, C, H, W]
        _, _, H, W = input_plane.shape

        # get feature sum from chunks
        features_corner, features_middle = self.get_features(input_plane)
        for i, f_corner in enumerate(features_corner):
            print(f"feature4x4[{i}]: \n{(f_corner * 128).int()}")
        for i, f_middle in enumerate(features_middle):
            print(f"feature4x3[{i}]: \n{(f_middle * 128).int()}")
        feature_corner = torch.cat(features_corner, dim=1)
        feature_middle = torch.cat(features_middle, dim=1)
        feature = feature_corner + feature_middle
        print(f"feature sum: \n{(feature * 128).int()}")

        # value head
        value = feature  # [B, dim_feature]
        for i, layer in enumerate(self.value_linears):
            value = torch.clamp(value, min=(-1 if i == 0 else 0), max=127 / 128)
            print(f"value input{i+1}: \n{(value * 128).int()}")
            value = layer(value)
            print(f"value output{i+1}: \n{(value * 128).int()}")

        policy = torch.zeros((feature.shape[0], H, W), dtype=feature.dtype, device=feature.device)
        return value, policy

    @property
    def weight_clipping(self):
        return [
            {
                "params": [f"value_linears.{i}.fc.weight" for i in range(len(self.value_linears))],
                "min_weight": -128 / 128,
                "max_weight": 127 / 128,
            }
        ]

    @property
    def name(self):
        m, f = self.model_size
        return f"flat_square7x7_nnue_v2_{m}m{f}f"


@MODELS.register("flat_square7x7_nnue_v3")
class FlatSquare7x7NNUEv3(nn.Module):
    def __init__(
        self, dim_middle=128, dim_feature=32, quant_int4=False, input_type="basic-nostm", value_no_draw=False
    ):
        super().__init__()
        self.model_size = (dim_middle, dim_feature)
        self.quant_int4 = quant_int4
        self.value_no_draw = value_no_draw
        self.input_plane = build_input_plane(input_type)
        self.mapping3x4_corner1 = self._make_3x4_mapping(dim_middle, dim_feature)
        self.mapping3x4_corner2 = self._make_3x4_mapping(dim_middle, dim_feature)
        self.mapping3x4_middle = self._make_3x4_mapping(dim_middle, dim_feature)

        # value head
        dim_vhead = 1 if value_no_draw else 3
        self.value_linears = nn.ModuleList(
            [
                LinearBlock(dim_feature * 4, 64, activation="none", quant=True, weight_quant_scale=256),
                LinearBlock(64, 32, activation="none", quant=True),
                LinearBlock(32, dim_vhead, activation="none", quant=True),
            ]
        )

    def _make_3x4_mapping(self, dim_middle, dim_mapping):
        return nn.Sequential(
            Conv2dBlock(self.input_plane.dim_plane, dim_middle, ks=2, st=1, activation="mish"),
            Conv2dBlock(dim_middle, dim_middle, ks=1, st=1, norm="bn", activation="mish"),
            Conv2dBlock(dim_middle, dim_middle, ks=2, st=1, norm="bn", activation="mish"),
            Conv2dBlock(dim_middle, dim_middle, ks=1, st=1, norm="bn", activation="mish"),
            Conv2dBlock(dim_middle, dim_middle, ks=(1, 2), st=1, norm="bn", activation="mish"),
            Conv2dBlock(dim_middle, dim_middle, ks=1, st=1, norm="bn", activation="mish"),
            Conv2dBlock(dim_middle, dim_mapping, ks=1, st=1, norm="bn", activation="none"),
        )

    def get_feature_sum(self, input_plane):
        fs_corner1, fs_corner2, fs_middle = [], [], []
        patterns = [
            [0, 3, 0, 4, 0, fs_corner1, self.mapping3x4_corner1],
            [0, 4, 4, 7, 1, fs_corner1, self.mapping3x4_corner1],
            [4, 7, 3, 7, 2, fs_corner1, self.mapping3x4_corner1],
            [3, 7, 0, 3, 3, fs_corner1, self.mapping3x4_corner1],
            [0, 4, 0, 3, 3, fs_corner2, self.mapping3x4_corner2],
            [0, 3, 3, 7, 0, fs_corner2, self.mapping3x4_corner2],
            [3, 7, 4, 7, 1, fs_corner2, self.mapping3x4_corner2],
            [4, 7, 0, 4, 2, fs_corner2, self.mapping3x4_corner2],
            [0, 4, 2, 5, 3, fs_middle, self.mapping3x4_middle],
            [2, 5, 3, 7, 0, fs_middle, self.mapping3x4_middle],
            [3, 7, 2, 5, 1, fs_middle, self.mapping3x4_middle],
            [2, 5, 0, 4, 2, fs_middle, self.mapping3x4_middle],
        ]
        for y0, y1, x0, x1, k, fs, mapping in patterns:
            chunk = torch.rot90(input_plane[:, :, y0:y1, x0:x1], k, (2, 3))
            feat = mapping(chunk)
            if self.quant_int4:  # int4 quant
                feat = fake_quant(torch.clamp(feat, min=-1, max=7 / 8), scale=8, num_bits=4)
            else:  # int8 quant
                feat = fake_quant(torch.clamp(feat, min=-1, max=127 / 128), scale=128)
            feat = feat.squeeze(-1).squeeze(-1)
            fs.append(feat)

        return fs_corner1, fs_corner2, fs_middle

    def forward(self, data):
        input_plane = self.input_plane(data)  # [B, C, H, W]
        _, _, H, W = input_plane.shape

        # get feature sum from chunks
        fs_corner1, fs_corner2, fs_middle = self.get_feature_sum(input_plane)
        f_corner1 = torch.cat(fs_corner1, dim=1)
        f_corner2 = torch.cat(fs_corner2, dim=1)
        f_middle = torch.cat(fs_middle, dim=1)
        feature = fake_quant((f_corner1 + f_corner2 + 1 / 128) / 2, floor=True) + f_middle

        # value head
        value = feature  # [B, dim_feature]
        for i, layer in enumerate(self.value_linears):
            value = torch.clamp(value, min=(-1 if i == 0 else 0), max=127 / 128)
            value = layer(value)

        policy = torch.zeros((feature.shape[0], H, W), dtype=feature.dtype, device=feature.device)
        return value, policy

    def forward_debug_print(self, data):
        input_plane = self.input_plane(data)  # [B, C, H, W]
        _, _, H, W = input_plane.shape

        # get feature sum from chunks
        fs_corner1, fs_corner2, fs_middle = self.get_feature_sum(input_plane)
        for i, f in enumerate(fs_corner1):
            print(f"fs_corner1[{i}]: \n{(f * 128).int()}")
        for i, f in enumerate(fs_corner2):
            print(f"fs_corner2[{i}]: \n{(f * 128).int()}")
        for i, f in enumerate(fs_middle):
            print(f"fs_middle[{i}]: \n{(f * 128).int()}")
        f_corner1 = torch.cat(fs_corner1, dim=1)
        f_corner2 = torch.cat(fs_corner2, dim=1)
        f_middle = torch.cat(fs_middle, dim=1)
        feature = fake_quant((f_corner1 + f_corner2 + 1 / 128) / 2, floor=True) + f_middle
        print(f"feature sum: \n{(feature * 128).int()}")

        # value head
        value = feature  # [B, dim_feature]
        for i, layer in enumerate(self.value_linears):
            value = torch.clamp(value, min=(-1 if i == 0 else 0), max=127 / 128)
            print(f"value input{i+1}: \n{(value * 128).int()}")
            value = layer(value)
            print(f"value output{i+1}: \n{(value * 128).int()}")

        policy = torch.zeros((feature.shape[0], H, W), dtype=feature.dtype, device=feature.device)
        return value, policy

    @property
    def weight_clipping(self):
        return [
            {
                "params": ["value_linears.1.fc.weight", "value_linears.2.fc.weight"],
                "min_weight": -128 / 128,
                "max_weight": 127 / 128,
            },
            {"params": [f"value_linears.0.fc.weight"], "min_weight": -128 / 256, "max_weight": 127 / 256},
        ]

    @property
    def name(self):
        m, f = self.model_size
        return f"flat_square7x7_nnue_v3_{m}m{f}f"


@MODELS.register("flat_square7x7_nnue_v4")
class FlatSquare7x7NNUEv4(nn.Module):
    def __init__(
        self, dim_middle=128, dim_feature=32, input_type="basic-nostm", value_no_draw=False, unit_feature_vector=False
    ):
        super().__init__()
        self.model_size = (dim_middle, dim_feature)
        self.value_no_draw = value_no_draw
        self.unit_feature_vector = unit_feature_vector
        self.input_plane = build_input_plane(input_type)
        self.mapping4x4 = nn.ModuleList([self._make_4x4_mapping(dim_middle, dim_feature) for _ in range(3)])

        # value head
        dim_vhead = 1 if value_no_draw else 3
        self.value_linears = nn.ModuleList(
            [
                LinearBlock(dim_feature, 32, activation="none", quant=True, weight_quant_scale=256),
                LinearBlock(32, 32, activation="none", quant=True),
                LinearBlock(32, dim_vhead, activation="none", quant=True),
            ]
        )

    def _make_4x4_mapping(self, dim_middle, dim_mapping):
        return nn.Sequential(
            Conv2dBlock(self.input_plane.dim_plane, dim_middle, ks=2, st=1, activation="relu"),
            Conv2dBlock(dim_middle, dim_middle, ks=1, st=1, norm="bn", activation="relu"),
            Conv2dBlock(dim_middle, dim_middle, ks=2, st=1, norm="bn", activation="relu"),
            Conv2dBlock(dim_middle, dim_middle, ks=1, st=1, norm="bn", activation="relu"),
            Conv2dBlock(dim_middle, dim_middle, ks=2, st=1, norm="bn", activation="relu"),
            Conv2dBlock(dim_middle, dim_mapping, ks=1, st=1, norm="bn", activation="none"),
        )

    def _do_vector_quantize(self, feature_groups):
        return feature_groups, {}, {}  # no vector quantization as default

    def get_features(self, input_plane):
        chunk_indices = [
            [
                [0, 4, 0, 4, 0, False],
                [0, 4, 3, 7, 1, False],
                [3, 7, 3, 7, 2, False],
                [3, 7, 0, 4, 3, False],
            ],
            [
                [0, 4, 1, 5, 0, False],
                [1, 5, 3, 7, 1, False],
                [3, 7, 2, 6, 2, False],
                [2, 6, 0, 4, 3, False],
                [1, 5, 0, 4, 0, True],
                [0, 4, 2, 6, 1, True],
                [2, 6, 3, 7, 2, True],
                [3, 7, 1, 5, 3, True],
            ],
            [
                [1, 5, 1, 5, 0, False],
                [1, 5, 2, 6, 1, False],
                [2, 6, 2, 6, 2, False],
                [2, 6, 1, 5, 3, False],
            ],
        ]

        feature_groups = []
        for chunk_idx, chunk_index in enumerate(chunk_indices):
            features = []
            for y0, y1, x0, x1, k, t in chunk_index:
                chunk = torch.rot90(input_plane[:, :, y0:y1, x0:x1], k, (2, 3))
                if t:
                    chunk = torch.transpose(chunk, 2, 3)
                feat = self.mapping4x4[chunk_idx](chunk)
                feat = feat.squeeze(-1).squeeze(-1)
                # normalize feature onto unit hypersphere
                if self.unit_feature_vector:
                    feat = F.normalize(feat, p=2, dim=-1)
                feat = torch.clamp(feat, min=-1, max=127 / 128)
                features.append(feat)
            feature_groups.append(torch.stack(features))

        return feature_groups

    def forward(self, data):
        input_plane = self.input_plane(data)  # [B, C, H, W]
        _, _, H, W = input_plane.shape

        # get feature sum from chunks
        feature_groups = self.get_features(input_plane)

        # do vector quantization
        feature_groups, aux_losses, aux_outputs = self._do_vector_quantize(feature_groups)

        # int8 quant and sum
        feature = torch.sum(fake_quant(torch.cat(feature_groups), scale=128), dim=0)

        # value head
        value = feature  # [B, dim_feature]
        for i, layer in enumerate(self.value_linears):
            value = torch.clamp(value, min=(-1 if i == 0 else 0), max=127 / 128)
            value = layer(value)

        policy = torch.zeros((feature.shape[0], H, W), dtype=feature.dtype, device=feature.device)

        return value, policy, aux_losses, aux_outputs

    def forward_debug_print(self, data):
        input_plane = self.input_plane(data)  # [B, C, H, W]
        _, _, H, W = input_plane.shape

        # get feature sum from chunks
        feature_groups = self.get_features(input_plane)

        # do vector quantization
        feature_groups, aux_losses, aux_outputs = self._do_vector_quantize(feature_groups)
        for i, f in enumerate(feature_groups):
            print(f"feature_group{i}: \n{(fake_quant(f, scale=128) * 128).int()}")

        # int8 quant and sum
        feature = torch.sum(fake_quant(torch.cat(feature_groups), scale=128), dim=0)
        print(f"feature sum: \n{(feature * 128).int()}")

        # value head
        value = feature  # [B, dim_feature]
        for i, layer in enumerate(self.value_linears):
            value = torch.clamp(value, min=(-1 if i == 0 else 0), max=127 / 128)
            value = layer(value)

        policy = torch.zeros((feature.shape[0], H, W), dtype=feature.dtype, device=feature.device)

        return value, policy, aux_losses, aux_outputs

    @property
    def weight_clipping(self):
        return [
            {
                "params": [f"value_linears.{i}.fc.weight" for i in range(len(self.value_linears))],
                "min_weight": -128 / 128,
                "max_weight": 127 / 128,
            }
        ]

    @property
    def name(self):
        m, f = self.model_size
        return f"flat_square7x7_nnue_v4_{m}m{f}f"


@MODELS.register("flat_square7x7_nnue_v4_vq")
class FlatSquare7x7NNUEv4VQ(FlatSquare7x7NNUEv4):
    def __init__(
        self,
        dim_middle=128,
        dim_feature=32,
        input_type="basic-nostm",
        value_no_draw=False,
        codebook_size=65536,
        use_cosine_sim=False,
        **vq_kwargs,
    ):
        super().__init__(dim_middle, dim_feature, input_type, value_no_draw, unit_feature_vector=use_cosine_sim)
        self.codebook_size = codebook_size
        self.use_cosine_sim = use_cosine_sim
        self.vq_layer = nn.ModuleList(
            [
                VectorQuantize(
                    codebook_size=codebook_size,
                    dim_feature=dim_feature,
                    use_cosine_sim=use_cosine_sim,
                    **vq_kwargs,
                )
                for _ in range(3)
            ]
        )

    def _do_vector_quantize(self, feature_groups):
        loss = []
        perplexity = []
        normalized_perplexity = []
        cluster_size_q10 = []
        cluster_size_q50 = []
        cluster_size_q90 = []

        for i, feature in enumerate(feature_groups):
            # fix feature if vq layer is not inited
            if not self.vq_layer[i].inited:
                feature = feature.detach()
            # Do vector quantization
            feature_quantized, info = self.vq_layer[i](feature.view(-1, feature.shape[-1]))
            feature_groups[i] = feature_quantized.view_as(feature)

            if info is not None:
                loss.append(info["loss"])
                perplexity.append(info["perplexity"])
                normalized_perplexity.append(info["normalized_perplexity"])
                cluster_size = self.vq_layer[i].normalized_cluster_size
                cluster_size_q10.append(torch.quantile(cluster_size, q=0.1))
                cluster_size_q50.append(torch.quantile(cluster_size, q=0.5))
                cluster_size_q90.append(torch.quantile(cluster_size, q=0.9))

        aux_losses = {}
        if len(loss) > 0:
            loss = torch.stack(loss).sum()
            aux_losses = {"vq": loss}

        aux_outputs = {}
        if len(perplexity) > 0:
            perplexity = torch.stack(perplexity).mean()
            normalized_perplexity = torch.stack(normalized_perplexity).mean()
            cluster_size_q10 = torch.stack(cluster_size_q10).mean()
            cluster_size_q50 = torch.stack(cluster_size_q50).mean()
            cluster_size_q90 = torch.stack(cluster_size_q90).mean()
            aux_outputs = {
                "vq_perplexity": perplexity,
                "vq_normed_perplexity": normalized_perplexity,
                "vq_cluster_size_q10": cluster_size_q10,
                "vq_cluster_size_q50": cluster_size_q50,
                "vq_cluster_size_q90": cluster_size_q90,
            }
        return feature_groups, aux_losses, aux_outputs
