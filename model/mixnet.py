import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from . import MODELS
from .layers.activation import ChannelWiseLeakyReLU, QuantPReLU, SwitchPReLU
from .layers.convolution import Conv2d, Conv2dBlock
from .layers.linear import LinearBlock
from .layers.normalization import build_norm2d_layer
from .input import build_input_plane
from .mixnet_components import Mapping, QuantizedHeadMixin
from .trace import TraceableModel
from .ops import (
    dynamic_pointwise_conv2d,
    mean_3x3_regions,
    quantized_avg4 as avg4,
    quantized_sum_3x3_regions,
    split_3x3_regions,
)
from utils.quant_utils import fake_quant


_MAPPING_GEMM_INDUCTOR_CONFIG = {"conv_1x1_as_mm": True}


@MODELS.register("mix6")
class Mix6Net(nn.Module):
    """Mix6Net adopted from https://github.com/hzyhhzy/gomoku_nnue/blob/87603e908cb1ae9106966e3596830376a637c21a/train_pytorch/model.py#L736"""

    def __init__(self, dim_middle=128, dim_policy=16, dim_value=32, map_max=30, input_type="basic-nostm"):
        super().__init__()
        self.model_size = (dim_middle, dim_policy, dim_value)
        self.map_max = map_max
        self.input_type = input_type
        dim_out = dim_policy + dim_value

        self.input_plane = build_input_plane(input_type)
        self.mapping = Mapping(self.input_plane.dim_plane, dim_middle, dim_out, use_channel_last=False)
        self.mapping_activation = ChannelWiseLeakyReLU(dim_out, bound=6)

        # policy nets
        self.policy_conv = Conv2dBlock(dim_policy, dim_policy, ks=3, st=1, padding=1, groups=dim_policy)
        self.policy_linear = Conv2dBlock(dim_policy, 1, ks=1, st=1, padding=0, activation="none", bias=False)
        self.policy_activation = ChannelWiseLeakyReLU(1, bias=False)

        # value nets
        self.value_activation = ChannelWiseLeakyReLU(dim_value, bias=False)
        self.value_linear = nn.Sequential(
            LinearBlock(dim_value, dim_value), LinearBlock(dim_value, dim_value)
        )
        self.value_linear_final = LinearBlock(dim_value, 3, activation="none")

    def forward(self, data):
        _, dim_policy, _ = self.model_size

        input_plane = self.input_plane(data)
        feature = self.mapping(input_plane)
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
        value = value + self.value_linear(value)
        value = self.value_linear_final(value)

        return {"value": value, "policy": policy}

    @property
    def name(self):
        m, p, v = self.model_size
        return f"mix6_{self.input_type}_{m}m{p}p{v}v" + (f"-{self.map_max}mm" if self.map_max != 0 else "")


@MODELS.register("mix7")
class Mix7Net(TraceableModel):
    def __init__(
        self,
        dim_middle=128,
        dim_policy=32,
        dim_value=32,
        dim_dwconv=None,
        map_max=30,
        input_type="basic-nostm",
        dwconv_kernel_size=3,
    ):
        super().__init__()
        self.model_size = (dim_middle, dim_policy, dim_value)
        self.map_max = map_max
        self.input_type = input_type
        self.dwconv_kernel_size = dwconv_kernel_size
        dim_out = max(dim_policy, dim_value)
        self.dim_dwconv = dim_out if dim_dwconv is None else dim_dwconv
        assert self.dim_dwconv <= dim_out, "Incorrect dim_dwconv!"

        self.input_plane = build_input_plane(input_type)
        self.mapping = Mapping(self.input_plane.dim_plane, dim_middle, dim_out, use_channel_last=False)
        self.mapping_activation = nn.PReLU(dim_out)

        # feature depth-wise conv
        self.feature_dwconv = Conv2dBlock(
            self.dim_dwconv,
            self.dim_dwconv,
            ks=dwconv_kernel_size,
            st=1,
            padding=dwconv_kernel_size // 2,
            groups=self.dim_dwconv,
        )

        # policy head (point-wise conv)
        self.policy_pwconv = Conv2dBlock(dim_policy, 1, ks=1, st=1, padding=0, activation="none", bias=False)
        self.policy_activation = nn.PReLU(1)

        # value head
        self.value_linear = nn.Sequential(
            LinearBlock(dim_value, dim_value),
            LinearBlock(dim_value, dim_value),
            LinearBlock(dim_value, 3, activation="none"),
        )

    def forward(self, data):
        _, dim_policy, dim_value = self.model_size

        input_plane = self.input_plane(data)
        feature = self.mapping(input_plane)
        # resize feature to range [-map_max, map_max]
        if self.map_max != 0:
            feature = self.map_max * torch.tanh(feature / self.map_max)
        self._trace("feature.directional", feature)
        # average feature across four directions
        feature = torch.mean(feature, dim=1)  # [B, max(PC,VC), H, W]
        self._trace("feature.reduced", feature)
        feature = self.mapping_activation(feature)
        self._trace("feature.activated", feature)

        # feature conv
        feat_dwconv = self.feature_dwconv(feature[:, : self.dim_dwconv])  # [B, dwconv, H, W]
        self._trace("feature.dwconv", feat_dwconv)
        feat_direct = feature[:, self.dim_dwconv :]  # [B, max(PC,VC)-dwconv, H, W]
        feature = torch.cat((feat_dwconv, feat_direct), dim=1)  # [B, max(PC,VC), H, W]
        self._trace("feature.output", feature)

        # policy head
        policy = feature[:, :dim_policy]
        self._trace("policy.input", policy)
        policy = self.policy_pwconv(policy)
        self._trace("policy.generated", policy)
        policy = self.policy_activation(policy)
        self._trace("policy.activated", policy)
        self._trace("policy.output", policy)

        # value head
        value = feature[:, :dim_value]
        value = torch.mean(value, dim=(2, 3))
        for i, linear in enumerate(self.value_linear):
            self._trace(f"value.linear.{i}.input", value)
            value = linear(value)
            self._trace(f"value.linear.{i}.output", value)

        return {"value": value, "policy": policy}

    @property
    def weight_clipping(self):
        # Clip prelu weight of mapping activation to [-1,1] to avoid overflow
        # In this range, prelu is the same as `max(x, ax)`.
        return [
            {
                "params": ["mapping_activation.weight"],
                "min_weight": -1.0,
                "max_weight": 1.0,
            },
            {
                "params": ["feature_dwconv.conv.weight"],
                "min_weight": -2.0,
                "max_weight": 2.0,
            },
        ]

    @property
    def name(self):
        m, p, v = self.model_size
        return f"mix7_{self.input_type}_{m}m{p}p{v}v" + (f"-{self.map_max}mm" if self.map_max != 0 else "")


@MODELS.register("mix8")
class Mix8Net(TraceableModel):
    def __init__(
        self,
        dim_middle=128,
        dim_feature=None,
        dim_policy=32,
        dim_value=64,
        dim_value_group=32,
        dim_dwconv=None,
        input_type="basicns",
    ):
        super().__init__()
        if dim_feature is None:
            dim_feature = max(dim_policy, dim_value)
        else:
            assert dim_feature == max(dim_policy, dim_value), (
                f"dim_feature ({dim_feature}) must equal"
                f" max(dim_policy, dim_value) ({max(dim_policy, dim_value)})"
            )
        self.model_size = (dim_middle, dim_feature, dim_policy, dim_value, dim_value_group)
        self.input_type = input_type
        self.dim_dwconv = dim_policy if dim_dwconv is None else dim_dwconv
        assert self.dim_dwconv <= dim_feature, f"Invalid dim_dwconv {self.dim_dwconv}"
        assert self.dim_dwconv >= dim_policy, "dim_dwconv must be not less than dim_policy"

        self.input_plane = build_input_plane(input_type)
        self.mapping = Mapping(self.input_plane.dim_plane, dim_middle, dim_feature, use_channel_last=False)
        self.mapping_activation = QuantPReLU(
            dim_feature,
            input_quant_scale=1024,
            input_quant_bits=16,
            weight_quant_scale=32768,
            weight_quant_bits=16,
        )

        # feature depth-wise conv
        self.feature_dwconv = Conv2dBlock(
            self.dim_dwconv,
            self.dim_dwconv,
            ks=3,
            st=1,
            padding=3 // 2,
            groups=self.dim_dwconv,
            activation="relu",
        )

        # policy head (point-wise conv)
        self.policy_pwconv_weight_linear = nn.Sequential(
            LinearBlock(dim_feature, dim_policy, activation="none"),
            nn.PReLU(dim_policy),
            LinearBlock(dim_policy, 4 * dim_policy, activation="none"),
        )
        self.policy_output = nn.Sequential(
            nn.PReLU(4),
            Conv2d(4, 1, 1),
        )

        # value head
        self.value_corner_linear = LinearBlock(dim_feature, dim_value_group, activation="none")
        self.value_corner_act = nn.PReLU(dim_value_group)
        self.value_edge_linear = LinearBlock(dim_feature, dim_value_group, activation="none")
        self.value_edge_act = nn.PReLU(dim_value_group)
        self.value_center_linear = LinearBlock(dim_feature, dim_value_group, activation="none")
        self.value_center_act = nn.PReLU(dim_value_group)
        self.value_quad_linear = LinearBlock(dim_value_group, dim_value_group, activation="none")
        self.value_quad_act = nn.PReLU(dim_value_group)
        self.value_linear = nn.Sequential(
            LinearBlock(dim_feature + 4 * dim_value_group, dim_value),
            LinearBlock(dim_value, dim_value),
            LinearBlock(dim_value, 3, activation="none"),
        )

    def get_feature(self, data, inv_side=False):
        # get the input plane from board and side to move input
        input_plane = self.input_plane(data, inv_side)  # [B, 2, H, W]
        # get per-point 4-direction cell features
        feature = self.mapping(input_plane)  # [B, 4, dim_feature, H, W]
        self._trace("feature.directional", feature)

        # clamp feature for int quantization
        feature = torch.clamp(feature, min=-32, max=32)  # int16, scale=255, [-32,32]
        feature = fake_quant(feature, scale=255, num_bits=16)
        # sum (and rescale) feature across four directions
        feature = torch.mean(feature, dim=1)  # [B, dim_feature, H, W] int16, scale=1020, [-32,32]
        self._trace("feature.reduced", feature)
        feature = self.mapping_activation(feature)  # int16, scale=1020, [-32,32]
        self._trace("feature.activated", feature)

        # apply feature depth-wise conv
        feat_dwconv = self.feature_dwconv(feature[:, : self.dim_dwconv])  # [B, dwconv, H, W]
        self._trace("feature.dwconv", feat_dwconv)
        feat_direct = feature[:, self.dim_dwconv :]  # [B, dim_feature-dwconv, H, W]
        feature = torch.cat([feat_dwconv, feat_direct], dim=1)  # [B, dim_feature, H, W]
        self._trace("feature.output", feature)

        return feature

    def forward(self, data):
        _, _, dim_policy, _, _ = self.model_size

        # get feature from single side
        feature = self.get_feature(data, False)  # [B, dim_feature, H, W]

        # value feature accumulator
        feature_mean = torch.mean(feature, dim=(2, 3))  # [B, dim_feature]
        self._trace("feature.sum.quantized", feature_mean)

        # value feature accumulator of nine regions
        B = feature.shape[0]
        (
            feature_00,
            feature_01,
            feature_02,
            feature_10,
            feature_11,
            feature_12,
            feature_20,
            feature_21,
            feature_22,
        ) = mean_3x3_regions(feature)
        self._trace("feature.region.00", feature_00)
        self._trace("feature.region.01", feature_01)
        self._trace("feature.region.02", feature_02)
        self._trace("feature.region.10", feature_10)
        self._trace("feature.region.11", feature_11)
        self._trace("feature.region.12", feature_12)
        self._trace("feature.region.20", feature_20)
        self._trace("feature.region.21", feature_21)
        self._trace("feature.region.22", feature_22)

        # policy head
        pwconv_weight = self.policy_pwconv_weight_linear(feature_mean)
        self._trace("policy.generated", pwconv_weight)
        pwconv_weight = pwconv_weight.reshape(B, 4 * dim_policy, 1, 1)
        policy = feature[:, :dim_policy]  # [B, dim_policy, H, W]
        self._trace("policy.input", policy)
        policy = dynamic_pointwise_conv2d(policy, pwconv_weight)
        self._trace("policy.dynamic", policy)
        policy = self.policy_output(policy)  # [B, 1, H, W]
        self._trace("policy.output", policy)

        # value head
        value_00 = self.value_corner_act(self.value_corner_linear(feature_00))
        value_01 = self.value_edge_act(self.value_edge_linear(feature_01))
        value_02 = self.value_corner_act(self.value_corner_linear(feature_02))
        value_10 = self.value_edge_act(self.value_edge_linear(feature_10))
        value_11 = self.value_center_act(self.value_center_linear(feature_11))
        value_12 = self.value_edge_act(self.value_edge_linear(feature_12))
        value_20 = self.value_corner_act(self.value_corner_linear(feature_20))
        value_21 = self.value_edge_act(self.value_edge_linear(feature_21))
        value_22 = self.value_corner_act(self.value_corner_linear(feature_22))
        self._trace("value.region.00", value_00)
        self._trace("value.region.01", value_01)
        self._trace("value.region.02", value_02)
        self._trace("value.region.10", value_10)
        self._trace("value.region.11", value_11)
        self._trace("value.region.12", value_12)
        self._trace("value.region.20", value_20)
        self._trace("value.region.21", value_21)
        self._trace("value.region.22", value_22)

        value_q00 = value_00 + value_01 + value_10 + value_11
        value_q01 = value_01 + value_02 + value_11 + value_12
        value_q10 = value_10 + value_11 + value_20 + value_21
        value_q11 = value_11 + value_12 + value_21 + value_22
        self._trace("value.quad.00.input", value_q00)
        self._trace("value.quad.01.input", value_q01)
        self._trace("value.quad.10.input", value_q10)
        self._trace("value.quad.11.input", value_q11)
        value_q00 = self.value_quad_act(self.value_quad_linear(value_q00))
        value_q01 = self.value_quad_act(self.value_quad_linear(value_q01))
        value_q10 = self.value_quad_act(self.value_quad_linear(value_q10))
        value_q11 = self.value_quad_act(self.value_quad_linear(value_q11))
        self._trace("value.quad.00.output", value_q00)
        self._trace("value.quad.01.output", value_q01)
        self._trace("value.quad.10.output", value_q10)
        self._trace("value.quad.11.output", value_q11)
        value = torch.cat(
            [
                feature_mean,
                value_q00,
                value_q01,
                value_q10,
                value_q11,
            ],
            1,
        )  # [B, dim_feature + 4 * dim_value_group]
        self._trace("value.concat", value)
        for i, linear in enumerate(self.value_linear):
            self._trace(f"value.linear.{i}.input", value)
            value = linear(value)
            self._trace(f"value.linear.{i}.output", value)

        return {"value": value, "policy": policy}

    @property
    def weight_clipping(self):
        # Clip prelu weight of mapping activation to [-1,1] to avoid overflow
        # In this range, prelu is the same as `max(x, ax)`.
        return [
            {
                "params": [
                    "mapping_activation.weight",
                    "policy_pwconv_weight_linear.1.weight",
                    "value_corner_act.weight",
                    "value_edge_act.weight",
                    "value_center_act.weight",
                    "value_quad_act.weight",
                ],
                "min_weight": -1.0,
                "max_weight": 1.0,
            },
            {
                "params": [f"feature_dwconv.conv.weight"],
                "min_weight": -1.5,
                "max_weight": 1.5,
            },
            {
                "params": [f"feature_dwconv.conv.bias"],
                "min_weight": -4.0,
                "max_weight": 4.0,
            },
        ]

    @property
    def name(self):
        _, f, p, v, q = self.model_size
        d = self.dim_dwconv
        return f"mix8_{self.input_type}_{f}f{p}p{v}v{q}q{d}d"


@MODELS.register("mix9")
class Mix9Net(QuantizedHeadMixin, TraceableModel):
    # Mapping contains many large 1x1 convolutions.  On Ada, explicit GEMM
    # lowering improves both forward and backward without changing parameters.
    inductor_config = _MAPPING_GEMM_INDUCTOR_CONFIG

    def __init__(
        self,
        dim_middle=128,
        dim_feature=64,
        dim_policy=32,
        dim_value=64,
        dim_dwconv=32,
        input_type="basicns",
        one_mapping=False,
        no_star_block=False,
        no_dynamic_pwconv=False,
        no_value_group=False,
        use_channel_last=True,
    ):
        super().__init__()
        self.model_size = (dim_middle, dim_feature, dim_policy, dim_value, dim_dwconv)
        self.input_type = input_type
        self.one_mapping = one_mapping
        self.no_star_block = no_star_block
        self.no_dynamic_pwconv = no_dynamic_pwconv
        self.no_value_group = no_value_group
        assert dim_dwconv <= dim_feature, f"Invalid dim_dwconv {dim_dwconv}"
        assert dim_dwconv >= dim_policy, "dim_dwconv must be not less than dim_policy"

        self.input_plane = build_input_plane(input_type)
        if one_mapping:
            self.mapping0 = Mapping(
                self.input_plane.dim_plane, dim_middle, dim_feature, use_channel_last=use_channel_last
            )
        else:
            self.mapping1 = Mapping(
                self.input_plane.dim_plane, dim_middle, dim_feature, use_channel_last=use_channel_last
            )
            self.mapping2 = Mapping(
                self.input_plane.dim_plane, dim_middle, dim_feature, use_channel_last=use_channel_last
            )

        # feature depth-wise conv
        self.feature_dwconv = Conv2dBlock(
            dim_dwconv,
            dim_dwconv,
            ks=3,
            st=1,
            padding=3 // 2,
            groups=dim_dwconv,
            activation="relu",
            quant="pixel-dwconv-floor",
            input_quant_scale=128,
            input_quant_bits=16,
            weight_quant_scale=65536,
            weight_quant_bits=16,
            bias_quant_scale=128,
            bias_quant_bits=16,
            weight_scale=0.25,
        )

        self._init_quantized_policy_head(dim_feature, dim_policy, no_dynamic_pwconv)
        self._init_quantized_grouped_value_head(dim_feature, dim_value, no_value_group, no_star_block)

    def get_feature(self, data, inv_side=False):
        # get the input plane from board and side to move input
        input_plane = self.input_plane(data, inv_side)  # [B, 2, H, W]
        # get per-point 4-direction cell features
        if self.one_mapping:
            feature = self.mapping0(input_plane)  # [B, 4, dim_feature, H, W]
        else:
            feature1 = self.mapping1(input_plane, dirs=[0, 1])  # [B, 2, dim_feature, H, W]
            feature2 = self.mapping2(input_plane, dirs=[2, 3])  # [B, 2, dim_feature, H, W]
            feature = torch.cat([feature1, feature2], dim=1)  # [B, 4, dim_feature, H, W]
        self._trace("feature.directional", feature)

        # clamp feature for int quantization
        feature = torch.clamp(feature, min=-16, max=511 / 32)  # int16, scale=32, [-16,16]
        feature = fake_quant(feature, scale=32, num_bits=16)
        # sum (and rescale) feature across four directions
        feature = torch.mean(feature, dim=1)  # [B, dim_feature, H, W] int16, scale=128, [-16,16]
        self._trace("feature.reduced", feature)
        # apply relu activation
        feature = F.relu(feature)  # [B, dim_feature, H, W] int16, scale=128, [0,16]
        self._trace("feature.activated", feature)

        # apply feature depth-wise conv
        _, _, _, _, dim_dwconv = self.model_size
        feat_dwconv = feature[:, :dim_dwconv]  # int16, scale=128, [0,16]
        feat_dwconv = self.feature_dwconv(feat_dwconv * 4)  # [B, dwconv, H, W] relu
        feat_dwconv = fake_quant(feat_dwconv, scale=128, num_bits=16)  # int16, scale=128, [0,9/2*16*4]
        self._trace("feature.dwconv", feat_dwconv)

        # apply activation for direct feature
        feat_direct = feature[:, dim_dwconv:]  # [B, dim_feature-dwconv, H, W] int16, scale=128, [0,16]
        feat_direct = fake_quant(feat_direct, scale=128, num_bits=16)  # int16, scale=128, [0,16]

        feature = torch.cat([feat_dwconv, feat_direct], dim=1)  # [B, dim_feature, H, W]
        self._trace("feature.output", feature)

        return feature

    def forward(self, data):
        _, _, dim_policy, _, _ = self.model_size

        # get feature from single side
        feature = self.get_feature(data, False)  # [B, dim_feature, H, W]
        B, _, H, W = feature.shape

        # global feature accumulator
        feature_sum = torch.sum(feature, dim=(2, 3))  # [B, dim_feature]
        self._trace("feature.sum.raw", feature_sum)
        feature_sum = fake_quant(feature_sum / 256, scale=128, num_bits=32, floor=True)  # srai 8
        self._trace("feature.sum.quantized", feature_sum)

        # policy head
        policy = self._forward_quantized_policy(feature, feature_sum, dim_policy, self.no_dynamic_pwconv)

        if self.no_value_group:
            value = feature_sum
            for i, linear in enumerate(self.value_linear):
                self._trace(f"value.linear.{i}.input", value)
                value = linear(value)
                self._trace(f"value.linear.{i}.output", value)
        else:
            # value feature accumulator of nine groups
            regions = (
                feature_00,
                feature_01,
                feature_02,
                feature_10,
                feature_11,
                feature_12,
                feature_20,
                feature_21,
                feature_22,
            ) = quantized_sum_3x3_regions(feature)
            self._trace("feature.region.00", feature_00)
            self._trace("feature.region.01", feature_01)
            self._trace("feature.region.02", feature_02)
            self._trace("feature.region.10", feature_10)
            self._trace("feature.region.11", feature_11)
            self._trace("feature.region.12", feature_12)
            self._trace("feature.region.20", feature_20)
            self._trace("feature.region.21", feature_21)
            self._trace("feature.region.22", feature_22)

            value = self._forward_quantized_grouped_value(feature_sum, regions)

        return {"value": value, "policy": policy}

    @property
    def weight_clipping(self):
        # Clip prelu weight of mapping activation to [-1,1] to avoid overflow
        # In this range, prelu is the same as `max(x, ax)`.
        weight_clipping_list = [
            {
                "params": ["feature_dwconv.conv.weight"],
                "min_weight": -32768 / 65536,
                "max_weight": 32767 / 65536,
            },
            *self._quantized_head_weight_clipping(
                self.no_value_group, self.no_star_block, self.no_dynamic_pwconv
            ),
        ]

        return weight_clipping_list

    @property
    def name(self):
        _, f, p, v, d = self.model_size
        return f"mix9_{self.input_type}_{f}f{p}p{v}v{d}d"


@MODELS.register("mix9s")
class Mix9sNet(QuantizedHeadMixin, TraceableModel):
    inductor_config = _MAPPING_GEMM_INDUCTOR_CONFIG

    def __init__(
        self,
        dim_middle=128,
        dim_feature=64,
        dim_policy=32,
        dim_value=64,
        dim_dwconv=32,
        input_type="basicns",
    ):
        super().__init__()
        self.model_size = (dim_middle, dim_feature, dim_policy, dim_value, dim_dwconv)
        self.input_type = input_type
        assert dim_dwconv <= dim_feature, f"Invalid dim_dwconv {dim_dwconv}"
        assert dim_dwconv >= dim_policy, "dim_dwconv must be not less than dim_policy"

        self.input_plane = build_input_plane(input_type)
        dim_input = self.input_plane.dim_plane
        self.mapping1 = Mapping(
            dim_input, dim_middle, dim_feature, use_channel_last=True, fix_direction_order=True
        )
        self.mapping2 = Mapping(
            dim_input, dim_middle, dim_feature, use_channel_last=True, fix_direction_order=True
        )

        # feature depth-wise conv
        self.feature_dwconv = Conv2dBlock(
            dim_dwconv,
            dim_dwconv,
            ks=3,
            st=1,
            padding=3 // 2,
            groups=dim_dwconv,
            activation="relu",
            quant="pixel-dwconv-floor",
            input_quant_scale=128,
            input_quant_bits=16,
            weight_quant_scale=65536,
            weight_quant_bits=16,
            bias_quant_scale=128,
            bias_quant_bits=16,
            weight_scale=0.25,
        )

        self._init_quantized_policy_head(dim_feature, dim_policy)
        self._init_quantized_grouped_value_head(dim_feature, dim_value)

    def do_feature_quantization(self, feature, data):
        return feature, {}, {}  # Not implemented

    def get_feature(self, data, inv_side=False):
        # get the input plane from board and side to move input
        input_plane = self.input_plane(data, inv_side)  # [B, 2, H, W]

        # get per-point 4-direction cell features
        feature1 = self.mapping1(input_plane, dirs=[0, 1])  # [B, 2, dim_feature, H, W]
        feature2 = self.mapping2(input_plane, dirs=[2, 3])  # [B, 2, dim_feature, H, W]
        feature = torch.cat([feature1, feature2], dim=1)  # [B, 4, dim_feature, H, W]
        self._trace("feature.directional", feature)
        feature = torch.clamp(feature, min=-511 / 32, max=511 / 32)  # [-511/32,511/32]

        # do feature quantization
        feature, aux_losses, aux_outputs = self.do_feature_quantization(feature, data)

        # clamp feature for int quantization
        feature = torch.clamp(feature, min=-16, max=511 / 32)  # [-512/32,511/32]
        feature = fake_quant(feature, scale=32, num_bits=16)  # int16, scale=32, [-16,511/32]
        # sum (and rescale) feature across four directions
        feature = torch.mean(feature, dim=1)  # [B, dim_feature, H, W] int16, scale=128, [-16,16]
        self._trace("feature.reduced", feature)
        # apply relu activation
        feature = F.relu(feature)  # [B, dim_feature, H, W] int16, scale=128, [0,16]
        self._trace("feature.activated", feature)

        # apply feature depth-wise conv
        _, _, _, _, dim_dwconv = self.model_size
        feat_dwconv = feature[:, :dim_dwconv]  # int16, scale=128, [0,16]
        feat_dwconv = self.feature_dwconv(feat_dwconv * 4)  # [B, dwconv, H, W] relu
        feat_dwconv = fake_quant(feat_dwconv, scale=128, num_bits=16)  # int16, scale=128, [0,9/2*16*4]
        self._trace("feature.dwconv", feat_dwconv)

        # apply activation for direct feature
        feat_direct = feature[:, dim_dwconv:]  # [B, dim_feature-dwconv, H, W] int16, scale=128, [0,16]
        feat_direct = fake_quant(feat_direct, scale=128, num_bits=16)  # int16, scale=128, [0,16]

        feature = torch.cat([feat_dwconv, feat_direct], dim=1)  # [B, dim_feature, H, W]
        self._trace("feature.output", feature)

        return feature, aux_losses, aux_outputs

    def forward(self, data):
        _, _, dim_policy, _, _ = self.model_size

        # get feature from single side
        feature, aux_losses, aux_outputs = self.get_feature(data, False)  # [B, dim_feature, H, W]
        B, _, H, W = feature.shape

        # global feature accumulator
        feature_sum = torch.sum(feature, dim=(2, 3))  # [B, dim_feature]
        self._trace("feature.sum.raw", feature_sum)
        feature_sum = fake_quant(feature_sum / 256, scale=128, num_bits=32, floor=True)  # srai 8
        self._trace("feature.sum.quantized", feature_sum)

        # policy head
        policy = self._forward_quantized_policy(feature, feature_sum, dim_policy)

        # value feature accumulator of nine groups
        regions = (
            feature_00,
            feature_01,
            feature_02,
            feature_10,
            feature_11,
            feature_12,
            feature_20,
            feature_21,
            feature_22,
        ) = quantized_sum_3x3_regions(feature)
        self._trace("feature.region.00", feature_00)
        self._trace("feature.region.01", feature_01)
        self._trace("feature.region.02", feature_02)
        self._trace("feature.region.10", feature_10)
        self._trace("feature.region.11", feature_11)
        self._trace("feature.region.12", feature_12)
        self._trace("feature.region.20", feature_20)
        self._trace("feature.region.21", feature_21)
        self._trace("feature.region.22", feature_22)

        value = self._forward_quantized_grouped_value(feature_sum, regions)

        return {"value": value, "policy": policy, "aux_losses": aux_losses, "aux_outputs": aux_outputs}

    @property
    def weight_clipping(self):
        weight_clipping_list = [
            {
                "params": ["feature_dwconv.conv.weight"],
                "min_weight": -32768 / 65536,
                "max_weight": 32767 / 65536,
            },
            *self._quantized_head_weight_clipping(),
        ]

        return weight_clipping_list

    @property
    def name(self):
        _, f, p, v, d = self.model_size
        return f"mix9s_{self.input_type}_{f}f{p}p{v}v{d}d"


@MODELS.register("mix9svq")
class Mix9sVQNet(Mix9sNet):
    # VQ search dominates this graph; forcing the inherited GEMM lowering was
    # slower in repeated complete-step measurements.
    inductor_config = {}

    def __init__(
        self,
        dim_middle=128,
        dim_feature=64,
        dim_policy=32,
        dim_value=64,
        dim_dwconv=32,
        input_type="basicns",
        codebook_size=16384,
        num_codebooks=1,
        **vq_kwargs,
    ):
        from .vq import ProductVectorQuantize, rotate_to

        super().__init__(
            dim_middle=dim_middle,
            dim_feature=dim_feature,
            dim_policy=dim_policy,
            dim_value=dim_value,
            dim_dwconv=dim_dwconv,
            input_type=input_type,
        )
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks
        self.vq_layers = nn.ModuleList(
            [
                ProductVectorQuantize(
                    codebook_size=codebook_size,
                    dim_feature=dim_feature,
                    num_codebooks=num_codebooks,
                    **vq_kwargs,
                )
                for _ in range(2)
            ]
        )
        # The VQ path itself has data-dependent unique-vector shapes and stays
        # eager, but its per-position rotation has a stable graph. Trainers
        # replace this eager reference through configure_compilation().
        self._rotate_to = rotate_to
        self._cached_positions = None
        self.register_buffer(
            "_vq_quantile_levels",
            torch.tensor([0.01, 0.1, 0.5, 0.9, 0.99]),
            persistent=False,
        )

    def configure_compilation(self, compile_fn):
        """Compile the rotation STE with a VQ-specific Dynamo policy."""
        from .vq import rotate_to

        # Inductor's max-autotune mode currently selects a slower forward and
        # backward kernel for the large [B*2*H*W, C] rotation workload. Keep
        # the configured backend, but use its default kernel policy here.
        self._rotate_to = compile_fn(
            rotate_to,
            mode="default",
            fullgraph=True,
            dynamic=True,
        )

    @torch.compiler.disable
    def _owned_unique_mask(self, unique_ids, unique_real):
        if not dist.is_available() or not dist.is_initialized():
            return unique_real
        real_ids = unique_ids[unique_real]
        local_count = torch.tensor(
            [len(real_ids)], dtype=torch.long, device=unique_ids.device
        )
        counts = [
            torch.zeros_like(local_count) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(counts, local_count)
        max_count = max(int(count.item()) for count in counts)
        if max_count == 0:
            return unique_real
        padded = unique_ids.new_zeros(max_count)
        if len(real_ids):
            padded[: len(real_ids)] = real_ids
        gathered = [torch.empty_like(padded) for _ in counts]
        dist.all_gather(gathered, padded)
        owned = unique_real.clone()
        for lower_rank in range(dist.get_rank()):
            count = int(counts[lower_rank].item())
            if count:
                owned &= ~torch.isin(
                    unique_ids, gathered[lower_rank][:count]
                )
        return owned

    @torch.compiler.disable
    def _quantize_line_feature(
        self, feature, line_encoding, vq_layer_idx, real_mask=None
    ):
        """
        Args:
            feature: [batch_size, num_directions, dim_feature, H, W].
            line_encoding: [batch_size, num_directions, H, W], dtype=int32.
        Returns:
            feature_vq: [batch_size, num_directions, dim_feature, H, W].
            info: A dict of quantization losses and aux outputs.
        """
        batch_size, num_directions, dim_feature, H, W = feature.shape
        feature = feature.permute(0, 1, 3, 4, 2).reshape(-1, dim_feature)  # [B*D*H*W, dim_feature]
        line_encoding = line_encoding.reshape(-1)  # [B*D*H*W]

        # remove duplicate line encoding and only keep the unique encodings
        line_encoding_unique, inverse_indices = torch.unique(
            line_encoding, sorted=True, return_inverse=True
        )  # [num_unique], [B*D*H*W]
        if real_mask is None:
            unique_real = None
        else:
            if real_mask.ndim != 1 or len(real_mask) != batch_size:
                raise ValueError(
                    "is_real must contain one flag per VQ evaluation row"
                )
            entry_real = (
                real_mask.bool()[:, None, None, None]
                .expand(batch_size, num_directions, H, W)
                .reshape(-1)
            )
            unique_real = torch.zeros(
                len(line_encoding_unique),
                dtype=torch.long,
                device=feature.device,
            )
            unique_real.scatter_reduce_(
                0,
                inverse_indices,
                entry_real.long(),
                reduce="amax",
                include_self=False,
            )
            unique_real = unique_real.bool()
            if not all(
                module._collect_eval_stats
                for module in self.vq_layers[vq_layer_idx].vq_modules
            ):
                unique_real.zero_()
            unique_real = self._owned_unique_mask(
                line_encoding_unique, unique_real
            )

        # get the first occurrence index of each unique line encoding
        # line_encoding_unique.shape == [num_unique]
        # inverse_indices.shape      == [N]   (N = batch_size*num_directions*H*W)
        N = inverse_indices.numel()
        if self._cached_positions is None or self._cached_positions.shape[0] != N:
            self._cached_positions = torch.arange(N, device=feature.device)

        # start with something larger than any real position
        first_occurrence_indices = torch.full_like(line_encoding_unique, fill_value=N, dtype=torch.long)

        # min-reduce the position for every unique id
        first_occurrence_indices.scatter_reduce_(
            0, inverse_indices, self._cached_positions, reduce="amin", include_self=False
        )

        # gather features based on the unique line encoding
        feature_per_line_encoding = feature[first_occurrence_indices]  # [num_unique, dim_feature]

        # quantize the features
        for module in self.vq_layers[vq_layer_idx].vq_modules:
            module.set_eval_entry_mask(unique_real)
        feature_per_line_encoding_vq, info, _ = self.vq_layers[vq_layer_idx](
            feature_per_line_encoding
        )

        # Gather quantized values back to all positions, but apply the
        # straight-through gradient at every occurrence. Mapping features are
        # a pure function of the 11-cell line encoding, so occurrences with
        # the same encoding are identical. This is gradient-equivalent for
        # mapping parameters and avoids the highly contended atomic scatter
        # produced by gather's backward into one representative occurrence.
        feature_vq_target = feature_per_line_encoding_vq[inverse_indices].detach()
        vq_module = self.vq_layers[vq_layer_idx].vq_modules[0]
        if vq_module.rotation_trick:
            feature_groups = feature.chunk(self.num_codebooks, dim=-1)
            target_groups = feature_vq_target.chunk(self.num_codebooks, dim=-1)
            feature_vq = torch.cat(
                [self._rotate_to(src, tgt) for src, tgt in zip(feature_groups, target_groups)],
                dim=-1,
            )
        else:
            feature_vq = feature + (feature_vq_target - feature).detach()
        feature_vq = feature_vq.reshape(batch_size, num_directions, H, W, dim_feature)
        feature_vq = feature_vq.permute(0, 1, 4, 2, 3)  # [B, num_directions, dim_feature, H, W]

        return feature_vq, info

    @torch.compiler.disable
    def do_feature_quantization(self, feature, data):
        line_encoding = data["line_encoding"]  # [B, 4, H, W]

        real_mask = None if self.training else data.get("is_real")
        if real_mask is None and not self.training:
            real_mask = torch.ones(
                len(line_encoding), dtype=torch.bool, device=line_encoding.device
            )
        feature_vq_1, info_1 = self._quantize_line_feature(
            feature[:, :2], line_encoding[:, :2], 0, real_mask
        )
        feature_vq_2, info_2 = self._quantize_line_feature(
            feature[:, 2:], line_encoding[:, 2:], 1, real_mask
        )
        feature_vq = torch.cat([feature_vq_1, feature_vq_2], dim=1)  # [B, 4, dim_feature, H, W]

        aux_losses = {}
        aux_outputs = {}
        if info_1 is not None and info_2 is not None:
            value = (info_1["loss"] + info_2["loss"]) / 2
            slots = []
            for layer_index, info in enumerate((info_1, info_2)):
                for codebook_index, stats in enumerate(info["loss_stats"]):
                    if stats is None:
                        continue
                    slots.append(
                        {
                            **stats,
                            "slot_id": (
                                f"layer{layer_index:02d}."
                                f"codebook{codebook_index:02d}"
                            ),
                            "slot_weight": 1.0
                            / (2 * self.num_codebooks),
                        }
                    )
            aux_losses = (
                {"vq": value}
                if self.training
                else {"vq": ("vq_loss", {"value": value, "slots": slots})}
            )
            cluster_size = torch.cat(
                [
                    self.vq_layers[0].normalized_cluster_size,
                    self.vq_layers[1].normalized_cluster_size,
                ]
            )
            cluster_size_quantiles = torch.quantile(
                cluster_size,
                q=self._vq_quantile_levels,
            )
            aux_outputs = {
                "vq_perplexity": (info_1["perplexity"] + info_2["perplexity"]) / 2,
                "vq_normed_perplexity": (info_1["normalized_perplexity"] + info_2["normalized_perplexity"])
                / 2,
                "vq_cluster_size_q01": cluster_size_quantiles[0],
                "vq_cluster_size_q10": cluster_size_quantiles[1],
                "vq_cluster_size_q50": cluster_size_quantiles[2],
                "vq_cluster_size_q90": cluster_size_quantiles[3],
                "vq_cluster_size_q99": cluster_size_quantiles[4],
                "vq_num_expired_codes": info_1["num_expired_codes"] + info_2["num_expired_codes"],
            }

        return feature_vq, aux_losses, aux_outputs

    @property
    def name(self):
        _, f, p, v, d = self.model_size
        return f"mix9svq_{self.input_type}_{f}f{p}p{v}v{d}d{self.codebook_size}c"


@MODELS.register("mix10")
class Mix10Net(TraceableModel):
    inductor_config = _MAPPING_GEMM_INDUCTOR_CONFIG

    def __init__(
        self,
        dim_middle=128,
        dim_feature=64,
        dim_dwconv=32,
        dim_value=64,
        input_type="basicns",
        mapping_norm="none",
        feature_norm="none",
        spherical_feature=False,
    ):
        super().__init__()
        self.model_size = (dim_middle, dim_feature, dim_dwconv, dim_value)
        self.input_type = input_type
        self.spherical_feature = spherical_feature
        assert dim_dwconv <= dim_feature, f"Invalid dim_dwconv {dim_dwconv}"
        assert dim_value <= dim_feature, f"Invalid dim_value {dim_value}"

        self.input_plane = build_input_plane(input_type)
        dim_input = self.input_plane.dim_plane
        self.mapping1 = Mapping(
            dim_input,
            dim_middle,
            dim_feature,
            use_channel_last=True,
            fix_direction_order=True,
            normalization=mapping_norm,
        )
        self.mapping2 = Mapping(
            dim_input,
            dim_middle,
            dim_feature,
            use_channel_last=True,
            fix_direction_order=True,
            normalization=mapping_norm,
        )

        # feature depth-wise conv
        self.feature_dwconv = Conv2dBlock(
            dim_dwconv,
            dim_dwconv,
            ks=3,
            st=1,
            padding=3 // 2,
            groups=dim_dwconv,
            activation="relu",
            quant="pixel-dwconv-floor",
            input_quant_scale=128,
            input_quant_bits=16,
            weight_quant_scale=65536,
            weight_quant_bits=16,
            bias_quant_scale=128,
            bias_quant_bits=16,
            weight_scale=0.25,
        )
        self.feature_norm = build_norm2d_layer(feature_norm, dim_dwconv)

        # policy head (point-wise conv) small
        dim_policy_small_in = max(dim_dwconv // 2, 16)
        dim_policy_small_out = max(dim_dwconv // 4, 16)
        self.policy_small_pwconv_weight = nn.Sequential(
            LinearBlock(dim_value, dim_value, quant=True),
            LinearBlock(
                dim_value,
                dim_policy_small_out * (dim_policy_small_in + 1),
                activation="none",
                quant=True,
            ),
        )
        self.policy_small_output = Conv2d(dim_policy_small_out, 1, 1)

        # policy head (point-wise conv) large
        dim_policy_large_in = max(dim_dwconv, 16)
        dim_policy_large_mid = max(dim_dwconv // 2, 16)
        dim_policy_large_out = max(dim_dwconv // 4, 16)
        self.policy_large_pwconv_weight_0 = LinearBlock(dim_value, dim_value, quant=True)
        self.policy_large_pwconv_weight_1 = LinearBlock(
            dim_value,
            dim_policy_large_mid * (dim_policy_large_in + 1),
            activation="none",
            quant=True,
        )
        self.policy_large_pwconv_weight_2 = LinearBlock(
            dim_value,
            dim_policy_large_out * (dim_policy_large_mid + 1),
            activation="none",
            quant=True,
        )
        self.policy_large_output = Conv2d(dim_policy_large_out, 1, 1)

        # value head small
        self.value_linear_small = nn.Sequential(
            LinearBlock(dim_feature, dim_value, quant=True),
            LinearBlock(dim_value, dim_value, quant=True),
        )
        self.value_small_output = LinearBlock(dim_value, 3, activation="none", quant=True)

        # value head large
        self.value_gate = LinearBlock(dim_value, dim_feature * 2, activation="none", quant=True)
        self.value_corner = LinearBlock(dim_feature, dim_value, quant=True)
        self.value_edge = LinearBlock(dim_feature, dim_value, quant=True)
        self.value_center = LinearBlock(dim_feature, dim_value, quant=True)
        self.value_quad = LinearBlock(dim_value, dim_value, quant=True)

        self.value_linear_large = nn.Sequential(
            LinearBlock(dim_value * 5, dim_value, quant=True),
            LinearBlock(dim_value, dim_value, quant=True),
        )
        self.value_large_output = LinearBlock(dim_value, 3, activation="none", quant=True)

    def do_feature_quantization(self, feature, data):
        return feature, {}, {}  # Not implemented

    def get_feature(self, data, inv_side=False):
        # get the input plane from board and side to move input
        input_plane = self.input_plane(data, inv_side)  # [B, 2, H, W]
        if not isinstance(input_plane, tuple):
            input_plane = (input_plane,)

        # get per-point 4-direction cell features
        feature1 = self.mapping1(*input_plane, dirs=[0, 1])  # [B, 2, dim_feature, H, W]
        feature2 = self.mapping2(*input_plane, dirs=[2, 3])  # [B, 2, dim_feature, H, W]
        feature = torch.cat([feature1, feature2], dim=1)  # [B, 4, dim_feature, H, W]
        self._trace("feature.directional", feature)
        # normalize feature onto hypersphere of radius 16
        if self.spherical_feature:
            feature = F.normalize(feature, p=2, dim=2) * (511 / 32)
        feature = torch.clamp(feature, min=-511 / 32, max=511 / 32)  # [-511/32,511/32]

        # do feature quantization
        feature, aux_losses, aux_outputs = self.do_feature_quantization(feature, data)

        # clamp feature for int quantization
        feature = torch.clamp(feature, min=-16, max=511 / 32)  # [-512/32,511/32]
        feature = fake_quant(feature, scale=32, num_bits=16)  # int16, scale=32, [-16,511/32]
        # sum (and rescale) feature across four directions
        feature = torch.mean(feature, dim=1)  # [B, dim_feature, H, W] int16, scale=128, [-16,16]
        self._trace("feature.reduced", feature)
        # apply relu activation
        feature = F.relu(feature)  # [B, dim_feature, H, W] int16, scale=128, [0,16]
        self._trace("feature.activated", feature)

        # apply feature depth-wise conv
        _, _, dim_dwconv, _ = self.model_size
        feat_dwconv = feature[:, :dim_dwconv]  # int16, scale=128, [0,16]
        feat_dwconv = self.feature_dwconv(feat_dwconv * 4)  # [B, dwconv, H, W] relu
        feat_dwconv = fake_quant(feat_dwconv, scale=128, num_bits=16)  # int16, scale=128, [0,9/2*16*4]
        self._trace("feature.dwconv", feat_dwconv)

        # apply mask to the feature after dwconv
        if self.feature_norm is not None:
            feat_dwconv = self.feature_norm(feat_dwconv, *input_plane[1:])

        # apply activation for direct feature
        feat_direct = feature[:, dim_dwconv:]  # [B, dim_feature-dwconv, H, W] int16, scale=128, [0,16]
        feat_direct = fake_quant(feat_direct, scale=128, num_bits=16)  # int16, scale=128, [0,16]

        feature = torch.cat([feat_dwconv, feat_direct], dim=1)  # [B, dim_feature, H, W]
        self._trace("feature.output", feature)

        return feature, aux_losses, aux_outputs, *input_plane[1:]

    def value_head_small(self, feature):
        # global feature accumulator
        feature_sum = torch.sum(feature, dim=(2, 3))  # [B, dim_feature]
        feature_sum = fake_quant(feature_sum / 256, scale=128, num_bits=32, floor=True)  # srai 8

        # (shared) small value head
        value_small_feature = self.value_linear_small(feature_sum)  # [B, dim_value]

        # small value output head
        value = self.value_small_output(value_small_feature)  # [B, 3]

        return value, value_small_feature

    def value_head_large(self, feature, value_small_feature):
        # value modulation
        feature_mod = self.value_gate(value_small_feature)  # [B, dim_feature * 2]
        feature_mod = fake_quant(feature_mod, scale=128, num_bits=8, floor=True)

        # value feature accumulator of nine groups
        def get_group_feature(region, mod):
            f = torch.sum(region, dim=(2, 3))  # [B, dim_feature]
            f = fake_quant(f / 32, scale=128, num_bits=8, floor=True)  # srai 5
            f = torch.cat([f, f], dim=1)  # [B, dim_feature * 2]
            # i16 dot product of two adjacent pairs of u8 and i8
            x = (f * mod).view(*f.shape[:-1], -1, 2).sum(-1)
            # clamp to int8, scale=128, [-1,1]
            x = fake_quant(x, scale=128, num_bits=8, floor=True)
            return x

        (
            feature_00,
            feature_01,
            feature_02,
            feature_10,
            feature_11,
            feature_12,
            feature_20,
            feature_21,
            feature_22,
        ) = tuple(get_group_feature(region, feature_mod) for region in split_3x3_regions(feature))
        value_00 = self.value_corner(feature_00)  # [B, dim_value]
        value_01 = self.value_edge(feature_01)  # [B, dim_value]
        value_02 = self.value_corner(feature_02)  # [B, dim_value]
        value_10 = self.value_edge(feature_10)  # [B, dim_value]
        value_11 = self.value_center(feature_11)  # [B, dim_value]
        value_12 = self.value_edge(feature_12)  # [B, dim_value]
        value_20 = self.value_corner(feature_20)  # [B, dim_value]
        value_21 = self.value_edge(feature_21)  # [B, dim_value]
        value_22 = self.value_corner(feature_22)  # [B, dim_value]

        value_q00 = avg4(value_00, value_01, value_10, value_11)
        value_q01 = avg4(value_01, value_02, value_11, value_12)
        value_q10 = avg4(value_10, value_11, value_20, value_21)
        value_q11 = avg4(value_11, value_12, value_21, value_22)
        value_q00 = self.value_quad(value_q00)
        value_q01 = self.value_quad(value_q01)
        value_q10 = self.value_quad(value_q10)
        value_q11 = self.value_quad(value_q11)

        # (shared) large value head
        value_large_feature = torch.cat(
            [value_small_feature, value_q00, value_q01, value_q10, value_q11], dim=1
        )  # [B, dim_value * 5]
        value_large_feature = self.value_linear_large(value_large_feature)  # [B, dim_feature]

        # large value output head
        value = self.value_large_output(value_large_feature)  # [B, 3]

        return value, value_large_feature

    def policy_head_small(self, feature, value_small_feature):
        _, _, dim_dwconv, _ = self.model_size
        dim_policy_small_in = max(dim_dwconv // 2, 16)
        dim_policy_small_out = max(dim_dwconv // 4, 16)
        num_policy_weight = dim_policy_small_in * dim_policy_small_out

        B, _, H, W = feature.shape
        pwconv_output = self.policy_small_pwconv_weight(value_small_feature)
        pwconv_weight = pwconv_output[:, :num_policy_weight].reshape(B, num_policy_weight, 1, 1)
        pwconv_weight = fake_quant(pwconv_weight, scale=128 * 128, num_bits=16, floor=True)
        pwconv_bias = pwconv_output[:, num_policy_weight:].reshape(
            B, dim_policy_small_out, 1, 1
        )  # int32, scale=128*128*128

        policy = fake_quant(
            feature[:, :dim_policy_small_in], scale=128, num_bits=16
        )  # [B, dim_policy_small_in, H, W]
        policy = dynamic_pointwise_conv2d(policy, pwconv_weight)
        policy = torch.clamp(
            policy + pwconv_bias, min=0
        )  # [B, dim_policy_small_out, H, W] int32, scale=128*128*128, relu
        policy = self.policy_small_output(policy)  # [B, 1, H, W]

        return policy

    def policy_head_large(self, feature, value_small_feature):
        _, _, dim_dwconv, _ = self.model_size
        dim_policy_large_in = max(dim_dwconv, 16)
        dim_policy_large_mid = max(dim_dwconv // 2, 16)
        dim_policy_large_out = max(dim_dwconv // 4, 16)
        num_policy_weight_1 = dim_policy_large_in * dim_policy_large_mid
        num_policy_weight_2 = dim_policy_large_mid * dim_policy_large_out

        B, _, H, W = feature.shape
        pwconv_shared = self.policy_large_pwconv_weight_0(value_small_feature)
        pwconv_output_1 = self.policy_large_pwconv_weight_1(pwconv_shared)
        pwconv_weight_1 = pwconv_output_1[:, :num_policy_weight_1].reshape(B, num_policy_weight_1, 1, 1)
        pwconv_weight_1 = fake_quant(pwconv_weight_1, scale=128 * 128, num_bits=16, floor=True)
        pwconv_bias_1 = pwconv_output_1[:, num_policy_weight_1:].reshape(
            B, dim_policy_large_mid, 1, 1
        )  # int32, scale=128*128*128

        policy = fake_quant(
            feature[:, :dim_policy_large_in], scale=128, num_bits=16
        )  # [B, dim_policy_large_in, H, W]
        policy = dynamic_pointwise_conv2d(policy, pwconv_weight_1)
        policy = torch.clamp(
            policy + pwconv_bias_1, min=0
        )  # [B, dim_policy_large_mid, H, W] int32, scale=128*128*128, relu

        pwconv_output_2 = self.policy_large_pwconv_weight_2(pwconv_shared)
        pwconv_weight_2 = pwconv_output_2[:, :num_policy_weight_2].reshape(B, num_policy_weight_2, 1, 1)
        pwconv_weight_2 = fake_quant(pwconv_weight_2, scale=128 * 128, num_bits=16, floor=True)
        pwconv_bias_2 = pwconv_output_2[:, num_policy_weight_2:].reshape(
            B, dim_policy_large_out, 1, 1
        )  # int32, scale=128*128*128

        policy = fake_quant(policy, scale=128, num_bits=16, floor=True)  # [B, dim_policy_large_mid, H, W]
        policy = dynamic_pointwise_conv2d(policy, pwconv_weight_2)
        policy = torch.clamp(
            policy + pwconv_bias_2, min=0
        )  # [B, dim_policy_large_out, H, W] int32, scale=128*128*128, relu

        policy = self.policy_large_output(policy)  # [B, 1, H, W]

        return policy

    def forward(self, data):
        # get feature from single side
        feature, aux_losses, aux_outputs, *extra = self.get_feature(data, False)  # [B, dim_feature, H, W]

        # value head
        value_small, value_small_feature = self.value_head_small(feature)
        self._trace("value.small.feature", value_small_feature)
        self._trace("value.small.output", value_small)
        value_large, value_large_feature = self.value_head_large(feature, value_small_feature)
        self._trace("value.large.feature", value_large_feature)
        self._trace("value.large.output", value_large)

        # policy head
        policy_small = self.policy_head_small(feature, value_small_feature)
        self._trace("policy.small.output", policy_small)
        policy_large = self.policy_head_large(feature, value_large_feature)
        self._trace("policy.large.output", policy_large)

        aux_losses.update(
            {
                "value_small": ("value_loss", value_small),
                "policy_small": ("policy_loss", policy_small),
                "policy_small_reg": ("policy_reg", policy_small),
            }
        )
        out = {
            "value": value_large,
            "policy": policy_large,
            "aux_losses": aux_losses,
            "aux_outputs": aux_outputs,
        }
        if extra:
            out["board_mask"] = extra[0]
        return out

    @property
    def weight_clipping(self):
        # Clip prelu weight of mapping activation to [-1,1] to avoid overflow
        # In this range, prelu is the same as `max(x, ax)`.
        return [
            {
                "params": ["feature_dwconv.conv.weight"],
                "min_weight": -32768 / 65536,
                "max_weight": 32767 / 65536,
            },
            {
                "params": [
                    "value_gate.fc.weight",
                    "value_corner.fc.weight",
                    "value_edge.fc.weight",
                    "value_center.fc.weight",
                    "value_quad.fc.weight",
                    "value_linear_small.0.fc.weight",
                    "value_linear_small.1.fc.weight",
                    "value_small_output.fc.weight",
                    "value_linear_large.0.fc.weight",
                    "value_linear_large.1.fc.weight",
                    "value_large_output.fc.weight",
                    "policy_small_pwconv_weight.0.fc.weight",
                    "policy_small_pwconv_weight.1.fc.weight",
                    "policy_large_pwconv_weight_0.fc.weight",
                    "policy_large_pwconv_weight_1.fc.weight",
                    "policy_large_pwconv_weight_2.fc.weight",
                ],
                "min_weight": -128 / 128,
                "max_weight": 127 / 128,
            },
        ]

    @property
    def name(self):
        _, f, d, v = self.model_size
        return f"mix10_{self.input_type}_{f}f{d}d{v}v"
