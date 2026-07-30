"""Reusable structural components shared by MixNet-family models."""

import torch
import torch.nn as nn

from .layers.activation import build_activation_layer
from .layers.container import SequentialWithExtraArguments
from .layers.convolution import Conv2d, Conv2dBlock, PointwiseConv2d
from .layers.directional import DirectionalConvLayer
from .layers.linear import LinearBlock
from .layers.normalization import build_norm2d_layer
from .layers.star import StarBlock
from .ops import dynamic_pointwise_conv2d, quantized_avg4
from utils.quant_utils import fake_quant


def _tuple_op(f, x):
    return tuple((f(xi) if xi is not None else None) for xi in x)


def _add_op(t):
    return None if (t[0] is None and t[1] is None) else t[0] + t[1]


class DirectionalConvResBlock(nn.Module):
    def __init__(self, dim, act, norm, **dirconv_kwargs):
        super().__init__()
        self.d_conv = DirectionalConvLayer(dim, dim, **dirconv_kwargs)
        self.norm = build_norm2d_layer(norm, dim)
        self.conv1x1 = PointwiseConv2d(dim, dim)
        self.activation = build_activation_layer(act)

    def forward(self, x, mask=None):
        residual = x
        x = self.d_conv(x)
        if self.norm is not None:
            x = _tuple_op(lambda t: self.norm(t) if mask is None else self.norm(t, mask=mask), x)
        x = _tuple_op(self.activation, x)
        x = _tuple_op(self.conv1x1, x)
        x = _tuple_op(self.activation, x)
        x = _tuple_op(_add_op, zip(x, residual))
        return x


class Conv0dResBlock(nn.Module):
    def __init__(self, dim, activation, middle_dim_multipler=1):
        super().__init__()
        middle_dim = middle_dim_multipler * dim
        self.conv1 = PointwiseConv2d(dim, middle_dim)
        self.conv2 = PointwiseConv2d(middle_dim, dim)
        self.activation = build_activation_layer(activation)

    def forward(self, x, mask=None):
        residual = x
        x = _tuple_op(self.conv1, x)
        x = _tuple_op(self.activation, x)
        x = _tuple_op(self.conv2, x)
        x = _tuple_op(self.activation, x)
        x = _tuple_op(_add_op, zip(x, residual))
        return x


class Mapping(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_middle,
        dim_out,
        use_channel_last=True,
        fix_direction_order=False,
        activation="silu",
        normalization="none",
        line_length=11,
    ):
        super().__init__()
        self.d_conv = DirectionalConvLayer(
            dim_in,
            dim_middle,
            use_nonzero_padding=False,
            use_channel_last=use_channel_last,
            fix_direction_order=fix_direction_order,
        )
        self.norm = build_norm2d_layer(normalization, dim_middle)
        self.convs = SequentialWithExtraArguments(
            *[
                DirectionalConvResBlock(
                    dim_middle,
                    activation,
                    normalization,
                    use_nonzero_padding=False,
                    use_channel_last=use_channel_last,
                    fix_direction_order=fix_direction_order,
                )
                for _ in range((line_length // 2) - 1)
            ],
            Conv0dResBlock(dim_middle, activation),
        )
        self.final_conv = PointwiseConv2d(dim_middle, dim_out)
        self.activation = build_activation_layer(activation)

    def forward(self, x, mask=None, dirs=[0, 1, 2, 3]):
        x = tuple((x if i in dirs else None) for i in range(4))
        x = self.d_conv(x)
        if self.norm is not None:
            x = _tuple_op(lambda t: self.norm(t) if mask is None else self.norm(t, mask=mask), x)
        x = _tuple_op(self.activation, x)
        x = self.convs(x, mask=mask)
        x = _tuple_op(self.final_conv, x)
        x = tuple(xi for xi in x if xi is not None)
        return torch.stack(x, dim=1)  # [B, <=4, dim_out, H, W]


class QuantizedHeadMixin:
    """Shared construction and execution for quantized policy and value heads."""

    def _init_quantized_policy_head(self, dim_feature, dim_policy, no_dynamic_pwconv=False):
        dim_pm = self.policy_middle_dim = 16
        if no_dynamic_pwconv:
            self.policy_pwconv = Conv2dBlock(
                dim_policy,
                dim_pm,
                ks=1,
                st=1,
                activation="relu",
                quant=True,
                input_quant_scale=128,
                input_quant_bits=16,
                weight_quant_scale=128 * 128,
                weight_quant_bits=16,
                bias_quant_scale=128 * 128 * 128,
                bias_quant_bits=32,
            )
        else:
            self.policy_pwconv_weight_linear = nn.Sequential(
                LinearBlock(dim_feature, dim_policy * 2, activation="relu", quant=True),
                LinearBlock(dim_policy * 2, dim_pm * dim_policy + dim_pm, activation="none", quant=True),
            )
        self.policy_output = Conv2d(dim_pm, 1, 1)

    def _init_quantized_grouped_value_head(
        self, dim_feature, dim_value, no_value_group=False, no_star_block=False
    ):
        if no_value_group:
            self.value_linear = nn.Sequential(
                LinearBlock(dim_feature, dim_value, activation="relu", quant=True),
                LinearBlock(dim_value, dim_value, activation="relu", quant=True),
                LinearBlock(dim_value, 3, activation="none", quant=True),
            )
        else:
            if no_star_block:
                self.value_corner = LinearBlock(dim_feature, dim_value, quant=True)
                self.value_edge = LinearBlock(dim_feature, dim_value, quant=True)
                self.value_center = LinearBlock(dim_feature, dim_value, quant=True)
                self.value_quad = LinearBlock(dim_value, dim_value, quant=True)
            else:
                self.value_corner = StarBlock(dim_feature, dim_value)
                self.value_edge = StarBlock(dim_feature, dim_value)
                self.value_center = StarBlock(dim_feature, dim_value)
                self.value_quad = StarBlock(dim_value, dim_value)
            self.value_linear = nn.Sequential(
                LinearBlock(dim_feature + 4 * dim_value, dim_value, activation="relu", quant=True),
                LinearBlock(dim_value, dim_value, activation="relu", quant=True),
                LinearBlock(dim_value, 3, activation="none", quant=True),
            )

    def _trace_quantized_head(self, name, tensor):
        trace = getattr(self, "_trace", None)
        if trace is not None:
            trace(name, tensor)

    def _forward_quantized_policy(self, feature, feature_sum, dim_policy, no_dynamic_pwconv=False):
        if no_dynamic_pwconv:
            policy = self.policy_pwconv(
                feature[:, :dim_policy]
            )  # [B, dim_pm, H, W] int32, scale=128*128*128, relu
            self._trace_quantized_head("policy.generated", policy)
            self._trace_quantized_head("policy.activated", policy)
        else:
            B = feature.shape[0]
            dim_pm = self.policy_middle_dim
            pwconv_output = self.policy_pwconv_weight_linear(feature_sum)
            self._trace_quantized_head("policy.generated", pwconv_output)
            pwconv_weight = pwconv_output[:, : dim_pm * dim_policy].reshape(B, dim_pm * dim_policy, 1, 1)
            pwconv_weight = fake_quant(pwconv_weight, scale=128 * 128, num_bits=16, floor=True)
            self._trace_quantized_head("policy.weight", pwconv_weight)
            policy = fake_quant(feature[:, :dim_policy], scale=128, num_bits=16)  # [B, dim_policy, H, W]
            self._trace_quantized_head("policy.input", policy)
            policy = dynamic_pointwise_conv2d(policy, pwconv_weight)
            self._trace_quantized_head("policy.dynamic", policy)
            pwconv_bias = pwconv_output[:, dim_pm * dim_policy :].reshape(
                B, dim_pm, 1, 1
            )  # int32, scale=128*128*128
            self._trace_quantized_head("policy.bias", pwconv_bias)
            policy = torch.clamp(
                policy + pwconv_bias, min=0
            )  # [B, dim_pm, H, W] int32, scale=128*128*128, relu
            self._trace_quantized_head("policy.activated", policy)

        policy = self.policy_output(policy)  # [B, 1, H, W]
        self._trace_quantized_head("policy.output", policy)
        return policy

    def _forward_quantized_grouped_value(self, feature_sum, regions):
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
        ) = regions

        value_00 = self.value_corner(feature_00)
        value_01 = self.value_edge(feature_01)
        value_02 = self.value_corner(feature_02)
        value_10 = self.value_edge(feature_10)
        value_11 = self.value_center(feature_11)
        value_12 = self.value_edge(feature_12)
        value_20 = self.value_corner(feature_20)
        value_21 = self.value_edge(feature_21)
        value_22 = self.value_corner(feature_22)
        self._trace_quantized_head("value.region.00", value_00)
        self._trace_quantized_head("value.region.01", value_01)
        self._trace_quantized_head("value.region.02", value_02)
        self._trace_quantized_head("value.region.10", value_10)
        self._trace_quantized_head("value.region.11", value_11)
        self._trace_quantized_head("value.region.12", value_12)
        self._trace_quantized_head("value.region.20", value_20)
        self._trace_quantized_head("value.region.21", value_21)
        self._trace_quantized_head("value.region.22", value_22)

        value_q00 = quantized_avg4(value_00, value_01, value_10, value_11)
        value_q01 = quantized_avg4(value_01, value_02, value_11, value_12)
        value_q10 = quantized_avg4(value_10, value_11, value_20, value_21)
        value_q11 = quantized_avg4(value_11, value_12, value_21, value_22)
        self._trace_quantized_head("value.quad.00.input", value_q00)
        self._trace_quantized_head("value.quad.01.input", value_q01)
        self._trace_quantized_head("value.quad.10.input", value_q10)
        self._trace_quantized_head("value.quad.11.input", value_q11)
        value_q00 = self.value_quad(value_q00)
        value_q01 = self.value_quad(value_q01)
        value_q10 = self.value_quad(value_q10)
        value_q11 = self.value_quad(value_q11)
        self._trace_quantized_head("value.quad.00.output", value_q00)
        self._trace_quantized_head("value.quad.01.output", value_q01)
        self._trace_quantized_head("value.quad.10.output", value_q10)
        self._trace_quantized_head("value.quad.11.output", value_q11)

        value = torch.cat(
            [
                feature_sum,
                value_q00,
                value_q01,
                value_q10,
                value_q11,
            ],
            1,
        )  # [B, dim_feature + dim_value * 4]
        self._trace_quantized_head("value.concat", value)
        for i, linear in enumerate(self.value_linear):
            self._trace_quantized_head(f"value.linear.{i}.input", value)
            value = linear(value)
            self._trace_quantized_head(f"value.linear.{i}.output", value)
        return value

    def _quantized_head_weight_clipping(
        self,
        no_value_group=False,
        no_star_block=False,
        no_dynamic_pwconv=False,
        combine_policy_with_value=False,
    ):
        if no_value_group:
            value_group_weights = []
        elif no_star_block:
            value_group_weights = [
                "value_corner.fc.weight",
                "value_edge.fc.weight",
                "value_center.fc.weight",
                "value_quad.fc.weight",
            ]
        else:
            value_group_weights = [
                "value_corner.up1.fc.weight",
                "value_corner.up2.fc.weight",
                "value_corner.down.fc.weight",
                "value_edge.up1.fc.weight",
                "value_edge.up2.fc.weight",
                "value_edge.down.fc.weight",
                "value_center.up1.fc.weight",
                "value_center.up2.fc.weight",
                "value_center.down.fc.weight",
                "value_quad.up1.fc.weight",
                "value_quad.up2.fc.weight",
                "value_quad.down.fc.weight",
            ]

        value_weights = [
            *value_group_weights,
            "value_linear.0.fc.weight",
            "value_linear.1.fc.weight",
            "value_linear.2.fc.weight",
        ]
        if no_dynamic_pwconv:
            policy_clipping = {
                "params": ["policy_pwconv.conv.weight"],
                "min_weight": -32768 / (128 * 128),
                "max_weight": 32767 / (128 * 128),
            }
        else:
            policy_clipping = {
                "params": [
                    "policy_pwconv_weight_linear.0.fc.weight",
                    "policy_pwconv_weight_linear.1.fc.weight",
                ],
                "min_weight": -128 / 128,
                "max_weight": 127 / 128,
            }

        if combine_policy_with_value:
            value_weights.extend(policy_clipping["params"])
            return [
                {
                    "params": value_weights,
                    "min_weight": -128 / 128,
                    "max_weight": 127 / 128,
                }
            ]

        return [
            {
                "params": value_weights,
                "min_weight": -128 / 128,
                "max_weight": 127 / 128,
            },
            policy_clipping,
        ]


__all__ = [
    "Conv0dResBlock",
    "DirectionalConvResBlock",
    "Mapping",
    "QuantizedHeadMixin",
]
