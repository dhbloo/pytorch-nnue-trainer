import torch
import torch.nn as nn
import torch.nn.functional as F

from . import MODELS
from .blocks import (
    Conv2dBlock,
    LinearBlock,
    ChannelWiseLeakyReLU,
    QuantPReLU,
    SwitchGate,
    SwitchLinearBlock,
    SwitchPReLU,
    SequentialWithExtraArguments,
    build_activation_layer,
    build_norm2d_layer,
)
from .input import build_input_plane
from utils.quant_utils import fake_quant


def tuple_op(f, x):
    return tuple((f(xi) if xi is not None else None) for xi in x)


def add_op(t):
    return None if (t[0] is None and t[1] is None) else t[0] + t[1]


def avg4(a, b, c, d):
    a = fake_quant(a, floor=True)
    b = fake_quant(b, floor=True)
    c = fake_quant(c, floor=True)
    d = fake_quant(d, floor=True)
    ab = fake_quant((a + b + 1 / 128) / 2, floor=True)
    cd = fake_quant((c + d + 1 / 128) / 2, floor=True)
    return fake_quant((ab + cd + 1 / 128) / 2, floor=True)


class DirectionalConvLayer(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        use_nonzero_padding=False,
        use_channel_last=True,
        fix_direction_order=False,
    ):
        super().__init__()
        weight_shape = (dim_out, dim_in, 3) if use_channel_last else (3, dim_out, dim_in)
        self.weight = nn.Parameter(torch.empty(weight_shape))
        self.bias = nn.Parameter(torch.zeros((dim_out,)))
        self.use_nonzero_padding = use_nonzero_padding
        self.use_channel_last = use_channel_last
        self.fix_direction_order = fix_direction_order

    def initialize(self):
        if self.use_channel_last:
            nn.init.kaiming_normal_(self.weight.data)
        else:
            nn.init.kaiming_normal_(self.weight.data.permute(1, 0, 2))
        self.bias.data.zero_()

    def _conv1d_direction(self, x, dir):
        if self.use_channel_last:
            dim_out, dim_in, kernel_size = self.weight.shape
        else:
            kernel_size, dim_out, dim_in = self.weight.shape
        zero = torch.zeros((dim_out, dim_in), dtype=self.weight.dtype, device=self.weight.device)

        if self.use_channel_last:
            w_0_, w_1_, w_2_ = self.weight[..., 0], self.weight[..., 1], self.weight[..., 2]
        else:
            w_0_, w_1_, w_2_ = self.weight[0], self.weight[1], self.weight[2]

        if self.fix_direction_order:
            weight_func_map = [
                lambda w: (zero, zero, zero, w_0_, w_1_, w_2_, zero, zero, zero),
                lambda w: (zero, w_0_, zero, zero, w_1_, zero, zero, w_2_, zero),
                lambda w: (zero, zero, w_2_, zero, w_1_, zero, w_0_, zero, zero),
                lambda w: (w_0_, zero, zero, zero, w_1_, zero, zero, zero, w_2_),
            ]
        else:
            weight_func_map = [
                lambda w: (zero, zero, zero, w_0_, w_1_, w_2_, zero, zero, zero),
                lambda w: (zero, w_0_, zero, zero, w_1_, zero, zero, w_2_, zero),
                lambda w: (w_0_, zero, zero, zero, w_1_, zero, zero, zero, w_2_),
                lambda w: (zero, zero, w_2_, zero, w_1_, zero, w_0_, zero, zero),
            ]

        weight = torch.stack(weight_func_map[dir](self.weight), dim=2)
        weight = weight.reshape(dim_out, dim_in, kernel_size, kernel_size)
        if self.use_nonzero_padding:
            x = F.pad(
                x,
                pad=(kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2),
                mode="constant",
                value=1,
            )
            return torch.conv2d(x, weight, self.bias, padding=0)
        else:
            return torch.conv2d(x, weight, self.bias, padding=kernel_size // 2)

    def forward(self, x):
        assert len(x) == 4, f"must be 4 directions, got {len(x)}"
        return tuple((self._conv1d_direction(xi, i) if xi is not None else None) for i, xi in enumerate(x))


class DirectionalConvResBlock(nn.Module):
    def __init__(self, dim, act, norm, **dirconv_kwargs):
        super().__init__()
        self.d_conv = DirectionalConvLayer(dim, dim, **dirconv_kwargs)
        self.norm = build_norm2d_layer(norm, dim)
        self.conv1x1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.activation = build_activation_layer(act)

    def forward(self, x, mask=None):
        residual = x
        x = self.d_conv(x)
        if self.norm is not None:
            x = tuple_op(lambda t: self.norm(t) if mask is None else self.norm(t, mask=mask), x)
        x = tuple_op(self.activation, x)
        x = tuple_op(self.conv1x1, x)
        x = tuple_op(self.activation, x)
        x = tuple_op(add_op, zip(x, residual))
        return x


class Conv0dResBlock(nn.Module):
    def __init__(self, dim, activation, middle_dim_multipler=1):
        super().__init__()
        middle_dim = middle_dim_multipler * dim
        self.conv1 = nn.Conv2d(dim, middle_dim, kernel_size=1)
        self.conv2 = nn.Conv2d(middle_dim, dim, kernel_size=1)
        self.activation = build_activation_layer(activation)

    def forward(self, x, mask=None):
        residual = x
        x = tuple_op(self.conv1, x)
        x = tuple_op(self.activation, x)
        x = tuple_op(self.conv2, x)
        x = tuple_op(self.activation, x)
        x = tuple_op(add_op, zip(x, residual))
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
        self.final_conv = nn.Conv2d(dim_middle, dim_out, kernel_size=1)
        self.activation = build_activation_layer(activation)

    def forward(self, x, mask=None, dirs=[0, 1, 2, 3]):
        x = tuple((x if i in dirs else None) for i in range(4))
        x = self.d_conv(x)
        if self.norm is not None:
            x = tuple_op(lambda t: self.norm(t) if mask is None else self.norm(t, mask=mask), x)
        x = tuple_op(self.activation, x)
        x = self.convs(x, mask=mask)
        x = tuple_op(self.final_conv, x)
        x = tuple(xi for xi in x if xi is not None)
        x = torch.stack(x, dim=1)  # [B, <=4, dim_out, H, W]
        return x


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

        return value, policy

    @property
    def name(self):
        m, p, v = self.model_size
        return f"mix6_{self.input_type}_{m}m{p}p{v}v" + (f"-{self.map_max}mm" if self.map_max != 0 else "")


@MODELS.register("mix7")
class Mix7Net(nn.Module):
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
        # average feature across four directions
        feature = torch.mean(feature, dim=1)  # [B, max(PC,VC), H, W]
        feature = self.mapping_activation(feature)

        # feature conv
        feat_dwconv = self.feature_dwconv(feature[:, : self.dim_dwconv])  # [B, dwconv, H, W]
        feat_direct = feature[:, self.dim_dwconv :]  # [B, max(PC,VC)-dwconv, H, W]
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

    def forward_debug_print(self, data):
        _, dim_policy, dim_value = self.model_size

        input_plane = self.input_plane(data)
        feature = self.mapping(input_plane)
        # resize feature to range [-map_max, map_max]
        if self.map_max != 0:
            feature = self.map_max * torch.tanh(feature / self.map_max)
        print(f"features at (0,0): \n{feature[..., 0, 0]}")
        # average feature across four directions
        feature = torch.mean(feature, dim=1)  # [B, PC+VC, H, W]
        print(f"feature mean at (0,0): \n{feature[..., 0, 0]}")
        feature = self.mapping_activation(feature)
        print(f"feature act at (0,0): \n{feature[..., 0, 0]}")

        # feature conv
        feat_dwconv = self.feature_dwconv(feature[:, : self.dim_dwconv])  # [B, dwconv, H, W]
        feat_direct = feature[:, self.dim_dwconv :]  # [B, max(PC,VC)-dwconv, H, W]
        feature = torch.cat((feat_dwconv, feat_direct), dim=1)  # [B, max(PC,VC), H, W]
        print(f"feature after dwconv at (0,0): \n{feature[..., 0, 0]}")

        # policy head
        policy = feature[:, :dim_policy]
        print(f"policy feature input at (0,0): \n{policy[..., 0, 0]}")
        policy = self.policy_pwconv(policy)
        print(f"policy after pwconv at (0,0): \n{policy[..., 0, 0]}")
        policy = self.policy_activation(policy)
        print(f"policy act at (0,0): \n{policy[..., 0, 0]}")

        # value head
        value = feature[:, :dim_value]
        print(f"value feature input at (0,0): \n{value[..., 0, 0]}")
        value = torch.mean(value, dim=(2, 3))
        print(f"value feature mean: \n{value}")
        value = self.value_linear(value)

        return value, policy

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
class Mix8Net(nn.Module):
    def __init__(
        self,
        dim_middle=128,
        dim_feature=64,
        dim_policy=32,
        dim_value=64,
        dim_value_group=32,
        dim_dwconv=None,
        input_type="basicns",
    ):
        super().__init__()
        dim_feature = max(dim_policy, dim_value)
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
            nn.Conv2d(4, 1, 1),
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

        # clamp feature for int quantization
        feature = torch.clamp(feature, min=-32, max=32)  # int16, scale=255, [-32,32]
        feature = fake_quant(feature, scale=255, num_bits=16)
        # sum (and rescale) feature across four directions
        feature = torch.mean(feature, dim=1)  # [B, dim_feature, H, W] int16, scale=1020, [-32,32]
        feature = self.mapping_activation(feature)  # int16, scale=1020, [-32,32]

        # apply feature depth-wise conv
        feat_dwconv = self.feature_dwconv(feature[:, : self.dim_dwconv])  # [B, dwconv, H, W]
        feat_direct = feature[:, self.dim_dwconv :]  # [B, dim_feature-dwconv, H, W]
        feature = torch.cat([feat_dwconv, feat_direct], dim=1)  # [B, dim_feature, H, W]

        return feature

    def forward(self, data):
        _, _, dim_policy, _, _ = self.model_size

        # get feature from single side
        feature = self.get_feature(data, False)  # [B, dim_feature, H, W]

        # value feature accumulator
        feature_mean = torch.mean(feature, dim=(2, 3))  # [B, dim_feature]

        # value feature accumulator of four quadrants
        B, _, H, W = feature.shape
        H0, W0 = 0, 0
        H1, W1 = (H // 3) + (H % 3 == 2), (W // 3) + (W % 3 == 2)
        H2, W2 = (H // 3) * 2 + (H % 3 > 0), (W // 3) * 2 + (W % 3 > 0)
        H3, W3 = H, W
        feature_00 = torch.mean(feature[:, :, H0:H1, W0:W1], dim=(2, 3))  # [B, dim_feature]
        feature_01 = torch.mean(feature[:, :, H0:H1, W1:W2], dim=(2, 3))  # [B, dim_feature]
        feature_02 = torch.mean(feature[:, :, H0:H1, W2:W3], dim=(2, 3))  # [B, dim_feature]
        feature_10 = torch.mean(feature[:, :, H1:H2, W0:W1], dim=(2, 3))  # [B, dim_feature]
        feature_11 = torch.mean(feature[:, :, H1:H2, W1:W2], dim=(2, 3))  # [B, dim_feature]
        feature_12 = torch.mean(feature[:, :, H1:H2, W2:W3], dim=(2, 3))  # [B, dim_feature]
        feature_20 = torch.mean(feature[:, :, H2:H3, W0:W1], dim=(2, 3))  # [B, dim_feature]
        feature_21 = torch.mean(feature[:, :, H2:H3, W1:W2], dim=(2, 3))  # [B, dim_feature]
        feature_22 = torch.mean(feature[:, :, H2:H3, W2:W3], dim=(2, 3))  # [B, dim_feature]

        # policy head
        pwconv_weight = self.policy_pwconv_weight_linear(feature_mean)
        pwconv_weight = pwconv_weight.reshape(B, 4 * dim_policy, 1, 1)
        policy = feature[:, :dim_policy]  # [B, dim_policy, H, W]
        policy = torch.cat(
            [
                F.conv2d(
                    input=policy.reshape(1, B * dim_policy, H, W),
                    weight=pwconv_weight[:, dim_policy * i : dim_policy * (i + 1)],
                    groups=B,
                ).reshape(B, 1, H, W)
                for i in range(4)
            ],
            dim=1,
        )
        policy = self.policy_output(policy)  # [B, 1, H, W]

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

        value_q00 = value_00 + value_01 + value_10 + value_11
        value_q01 = value_01 + value_02 + value_11 + value_12
        value_q10 = value_10 + value_11 + value_20 + value_21
        value_q11 = value_11 + value_12 + value_21 + value_22
        value_q00 = self.value_quad_act(self.value_quad_linear(value_q00))
        value_q01 = self.value_quad_act(self.value_quad_linear(value_q01))
        value_q10 = self.value_quad_act(self.value_quad_linear(value_q10))
        value_q11 = self.value_quad_act(self.value_quad_linear(value_q11))
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
        value = self.value_linear(value)

        return value, policy

    def forward_debug_print(self, data):
        _, _, dim_policy, _, _ = self.model_size

        # get feature from single side
        feature = self.get_feature(data, False)
        print(f"feature after dwconv at (0,0): \n{feature[..., 0, 0]}")

        # value feature accumulator
        feature_mean = torch.mean(feature, dim=(2, 3))  # [B, dim_feature, H, W]
        print(f"feature mean: \n{feature_mean}")

        # value feature accumulator of four quadrants
        B, _, H, W = feature.shape
        H0, W0 = 0, 0
        H1, W1 = (H // 3) + (H % 3 == 2), (W // 3) + (W % 3 == 2)
        H2, W2 = (H // 3) * 2 + (H % 3 > 0), (W // 3) * 2 + (W % 3 > 0)
        H3, W3 = H, W
        feature_00 = torch.mean(feature[:, :, H0:H1, W0:W1], dim=(2, 3))  # [B, dim_feature]
        feature_01 = torch.mean(feature[:, :, H0:H1, W1:W2], dim=(2, 3))  # [B, dim_feature]
        feature_02 = torch.mean(feature[:, :, H0:H1, W2:W3], dim=(2, 3))  # [B, dim_feature]
        feature_10 = torch.mean(feature[:, :, H1:H2, W0:W1], dim=(2, 3))  # [B, dim_feature]
        feature_11 = torch.mean(feature[:, :, H1:H2, W1:W2], dim=(2, 3))  # [B, dim_feature]
        feature_12 = torch.mean(feature[:, :, H1:H2, W2:W3], dim=(2, 3))  # [B, dim_feature]
        feature_20 = torch.mean(feature[:, :, H2:H3, W0:W1], dim=(2, 3))  # [B, dim_feature]
        feature_21 = torch.mean(feature[:, :, H2:H3, W1:W2], dim=(2, 3))  # [B, dim_feature]
        feature_22 = torch.mean(feature[:, :, H2:H3, W2:W3], dim=(2, 3))  # [B, dim_feature]
        print(f"feature 00 mean: \n{feature_00}")
        print(f"feature 01 mean: \n{feature_01}")
        print(f"feature 02 mean: \n{feature_02}")
        print(f"feature 10 mean: \n{feature_10}")
        print(f"feature 11 mean: \n{feature_11}")
        print(f"feature 12 mean: \n{feature_12}")
        print(f"feature 20 mean: \n{feature_20}")
        print(f"feature 21 mean: \n{feature_21}")
        print(f"feature 22 mean: \n{feature_22}")

        # policy head
        pwconv_weight = self.policy_pwconv_weight_linear(feature_mean)  # [B, dim_policy]
        print(f"policy weight: \n{pwconv_weight}")
        pwconv_weight = pwconv_weight.reshape(B, 4 * dim_policy, 1, 1)
        policy = feature[:, :dim_policy]  # [B, dim_policy, H, W]
        print(f"policy after dwconv at (0,0): \n{policy[..., 0, 0]}")
        policy = torch.cat(
            [
                F.conv2d(
                    input=policy.reshape(1, B * dim_policy, H, W),
                    weight=pwconv_weight[:, dim_policy * i : dim_policy * (i + 1)],
                    groups=B,
                ).reshape(B, 1, H, W)
                for i in range(4)
            ],
            dim=1,
        )
        print(f"policy after dynamic pwconv at (0,0): \n{policy[..., 0, 0]}")
        policy = self.policy_output(policy)  # [B, 1, H, W]
        print(f"policy output at (0,0): \n{policy[..., 0, 0]}")

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
        print(f"value_00: \n{value_00}")
        print(f"value_01: \n{value_01}")
        print(f"value_02: \n{value_02}")
        print(f"value_10: \n{value_10}")
        print(f"value_11: \n{value_11}")
        print(f"value_12: \n{value_12}")
        print(f"value_20: \n{value_20}")
        print(f"value_21: \n{value_21}")
        print(f"value_22: \n{value_22}")
        value_q00 = value_00 + value_01 + value_10 + value_11
        value_q01 = value_01 + value_02 + value_11 + value_12
        value_q10 = value_10 + value_11 + value_20 + value_21
        value_q11 = value_11 + value_12 + value_21 + value_22
        print(f"value_quad00 sum: \n{value_q00}")
        print(f"value_quad01 sum: \n{value_q01}")
        print(f"value_quad10 sum: \n{value_q10}")
        print(f"value_quad11 sum: \n{value_q11}")
        value_q00 = self.value_quad_act(self.value_quad_linear(value_q00))
        value_q01 = self.value_quad_act(self.value_quad_linear(value_q01))
        value_q10 = self.value_quad_act(self.value_quad_linear(value_q10))
        value_q11 = self.value_quad_act(self.value_quad_linear(value_q11))
        print(f"value_quad00: \n{value_q00}")
        print(f"value_quad01: \n{value_q01}")
        print(f"value_quad10: \n{value_q10}")
        print(f"value_quad11: \n{value_q11}")
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
        print(f"value feature input: \n{value}")
        for i, linear in enumerate(self.value_linear):
            value = linear(value)
            print(f"value feature after layer {i}: \n{value}")

        return value, policy

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


class StarBlock(nn.Module):
    def __init__(self, dim_in, dim_out, expand=1):
        super().__init__()
        self.up1 = LinearBlock(dim_in, dim_out * 2 * expand, activation="relu", quant=True)
        self.up2 = LinearBlock(dim_in, dim_out * 2 * expand, activation="none", quant=True)
        self.down = LinearBlock(dim_out * expand, dim_out, activation="relu", quant=True)

    def forward(self, x):
        x1 = self.up1(x)
        x2 = self.up2(x)
        x1 = fake_quant(x1, scale=128, num_bits=8, floor=True)
        x2 = fake_quant(x2, scale=128, num_bits=8, floor=True)
        # i32 dot product of two adjacent pairs of u8 and i8
        x = (x1 * x2).view(*x.shape[:-1], -1, 2).sum(-1)
        # clamp to int8, scale=128, [-1,1]
        x = fake_quant(x, scale=128, num_bits=8, floor=True)
        x = self.down(x)  # int8 linear layer with signed input
        return x

    def forward_debug_print(self, x, name="starblock"):
        x1 = self.up1(x)
        x2 = self.up2(x)
        x1 = fake_quant(x1, scale=128, num_bits=8, floor=True)
        x2 = fake_quant(x2, scale=128, num_bits=8, floor=True)
        print(f"{name} up1: \n{(x1*128).int()}")
        print(f"{name} up2: \n{(x2*128).int()}")
        # i32 dot product of two adjacent pairs of u8 and i8
        x = (x1 * x2).view(*x.shape[:-1], -1, 2).sum(-1)
        x = fake_quant(x, scale=128, num_bits=8, floor=True)
        print(f"{name} dot2 product: \n{(x*128).int()}")
        x = self.down(x)  # int8 linear layer with signed input
        print(f"{name} down: \n{(x*128).int()}")
        return x


@MODELS.register("mix9")
class Mix9Net(nn.Module):
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
        )

        # policy head (point-wise conv)
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

        self.policy_output = nn.Conv2d(dim_pm, 1, 1)

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

    def initialize(self):
        self.feature_dwconv.conv.weight.data.mul_(0.25)

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

        # clamp feature for int quantization
        feature = torch.clamp(feature, min=-16, max=511 / 32)  # int16, scale=32, [-16,16]
        feature = fake_quant(feature, scale=32, num_bits=16)
        # sum (and rescale) feature across four directions
        feature = torch.mean(feature, dim=1)  # [B, dim_feature, H, W] int16, scale=128, [-16,16]
        # apply relu activation
        feature = F.relu(feature)  # [B, dim_feature, H, W] int16, scale=128, [0,16]

        # apply feature depth-wise conv
        _, _, _, _, dim_dwconv = self.model_size
        feat_dwconv = feature[:, :dim_dwconv]  # int16, scale=128, [0,16]
        feat_dwconv = self.feature_dwconv(feat_dwconv * 4)  # [B, dwconv, H, W] relu
        feat_dwconv = fake_quant(feat_dwconv, scale=128, num_bits=16)  # int16, scale=128, [0,9/2*16*4]

        # apply activation for direct feature
        feat_direct = feature[:, dim_dwconv:]  # [B, dim_feature-dwconv, H, W] int16, scale=128, [0,16]
        feat_direct = fake_quant(feat_direct, scale=128, num_bits=16)  # int16, scale=128, [0,16]

        feature = torch.cat([feat_dwconv, feat_direct], dim=1)  # [B, dim_feature, H, W]

        return feature

    def forward(self, data):
        _, _, dim_policy, _, _ = self.model_size

        # get feature from single side
        feature = self.get_feature(data, False)  # [B, dim_feature, H, W]
        B, _, H, W = feature.shape

        # global feature accumulator
        feature_sum = torch.sum(feature, dim=(2, 3))  # [B, dim_feature]
        feature_sum = fake_quant(feature_sum / 256, scale=128, num_bits=32, floor=True)  # srai 8

        # policy head
        if self.no_dynamic_pwconv:
            policy = self.policy_pwconv(
                feature[:, :dim_policy]
            )  # [B, dim_pm, H, W] int32, scale=128*128*128, relu
        else:
            dim_pm = self.policy_middle_dim
            pwconv_output = self.policy_pwconv_weight_linear(feature_sum)
            pwconv_weight = pwconv_output[:, : dim_pm * dim_policy].reshape(B, dim_pm * dim_policy, 1, 1)
            pwconv_weight = fake_quant(pwconv_weight, scale=128 * 128, num_bits=16, floor=True)
            policy = fake_quant(feature[:, :dim_policy], scale=128, num_bits=16)  # [B, dim_policy, H, W]
            policy = torch.cat(
                [
                    F.conv2d(
                        input=policy.reshape(1, B * dim_policy, H, W),
                        weight=pwconv_weight[:, dim_policy * i : dim_policy * (i + 1)],
                        groups=B,
                    ).reshape(B, 1, H, W)
                    for i in range(dim_pm)
                ],
                1,
            )
            pwconv_bias = pwconv_output[:, dim_pm * dim_policy :].reshape(
                B, dim_pm, 1, 1
            )  # int32, scale=128*128*128
            policy = torch.clamp(
                policy + pwconv_bias, min=0
            )  # [B, dim_pm, H, W] int32, scale=128*128*128, relu

        policy = self.policy_output(policy)  # [B, 1, H, W]

        if self.no_value_group:
            value = self.value_linear(feature_sum)
        else:
            # value feature accumulator of nine groups
            H0, W0 = 0, 0
            H1, W1 = (H // 3) + (H % 3 == 2), (W // 3) + (W % 3 == 2)
            H2, W2 = (H // 3) * 2 + (H % 3 > 0), (W // 3) * 2 + (W % 3 > 0)
            H3, W3 = H, W
            feature_00 = torch.sum(feature[:, :, H0:H1, W0:W1], dim=(2, 3))  # [B, dim_feature]
            feature_01 = torch.sum(feature[:, :, H0:H1, W1:W2], dim=(2, 3))  # [B, dim_feature]
            feature_02 = torch.sum(feature[:, :, H0:H1, W2:W3], dim=(2, 3))  # [B, dim_feature]
            feature_10 = torch.sum(feature[:, :, H1:H2, W0:W1], dim=(2, 3))  # [B, dim_feature]
            feature_11 = torch.sum(feature[:, :, H1:H2, W1:W2], dim=(2, 3))  # [B, dim_feature]
            feature_12 = torch.sum(feature[:, :, H1:H2, W2:W3], dim=(2, 3))  # [B, dim_feature]
            feature_20 = torch.sum(feature[:, :, H2:H3, W0:W1], dim=(2, 3))  # [B, dim_feature]
            feature_21 = torch.sum(feature[:, :, H2:H3, W1:W2], dim=(2, 3))  # [B, dim_feature]
            feature_22 = torch.sum(feature[:, :, H2:H3, W2:W3], dim=(2, 3))  # [B, dim_feature]
            feature_00 = fake_quant(feature_00 / 32, scale=128, num_bits=32, floor=True)  # srai 5
            feature_01 = fake_quant(feature_01 / 32, scale=128, num_bits=32, floor=True)  # srai 5
            feature_02 = fake_quant(feature_02 / 32, scale=128, num_bits=32, floor=True)  # srai 5
            feature_10 = fake_quant(feature_10 / 32, scale=128, num_bits=32, floor=True)  # srai 5
            feature_11 = fake_quant(feature_11 / 32, scale=128, num_bits=32, floor=True)  # srai 5
            feature_12 = fake_quant(feature_12 / 32, scale=128, num_bits=32, floor=True)  # srai 5
            feature_20 = fake_quant(feature_20 / 32, scale=128, num_bits=32, floor=True)  # srai 5
            feature_21 = fake_quant(feature_21 / 32, scale=128, num_bits=32, floor=True)  # srai 5
            feature_22 = fake_quant(feature_22 / 32, scale=128, num_bits=32, floor=True)  # srai 5

            # value head
            value_00 = self.value_corner(feature_00)
            value_01 = self.value_edge(feature_01)
            value_02 = self.value_corner(feature_02)
            value_10 = self.value_edge(feature_10)
            value_11 = self.value_center(feature_11)
            value_12 = self.value_edge(feature_12)
            value_20 = self.value_corner(feature_20)
            value_21 = self.value_edge(feature_21)
            value_22 = self.value_corner(feature_22)

            value_q00 = avg4(value_00, value_01, value_10, value_11)
            value_q01 = avg4(value_01, value_02, value_11, value_12)
            value_q10 = avg4(value_10, value_11, value_20, value_21)
            value_q11 = avg4(value_11, value_12, value_21, value_22)
            value_q00 = self.value_quad(value_q00)
            value_q01 = self.value_quad(value_q01)
            value_q10 = self.value_quad(value_q10)
            value_q11 = self.value_quad(value_q11)

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
            value = self.value_linear(value)

        return value, policy

    def forward_debug_print(self, data):
        _, _, dim_policy, _, _ = self.model_size

        # get feature from single side
        feature = self.get_feature(data, False)  # [B, dim_feature, H, W]
        B, _, H, W = feature.shape
        print(f"feature after dwconv at (0,0): \n{(feature[..., 0, 0]*128).int()}")

        # global feature accumulator
        feature_sum = torch.sum(feature, dim=(2, 3))  # [B, dim_feature]
        print(f"feature sum before scale: \n{(feature_sum*128).int()}")
        feature_sum = fake_quant(feature_sum / 256, scale=128, num_bits=32, floor=True)  # srai 8
        print(f"feature sum: \n{(feature_sum*128).int()}")

        # policy head
        if self.no_dynamic_pwconv:
            policy = self.policy_pwconv(
                feature[:, :dim_policy]
            )  # [B, dim_pm, H, W] int32, scale=128*128*128, relu
            print(f"policy pwconv output at (0,0): \n{(policy[..., 0, 0]*128*128*128).int()}")
        else:
            dim_pm = self.policy_middle_dim
            pwconv_output = self.policy_pwconv_weight_linear(feature_sum)
            print(f"policy pwconv output: \n{(pwconv_output*128*128).int()}")
            pwconv_weight = pwconv_output[:, : dim_pm * dim_policy].reshape(B, dim_pm * dim_policy, 1, 1)
            pwconv_weight = fake_quant(pwconv_weight, scale=128 * 128, num_bits=16, floor=True)
            print(f"policy pwconv weight: \n{(pwconv_weight.flatten(1, -1)*128*128).int()}")
            policy = fake_quant(feature[:, :dim_policy], scale=128, num_bits=16)  # [B, dim_policy, H, W]
            print(f"policy after dwconv at (0,0): \n{(policy[..., 0, 0]*128).int()}")
            policy = torch.cat(
                [
                    F.conv2d(
                        input=policy.reshape(1, B * dim_policy, H, W),
                        weight=pwconv_weight[:, dim_policy * i : dim_policy * (i + 1)],
                        groups=B,
                    ).reshape(B, 1, H, W)
                    for i in range(dim_pm)
                ],
                1,
            )
            print(f"policy after dynamic pwconv at (0,0): \n{(policy[..., 0, 0]*128*128*128).int()}")
            pwconv_bias = pwconv_output[:, dim_pm * dim_policy :].reshape(
                B, dim_pm, 1, 1
            )  # int32, scale=128*128*128
            print(f"policy pwconv bias: \n{(pwconv_bias.flatten(1, -1)*128*128*128).int()}")
            policy = torch.clamp(
                policy + pwconv_bias, min=0
            )  # [B, dim_pm, H, W] int32, scale=128*128*128, relu
            print(f"policy pwconv output at (0,0): \n{(policy[..., 0, 0]*128*128*128).int()}")
        policy = self.policy_output(policy)  # [B, 1, H, W]
        print(f"policy output at (0,0): \n{policy[..., 0, 0]}")

        if self.no_value_group:
            value = feature_sum  # [B, dim_feature]
            print(f"value feature input: \n{(value*128).int()}")
            for i, linear in enumerate(self.value_linear):
                value = linear(value)
                print(f"value feature after layer {i}: \n{(value*128).int()}")
        else:
            # value feature accumulator of nine groups
            H0, W0 = 0, 0
            H1, W1 = (H // 3) + (H % 3 == 2), (W // 3) + (W % 3 == 2)
            H2, W2 = (H // 3) * 2 + (H % 3 > 0), (W // 3) * 2 + (W % 3 > 0)
            H3, W3 = H, W
            feature_00 = torch.sum(feature[:, :, H0:H1, W0:W1], dim=(2, 3))  # [B, dim_feature]
            feature_01 = torch.sum(feature[:, :, H0:H1, W1:W2], dim=(2, 3))  # [B, dim_feature]
            feature_02 = torch.sum(feature[:, :, H0:H1, W2:W3], dim=(2, 3))  # [B, dim_feature]
            feature_10 = torch.sum(feature[:, :, H1:H2, W0:W1], dim=(2, 3))  # [B, dim_feature]
            feature_11 = torch.sum(feature[:, :, H1:H2, W1:W2], dim=(2, 3))  # [B, dim_feature]
            feature_12 = torch.sum(feature[:, :, H1:H2, W2:W3], dim=(2, 3))  # [B, dim_feature]
            feature_20 = torch.sum(feature[:, :, H2:H3, W0:W1], dim=(2, 3))  # [B, dim_feature]
            feature_21 = torch.sum(feature[:, :, H2:H3, W1:W2], dim=(2, 3))  # [B, dim_feature]
            feature_22 = torch.sum(feature[:, :, H2:H3, W2:W3], dim=(2, 3))  # [B, dim_feature]
            feature_00 = fake_quant(feature_00 / 32, scale=128, num_bits=32, floor=True)  # srai 5
            feature_01 = fake_quant(feature_01 / 32, scale=128, num_bits=32, floor=True)  # srai 5
            feature_02 = fake_quant(feature_02 / 32, scale=128, num_bits=32, floor=True)  # srai 5
            feature_10 = fake_quant(feature_10 / 32, scale=128, num_bits=32, floor=True)  # srai 5
            feature_11 = fake_quant(feature_11 / 32, scale=128, num_bits=32, floor=True)  # srai 5
            feature_12 = fake_quant(feature_12 / 32, scale=128, num_bits=32, floor=True)  # srai 5
            feature_20 = fake_quant(feature_20 / 32, scale=128, num_bits=32, floor=True)  # srai 5
            feature_21 = fake_quant(feature_21 / 32, scale=128, num_bits=32, floor=True)  # srai 5
            feature_22 = fake_quant(feature_22 / 32, scale=128, num_bits=32, floor=True)  # srai 5
            print(f"feature 00 sum: \n{(feature_00*128).int()}")
            print(f"feature 01 sum: \n{(feature_01*128).int()}")
            print(f"feature 02 sum: \n{(feature_02*128).int()}")
            print(f"feature 10 sum: \n{(feature_10*128).int()}")
            print(f"feature 11 sum: \n{(feature_11*128).int()}")
            print(f"feature 12 sum: \n{(feature_12*128).int()}")
            print(f"feature 20 sum: \n{(feature_20*128).int()}")
            print(f"feature 21 sum: \n{(feature_21*128).int()}")
            print(f"feature 22 sum: \n{(feature_22*128).int()}")

            # value head
            if self.no_star_block:
                value_00 = fake_quant(self.value_corner(feature_00), floor=True)
                value_01 = fake_quant(self.value_edge(feature_01), floor=True)
                value_02 = fake_quant(self.value_corner(feature_02), floor=True)
                value_10 = fake_quant(self.value_edge(feature_10), floor=True)
                value_11 = fake_quant(self.value_center(feature_11), floor=True)
                value_12 = fake_quant(self.value_edge(feature_12), floor=True)
                value_20 = fake_quant(self.value_corner(feature_20), floor=True)
                value_21 = fake_quant(self.value_edge(feature_21), floor=True)
                value_22 = fake_quant(self.value_corner(feature_22), floor=True)
                print(f"value_00: \n{(value_00*128).int()}")
                print(f"value_01: \n{(value_01*128).int()}")
                print(f"value_02: \n{(value_02*128).int()}")
                print(f"value_10: \n{(value_10*128).int()}")
                print(f"value_11: \n{(value_11*128).int()}")
                print(f"value_12: \n{(value_12*128).int()}")
                print(f"value_20: \n{(value_20*128).int()}")
                print(f"value_21: \n{(value_21*128).int()}")
                print(f"value_22: \n{(value_22*128).int()}")
            else:
                value_00 = self.value_corner.forward_debug_print(feature_00, "value_00")
                value_01 = self.value_edge.forward_debug_print(feature_01, "value_01")
                value_02 = self.value_corner.forward_debug_print(feature_02, "value_02")
                value_10 = self.value_edge.forward_debug_print(feature_10, "value_10")
                value_11 = self.value_center.forward_debug_print(feature_11, "value_11")
                value_12 = self.value_edge.forward_debug_print(feature_12, "value_12")
                value_20 = self.value_corner.forward_debug_print(feature_20, "value_20")
                value_21 = self.value_edge.forward_debug_print(feature_21, "value_21")
                value_22 = self.value_corner.forward_debug_print(feature_22, "value_22")

            value_q00 = avg4(value_00, value_01, value_10, value_11)
            value_q01 = avg4(value_01, value_02, value_11, value_12)
            value_q10 = avg4(value_10, value_11, value_20, value_21)
            value_q11 = avg4(value_11, value_12, value_21, value_22)
            print(f"value_q00 avg: \n{(value_q00*128).int()}")
            print(f"value_q01 avg: \n{(value_q01*128).int()}")
            print(f"value_q10 avg: \n{(value_q10*128).int()}")
            print(f"value_q11 avg: \n{(value_q11*128).int()}")

            if self.no_star_block:
                value_q00 = fake_quant(self.value_quad(value_q00), floor=True)
                value_q01 = fake_quant(self.value_quad(value_q01), floor=True)
                value_q10 = fake_quant(self.value_quad(value_q10), floor=True)
                value_q11 = fake_quant(self.value_quad(value_q11), floor=True)
                print(f"value_q00: \n{(value_q00*128).int()}")
                print(f"value_q01: \n{(value_q01*128).int()}")
                print(f"value_q10: \n{(value_q10*128).int()}")
                print(f"value_q11: \n{(value_q11*128).int()}")
            else:
                value_q00 = self.value_quad.forward_debug_print(value_q00, "value_q00")
                value_q01 = self.value_quad.forward_debug_print(value_q01, "value_q01")
                value_q10 = self.value_quad.forward_debug_print(value_q10, "value_q10")
                value_q11 = self.value_quad.forward_debug_print(value_q11, "value_q11")

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
            print(f"value feature input: \n{(value*128).int()}")
            for i, linear in enumerate(self.value_linear):
                value = linear(value)
                print(f"value feature after layer {i}: \n{(value*128).int()}")

        return value, policy

    @property
    def weight_clipping(self):
        # Clip prelu weight of mapping activation to [-1,1] to avoid overflow
        # In this range, prelu is the same as `max(x, ax)`.
        if self.no_value_group:
            value_group_weights = []
        elif self.no_star_block:
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

        weight_clipping_list = [
            {
                "params": ["feature_dwconv.conv.weight"],
                "min_weight": -32768 / 65536,
                "max_weight": 32767 / 65536,
            },
            {
                "params": [
                    *value_group_weights,
                    "value_linear.0.fc.weight",
                    "value_linear.1.fc.weight",
                    "value_linear.2.fc.weight",
                ],
                "min_weight": -128 / 128,
                "max_weight": 127 / 128,
            },
        ]

        if self.no_dynamic_pwconv:
            weight_clipping_list.append(
                {
                    "params": ["policy_pwconv.conv.weight"],
                    "min_weight": -32768 / (128 * 128),
                    "max_weight": 32767 / (128 * 128),
                }
            )
        else:
            weight_clipping_list.append(
                {
                    "params": [
                        "policy_pwconv_weight_linear.0.fc.weight",
                        "policy_pwconv_weight_linear.1.fc.weight",
                    ],
                    "min_weight": -128 / 128,
                    "max_weight": 127 / 128,
                }
            )

        return weight_clipping_list

    @property
    def name(self):
        _, f, p, v, d = self.model_size
        return f"mix9_{self.input_type}_{f}f{p}p{v}v{d}d"


@MODELS.register("mix9s")
class Mix9sNet(nn.Module):
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
        )

        # policy head (point-wise conv)
        dim_pm = self.policy_middle_dim = 16
        self.policy_pwconv_weight_linear = nn.Sequential(
            LinearBlock(dim_feature, dim_policy * 2, activation="relu", quant=True),
            LinearBlock(dim_policy * 2, dim_pm * dim_policy + dim_pm, activation="none", quant=True),
        )
        self.policy_output = nn.Conv2d(dim_pm, 1, 1)

        self.value_corner = StarBlock(dim_feature, dim_value)
        self.value_edge = StarBlock(dim_feature, dim_value)
        self.value_center = StarBlock(dim_feature, dim_value)
        self.value_quad = StarBlock(dim_value, dim_value)
        self.value_linear = nn.Sequential(
            LinearBlock(dim_feature + 4 * dim_value, dim_value, activation="relu", quant=True),
            LinearBlock(dim_value, dim_value, activation="relu", quant=True),
            LinearBlock(dim_value, 3, activation="none", quant=True),
        )

    def initialize(self):
        self.feature_dwconv.conv.weight.data.mul_(0.25)

    def do_feature_quantization(self, feature, data):
        return feature, {}, {}  # Not implemented

    def get_feature(self, data, inv_side=False):
        # get the input plane from board and side to move input
        input_plane = self.input_plane(data, inv_side)  # [B, 2, H, W]

        # get per-point 4-direction cell features
        feature1 = self.mapping1(input_plane, dirs=[0, 1])  # [B, 2, dim_feature, H, W]
        feature2 = self.mapping2(input_plane, dirs=[2, 3])  # [B, 2, dim_feature, H, W]
        feature = torch.cat([feature1, feature2], dim=1)  # [B, 4, dim_feature, H, W]
        feature = torch.clamp(feature, min=-511 / 32, max=511 / 32)  # [-511/32,511/32]

        # do feature quantization
        feature, aux_losses, aux_outputs = self.do_feature_quantization(feature, data)

        # clamp feature for int quantization
        feature = torch.clamp(feature, min=-16, max=511 / 32)  # [-512/32,511/32]
        feature = fake_quant(feature, scale=32, num_bits=16)  # int16, scale=32, [-16,511/32]
        # sum (and rescale) feature across four directions
        feature = torch.mean(feature, dim=1)  # [B, dim_feature, H, W] int16, scale=128, [-16,16]
        # apply relu activation
        feature = F.relu(feature)  # [B, dim_feature, H, W] int16, scale=128, [0,16]

        # apply feature depth-wise conv
        _, _, _, _, dim_dwconv = self.model_size
        feat_dwconv = feature[:, :dim_dwconv]  # int16, scale=128, [0,16]
        feat_dwconv = self.feature_dwconv(feat_dwconv * 4)  # [B, dwconv, H, W] relu
        feat_dwconv = fake_quant(feat_dwconv, scale=128, num_bits=16)  # int16, scale=128, [0,9/2*16*4]

        # apply activation for direct feature
        feat_direct = feature[:, dim_dwconv:]  # [B, dim_feature-dwconv, H, W] int16, scale=128, [0,16]
        feat_direct = fake_quant(feat_direct, scale=128, num_bits=16)  # int16, scale=128, [0,16]

        feature = torch.cat([feat_dwconv, feat_direct], dim=1)  # [B, dim_feature, H, W]

        return feature, aux_losses, aux_outputs

    def forward(self, data):
        _, _, dim_policy, _, _ = self.model_size

        # get feature from single side
        feature, aux_losses, aux_outputs = self.get_feature(data, False)  # [B, dim_feature, H, W]
        B, _, H, W = feature.shape

        # global feature accumulator
        feature_sum = torch.sum(feature, dim=(2, 3))  # [B, dim_feature]
        feature_sum = fake_quant(feature_sum / 256, scale=128, num_bits=32, floor=True)  # srai 8

        # policy head
        dim_pm = self.policy_middle_dim
        pwconv_output = self.policy_pwconv_weight_linear(feature_sum)
        pwconv_weight = pwconv_output[:, : dim_pm * dim_policy].reshape(B, dim_pm * dim_policy, 1, 1)
        pwconv_weight = fake_quant(pwconv_weight, scale=128 * 128, num_bits=16, floor=True)
        policy = fake_quant(feature[:, :dim_policy], scale=128, num_bits=16)  # [B, dim_policy, H, W]
        policy = torch.cat(
            [
                F.conv2d(
                    input=policy.reshape(1, B * dim_policy, H, W),
                    weight=pwconv_weight[:, dim_policy * i : dim_policy * (i + 1)],
                    groups=B,
                ).reshape(B, 1, H, W)
                for i in range(dim_pm)
            ],
            1,
        )
        pwconv_bias = pwconv_output[:, dim_pm * dim_policy :].reshape(
            B, dim_pm, 1, 1
        )  # int32, scale=128*128*128
        policy = torch.clamp(policy + pwconv_bias, min=0)  # [B, dim_pm, H, W] int32, scale=128*128*128, relu
        policy = self.policy_output(policy)  # [B, 1, H, W]

        # value feature accumulator of nine groups
        H0, W0 = 0, 0
        H1, W1 = (H // 3) + (H % 3 == 2), (W // 3) + (W % 3 == 2)
        H2, W2 = (H // 3) * 2 + (H % 3 > 0), (W // 3) * 2 + (W % 3 > 0)
        H3, W3 = H, W
        feature_00 = torch.sum(feature[:, :, H0:H1, W0:W1], dim=(2, 3))  # [B, dim_feature]
        feature_01 = torch.sum(feature[:, :, H0:H1, W1:W2], dim=(2, 3))  # [B, dim_feature]
        feature_02 = torch.sum(feature[:, :, H0:H1, W2:W3], dim=(2, 3))  # [B, dim_feature]
        feature_10 = torch.sum(feature[:, :, H1:H2, W0:W1], dim=(2, 3))  # [B, dim_feature]
        feature_11 = torch.sum(feature[:, :, H1:H2, W1:W2], dim=(2, 3))  # [B, dim_feature]
        feature_12 = torch.sum(feature[:, :, H1:H2, W2:W3], dim=(2, 3))  # [B, dim_feature]
        feature_20 = torch.sum(feature[:, :, H2:H3, W0:W1], dim=(2, 3))  # [B, dim_feature]
        feature_21 = torch.sum(feature[:, :, H2:H3, W1:W2], dim=(2, 3))  # [B, dim_feature]
        feature_22 = torch.sum(feature[:, :, H2:H3, W2:W3], dim=(2, 3))  # [B, dim_feature]
        feature_00 = fake_quant(feature_00 / 32, scale=128, num_bits=32, floor=True)  # srai 5
        feature_01 = fake_quant(feature_01 / 32, scale=128, num_bits=32, floor=True)  # srai 5
        feature_02 = fake_quant(feature_02 / 32, scale=128, num_bits=32, floor=True)  # srai 5
        feature_10 = fake_quant(feature_10 / 32, scale=128, num_bits=32, floor=True)  # srai 5
        feature_11 = fake_quant(feature_11 / 32, scale=128, num_bits=32, floor=True)  # srai 5
        feature_12 = fake_quant(feature_12 / 32, scale=128, num_bits=32, floor=True)  # srai 5
        feature_20 = fake_quant(feature_20 / 32, scale=128, num_bits=32, floor=True)  # srai 5
        feature_21 = fake_quant(feature_21 / 32, scale=128, num_bits=32, floor=True)  # srai 5
        feature_22 = fake_quant(feature_22 / 32, scale=128, num_bits=32, floor=True)  # srai 5

        # value head
        value_00 = self.value_corner(feature_00)
        value_01 = self.value_edge(feature_01)
        value_02 = self.value_corner(feature_02)
        value_10 = self.value_edge(feature_10)
        value_11 = self.value_center(feature_11)
        value_12 = self.value_edge(feature_12)
        value_20 = self.value_corner(feature_20)
        value_21 = self.value_edge(feature_21)
        value_22 = self.value_corner(feature_22)

        value_q00 = avg4(value_00, value_01, value_10, value_11)
        value_q01 = avg4(value_01, value_02, value_11, value_12)
        value_q10 = avg4(value_10, value_11, value_20, value_21)
        value_q11 = avg4(value_11, value_12, value_21, value_22)
        value_q00 = self.value_quad(value_q00)
        value_q01 = self.value_quad(value_q01)
        value_q10 = self.value_quad(value_q10)
        value_q11 = self.value_quad(value_q11)

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
        value = self.value_linear(value)

        return value, policy, aux_losses, aux_outputs

    def forward_debug_print(self, data):
        _, _, dim_policy, _, _ = self.model_size

        # get feature from single side
        feature, aux_losses, aux_outputs = self.get_feature(data, False)  # [B, dim_feature, H, W]
        B, _, H, W = feature.shape
        print(f"feature after dwconv at (0,0): \n{(feature[..., 0, 0]*128).int()}")

        # global feature accumulator
        feature_sum = torch.sum(feature, dim=(2, 3))  # [B, dim_feature]
        print(f"feature sum before scale: \n{(feature_sum*128).int()}")
        feature_sum = fake_quant(feature_sum / 256, scale=128, num_bits=32, floor=True)  # srai 8
        print(f"feature sum: \n{(feature_sum*128).int()}")

        # policy head
        dim_pm = self.policy_middle_dim
        pwconv_output = self.policy_pwconv_weight_linear(feature_sum)
        print(f"policy pwconv output: \n{(pwconv_output*128*128).int()}")
        pwconv_weight = pwconv_output[:, : dim_pm * dim_policy].reshape(B, dim_pm * dim_policy, 1, 1)
        pwconv_weight = fake_quant(pwconv_weight, scale=128 * 128, num_bits=16, floor=True)
        print(f"policy pwconv weight: \n{(pwconv_weight.flatten(1, -1)*128*128).int()}")
        policy = fake_quant(feature[:, :dim_policy], scale=128, num_bits=16)  # [B, dim_policy, H, W]
        print(f"policy after dwconv at (0,0): \n{(policy[..., 0, 0]*128).int()}")
        policy = torch.cat(
            [
                F.conv2d(
                    input=policy.reshape(1, B * dim_policy, H, W),
                    weight=pwconv_weight[:, dim_policy * i : dim_policy * (i + 1)],
                    groups=B,
                ).reshape(B, 1, H, W)
                for i in range(dim_pm)
            ],
            1,
        )
        print(f"policy after dynamic pwconv at (0,0): \n{(policy[..., 0, 0]*128*128*128).int()}")
        pwconv_bias = pwconv_output[:, dim_pm * dim_policy :].reshape(
            B, dim_pm, 1, 1
        )  # int32, scale=128*128*128
        print(f"policy pwconv bias: \n{(pwconv_bias.flatten(1, -1)*128*128*128).int()}")
        policy = torch.clamp(policy + pwconv_bias, min=0)  # [B, dim_pm, H, W] int32, scale=128*128*128, relu
        print(f"policy pwconv output at (0,0): \n{(policy[..., 0, 0]*128*128*128).int()}")
        policy = self.policy_output(policy)  # [B, 1, H, W]
        print(f"policy output at (0,0): \n{policy[..., 0, 0]}")

        # value feature accumulator of nine groups
        H0, W0 = 0, 0
        H1, W1 = (H // 3) + (H % 3 == 2), (W // 3) + (W % 3 == 2)
        H2, W2 = (H // 3) * 2 + (H % 3 > 0), (W // 3) * 2 + (W % 3 > 0)
        H3, W3 = H, W
        feature_00 = torch.sum(feature[:, :, H0:H1, W0:W1], dim=(2, 3))  # [B, dim_feature]
        feature_01 = torch.sum(feature[:, :, H0:H1, W1:W2], dim=(2, 3))  # [B, dim_feature]
        feature_02 = torch.sum(feature[:, :, H0:H1, W2:W3], dim=(2, 3))  # [B, dim_feature]
        feature_10 = torch.sum(feature[:, :, H1:H2, W0:W1], dim=(2, 3))  # [B, dim_feature]
        feature_11 = torch.sum(feature[:, :, H1:H2, W1:W2], dim=(2, 3))  # [B, dim_feature]
        feature_12 = torch.sum(feature[:, :, H1:H2, W2:W3], dim=(2, 3))  # [B, dim_feature]
        feature_20 = torch.sum(feature[:, :, H2:H3, W0:W1], dim=(2, 3))  # [B, dim_feature]
        feature_21 = torch.sum(feature[:, :, H2:H3, W1:W2], dim=(2, 3))  # [B, dim_feature]
        feature_22 = torch.sum(feature[:, :, H2:H3, W2:W3], dim=(2, 3))  # [B, dim_feature]
        feature_00 = fake_quant(feature_00 / 32, scale=128, num_bits=32, floor=True)  # srai 5
        feature_01 = fake_quant(feature_01 / 32, scale=128, num_bits=32, floor=True)  # srai 5
        feature_02 = fake_quant(feature_02 / 32, scale=128, num_bits=32, floor=True)  # srai 5
        feature_10 = fake_quant(feature_10 / 32, scale=128, num_bits=32, floor=True)  # srai 5
        feature_11 = fake_quant(feature_11 / 32, scale=128, num_bits=32, floor=True)  # srai 5
        feature_12 = fake_quant(feature_12 / 32, scale=128, num_bits=32, floor=True)  # srai 5
        feature_20 = fake_quant(feature_20 / 32, scale=128, num_bits=32, floor=True)  # srai 5
        feature_21 = fake_quant(feature_21 / 32, scale=128, num_bits=32, floor=True)  # srai 5
        feature_22 = fake_quant(feature_22 / 32, scale=128, num_bits=32, floor=True)  # srai 5
        print(f"feature 00 sum: \n{(feature_00*128).int()}")
        print(f"feature 01 sum: \n{(feature_01*128).int()}")
        print(f"feature 02 sum: \n{(feature_02*128).int()}")
        print(f"feature 10 sum: \n{(feature_10*128).int()}")
        print(f"feature 11 sum: \n{(feature_11*128).int()}")
        print(f"feature 12 sum: \n{(feature_12*128).int()}")
        print(f"feature 20 sum: \n{(feature_20*128).int()}")
        print(f"feature 21 sum: \n{(feature_21*128).int()}")
        print(f"feature 22 sum: \n{(feature_22*128).int()}")

        # value head
        value_00 = self.value_corner(feature_00)
        value_01 = self.value_edge(feature_01)
        value_02 = self.value_corner(feature_02)
        value_10 = self.value_edge(feature_10)
        value_11 = self.value_center(feature_11)
        value_12 = self.value_edge(feature_12)
        value_20 = self.value_corner(feature_20)
        value_21 = self.value_edge(feature_21)
        value_22 = self.value_corner(feature_22)

        value_q00 = avg4(value_00, value_01, value_10, value_11)
        value_q01 = avg4(value_01, value_02, value_11, value_12)
        value_q10 = avg4(value_10, value_11, value_20, value_21)
        value_q11 = avg4(value_11, value_12, value_21, value_22)
        value_q00 = self.value_quad(value_q00)
        value_q01 = self.value_quad(value_q01)
        value_q10 = self.value_quad(value_q10)
        value_q11 = self.value_quad(value_q11)

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
        print(f"value feature input: \n{(value*128).int()}")
        for i, linear in enumerate(self.value_linear):
            value = linear(value)
            print(f"value feature after layer {i}: \n{(value*128).int()}")

        return value, policy, aux_losses, aux_outputs

    @property
    def weight_clipping(self):
        weight_clipping_list = [
            {
                "params": ["feature_dwconv.conv.weight"],
                "min_weight": -32768 / 65536,
                "max_weight": 32767 / 65536,
            },
            {
                "params": [
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
                    "value_linear.0.fc.weight",
                    "value_linear.1.fc.weight",
                    "value_linear.2.fc.weight",
                ],
                "min_weight": -128 / 128,
                "max_weight": 127 / 128,
            },
            {
                "params": [
                    "policy_pwconv_weight_linear.0.fc.weight",
                    "policy_pwconv_weight_linear.1.fc.weight",
                ],
                "min_weight": -128 / 128,
                "max_weight": 127 / 128,
            },
        ]

        return weight_clipping_list

    @property
    def name(self):
        _, f, p, v, d = self.model_size
        return f"mix9s_{self.input_type}_{f}f{p}p{v}v{d}d"


@MODELS.register("mix9svq")
class Mix9sVQNet(Mix9sNet):
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
        from .vq import ProductVectorQuantize

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
        self._cached_positions = None

    @torch.compiler.disable
    def _quantize_line_feature(self, feature, line_encoding, vq_layer_idx):
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
        feature_per_line_encoding_vq, info, _ = self.vq_layers[vq_layer_idx](feature_per_line_encoding)

        # gather the quantized features back to the original shape
        feature_vq = feature_per_line_encoding_vq[inverse_indices]  # [N, dim_feature]
        feature_vq = feature_vq.reshape(batch_size, num_directions, H, W, dim_feature)
        feature_vq = feature_vq.permute(0, 1, 4, 2, 3)  # [B, num_directions, dim_feature, H, W]

        return feature_vq, info

    @torch.compiler.disable
    def do_feature_quantization(self, feature, data):
        line_encoding = data["line_encoding"]  # [B, 4, H, W]

        feature_vq_1, info_1 = self._quantize_line_feature(feature[:, :2], line_encoding[:, :2], 0)
        feature_vq_2, info_2 = self._quantize_line_feature(feature[:, 2:], line_encoding[:, 2:], 1)
        feature_vq = torch.cat([feature_vq_1, feature_vq_2], dim=1)  # [B, 4, dim_feature, H, W]

        aux_losses = {}
        aux_outputs = {}
        if info_1 is not None and info_2 is not None:
            aux_losses = {"vq": (info_1["loss"] + info_2["loss"]) / 2}
            cluster_size = torch.cat(
                [
                    self.vq_layers[0].normalized_cluster_size,
                    self.vq_layers[1].normalized_cluster_size,
                ]
            )
            aux_outputs = {
                "vq_perplexity": (info_1["perplexity"] + info_2["perplexity"]) / 2,
                "vq_normed_perplexity": (info_1["normalized_perplexity"] + info_2["normalized_perplexity"])
                / 2,
                "vq_cluster_size_q01": torch.quantile(cluster_size, q=0.01),
                "vq_cluster_size_q10": torch.quantile(cluster_size, q=0.1),
                "vq_cluster_size_q50": torch.quantile(cluster_size, q=0.5),
                "vq_cluster_size_q90": torch.quantile(cluster_size, q=0.9),
                "vq_cluster_size_q99": torch.quantile(cluster_size, q=0.99),
                "vq_num_expired_codes": info_1["num_expired_codes"] + info_2["num_expired_codes"],
            }

        return feature_vq, aux_losses, aux_outputs

    @property
    def name(self):
        _, f, p, v, d = self.model_size
        return f"mix9svq_{self.input_type}_{f}f{p}p{v}v{d}d{self.codebook_size}c"


@MODELS.register("mix10")
class Mix10Net(nn.Module):
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
        self.policy_small_output = nn.Conv2d(dim_policy_small_out, 1, 1)

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
        self.policy_large_output = nn.Conv2d(dim_policy_large_out, 1, 1)

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

    def initialize(self):
        self.feature_dwconv.conv.weight.data.mul_(0.25)

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
        # apply relu activation
        feature = F.relu(feature)  # [B, dim_feature, H, W] int16, scale=128, [0,16]

        # apply feature depth-wise conv
        _, _, dim_dwconv, _ = self.model_size
        feat_dwconv = feature[:, :dim_dwconv]  # int16, scale=128, [0,16]
        feat_dwconv = self.feature_dwconv(feat_dwconv * 4)  # [B, dwconv, H, W] relu
        feat_dwconv = fake_quant(feat_dwconv, scale=128, num_bits=16)  # int16, scale=128, [0,9/2*16*4]

        # apply mask to the feature after dwconv
        if self.feature_norm is not None:
            feat_dwconv = self.feature_norm(feat_dwconv, *input_plane[1:])

        # apply activation for direct feature
        feat_direct = feature[:, dim_dwconv:]  # [B, dim_feature-dwconv, H, W] int16, scale=128, [0,16]
        feat_direct = fake_quant(feat_direct, scale=128, num_bits=16)  # int16, scale=128, [0,16]

        feature = torch.cat([feat_dwconv, feat_direct], dim=1)  # [B, dim_feature, H, W]

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
        def get_group_feature(y0, y1, x0, x1, mod):
            f = torch.sum(feature[:, :, y0:y1, x0:x1], dim=(2, 3))  # [B, dim_feature]
            f = fake_quant(f / 32, scale=128, num_bits=8, floor=True)  # srai 5
            f = torch.cat([f, f], dim=1)  # [B, dim_feature * 2]
            # i16 dot product of two adjacent pairs of u8 and i8
            x = (f * mod).view(*f.shape[:-1], -1, 2).sum(-1)
            # clamp to int8, scale=128, [-1,1]
            x = fake_quant(x, scale=128, num_bits=8, floor=True)
            return x

        _, _, H, W = feature.shape
        H0, W0 = 0, 0
        H1, W1 = (H // 3) + (H % 3 == 2), (W // 3) + (W % 3 == 2)
        H2, W2 = (H // 3) * 2 + (H % 3 > 0), (W // 3) * 2 + (W % 3 > 0)
        H3, W3 = H, W
        feature_00 = get_group_feature(H0, H1, W0, W1, feature_mod)  # [B, dim_feature]
        feature_01 = get_group_feature(H0, H1, W1, W2, feature_mod)  # [B, dim_feature]
        feature_02 = get_group_feature(H0, H1, W2, W3, feature_mod)  # [B, dim_feature]
        feature_10 = get_group_feature(H1, H2, W0, W1, feature_mod)  # [B, dim_feature]
        feature_11 = get_group_feature(H1, H2, W1, W2, feature_mod)  # [B, dim_feature]
        feature_12 = get_group_feature(H1, H2, W2, W3, feature_mod)  # [B, dim_feature]
        feature_20 = get_group_feature(H2, H3, W0, W1, feature_mod)  # [B, dim_feature]
        feature_21 = get_group_feature(H2, H3, W1, W2, feature_mod)  # [B, dim_feature]
        feature_22 = get_group_feature(H2, H3, W2, W3, feature_mod)  # [B, dim_feature]
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
        policy = torch.cat(
            [
                F.conv2d(
                    input=policy.reshape(1, B * dim_policy_small_in, H, W),
                    weight=pwconv_weight[:, dim_policy_small_in * i : dim_policy_small_in * (i + 1)],
                    groups=B,
                ).reshape(B, 1, H, W)
                for i in range(dim_policy_small_out)
            ],
            1,
        )
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
        policy = torch.cat(
            [
                F.conv2d(
                    input=policy.reshape(1, B * dim_policy_large_in, H, W),
                    weight=pwconv_weight_1[:, dim_policy_large_in * i : dim_policy_large_in * (i + 1)],
                    groups=B,
                ).reshape(B, 1, H, W)
                for i in range(dim_policy_large_mid)
            ],
            1,
        )
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
        policy = torch.cat(
            [
                F.conv2d(
                    input=policy.reshape(1, B * dim_policy_large_mid, H, W),
                    weight=pwconv_weight_2[:, dim_policy_large_mid * i : dim_policy_large_mid * (i + 1)],
                    groups=B,
                ).reshape(B, 1, H, W)
                for i in range(dim_policy_large_out)
            ],
            1,
        )
        policy = torch.clamp(
            policy + pwconv_bias_2, min=0
        )  # [B, dim_policy_large_out, H, W] int32, scale=128*128*128, relu

        policy = self.policy_large_output(policy)  # [B, 1, H, W]

        return policy

    def forward(self, data):
        # get feature from single side
        feature, *retvals = self.get_feature(data, False)  # [B, dim_feature, H, W]

        # value head
        value_small, value_small_feature = self.value_head_small(feature)
        value_large, value_large_feature = self.value_head_large(feature, value_small_feature)

        # policy head
        policy_small = self.policy_head_small(feature, value_small_feature)
        policy_large = self.policy_head_large(feature, value_large_feature)

        retvals[0].update(
            {
                "value_small": ("value_loss", value_small),
                "policy_small": ("policy_loss", policy_small),
                "policy_small_reg": ("policy_reg", policy_small),
            }
        )
        return value_large, policy_large, *retvals

    def forward_debug_print(self, data):
        # get feature from single side
        feature, *retvals = self.get_feature(data, False)  # [B, dim_feature, H, W]

        # value head
        value_small, value_small_feature = self.value_head_small(feature)
        value_large, value_large_feature = self.value_head_large(feature, value_small_feature)
        print(f"value_small_feature: \n{(value_small_feature*128).int()}")
        print(f"value_small: \n{value_small}")
        print(f"value_large_feature: \n{(value_large_feature*128).int()}")
        print(f"value_large: \n{value_large}")

        # policy head
        policy_small = self.policy_head_small(feature, value_small_feature)
        policy_large = self.policy_head_large(feature, value_large_feature)
        print(f"Raw Policy Small: \n{(policy_small[0, 0]*32).int()}")
        print(f"Raw Policy Large: \n{(policy_large[0, 0]*32).int()}")

        retvals[0].update(
            {
                "value_small": ("value_loss", value_small),
                "policy_small": ("policy_loss", policy_small),
                "policy_small_reg": ("policy_reg", policy_small),
            }
        )
        return value_large, policy_large, *retvals

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
