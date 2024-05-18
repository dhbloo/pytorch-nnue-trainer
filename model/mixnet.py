import torch
import torch.nn as nn
import torch.nn.functional as F

from . import MODELS
from .blocks import Conv2dBlock, LinearBlock, ChannelWiseLeakyReLU, QuantPReLU, \
    SwitchGate, SwitchLinearBlock, SwitchPReLU, SequentialWithExtraArguments
from .input import build_input_plane
from utils.quant_utils import fake_quant

###########################################################
# Mix6Net adopted from https://github.com/hzyhhzy/gomoku_nnue/blob/87603e908cb1ae9106966e3596830376a637c21a/train_pytorch/model.py#L736
###########################################################


def tuple_op(f, x):
    return tuple((f(xi) if xi is not None else None) for xi in x)

def add_op(t):
    return None if (t[0] is None and t[1] is None) else t[0] + t[1]


class DirectionalConvLayer(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((3, dim_out, dim_in)))
        self.bias = nn.Parameter(torch.zeros((dim_out, )))
        nn.init.kaiming_normal_(self.weight)

    def _conv1d_direction(self, x, dir):
        kernel_size, dim_out, dim_in = self.weight.shape
        zero = torch.zeros((dim_out, dim_in), dtype=self.weight.dtype, device=self.weight.device)

        weight_func_map = [
            lambda w: (zero, zero, zero, w[0], w[1], w[2], zero, zero, zero),
            lambda w: (zero, w[0], zero, zero, w[1], zero, zero, w[2], zero),
            lambda w: (w[0], zero, zero, zero, w[1], zero, zero, zero, w[2]),
            lambda w: (zero, zero, w[2], zero, w[1], zero, w[0], zero, zero),
        ]
        weight = torch.stack(weight_func_map[dir](self.weight), dim=2)
        weight = weight.reshape(dim_out, dim_in, kernel_size, kernel_size)
        return torch.conv2d(x, weight, self.bias, padding=kernel_size // 2)

    def forward(self, x):
        assert len(x) == 4, f"must be 4 directions, got {len(x)}"
        return tuple((self._conv1d_direction(xi, i) if xi is not None else None) for i, xi in enumerate(x))


class DirectionalConvResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.d_conv = DirectionalConvLayer(dim, dim)
        self.conv1x1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.d_conv(x)
        x = tuple_op(self.activation, x)
        x = tuple_op(self.conv1x1, x)
        x = tuple_op(self.activation, x)
        x = tuple_op(add_op, zip(x, residual))
        return x


class Conv0dResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        residual = x
        x = tuple_op(self.conv1, x)
        x = tuple_op(self.activation, x)
        x = tuple_op(self.conv2, x)
        x = tuple_op(self.activation, x)
        x = tuple_op(add_op, zip(x, residual))
        return x


class Mapping(nn.Module):
    def __init__(self, dim_in, dim_middle, dim_out):
        super().__init__()
        self.d_conv = DirectionalConvLayer(dim_in, dim_middle)
        self.convs = nn.Sequential(
            *[DirectionalConvResBlock(dim_middle) for _ in range(4)],
            Conv0dResBlock(dim_middle),
        )
        self.final_conv = nn.Conv2d(dim_middle, dim_out, kernel_size=1)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x, dirs=[0, 1, 2, 3]):
        x = tuple((x if i in dirs else None) for i in range(4))
        x = self.d_conv(x)
        x = tuple_op(self.activation, x)
        x = self.convs(x)
        x = tuple_op(self.final_conv, x)
        x = tuple(xi for xi in x if xi is not None)
        x = torch.stack(x, dim=1)  # [B, <=4, dim_out, H, W]
        return x


class RotatedConv2d3x3(nn.Module):
    def __init__(self, dim_in, dim_out) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty((9, dim_out, dim_in)))
        self.bias = nn.Parameter(torch.zeros((dim_out, )))
        nn.init.kaiming_normal_(self.weight)

    def forward(self, x):
        w = self.weight
        _, dim_out, dim_in = w.shape
        zero = torch.zeros((dim_out, dim_in), dtype=w.dtype, device=w.device)
        weight7x7 = [
            zero, zero, zero, zero, w[0], zero, zero, \
            zero, zero, w[1], zero, zero, zero, zero, \
            w[2], zero, zero, zero, zero, w[3], zero, \
            zero, zero, zero, w[4], zero, zero, zero, \
            zero, w[5], zero, zero, zero, zero, w[6], \
            zero, zero, zero, zero, w[7], zero, zero, \
            zero, zero, w[8], zero, zero, zero, zero,
        ]
        weight7x7 = torch.stack(weight7x7, dim=2)
        weight7x7 = weight7x7.reshape(dim_out, dim_in, 7, 7)
        return torch.conv2d(x, weight7x7, self.bias, padding=7 // 2)


@MODELS.register('mix6')
class Mix6Net(nn.Module):
    def __init__(self,
                 dim_middle=128,
                 dim_policy=16,
                 dim_value=32,
                 map_max=30,
                 input_type='basic-nostm'):
        super().__init__()
        self.model_size = (dim_middle, dim_policy, dim_value)
        self.map_max = map_max
        self.input_type = input_type
        dim_out = dim_policy + dim_value

        self.input_plane = build_input_plane(input_type)
        self.mapping = Mapping(self.input_plane.dim_plane, dim_middle, dim_out)
        self.mapping_activation = ChannelWiseLeakyReLU(dim_out, bound=6)

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
        return f"mix6_{self.input_type}_{m}m{p}p{v}v" + (f"-{self.map_max}mm"
                                                         if self.map_max != 0 else "")


@MODELS.register('mix6q')
class Mix6QNet(nn.Module):  # Q -> quant
    def __init__(self,
                 dim_middle=128,
                 dim_policy=32,
                 dim_value=32,
                 input_type='basic',
                 quantization=True,
                 scale_feature=4):
        super().__init__()
        self.model_size = (dim_middle, dim_policy, dim_value)
        self.input_type = input_type
        dim_out = dim_policy + dim_value

        self.input_plane = build_input_plane(input_type)
        self.mapping = Mapping(self.input_plane.dim_plane, dim_middle, dim_out)
        self.mapping_activation = nn.PReLU(dim_out)

        # policy nets
        self.policy_dw_conv = Conv2dBlock(dim_policy,
                                          dim_policy,
                                          ks=3,
                                          st=1,
                                          padding=1,
                                          groups=dim_policy,
                                          activation='lrelu/16')
        self.policy_pw_conv = Conv2dBlock(dim_policy,
                                          1,
                                          ks=1,
                                          st=1,
                                          padding=0,
                                          activation='lrelu/16',
                                          bias=False)

        # value nets
        self.value_activation = nn.ReLU(inplace=True)
        self.value_linear1 = LinearBlock(dim_value, dim_value, activation='relu')
        self.value_linear2 = LinearBlock(dim_value, dim_value, activation='relu')
        self.value_linear_final = LinearBlock(dim_value, 3, activation='none')

        # adaptive scales
        self.policy_output_scale = nn.Parameter(torch.ones(1), True)
        self.value_output_scale = nn.Parameter(torch.ones(1), True)

        # quantization constants
        self.quantization = quantization
        self.scale_weight = 64
        self.scale_feature = scale_feature
        self.scale_feature_after_mean = self.scale_feature / 16 * 3600 / 256
        self.maxf_i8_w = 127 / self.scale_weight  # 1.98438
        self.maxf_i8_f = 127 / self.scale_feature
        self.maxf_i8_f_after_mean = 127 / self.scale_feature_after_mean

    def forward(self, data):
        _, dim_policy, _ = self.model_size

        input_plane = self.input_plane(data)
        feature = self.mapping(input_plane)
        # rescale to range [-maxf_i8_f, maxf_i8_f]
        feature = torch.tanh(feature / self.maxf_i8_f) * self.maxf_i8_f

        # sum feature across four directions
        feature = torch.sum(feature, dim=1)  # [B, PC+VC, H, W], range [-maxf_i8_f*4, maxf_i8_f*4]
        feature = self.mapping_activation(feature)  # range [-maxf_i8_f*4, maxf_i8_f*4]
        feature = feature / 4  # range [-maxf_i8_f, maxf_i8_f]

        # policy head
        policy = feature[:, :dim_policy]  # range [-maxf_i8_f, maxf_i8_f]
        policy = self.policy_dw_conv(policy)
        if self.quantization:
            # Clipped LeakyReLU, range [-maxf_i8_f, maxf_i8_f]
            policy = torch.clamp(policy, min=-self.maxf_i8_f, max=self.maxf_i8_f)
        policy = self.policy_pw_conv(policy)
        policy = policy * self.policy_output_scale

        # value head
        value = feature[:, dim_policy:]  # range [-maxf_i8_f, maxf_i8_f]
        value = torch.mean(value, dim=(2, 3))  # range [-maxf_i8_f_after_mean, maxf_i8_f_after_mean]
        value = self.value_activation(value)  # range [0, maxf_i8_f_after_mean]
        value = self.value_linear1(value)
        if self.quantization:
            # ClippedReLU, range [0, maxf_i8_f_after_mean]
            value = torch.clamp(value, max=self.maxf_i8_f_after_mean)
        value = self.value_linear2(value)
        if self.quantization:
            # ClippedReLU, range [0, maxf_i8_f_after_mean]
            value = torch.clamp(value, max=self.maxf_i8_f_after_mean)
        value = self.value_linear_final(value)
        value = value * self.value_output_scale

        return value, policy

    @property
    def weight_clipping(self):
        return [] if not self.quantization else [
            {
                'params': ['mapping_activation.weight'],
                'min_weight': -64 / 64,
                'max_weight': 64 / 64
            },
            {
                'params': [
                    'policy_dw_conv.conv.weight',
                    'policy_pw_conv.conv.weight',
                    'value_linear1.fc.weight',
                    'value_linear2.fc.weight',
                    'value_linear_final.fc.weight',
                ],
                'min_weight':
                -self.maxf_i8_w,
                'max_weight':
                self.maxf_i8_w
            },
        ]

    @property
    def name(self):
        m, p, v = self.model_size
        return f"mix6q_{self.input_type}_{m}m{p}p{v}v"


@MODELS.register('mix7')
class Mix7Net(nn.Module):
    def __init__(self,
                 dim_middle=128,
                 dim_policy=32,
                 dim_value=32,
                 dim_dwconv=None,
                 map_max=30,
                 input_type='basic-nostm',
                 dwconv_kernel_size=3):
        super().__init__()
        self.model_size = (dim_middle, dim_policy, dim_value)
        self.map_max = map_max
        self.input_type = input_type
        self.dwconv_kernel_size = dwconv_kernel_size
        dim_out = max(dim_policy, dim_value)
        self.dim_dwconv = dim_out if dim_dwconv is None else dim_dwconv
        assert self.dim_dwconv <= dim_out, "Incorrect dim_dwconv!"

        self.input_plane = build_input_plane(input_type)
        self.mapping = Mapping(self.input_plane.dim_plane, dim_middle, dim_out)
        self.mapping_activation = nn.PReLU(dim_out)

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
        feat_dwconv = self.feature_dwconv(feature[:, :self.dim_dwconv])  # [B, dwconv, H, W]
        feat_direct = feature[:, self.dim_dwconv:]  # [B, max(PC,VC)-dwconv, H, W]
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
        m, p, v = self.model_size
        return f"mix7_{self.input_type}_{m}m{p}p{v}v" + (f"-{self.map_max}mm"
                                                         if self.map_max != 0 else "")


@MODELS.register('mix8')
class Mix8Net(nn.Module):
    def __init__(self,
                 dim_middle=128,
                 dim_feature=64,
                 dim_policy=32,
                 dim_value=64,
                 dim_value_group=32,
                 dim_dwconv=None,
                 input_type='basicns'):
        super().__init__()
        dim_feature = max(dim_policy, dim_value)
        self.model_size = (dim_middle, dim_feature, dim_policy, dim_value, dim_value_group)
        self.input_type = input_type
        self.dim_dwconv = dim_policy if dim_dwconv is None else dim_dwconv
        assert self.dim_dwconv <= dim_feature, f"Invalid dim_dwconv {self.dim_dwconv}"
        assert self.dim_dwconv >= dim_policy, "dim_dwconv must be not less than dim_policy"

        self.input_plane = build_input_plane(input_type)
        self.mapping = Mapping(self.input_plane.dim_plane, dim_middle, dim_feature)
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
            activation='relu',
        )

        # policy head (point-wise conv)
        self.policy_pwconv_weight_linear = nn.Sequential(
            LinearBlock(dim_feature, dim_policy, activation='none'),
            nn.PReLU(dim_policy),
            LinearBlock(dim_policy, 4 * dim_policy, activation='none'),
        )
        self.policy_output = nn.Sequential(
            nn.PReLU(4),
            nn.Conv2d(4, 1, 1),
        )

        # value head
        self.value_corner_linear = LinearBlock(dim_feature, dim_value_group, activation='none')
        self.value_corner_act = nn.PReLU(dim_value_group)
        self.value_edge_linear = LinearBlock(dim_feature, dim_value_group, activation='none')
        self.value_edge_act = nn.PReLU(dim_value_group)
        self.value_center_linear = LinearBlock(dim_feature, dim_value_group, activation='none')
        self.value_center_act = nn.PReLU(dim_value_group)
        self.value_quad_linear = LinearBlock(dim_value_group, dim_value_group, activation='none')
        self.value_quad_act = nn.PReLU(dim_value_group)
        self.value_linear = nn.Sequential(
            LinearBlock(dim_feature + 4 * dim_value_group, dim_value),
            LinearBlock(dim_value, dim_value),
            LinearBlock(dim_value, 3, activation='none'),
        )

    def get_feature(self, data, inv_side=False):
        # get the input plane from board and side to move input
        input_plane = self.input_plane({
            'board_input':
            torch.flip(data['board_input'], dims=[1]) if inv_side else data['board_input'],
            'stm_input':
            -data['stm_input'] if inv_side else data['stm_input']
        })  # [B, 2, H, W]
        # get per-point 4-direction cell features
        feature = self.mapping(input_plane)  # [B, 4, dim_feature, H, W]

        # clamp feature for int quantization
        feature = torch.clamp(feature, min=-32, max=32)  # int16, scale=255, [-32,32]
        feature = fake_quant(feature, scale=255, num_bits=16)
        # sum (and rescale) feature across four directions
        feature = torch.mean(feature, dim=1)  # [B, dim_feature, H, W] int16, scale=1020, [-32,32]
        feature = self.mapping_activation(feature)  # int16, scale=1020, [-32,32]

        # apply feature depth-wise conv
        feat_dwconv = self.feature_dwconv(feature[:, :self.dim_dwconv])  # [B, dwconv, H, W]
        feat_direct = feature[:, self.dim_dwconv:]  # [B, dim_feature-dwconv, H, W]
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
        policy = torch.cat([
            F.conv2d(input=policy.reshape(1, B * dim_policy, H, W),
                     weight=pwconv_weight[:, dim_policy * i:dim_policy * (i + 1)],
                     groups=B).reshape(B, 1, H, W) for i in range(4)
        ],
                           dim=1)
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
        value = torch.cat([
            feature_mean,
            value_q00,
            value_q01,
            value_q10,
            value_q11,
        ], 1)  # [B, dim_feature + 4 * dim_value_group]
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
        policy = torch.cat([
            F.conv2d(input=policy.reshape(1, B * dim_policy, H, W),
                     weight=pwconv_weight[:, dim_policy * i:dim_policy * (i + 1)],
                     groups=B).reshape(B, 1, H, W) for i in range(4)
        ],
                           dim=1)
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
        value = torch.cat([
            feature_mean,
            value_q00,
            value_q01,
            value_q10,
            value_q11,
        ], 1)  # [B, dim_feature + 4 * dim_value_group]
        print(f"value feature input: \n{value}")
        for i, linear in enumerate(self.value_linear):
            value = linear(value)
            print(f"value feature after layer {i}: \n{value}")

        return value, policy

    @property
    def weight_clipping(self):
        # Clip prelu weight of mapping activation to [-1,1] to avoid overflow
        # In this range, prelu is the same as `max(x, ax)`.
        return [{
            'params': [
                'mapping_activation.weight', 'policy_pwconv_weight_linear.1.weight',
                'value_corner_act.weight', 'value_edge_act.weight', 'value_center_act.weight',
                'value_quad_act.weight'
            ],
            'min_weight':
            -1.0,
            'max_weight':
            1.0,
        }, {
            'params': [f'feature_dwconv.conv.weight'],
            'min_weight': -1.5,
            'max_weight': 1.5,
        }, {
            'params': [f'feature_dwconv.conv.bias'],
            'min_weight': -4.0,
            'max_weight': 4.0,
        }]

    @property
    def name(self):
        _, f, p, v, q = self.model_size
        d = self.dim_dwconv
        return f"mix8_{self.input_type}_{f}f{p}p{v}v{q}q{d}d"


class StarBlock(nn.Module):
    def __init__(self, dim_in, dim_out, expand=1):
        super().__init__()
        self.up1 = LinearBlock(dim_in, dim_out * 2 * expand, activation='relu', quant=True)
        self.up2 = LinearBlock(dim_in, dim_out * 2 * expand, activation='none', quant=True)
        self.down = LinearBlock(dim_out * expand, dim_out, activation='relu', quant=True)

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
    
    def forward_debug_print(self, x, name='starblock'):
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


@MODELS.register('mix9')
class Mix9Net(nn.Module):
    def __init__(self,
                 dim_middle=128,
                 dim_feature=64,
                 dim_policy=32,
                 dim_value=64,
                 dim_dwconv=32,
                 input_type='basicns'):
        super().__init__()
        self.model_size = (dim_middle, dim_feature, dim_policy, dim_value, dim_dwconv)
        self.input_type = input_type
        assert dim_dwconv <= dim_feature, f"Invalid dim_dwconv {dim_dwconv}"
        assert dim_dwconv >= dim_policy, "dim_dwconv must be not less than dim_policy"

        self.input_plane = build_input_plane(input_type)
        self.mapping1 = Mapping(self.input_plane.dim_plane, dim_middle, dim_feature)
        self.mapping2 = Mapping(self.input_plane.dim_plane, dim_middle, dim_feature)

        # feature depth-wise conv
        self.feature_dwconv = Conv2dBlock(
            dim_dwconv,
            dim_dwconv,
            ks=3,
            st=1,
            padding=3 // 2,
            groups=dim_dwconv,
            activation='relu',
            quant='pixel-dwconv-floor',
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
            LinearBlock(dim_feature, dim_policy * 2, activation='relu', quant=True),
            LinearBlock(dim_policy * 2, dim_pm * dim_policy + dim_pm, activation='none', quant=True),
        )
        self.policy_output = nn.Conv2d(dim_pm, 1, 1)

        self.value_corner = StarBlock(dim_feature, dim_value)
        self.value_edge = StarBlock(dim_feature, dim_value)
        self.value_center = StarBlock(dim_feature, dim_value)
        self.value_quad = StarBlock(dim_value, dim_value)
        self.value_linear = nn.Sequential(
            LinearBlock(dim_feature + 4 * dim_value, dim_value, activation='relu', quant=True),
            LinearBlock(dim_value, dim_value, activation='relu', quant=True),
            LinearBlock(dim_value, 3, activation='none', quant=True),
        )

    def custom_init(self):
        self.feature_dwconv.conv.weight.data.mul_(0.25)

    def get_feature(self, data, inv_side=False):
        # get the input plane from board and side to move input
        input_plane = self.input_plane({
            'board_input':
            torch.flip(data['board_input'], dims=[1]) if inv_side else data['board_input'],
            'stm_input':
            -data['stm_input'] if inv_side else data['stm_input']
        })  # [B, 2, H, W]
        # get per-point 4-direction cell features
        feature1 = self.mapping1(input_plane, dirs=[0, 1])  # [B, 2, dim_feature, H, W]
        feature2 = self.mapping2(input_plane, dirs=[2, 3])  # [B, 2, dim_feature, H, W]
        feature = torch.cat([feature1, feature2], dim=1)  # [B, 4, dim_feature, H, W]

        # clamp feature for int quantization
        feature = torch.clamp(feature, min=-16, max=16)  # int16, scale=32, [-16,16]
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

        # value feature accumulator
        feature_sum = torch.sum(feature, dim=(2, 3))  # [B, dim_feature]
        feature_sum = fake_quant(feature_sum / 256, scale=128, num_bits=32, floor=True)  # srai 8

        # value feature accumulator of nine groups
        B, _, H, W = feature.shape
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

        # policy head
        dim_pm = self.policy_middle_dim
        pwconv_output = self.policy_pwconv_weight_linear(feature_sum)
        pwconv_weight = pwconv_output[:, :dim_pm * dim_policy].reshape(B, dim_pm * dim_policy, 1, 1)
        pwconv_weight = fake_quant(pwconv_weight, scale=128*128, num_bits=16, floor=True)
        policy = fake_quant(feature[:, :dim_policy], scale=128, num_bits=16)  # [B, dim_policy, H, W]
        policy = torch.cat([
            F.conv2d(input=policy.reshape(1, B * dim_policy, H, W),
                     weight=pwconv_weight[:, dim_policy * i:dim_policy * (i + 1)],
                     groups=B).reshape(B, 1, H, W) for i in range(dim_pm)
        ], 1)
        pwconv_bias = pwconv_output[:, dim_pm * dim_policy:].reshape(B, dim_pm, 1, 1)  # int32, scale=128*128*128
        policy = torch.clamp(policy + pwconv_bias, min=0)  # [B, dim_pm, H, W] int32, scale=128*128*128, relu
        policy = self.policy_output(policy)  # [B, 1, H, W]

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

        def avg4(a, b, c, d):
            a = fake_quant(a, floor=True)
            b = fake_quant(b, floor=True)
            c = fake_quant(c, floor=True)
            d = fake_quant(d, floor=True)
            ab = fake_quant((a + b + 1/128) / 2, floor=True)
            cd = fake_quant((c + d + 1/128) / 2, floor=True)
            return fake_quant((ab + cd + 1/128) / 2, floor=True)

        value_q00 = avg4(value_00, value_01, value_10, value_11)
        value_q01 = avg4(value_01, value_02, value_11, value_12)
        value_q10 = avg4(value_10, value_11, value_20, value_21)
        value_q11 = avg4(value_11, value_12, value_21, value_22)
        value_q00 = self.value_quad(value_q00)
        value_q01 = self.value_quad(value_q01)
        value_q10 = self.value_quad(value_q10)
        value_q11 = self.value_quad(value_q11)

        value = torch.cat([
            feature_sum,
            value_q00, value_q01, value_q10, value_q11,
        ], 1)  # [B, dim_feature + dim_value * 4]
        value = self.value_linear(value)

        return value, policy

    def forward_debug_print(self, data):
        _, _, dim_policy, _, _ = self.model_size

        # get feature from single side
        feature = self.get_feature(data, False)  # [B, dim_feature, H, W]
        print(f"feature after dwconv at (0,0): \n{(feature[..., 0, 0]*128).int()}")

        # value feature accumulator
        feature_sum = torch.sum(feature, dim=(2, 3))  # [B, dim_feature]
        print(f"feature sum before scale: \n{(feature_sum*128).int()}")
        feature_sum = fake_quant(feature_sum / 256, scale=128, num_bits=32, floor=True)  # srai 8
        print(f"feature sum: \n{(feature_sum*128).int()}")

        # value feature accumulator of nine groups
        B, _, H, W = feature.shape
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

        # policy head
        dim_pm = self.policy_middle_dim
        pwconv_output = self.policy_pwconv_weight_linear(feature_sum)
        print(f"policy pwconv output: \n{(pwconv_output*128*128).int()}")
        pwconv_weight = pwconv_output[:, :dim_pm * dim_policy].reshape(B, dim_pm * dim_policy, 1, 1)
        pwconv_weight = fake_quant(pwconv_weight, scale=128*128, num_bits=16, floor=True)
        print(f"policy pwconv weight: \n{(pwconv_weight.flatten(1, -1)*128*128).int()}")
        policy = fake_quant(feature[:, :dim_policy], scale=128, num_bits=16)  # [B, dim_policy, H, W]
        print(f"policy after dwconv at (0,0): \n{(policy[..., 0, 0]*128).int()}")
        policy = torch.cat([
            F.conv2d(input=policy.reshape(1, B * dim_policy, H, W),
                     weight=pwconv_weight[:, dim_policy * i:dim_policy * (i + 1)],
                     groups=B).reshape(B, 1, H, W) for i in range(dim_pm)
        ], 1)
        print(f"policy after dynamic pwconv at (0,0): \n{(policy[..., 0, 0]*128*128*128).int()}")
        pwconv_bias = pwconv_output[:, dim_pm * dim_policy:].reshape(B, dim_pm, 1, 1)  # int32, scale=128*128*128
        print(f"policy pwconv bias: \n{(pwconv_bias.flatten(1, -1)*128*128*128).int()}")
        policy = torch.clamp(policy + pwconv_bias, min=0)  # [B, dim_pm, H, W] int32, scale=128*128*128, relu
        print(f"policy pwconv output at (0,0): \n{(policy[..., 0, 0]*128*128*128).int()}")
        policy = self.policy_output(policy)  # [B, 1, H, W]
        print(f"policy output at (0,0): \n{policy[..., 0, 0]}")

        # value head
        value_00 = self.value_corner.forward_debug_print(feature_00, "value_00")
        value_01 = self.value_edge.forward_debug_print(feature_01, "value_01")
        value_02 = self.value_corner.forward_debug_print(feature_02, "value_02")
        value_10 = self.value_edge.forward_debug_print(feature_10, "value_10")
        value_11 = self.value_center.forward_debug_print(feature_11, "value_11")
        value_12 = self.value_edge.forward_debug_print(feature_12, "value_12")
        value_20 = self.value_corner.forward_debug_print(feature_20, "value_20")
        value_21 = self.value_edge.forward_debug_print(feature_21, "value_21")
        value_22 = self.value_corner.forward_debug_print(feature_22, "value_22")

        def avg4(a, b, c, d):
            a = fake_quant(a, floor=True)
            b = fake_quant(b, floor=True)
            c = fake_quant(c, floor=True)
            d = fake_quant(d, floor=True)
            ab = fake_quant((a + b + 1/128) / 2, floor=True)
            cd = fake_quant((c + d + 1/128) / 2, floor=True)
            return fake_quant((ab + cd + 1/128) / 2, floor=True)

        value_q00 = avg4(value_00, value_01, value_10, value_11)
        value_q01 = avg4(value_01, value_02, value_11, value_12)
        value_q10 = avg4(value_10, value_11, value_20, value_21)
        value_q11 = avg4(value_11, value_12, value_21, value_22)
        print(f"value_q00 avg: \n{(value_q00*128).int()}")
        print(f"value_q01 avg: \n{(value_q01*128).int()}")
        print(f"value_q10 avg: \n{(value_q10*128).int()}")
        print(f"value_q11 avg: \n{(value_q11*128).int()}")
        value_q00 = self.value_quad.forward_debug_print(value_q00, "value_q00")
        value_q01 = self.value_quad.forward_debug_print(value_q01, "value_q01")
        value_q10 = self.value_quad.forward_debug_print(value_q10, "value_q10")
        value_q11 = self.value_quad.forward_debug_print(value_q11, "value_q11")

        value = torch.cat([
            feature_sum,
            value_q00, value_q01, value_q10, value_q11,
        ], 1)  # [B, dim_feature + dim_value * 4]
        print(f"value feature input: \n{(value*128).int()}")
        for i, linear in enumerate(self.value_linear):
            value = linear(value)
            print(f"value feature after layer {i}: \n{(value*128).int()}")

        return value, policy

    @property
    def weight_clipping(self):
        # Clip prelu weight of mapping activation to [-1,1] to avoid overflow
        # In this range, prelu is the same as `max(x, ax)`.
        return [{
            'params': ['feature_dwconv.conv.weight'],
            'min_weight': -32768 / 65536,
            'max_weight': 32767 / 65536
        },
        {
            'params': ['value_corner.up1.fc.weight', 
                       'value_corner.up2.fc.weight', 
                       'value_corner.down.fc.weight', 
                       'value_edge.up1.fc.weight', 
                       'value_edge.up2.fc.weight', 
                       'value_edge.down.fc.weight', 
                       'value_center.up1.fc.weight', 
                       'value_center.up2.fc.weight', 
                       'value_center.down.fc.weight', 
                       'value_quad.up1.fc.weight', 
                       'value_quad.up2.fc.weight', 
                       'value_quad.down.fc.weight', 
                       'value_linear.0.fc.weight',
                       'value_linear.1.fc.weight',
                       'value_linear.2.fc.weight',
                       'policy_pwconv_weight_linear.0.fc.weight',
                       'policy_pwconv_weight_linear.1.fc.weight'],
            'min_weight': -128 / 128,
            'max_weight': 127 / 128
        }]

    @property
    def name(self):
        _, f, p, v, d = self.model_size
        return f"mix9_{self.input_type}_{f}f{p}p{v}v{d}d"

