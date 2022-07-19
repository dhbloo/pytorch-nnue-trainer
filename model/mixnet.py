import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.quantization import QuantStub, DeQuantStub

from . import MODELS
from .blocks import Conv2dBlock, LinearBlock
from .input import build_input_plane

###########################################################
# Mix6Net adopted from https://github.com/hzyhhzy/gomoku_nnue/blob/87603e908cb1ae9106966e3596830376a637c21a/train_pytorch/model.py#L736
###########################################################


def tuple_op(f, x):
    return tuple(map(f, x))


class DirectionalConvLayer(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.weight = Parameter(torch.empty((3, dim_out, dim_in)))
        self.bias = Parameter(torch.zeros((dim_out, )))
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
        return tuple(self._conv1d_direction(xi, i) for i, xi in enumerate(x))


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
        x = tuple_op(lambda t: t[0] + t[1], zip(x, residual))
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
        x = tuple_op(lambda t: t[0] + t[1], zip(x, residual))
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

    def forward(self, x):
        x = self.d_conv((x, x, x, x))
        x = tuple_op(self.activation, x)
        x = self.convs(x)
        x = tuple_op(self.final_conv, x)
        x = torch.stack(x, dim=1)  # [B, 4, dim_out, H, W]
        return x


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
class Mix6QNet(nn.Module):
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
        self.policy_output_scale = Parameter(torch.ones(1), True)
        self.value_output_scale = Parameter(torch.ones(1), True)

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
        value = torch.mean(value,
                           dim=(2, 3))  # range [-maxf_i8_f_after_mean, maxf_i8_f_after_mean]
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
                 map_max=30,
                 input_type='basic-nostm',
                 reuse_feature=False,
                 dwconv_kernel_size=3):
        super().__init__()
        self.model_size = (dim_middle, dim_policy, dim_value)
        self.map_max = map_max
        self.input_type = input_type
        self.reuse_feature = reuse_feature
        dim_out = max(dim_policy, dim_value) if reuse_feature else dim_policy + dim_value

        self.input_plane = build_input_plane(input_type)
        self.mapping = Mapping(self.input_plane.dim_plane, dim_middle, dim_out)
        self.mapping_activation = nn.PReLU(dim_out)

        # feature depth-wise conv
        self.feature_dwconv = Conv2dBlock(dim_out,
                                          dim_out,
                                          ks=dwconv_kernel_size,
                                          st=1,
                                          padding=dwconv_kernel_size // 2,
                                          groups=dim_out)

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
        self.value_activation = nn.PReLU(dim_value)
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
        feature = torch.mean(feature, dim=1)  # [B, PC+VC, H, W]
        feature = self.mapping_activation(feature)

        # feature conv
        feature = self.feature_dwconv(feature)  # [B, PC+VC, H, W]

        # policy head
        policy = feature[:, :dim_policy]
        policy = self.policy_pwconv(policy)
        policy = self.policy_activation(policy)

        # value head
        value = feature[:, :dim_value] if self.reuse_feature else feature[:, dim_policy:]
        value = torch.mean(value, dim=(2, 3))
        value = self.value_activation(value)
        value = self.value_linear(value)

        return value, policy

    @property
    def name(self):
        m, p, v = self.model_size
        return f"mix7_{self.input_type}_{m}m{p}p{v}v" + (f"-{self.map_max}mm"
                                                         if self.map_max != 0 else "")
