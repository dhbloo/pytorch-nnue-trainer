import torch
import torch.nn as nn
import torch.nn.functional as F

from . import MODELS
from .blocks import Conv2dBlock, LinearBlock, build_activation_layer
from .mixnet import StarBlock
from utils.quant_utils import fake_quant


def get_total_num_line_encoding(length):
    half = length // 2
    code = 2 * 3**length
    for i in range(0, half + 1):
        code += 2 * 3**i
    for i in range(half + 2, length):
        code += 1 * 3**i
    return code + 1


class LineEncodingMapping(nn.Module):
    def __init__(self, dim_middle, dim_out, line_length=11, activation="silu"):
        super().__init__()
        self.line_length = line_length
        self.dim_middle = dim_middle
        self.dim_out = dim_out
        self.total_num = 4**line_length
        assert line_length % 2 == 1, "Line length must be odd"
        assert line_length >= 5, "Line length must be at least 5"

        self.act = build_activation_layer(activation)
        self.input_conv = nn.Conv1d(4, dim_middle, kernel_size=5, padding=2)
        self.conv1x1_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        for _ in range(line_length // 2 - 2):
            conv1x1 = nn.Conv1d(dim_middle, dim_middle, kernel_size=1)
            conv = nn.Conv1d(dim_middle, dim_middle, kernel_size=3, padding=1)
            self.conv1x1_layers.append(conv1x1)
            self.conv_layers.append(conv)
        self.output_fc1 = nn.Linear(dim_middle, dim_middle)
        self.output_fc2 = nn.Linear(dim_middle, dim_out)

    def convert_index_to_feature(self, line_index):
        fs_indices = []
        for i in range(self.line_length):
            f = torch.bitwise_right_shift(line_index, i * 2)
            f = torch.bitwise_and(f, 0b11)
            fs_indices.append(f)
        fs_indices = torch.stack(fs_indices, dim=-1)  # [..., line_length]
        fs_onehot = torch.stack(
            [
                fs_indices == 0,
                fs_indices == 1,
                fs_indices == 2,
                fs_indices == 3,
            ],
            dim=-2,
        )  # [..., 4, line_length]
        return fs_onehot.float()

    def forward_line_encoding(self, line_encoding):
        fs = self.convert_index_to_feature(line_encoding)  # [N, 4, line_length]
        fs = self.act(self.input_conv(fs))  # [N, dim_middle, line_length]
        for conv1x1, conv in zip(self.conv1x1_layers, self.conv_layers):
            fs_redisual = fs
            fs = self.act(conv1x1(fs))
            fs = self.act(conv(fs))
            fs = fs + fs_redisual
        fs = fs[..., self.line_length // 2 : self.line_length // 2 + 1]  # [N, dim_middle, 1]
        fs = torch.squeeze(fs, dim=-1)  # [N, dim_middle]
        fs = self.act(self.output_fc1(fs))  # [N, dim_middle]
        fs = self.output_fc2(fs)  # [N, dim_out]
        return fs

    def forward(self, data):
        # Must be raw code
        assert torch.all(
            data["line_encoding_total_num"] == self.total_num
        ), "Incorrect total number of line encoding, must set raw_code=True"
        # line_encoding: [B, 4, H, W]
        # enc_unique: [N_unique, 4, line_length]
        # inv_indices: [B, 4, H, W]
        enc_unique, inv_indices = torch.unique(data["line_encoding"], return_inverse=True)
        fs = self.forward_line_encoding(enc_unique)  # [N_unique, dim_out]
        fs = F.embedding(inv_indices.flatten(), fs)  # [B*4*H*W, dim_out]
        fs = fs.view(*inv_indices.shape, self.dim_out)  # [B, 4, H, W, dim_out]
        fs = fs.permute(0, 1, 4, 2, 3)  # [B, 4, dim_out, H, W]
        return fs


@MODELS.register("linennuev1")
class LineNNUEv1(nn.Module):
    LINE_LENGTH = 11

    def __init__(self, dim_feature=64, dim_policy=32, dim_value=64, dim_dwconv=32):
        super().__init__()
        self.model_size = (dim_feature, dim_policy, dim_value, dim_dwconv)
        self.line_encoding_total_num = get_total_num_line_encoding(self.LINE_LENGTH)

        self.mapping = nn.Embedding(self.line_encoding_total_num, dim_feature)

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

    def custom_init(self):
        self.feature_dwconv.conv.weight.data.mul_(0.25)

    def get_feature(self, data):
        # get per-point 4-direction cell features
        feature_index = data["line_encoding"]  # [B, 4, H, W]
        feature = self.mapping(feature_index)  # [B, 4, H, W, dim_feature]
        feature = torch.permute(feature, (0, 1, 4, 2, 3))  # [B, 4, dim_feature, H, W]

        # clamp feature for int quantization
        feature = torch.clamp(feature, min=-16, max=16)  # int16, scale=32, [-16,16]
        feature = fake_quant(feature, scale=32, num_bits=16)
        # sum (and rescale) feature across four directions
        feature = torch.mean(feature, dim=1)  # [B, dim_feature, H, W] int16, scale=128, [-16,16]
        # apply relu activation
        feature = F.relu(feature)  # [B, dim_feature, H, W] int16, scale=128, [0,16]

        # apply feature depth-wise conv
        _, _, _, dim_dwconv = self.model_size
        feat_dwconv = feature[:, :dim_dwconv]  # int16, scale=128, [0,16]
        feat_dwconv = self.feature_dwconv(feat_dwconv * 4)  # [B, dwconv, H, W] relu
        feat_dwconv = fake_quant(feat_dwconv, scale=128, num_bits=16)  # int16, scale=128, [0,9/2*16*4]

        # apply activation for direct feature
        feat_direct = feature[:, dim_dwconv:]  # [B, dim_feature-dwconv, H, W] int16, scale=128, [0,16]
        feat_direct = fake_quant(feat_direct, scale=128, num_bits=16)  # int16, scale=128, [0,16]

        feature = torch.cat([feat_dwconv, feat_direct], dim=1)  # [B, dim_feature, H, W]

        return feature

    def forward(self, data):
        assert torch.all(data["line_encoding_total_num"] == self.line_encoding_total_num)
        _, dim_policy, _, _ = self.model_size

        # get feature from single side
        feature = self.get_feature(data)  # [B, dim_feature, H, W]

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
        pwconv_bias = pwconv_output[:, dim_pm * dim_policy :].reshape(B, dim_pm, 1, 1)  # int32, scale=128*128*128
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
            ab = fake_quant((a + b + 1 / 128) / 2, floor=True)
            cd = fake_quant((c + d + 1 / 128) / 2, floor=True)
            return fake_quant((ab + cd + 1 / 128) / 2, floor=True)

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

    @property
    def weight_clipping(self):
        # Clip prelu weight of mapping activation to [-1,1] to avoid overflow
        # In this range, prelu is the same as `max(x, ax)`.
        return [
            {"params": ["feature_dwconv.conv.weight"], "min_weight": -32768 / 65536, "max_weight": 32767 / 65536},
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
                    "policy_pwconv_weight_linear.0.fc.weight",
                    "policy_pwconv_weight_linear.1.fc.weight",
                ],
                "min_weight": -128 / 128,
                "max_weight": 127 / 128,
            },
        ]

    @property
    def name(self):
        f, p, v, d = self.model_size
        return f"linennuev1_{f}f{p}p{v}v{d}d"
