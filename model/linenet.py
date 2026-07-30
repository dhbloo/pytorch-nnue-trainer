import torch
import torch.nn as nn
import torch.nn.functional as F

from . import MODELS
from .layers.convolution import Conv2dBlock
from .mixnet_components import QuantizedHeadMixin
from .ops import quantized_sum_3x3_regions
from .validation import validate_batch_shared_value
from dataset.pipeline.line_encoding import get_total_num_encoding
from utils.quant_utils import fake_quant



@MODELS.register("linennuev1")
class LineNNUEv1(QuantizedHeadMixin, nn.Module):
    LINE_LENGTH = 11

    def __init__(self, dim_feature=64, dim_policy=32, dim_value=64, dim_dwconv=32):
        super().__init__()
        self.model_size = (dim_feature, dim_policy, dim_value, dim_dwconv)
        self.line_encoding_total_num = get_total_num_encoding(self.LINE_LENGTH)

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
            weight_scale=0.25,
        )

        self._init_quantized_policy_head(dim_feature, dim_policy)
        self._init_quantized_grouped_value_head(dim_feature, dim_value)

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
        validate_batch_shared_value(
            "line_encoding_total_num",
            data["line_encoding_total_num"],
            self.line_encoding_total_num,
        )
        _, dim_policy, _, _ = self.model_size

        # get feature from single side
        feature = self.get_feature(data)  # [B, dim_feature, H, W]

        # value feature accumulator
        feature_sum = torch.sum(feature, dim=(2, 3))  # [B, dim_feature]
        feature_sum = fake_quant(feature_sum / 256, scale=128, num_bits=32, floor=True)  # srai 8

        # value feature accumulator of nine groups
        regions = quantized_sum_3x3_regions(feature)

        # policy head
        policy = self._forward_quantized_policy(feature, feature_sum, dim_policy)

        # value head
        value = self._forward_quantized_grouped_value(feature_sum, regions)

        return {"value": value, "policy": policy}

    @property
    def weight_clipping(self):
        # Clip prelu weight of mapping activation to [-1,1] to avoid overflow
        # In this range, prelu is the same as `max(x, ax)`.
        return [
            {"params": ["feature_dwconv.conv.weight"], "min_weight": -32768 / 65536, "max_weight": 32767 / 65536},
            *self._quantized_head_weight_clipping(combine_policy_with_value=True),
        ]

    @property
    def name(self):
        f, p, v, d = self.model_size
        return f"linennuev1_{f}f{p}p{v}v{d}d"
