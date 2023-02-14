import torch
import numpy as np
import io
from zlib import crc32
from math import ceil
from utils.misc_utils import ascii_hist
from dataset.pipeline.line_encoding import get_encoding_usage_flags
from . import BaseSerializer, SERIALIZERS
from ..flatnet import FlatLadder7x7NNUEv1


@SERIALIZERS.register('flat_ladder7x7_nnue_v1')
class FlatLadder7x7NNUEv1Serializer(BaseSerializer):
    """
    FlatLadder7x7NNUEv1 binary serializer.

    The corresponding C-language struct layout: 
    struct FlatLadder7x7NNUEv1Weight {
        // mapping features
        int8_t  mapping[59049][ValueDim];

        // nnue layers weights
        int8_t  l1_weight[ValueDim][ValueDim];
        int32_t l1_bias[ValueDim];
        int8_t  l2_weight[ValueDim][ValueDim];
        int32_t l2_bias[ValueDim];
        int8_t  l3_weight[ValueDim][ValueDim];
        int32_t l3_bias[ValueDim];
        int8_t  l4_weight[4][ValueDim];
        int32_t l4_bias[4];
    };
    """
    def __init__(self, s_weight=128, s_output=128, **kwargs):
        super().__init__(**kwargs)
        self.s_weight = s_weight
        self.s_output = s_output

    @property
    def is_binary(self) -> bool:
        return True

    @property
    def needs_header(self) -> bool:
        return False

    def arch_hash(self, model) -> int:
        raise NotImplementedError()

    def _index_to_chunk(self, index: int):
        l = []
        while True:
            index, reminder = divmod(index, 3)
            l.append(reminder)
            if index == 0: break
        l = l[::-1]
        for _ in range(10 - len(l)):
            l.insert(0, 0)
        return l

    def _get_mapping_input(self, chunk: list[int]):
        inputs = torch.zeros(2, 4, 4)
        lower_triangle_index = [0, 4, 5, 8, 9, 10, 12, 13, 14, 15]
        for i in range(10):
            y = lower_triangle_index[i] // 4
            x = lower_triangle_index[i] % 4
            if chunk[i] > 0: inputs[chunk[i] - 1][y][x] = 1
        return inputs

    def _export_feature_map(self, model: FlatLadder7x7NNUEv1, device: torch.device):
        _, dim_policy, dim_value = model.model_size
        chunk_input = torch.zeros(3**10, 2, 4, 4)
        for i in range(3**10):
            chunk = self._index_to_chunk(i)
            chunk_input[i] = self._get_mapping_input(chunk)
        feature_map = model.mapping(chunk_input.to(device))
        feature_map = torch.clamp(feature_map, min=-1, max=127 / 128)
        feature_map = feature_map.view((3**10, dim_policy + dim_value)).cpu().numpy()
        ascii_hist("feature_map", feature_map)
        feature_map = np.clip(feature_map * 128, -128, 127).astype(np.int8)
        return feature_map

    def _export_value(self, model: FlatLadder7x7NNUEv1):
        linear1_weight = model.value_linears[0].fc.weight.cpu().numpy()
        linear1_bias = model.value_linears[0].fc.bias.cpu().numpy()
        linear2_weight = model.value_linears[1].fc.weight.cpu().numpy()
        linear2_bias = model.value_linears[1].fc.bias.cpu().numpy()
        linear3_weight = model.value_linears[2].fc.weight.cpu().numpy()
        linear3_bias = model.value_linears[2].fc.bias.cpu().numpy()
        linear4_weight = model.value_linears[3].fc.weight.cpu().numpy()
        linear4_bias = model.value_linears[3].fc.bias.cpu().numpy()

        ascii_hist("value: linear1_weight", linear1_weight)
        ascii_hist("value: linear1_bias", linear1_bias)
        ascii_hist("value: linear2_weight", linear2_weight)
        ascii_hist("value: linear2_bias", linear2_bias)
        ascii_hist("value: linear3_weight", linear3_weight)
        ascii_hist("value: linear3_bias", linear3_bias)
        ascii_hist("value: linear4_weight", linear4_weight)
        ascii_hist("value: linear4_bias", linear4_bias)

        assert int(linear1_weight.min() * self.s_weight) >= -128
        assert int(linear1_weight.max() * self.s_weight) <= 127
        assert int(linear2_weight.min() * self.s_weight) >= -128
        assert int(linear2_weight.max() * self.s_weight) <= 127
        assert int(linear3_weight.min() * self.s_weight) >= -128
        assert int(linear3_weight.max() * self.s_weight) <= 127
        assert int(linear4_weight.min() * self.s_output) >= -128
        assert int(linear4_weight.max() * self.s_output) <= 127

        if model.value_no_draw:
            linear4_weight = np.concatenate([linear4_weight, np.zeros((3, 128))], axis=0)
            linear4_bias = np.concatenate([linear4_bias, np.zeros((3, ))], axis=0)
        else:
            linear4_weight = np.concatenate([linear4_weight, np.zeros((1, 128))], axis=0)
            linear4_bias = np.concatenate([linear4_bias, np.zeros((1, ))], axis=0)

        return (
            np.clip(linear1_weight * self.s_weight, -128, 127).astype(np.int8),
            np.clip(linear1_bias * self.s_weight * 128, -2**31, 2**31 - 1).astype(np.int32),
            np.clip(linear2_weight * self.s_weight, -128, 127).astype(np.int8),
            np.clip(linear2_bias * self.s_weight * 128, -2**31, 2**31 - 1).astype(np.int32),
            np.clip(linear3_weight * self.s_weight, -128, 127).astype(np.int8),
            np.clip(linear3_bias * self.s_weight * 128, -2**31, 2**31 - 1).astype(np.int32),
            np.clip(linear4_weight * self.s_output, -128, 127).astype(np.int8),
            np.clip(linear4_bias * self.s_output * 128, -2**31, 2**31 - 1).astype(np.int32),
        )

    def serialize(self, out: io.IOBase, model: FlatLadder7x7NNUEv1, device: torch.device):
        feature_map = self._export_feature_map(model, device)
        linear1_weight, linear1_bias, \
        linear2_weight, linear2_bias, \
        linear3_weight, linear3_bias, \
        linear4_weight, linear4_bias = self._export_value(model)

        o: io.RawIOBase = out

        # int8_t  mapping[59049][ValueDim];
        o.write(feature_map.astype('<i1').tobytes())  # (59049, 128)

        # float l1_weight[ValueDim][ValueDim];
        # float l1_bias[ValueDim];
        o.write(linear1_weight.astype('<i1').tobytes())  # (128, 128)
        o.write(linear1_bias.astype('<i4').tobytes())  # (128,)

        # float l2_weight[ValueDim][ValueDim];
        # float l2_bias[ValueDim];
        o.write(linear2_weight.astype('<i1').tobytes())  # (128, 128)
        o.write(linear2_bias.astype('<i4').tobytes())  # (128,)

        # float l3_weight[ValueDim][ValueDim];
        # float l3_bias[ValueDim];
        o.write(linear3_weight.astype('<i1').tobytes())  # (128, 128)
        o.write(linear3_bias.astype('<i4').tobytes())  # (128,)

        # float l4_weight[4][ValueDim];
        # float l4_bias[4];
        o.write(linear4_weight.astype('<i1').tobytes())  # (4, 128)
        o.write(linear4_bias.astype('<i4').tobytes())  # (4,)
