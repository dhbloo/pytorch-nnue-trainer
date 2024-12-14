import torch
import numpy as np
import io
from tqdm import tqdm
from utils.misc_utils import ascii_hist
from . import BaseSerializer, SERIALIZERS
from ..flatnet import *


@SERIALIZERS.register('flat_ladder7x7_nnue_v1')
class FlatLadder7x7NNUEv1Serializer(BaseSerializer):
    """
    FlatLadder7x7NNUEv1 binary serializer.

    The corresponding C-language struct layout: 
    struct FlatLadder7x7NNUEv1Weight {
        // mapping features
        int8_t  mapping[59049][ValueDim];

        // nnue layers weights
        int8_t  l1_weight[32][ValueDim];
        int32_t l1_bias[32];
        int8_t  l2_weight[32][32];
        int32_t l2_bias[32];
        int8_t  l3_weight[32][32];
        int32_t l3_bias[32];
        int8_t  l4_weight[4][32];
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

    @torch.no_grad()
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
        feature_map = np.clip(np.around(feature_map * 128), -128, 127).astype(np.int8)
        return feature_map

    def _export_value(self, model: FlatLadder7x7NNUEv1):
        l1_weight = model.value_linears[0].fc.weight.cpu().numpy()
        l1_bias = model.value_linears[0].fc.bias.cpu().numpy()
        l2_weight = model.value_linears[1].fc.weight.cpu().numpy()
        l2_bias = model.value_linears[1].fc.bias.cpu().numpy()
        l3_weight = model.value_linears[2].fc.weight.cpu().numpy()
        l3_bias = model.value_linears[2].fc.bias.cpu().numpy()
        l4_weight = model.value_linears[3].fc.weight.cpu().numpy()
        l4_bias = model.value_linears[3].fc.bias.cpu().numpy()

        ascii_hist("value: linear1_weight", l1_weight)
        ascii_hist("value: linear1_bias", l1_bias)
        ascii_hist("value: linear2_weight", l2_weight)
        ascii_hist("value: linear2_bias", l2_bias)
        ascii_hist("value: linear3_weight", l3_weight)
        ascii_hist("value: linear3_bias", l3_bias)
        ascii_hist("value: linear4_weight", l4_weight)
        ascii_hist("value: linear4_bias", l4_bias)

        assert int(l1_weight.min() * self.s_weight) >= -128
        assert int(l1_weight.max() * self.s_weight) <= 127
        assert int(l2_weight.min() * self.s_weight) >= -128
        assert int(l2_weight.max() * self.s_weight) <= 127
        assert int(l3_weight.min() * self.s_weight) >= -128
        assert int(l3_weight.max() * self.s_weight) <= 127
        assert int(l4_weight.min() * self.s_output) >= -128
        assert int(l4_weight.max() * self.s_output) <= 127

        if model.value_no_draw:
            l4_weight = np.concatenate([l4_weight, np.zeros((3, 32))], axis=0)
            l4_bias = np.concatenate([l4_bias, np.zeros((3, ))], axis=0)
        else:
            l4_weight = np.concatenate([l4_weight, np.zeros((1, 32))], axis=0)
            l4_bias = np.concatenate([l4_bias, np.zeros((1, ))], axis=0)

        return (
            np.clip(np.around(l1_weight * self.s_weight), -128, 127).astype(np.int8),
            np.clip(np.around(l1_bias * self.s_weight * 128), -2**31, 2**31 - 1).astype(np.int32),
            np.clip(np.around(l2_weight * self.s_weight), -128, 127).astype(np.int8),
            np.clip(np.around(l2_bias * self.s_weight * 128), -2**31, 2**31 - 1).astype(np.int32),
            np.clip(np.around(l3_weight * self.s_weight), -128, 127).astype(np.int8),
            np.clip(np.around(l3_bias * self.s_weight * 128), -2**31, 2**31 - 1).astype(np.int32),
            np.clip(np.around(l4_weight * self.s_output), -128, 127).astype(np.int8),
            np.clip(np.around(l4_bias * self.s_output * 128), -2**31, 2**31 - 1).astype(np.int32),
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

        # float l1_weight[32][ValueDim];
        # float l1_bias[32];
        o.write(linear1_weight.astype('<i1').tobytes())  # (32, 128)
        o.write(linear1_bias.astype('<i4').tobytes())  # (32,)

        # float l2_weight[32][32];
        # float l2_bias[32];
        o.write(linear2_weight.astype('<i1').tobytes())  # (32, 32)
        o.write(linear2_bias.astype('<i4').tobytes())  # (32,)

        # float l3_weight[32][32];
        # float l3_bias[32];
        o.write(linear3_weight.astype('<i1').tobytes())  # (32, 32)
        o.write(linear3_bias.astype('<i4').tobytes())  # (32,)

        # float l4_weight[4][32];
        # float l4_bias[4];
        o.write(linear4_weight.astype('<i1').tobytes())  # (4, 32)
        o.write(linear4_bias.astype('<i4').tobytes())  # (4,)


@SERIALIZERS.register('flat_ladder7x7_nnue_v2')
class FlatLadder7x7NNUEv2Serializer(BaseSerializer):
    """
    FlatLadder7x7NNUEv1 binary serializer.

    The corresponding C-language struct layout: 
    struct FlatLadder7x7NNUEv1Weight {
        // mapping features
        int8_t  mapping[2][59049][ValueDim];

        // nnue layers weights
        int8_t  l1_weight[32][ValueDim];
        int32_t l1_bias[32];
        int8_t  l2_weight[32][32];
        int32_t l2_bias[32];
        int8_t  l3_weight[32][32];
        int32_t l3_bias[32];
        int8_t  l4_weight[4][32];
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

    @torch.no_grad()
    def _export_feature_map(self, model: FlatLadder7x7NNUEv2, device: torch.device):
        _, dim_policy, dim_value = model.model_size
        chunk_input = torch.zeros(3**10, 2, 4, 4)
        for i in range(3**10):
            chunk = self._index_to_chunk(i)
            chunk_input[i] = self._get_mapping_input(chunk)
        mapping1 = model.mapping1(chunk_input.to(device))
        mapping2 = model.mapping2(chunk_input.to(device))
        mapping1 = torch.clamp(mapping1, min=-1, max=127 / 128)
        mapping2 = torch.clamp(mapping2, min=-1, max=127 / 128)
        mapping1 = mapping1.view((3**10, dim_policy + dim_value)).cpu().numpy()
        mapping2 = mapping2.view((3**10, dim_policy + dim_value)).cpu().numpy()
        ascii_hist("mapping1", mapping1)
        ascii_hist("mapping2", mapping2)
        mapping1 = np.clip(np.around(mapping1 * 128), -128, 127).astype(np.int8)
        mapping2 = np.clip(np.around(mapping2 * 128), -128, 127).astype(np.int8)
        return mapping1, mapping2

    def _export_value(self, model: FlatLadder7x7NNUEv2):
        l1_weight = model.value_linears[0].fc.weight.cpu().numpy()
        l1_bias = model.value_linears[0].fc.bias.cpu().numpy()
        l2_weight = model.value_linears[1].fc.weight.cpu().numpy()
        l2_bias = model.value_linears[1].fc.bias.cpu().numpy()
        l3_weight = model.value_linears[2].fc.weight.cpu().numpy()
        l3_bias = model.value_linears[2].fc.bias.cpu().numpy()
        l4_weight = model.value_linears[3].fc.weight.cpu().numpy()
        l4_bias = model.value_linears[3].fc.bias.cpu().numpy()

        ascii_hist("value: linear1_weight", l1_weight)
        ascii_hist("value: linear1_bias", l1_bias)
        ascii_hist("value: linear2_weight", l2_weight)
        ascii_hist("value: linear2_bias", l2_bias)
        ascii_hist("value: linear3_weight", l3_weight)
        ascii_hist("value: linear3_bias", l3_bias)
        ascii_hist("value: linear4_weight", l4_weight)
        ascii_hist("value: linear4_bias", l4_bias)

        assert int(l1_weight.min() * self.s_weight) >= -128
        assert int(l1_weight.max() * self.s_weight) <= 127
        assert int(l2_weight.min() * self.s_weight) >= -128
        assert int(l2_weight.max() * self.s_weight) <= 127
        assert int(l3_weight.min() * self.s_weight) >= -128
        assert int(l3_weight.max() * self.s_weight) <= 127
        assert int(l4_weight.min() * self.s_output) >= -128
        assert int(l4_weight.max() * self.s_output) <= 127

        if model.value_no_draw:
            l4_weight = np.concatenate([l4_weight, np.zeros((3, 32))], axis=0)
            l4_bias = np.concatenate([l4_bias, np.zeros((3, ))], axis=0)
        else:
            l4_weight = np.concatenate([l4_weight, np.zeros((1, 32))], axis=0)
            l4_bias = np.concatenate([l4_bias, np.zeros((1, ))], axis=0)

        return (
            np.clip(np.around(l1_weight * self.s_weight), -128, 127).astype(np.int8),
            np.clip(np.around(l1_bias * self.s_weight * 128), -2**31, 2**31 - 1).astype(np.int32),
            np.clip(np.around(l2_weight * self.s_weight), -128, 127).astype(np.int8),
            np.clip(np.around(l2_bias * self.s_weight * 128), -2**31, 2**31 - 1).astype(np.int32),
            np.clip(np.around(l3_weight * self.s_weight), -128, 127).astype(np.int8),
            np.clip(np.around(l3_bias * self.s_weight * 128), -2**31, 2**31 - 1).astype(np.int32),
            np.clip(np.around(l4_weight * self.s_output), -128, 127).astype(np.int8),
            np.clip(np.around(l4_bias * self.s_output * 128), -2**31, 2**31 - 1).astype(np.int32),
        )

    def serialize(self, out: io.IOBase, model: FlatLadder7x7NNUEv2, device: torch.device):
        mapping1, mapping2 = self._export_feature_map(model, device)
        linear1_weight, linear1_bias, \
        linear2_weight, linear2_bias, \
        linear3_weight, linear3_bias, \
        linear4_weight, linear4_bias = self._export_value(model)

        o: io.RawIOBase = out

        # int8_t mapping[2][59049][FeatureDim];
        o.write(mapping1.astype('<i1').tobytes())  # (59049, 128)
        o.write(mapping2.astype('<i1').tobytes())  # (59049, 128)

        # float l1_weight[32][ValueDim];
        # float l1_bias[32];
        o.write(linear1_weight.astype('<i1').tobytes())  # (32, 128)
        o.write(linear1_bias.astype('<i4').tobytes())  # (32,)

        # float l2_weight[32][32];
        # float l2_bias[32];
        o.write(linear2_weight.astype('<i1').tobytes())  # (32, 32)
        o.write(linear2_bias.astype('<i4').tobytes())  # (32,)

        # float l3_weight[32][32];
        # float l3_bias[32];
        o.write(linear3_weight.astype('<i1').tobytes())  # (32, 32)
        o.write(linear3_bias.astype('<i4').tobytes())  # (32,)

        # float l4_weight[4][32];
        # float l4_bias[4];
        o.write(linear4_weight.astype('<i1').tobytes())  # (4, 32)
        o.write(linear4_bias.astype('<i4').tobytes())  # (4,)


@SERIALIZERS.register('flat_square7x7_nnue_v2')
class FlatSquare7x7NNUEv2Serializer(BaseSerializer):
    """
    FlatSquare7x7NNUEv2 binary serializer.

    The corresponding C-language struct layout: 
    struct FlatSquare7x7NNUEv2Weight {
        // mapping features
        int8_t  mapping4x4[43046721][FeatureDim or FeatureDim//2];
        int8_t  mapping4x3[531441][FeatureDim or FeatureDim//2];

        // nnue layers weights
        int8_t  l1_weight[32][FeatureDim * 4];
        int32_t l1_bias[32];
        int8_t  l2_weight[32][32];
        int32_t l2_bias[32];
        int8_t  l3_weight[4][32];
        int32_t l3_bias[4];
    };
    """
    def __init__(self, s_input=256, s_weight=128, **kwargs):
        super().__init__(**kwargs)
        self.s_input = s_input
        self.s_weight = s_weight
        
    @property
    def is_binary(self) -> bool:
        return True

    @property
    def needs_header(self) -> bool:
        return False

    def arch_hash(self, model) -> int:
        raise NotImplementedError()
    
    def _get_mapping4x4_input(self, start_idx: int, end_idx: int) -> torch.Tensor:
        indices = torch.arange(start_idx, end_idx, dtype=torch.int32)
        inputs = torch.zeros(indices.shape[0], 2, 4, 4)
        for i in range(4 * 4):
            bit = (indices // 3**i) % 3
            inputs[:, 0, i // 4, i % 4] = bit == 1
            inputs[:, 1, i // 4, i % 4] = bit == 2
        return inputs
    
    def _get_mapping4x3_input(self, start_idx: int, end_idx: int) -> torch.Tensor:
        indices = torch.arange(start_idx, end_idx, dtype=torch.int32)
        inputs = torch.zeros(indices.shape[0], 2, 4, 3)
        for i in range(4 * 3):
            bit = (indices // 3**i) % 3
            inputs[:, 0, i // 3, i % 3] = bit == 1
            inputs[:, 1, i // 3, i % 3] = bit == 2
        return inputs
    
    @torch.no_grad()
    def _export_feature_map(self, model: FlatSquare7x7NNUEv2, device: torch.device):
        _, dim_feature = model.model_size
        batch_size = 4096
        
        mapping4x4 = []
        for i_start in tqdm(range(0, 3**16, batch_size)):
            i_end = min(i_start + batch_size, 3**16)
            chunk4x4_input = self._get_mapping4x4_input(i_start, i_end).to(device)
            mapping4x4.append(model.mapping4x4(chunk4x4_input))
        mapping4x4 = torch.cat(mapping4x4, dim=0).squeeze(-1).squeeze(-1)
        
        mapping4x3 = []
        for i_start in tqdm(range(0, 3**12, batch_size)):
            i_end = min(i_start + batch_size, 3**12)
            chunk4x3_input = self._get_mapping4x3_input(i_start, i_end).to(device)
            mapping4x3.append(model.mapping4x3(chunk4x3_input))
        mapping4x3 = torch.cat(mapping4x3, dim=0).squeeze(-1).squeeze(-1)
        
        if model.quant_int4:  # int4 quant
            mapping4x4 = torch.clamp(mapping4x4, min=-1, max=7 / 8)
        else:  # int8 quant
            mapping4x4 = torch.clamp(mapping4x4, min=-1, max=127 / 128)
        mapping4x3 = torch.clamp(mapping4x3, min=-1, max=127 / 128)
        
        mapping4x4 = mapping4x4.view((3**16, dim_feature)).cpu().numpy()
        mapping4x3 = mapping4x3.view((3**12, dim_feature)).cpu().numpy()
        ascii_hist("mapping4x4", mapping4x4)
        ascii_hist("mapping4x3", mapping4x3)
        
        if model.quant_int4:  # int4 quant
            mapping4x4 = np.clip(np.around(mapping4x4 * 8), -8, 7).astype(np.int8)
        else:  # int8 quant
            mapping4x4 = np.clip(np.around(mapping4x4 * 128), -128, 127).astype(np.int8)
        mapping4x3 = np.clip(np.around(mapping4x3 * 128), -128, 127).astype(np.int8)
        return mapping4x4, mapping4x3
    
    def _export_value(self, model: FlatSquare7x7NNUEv2):
        l1_weight = model.value_linears[0].fc.weight.cpu().numpy()
        l1_bias = model.value_linears[0].fc.bias.cpu().numpy()
        l2_weight = model.value_linears[1].fc.weight.cpu().numpy()
        l2_bias = model.value_linears[1].fc.bias.cpu().numpy()
        l3_weight = model.value_linears[2].fc.weight.cpu().numpy()
        l3_bias = model.value_linears[2].fc.bias.cpu().numpy()

        ascii_hist("value: linear1_weight", l1_weight)
        ascii_hist("value: linear1_bias", l1_bias)
        ascii_hist("value: linear2_weight", l2_weight)
        ascii_hist("value: linear2_bias", l2_bias)
        ascii_hist("value: linear3_weight", l3_weight)
        ascii_hist("value: linear3_bias", l3_bias)

        if model.value_no_draw:
            l3_weight = np.concatenate([l3_weight, np.zeros((3, 32))], axis=0)
            l3_bias = np.concatenate([l3_bias, np.zeros((3, ))], axis=0)
        else:
            l3_weight = np.concatenate([l3_weight, np.zeros((1, 32))], axis=0)
            l3_bias = np.concatenate([l3_bias, np.zeros((1, ))], axis=0)

        return (
            np.clip(np.around(l1_weight * self.s_input), -128, 127).astype(np.int8),
            np.clip(np.around(l1_bias * self.s_input * 128), -2**31, 2**31 - 1).astype(np.int32),
            np.clip(np.around(l2_weight * self.s_weight), -128, 127).astype(np.int8),
            np.clip(np.around(l2_bias * self.s_weight * 128), -2**31, 2**31 - 1).astype(np.int32),
            np.clip(np.around(l3_weight * self.s_weight), -128, 127).astype(np.int8),
            np.clip(np.around(l3_bias * self.s_weight * 128), -2**31, 2**31 - 1).astype(np.int32),
        )
    
    def serialize(self, out: io.IOBase, model: FlatSquare7x7NNUEv2, device: torch.device):
        mapping4x4, mapping4x3 = self._export_feature_map(model, device)
        linear1_weight, linear1_bias, \
        linear2_weight, linear2_bias, \
        linear3_weight, linear3_bias = self._export_value(model)

        o: io.RawIOBase = out

        if model.quant_int4:
            # int8_t mapping4x4[43046721][FeatureDim // 2];
            mapping4x4 = mapping4x4.astype('<i1')
            mapping4x4_u8_lo = mapping4x4[:, 0::2] + 8
            mapping4x4_u8_hi = mapping4x4[:, 1::2] + 8
            assert mapping4x4_u8_lo.min() >= 0 and mapping4x4_u8_lo.max() <= 15
            assert mapping4x4_u8_hi.min() >= 0 and mapping4x4_u8_hi.max() <= 15
            mapping4x4_u8 = np.bitwise_or(np.left_shift(mapping4x4_u8_hi, 4), mapping4x4_u8_lo)
            o.write(mapping4x4_u8.tobytes())  # (43046721, F//2)
        else:
            # int8_t mapping4x4[43046721][FeatureDim];
            o.write(mapping4x4.astype('<i1').tobytes())  # (43046721, F)
            
        # int8_t mapping4x3[531441][FeatureDim];
        o.write(mapping4x3.astype('<i1').tobytes())  # (531441, F)

        # float l1_weight[32][FeatureDim * 4];
        # float l1_bias[32];
        o.write(linear1_weight.astype('<i1').tobytes())  # (32, F)
        o.write(linear1_bias.astype('<i4').tobytes())  # (32,)

        # float l2_weight[32][32];
        # float l2_bias[32];
        o.write(linear2_weight.astype('<i1').tobytes())  # (32, 32)
        o.write(linear2_bias.astype('<i4').tobytes())  # (32,)

        # float l3_weight[4][32];
        # float l3_bias[4];
        o.write(linear3_weight.astype('<i1').tobytes())  # (4, 32)
        o.write(linear3_bias.astype('<i4').tobytes())  # (4,)


@SERIALIZERS.register('flat_square7x7_nnue_v3')
class FlatSquare7x7NNUEv3Serializer(BaseSerializer):
    """
    FlatSquare7x7NNUEv3 binary serializer.

    The corresponding C-language struct layout: 
    struct FlatSquare7x7NNUEv3Weight {
        // mapping features
        int8_t  mapping3x4[3][531441][FeatureDim];

        // nnue layers weights
        int8_t  l1_weight[64][FeatureDim * 4];
        int32_t l1_bias[64];
        int8_t  l2_weight[32][64];
        int32_t l2_bias[32];
        int8_t  l3_weight[4][32];
        int32_t l3_bias[4];
    };
    """
    def __init__(self, s_input=256, s_weight=128, **kwargs):
        super().__init__(**kwargs)
        self.s_input = s_input
        self.s_weight = s_weight
        
    @property
    def is_binary(self) -> bool:
        return True

    @property
    def needs_header(self) -> bool:
        return False

    def arch_hash(self, model) -> int:
        raise NotImplementedError()
    
    def _get_mapping3x4_input(self, start_idx: int, end_idx: int) -> torch.Tensor:
        indices = torch.arange(start_idx, end_idx, dtype=torch.int32)
        inputs = torch.zeros(indices.shape[0], 2, 3, 4)
        for i in range(3 * 4):
            bit = (indices // 3**i) % 3
            inputs[:, 0, i // 4, i % 4] = bit == 1
            inputs[:, 1, i // 4, i % 4] = bit == 2
        return inputs
    
    @torch.no_grad()
    def _export_feature_map(self, model: FlatSquare7x7NNUEv3, device: torch.device):
        _, dim_feature = model.model_size
        batch_size = 4096
        
        mapping3x4_corner1 = []
        mapping3x4_corner2 = []
        mapping3x4_middle = []
        for i_start in tqdm(range(0, 3**12, batch_size)):
            i_end = min(i_start + batch_size, 3**12)
            chunk3x4_input = self._get_mapping3x4_input(i_start, i_end).to(device)
            mapping3x4_corner1.append(model.mapping3x4_corner1(chunk3x4_input))
            mapping3x4_corner2.append(model.mapping3x4_corner2(chunk3x4_input))
            mapping3x4_middle.append(model.mapping3x4_middle(chunk3x4_input))

        def export_mapping3x4(name, mapping3x4):
            mapping3x4 = torch.cat(mapping3x4, dim=0).squeeze(-1).squeeze(-1)
            
            if model.quant_int4:  # int4 quant
                mapping3x4 = torch.clamp(mapping3x4, min=-1, max=7 / 8)
            else:  # int8 quant
                mapping3x4 = torch.clamp(mapping3x4, min=-1, max=127 / 128)
        
            mapping3x4 = mapping3x4.view((3**12, dim_feature)).cpu().numpy()
            ascii_hist(f"mapping3x4[{name}]", mapping3x4)
        
            if model.quant_int4:  # int4 quant
                mapping3x4 = np.clip(np.around(mapping3x4 * 8), -8, 7).astype(np.int8)
            else:  # int8 quant
                mapping3x4 = np.clip(np.around(mapping3x4 * 128), -128, 127).astype(np.int8)
                
            return mapping3x4
        
        mapping3x4_corner1 = export_mapping3x4("corner1", mapping3x4_corner1)
        mapping3x4_corner2 = export_mapping3x4("corner2", mapping3x4_corner2)
        mapping3x4_middle = export_mapping3x4("middle", mapping3x4_middle)
        return np.stack([mapping3x4_corner1, mapping3x4_corner2, mapping3x4_middle])
    
    def _export_value(self, model: FlatSquare7x7NNUEv3):
        l1_weight = model.value_linears[0].fc.weight.cpu().numpy()
        l1_bias = model.value_linears[0].fc.bias.cpu().numpy()
        l2_weight = model.value_linears[1].fc.weight.cpu().numpy()
        l2_bias = model.value_linears[1].fc.bias.cpu().numpy()
        l3_weight = model.value_linears[2].fc.weight.cpu().numpy()
        l3_bias = model.value_linears[2].fc.bias.cpu().numpy()

        ascii_hist("value: linear1_weight", l1_weight)
        ascii_hist("value: linear1_bias", l1_bias)
        ascii_hist("value: linear2_weight", l2_weight)
        ascii_hist("value: linear2_bias", l2_bias)
        ascii_hist("value: linear3_weight", l3_weight)
        ascii_hist("value: linear3_bias", l3_bias)

        if model.value_no_draw:
            l3_weight = np.concatenate([l3_weight, np.zeros((3, 32))], axis=0)
            l3_bias = np.concatenate([l3_bias, np.zeros((3, ))], axis=0)
        else:
            l3_weight = np.concatenate([l3_weight, np.zeros((1, 32))], axis=0)
            l3_bias = np.concatenate([l3_bias, np.zeros((1, ))], axis=0)

        return (
            np.clip(np.around(l1_weight * self.s_input), -128, 127).astype(np.int8),
            np.clip(np.around(l1_bias * self.s_input * 128), -2**31, 2**31 - 1).astype(np.int32),
            np.clip(np.around(l2_weight * self.s_weight), -128, 127).astype(np.int8),
            np.clip(np.around(l2_bias * self.s_weight * 128), -2**31, 2**31 - 1).astype(np.int32),
            np.clip(np.around(l3_weight * self.s_weight), -128, 127).astype(np.int8),
            np.clip(np.around(l3_bias * self.s_weight * 128), -2**31, 2**31 - 1).astype(np.int32),
        )
    
    def serialize(self, out: io.IOBase, model: FlatSquare7x7NNUEv3, device: torch.device):
        mapping3x4 = self._export_feature_map(model, device)
        linear1_weight, linear1_bias, \
        linear2_weight, linear2_bias, \
        linear3_weight, linear3_bias = self._export_value(model)

        o: io.RawIOBase = out

        if model.quant_int4:
            # int8_t mapping3x4[3][531441][FeatureDim // 2];
            mapping3x4 = mapping3x4.astype('<i1')
            mapping3x4_u8_lo = mapping3x4[..., 0::2] + 8
            mapping3x4_u8_hi = mapping3x4[..., 1::2] + 8
            assert mapping3x4_u8_lo.min() >= 0 and mapping3x4_u8_lo.max() <= 15
            assert mapping3x4_u8_hi.min() >= 0 and mapping3x4_u8_hi.max() <= 15
            mapping3x4_u8 = np.bitwise_or(np.left_shift(mapping3x4_u8_hi, 4), mapping3x4_u8_lo)
            o.write(mapping3x4_u8.tobytes())  # (3, 531441, F//2)
        else:
            # int8_t mapping3x4[3][531441][FeatureDim];
            o.write((mapping3x4[0] + 128).astype('<u1').tobytes())  # (531441, F)
            o.write((mapping3x4[1] + 128).astype('<u1').tobytes())  # (531441, F)
            o.write(mapping3x4[2].astype('<i1').tobytes())  # (531441, F)

        # float l1_weight[64][FeatureDim * 4];
        # float l1_bias[64];
        o.write(linear1_weight.astype('<i1').tobytes())  # (64, F)
        o.write(linear1_bias.astype('<i4').tobytes())  # (64,)

        # float l2_weight[32][64];
        # float l2_bias[32];
        o.write(linear2_weight.astype('<i1').tobytes())  # (32, 64)
        o.write(linear2_bias.astype('<i4').tobytes())  # (32,)

        # float l3_weight[4][32];
        # float l3_bias[4];
        o.write(linear3_weight.astype('<i1').tobytes())  # (4, 32)
        o.write(linear3_bias.astype('<i4').tobytes())  # (4,)


@SERIALIZERS.register('flat_square7x7_nnue_v4')
class FlatSquare7x7NNUEv4Serializer(BaseSerializer):
    """
    FlatSquare7x7NNUEv4 binary serializer.

    The corresponding C-language struct layout: 
    struct FlatSquare7x7NNUEv4Weight {
        // mapping features
        int8_t  mapping4x4[3][43046721][FeatureDim];

        // nnue layers weights
        int8_t  l1_weight[32][FeatureDim];
        int32_t l1_bias[32];
        int8_t  l2_weight[32][32];
        int32_t l2_bias[32];
        int8_t  l3_weight[4][32];
        int32_t l3_bias[4];
    };
    """
    def __init__(self, s_input=256, s_weight=128, batch_size=16384, **kwargs):
        super().__init__(**kwargs)
        self.s_input = s_input
        self.s_weight = s_weight
        self.batch_size = batch_size
        
    @property
    def is_binary(self) -> bool:
        return True

    @property
    def needs_header(self) -> bool:
        return False

    def arch_hash(self, model) -> int:
        raise NotImplementedError()
    
    def _get_mapping4x4_input(self, start_idx: int, end_idx: int) -> torch.Tensor:
        indices = torch.arange(start_idx, end_idx, dtype=torch.int32)
        inputs = torch.zeros(indices.shape[0], 2, 4, 4)
        for i in range(4 * 4):
            bit = (indices // 3**i) % 3
            inputs[:, 0, i // 4, i % 4] = bit == 1
            inputs[:, 1, i // 4, i % 4] = bit == 2
        return inputs
    
    @torch.no_grad()
    def _export_feature_map(self, model: FlatSquare7x7NNUEv4, device: torch.device):
        num_mappings = len(model.mapping4x4)
        dim_feature = model.model_size[1]
        mapping4x4 = torch.zeros((num_mappings, 3**16, dim_feature), device=device)
        for i_start in tqdm(range(0, 3**16, self.batch_size)):
            i_end = min(i_start + self.batch_size, 3**16)
            chunk4x4_input = self._get_mapping4x4_input(i_start, i_end).to(device)
            for j in range(num_mappings):
                feature = model.mapping4x4[j](chunk4x4_input)
                feature = feature.squeeze(-1).squeeze(-1)
                if model.unit_feature_vector:
                    feature = F.normalize(feature, p=2, dim=-1)
                feature = torch.clamp(feature, min=-1, max=127 / 128)
                mapping4x4[j, i_start:i_end] = feature

        mapping4x4 = mapping4x4.cpu().numpy()
        ascii_hist(f"mapping4x4", mapping4x4)
        mapping4x4 = np.clip(np.around(mapping4x4 * 128), -128, 127).astype(np.int8)
        
        return mapping4x4
    
    def _export_value(self, model: FlatSquare7x7NNUEv4):
        l1_weight = model.value_linears[0].fc.weight.cpu().numpy()
        l1_bias = model.value_linears[0].fc.bias.cpu().numpy()
        l2_weight = model.value_linears[1].fc.weight.cpu().numpy()
        l2_bias = model.value_linears[1].fc.bias.cpu().numpy()
        l3_weight = model.value_linears[2].fc.weight.cpu().numpy()
        l3_bias = model.value_linears[2].fc.bias.cpu().numpy()

        ascii_hist("value: linear1_weight", l1_weight)
        ascii_hist("value: linear1_bias", l1_bias)
        ascii_hist("value: linear2_weight", l2_weight)
        ascii_hist("value: linear2_bias", l2_bias)
        ascii_hist("value: linear3_weight", l3_weight)
        ascii_hist("value: linear3_bias", l3_bias)

        if model.value_no_draw:
            l3_weight = np.concatenate([l3_weight, np.zeros((3, 32))], axis=0)
            l3_bias = np.concatenate([l3_bias, np.zeros((3, ))], axis=0)
        else:
            l3_weight = np.concatenate([l3_weight, np.zeros((1, 32))], axis=0)
            l3_bias = np.concatenate([l3_bias, np.zeros((1, ))], axis=0)

        return (
            np.clip(np.around(l1_weight * self.s_input), -128, 127).astype(np.int8),
            np.clip(np.around(l1_bias * self.s_input * 128), -2**31, 2**31 - 1).astype(np.int32),
            np.clip(np.around(l2_weight * self.s_weight), -128, 127).astype(np.int8),
            np.clip(np.around(l2_bias * self.s_weight * 128), -2**31, 2**31 - 1).astype(np.int32),
            np.clip(np.around(l3_weight * self.s_weight), -128, 127).astype(np.int8),
            np.clip(np.around(l3_bias * self.s_weight * 128), -2**31, 2**31 - 1).astype(np.int32),
        )
    
    def serialize(self, out: io.IOBase, model: FlatSquare7x7NNUEv4, device: torch.device):
        mapping4x4 = self._export_feature_map(model, device)
        linear1_weight, linear1_bias, \
        linear2_weight, linear2_bias, \
        linear3_weight, linear3_bias = self._export_value(model)

        o: io.RawIOBase = out

        # int8_t mapping4x4[3][43046721][FeatureDim];
        o.write(mapping4x4.astype('<i1').tobytes())  # (3, 43046721, F)

        # float l1_weight[32][FeatureDim];
        # float l1_bias[32];
        o.write(linear1_weight.astype('<i1').tobytes())  # (32, F)
        o.write(linear1_bias.astype('<i4').tobytes())  # (32,)

        # float l2_weight[32][32];
        # float l2_bias[32];
        o.write(linear2_weight.astype('<i1').tobytes())  # (32, 32)
        o.write(linear2_bias.astype('<i4').tobytes())  # (32,)

        # float l3_weight[4][32];
        # float l3_bias[4];
        o.write(linear3_weight.astype('<i1').tobytes())  # (4, 32)
        o.write(linear3_bias.astype('<i4').tobytes())  # (4,)


@SERIALIZERS.register('flat_square7x7_nnue_v4_vq')
class FlatSquare7x7NNUEv4VQSerializer(BaseSerializer):
    """
    FlatSquare7x7NNUEv4VQ binary serializer.

    The corresponding C-language struct layout: 
    struct FlatSquare7x7NNUEv4VQWeight {
        // mapping features
        uint16_t mapping4x4_indices[3][43046721];
        int8_t   mapping4x4_features[3][CodebookSize][FeatureDim];

        // nnue layers weights
        int8_t  l1_weight[32][FeatureDim];
        int32_t l1_bias[32];
        int8_t  l2_weight[32][32];
        int32_t l2_bias[32];
        int8_t  l3_weight[4][32];
        int32_t l3_bias[4];
    };
    """
    def __init__(self, s_input=256, s_weight=128, batch_size=16384, **kwargs):
        super().__init__(**kwargs)
        self.s_input = s_input
        self.s_weight = s_weight
        self.batch_size = batch_size
        
    @property
    def is_binary(self) -> bool:
        return True

    @property
    def needs_header(self) -> bool:
        return False

    def arch_hash(self, model) -> int:
        raise NotImplementedError()
    
    def _get_mapping4x4_input(self, start_idx: int, end_idx: int) -> torch.Tensor:
        indices = torch.arange(start_idx, end_idx, dtype=torch.int32)
        inputs = torch.zeros(indices.shape[0], 2, 4, 4)
        for i in range(4 * 4):
            bit = (indices // 3**i) % 3
            inputs[:, 0, i // 4, i % 4] = bit == 1
            inputs[:, 1, i // 4, i % 4] = bit == 2
        return inputs
    
    @torch.no_grad()
    def _export_feature_map(self, model: FlatSquare7x7NNUEv4VQ, device: torch.device):
        num_mappings = len(model.mapping4x4)
        num_codes = model.codebook_size
        dim_feature = model.model_size[1]
        assert num_codes <= 2**16, f"Only support maximum {2**16} codes now."

        # compute mapping4x4's indices
        mapping4x4_indices = torch.zeros((num_mappings, 3**16), dtype=torch.int, device=device)
        for i_start in tqdm(range(0, 3**16, self.batch_size)):
            i_end = min(i_start + self.batch_size, 3**16)
            chunk4x4_input = self._get_mapping4x4_input(i_start, i_end).to(device)
            for j in range(num_mappings):
                feature = model.mapping4x4[j](chunk4x4_input)
                feature = feature.squeeze(-1).squeeze(-1)
                if model.unit_feature_vector:
                    feature = F.normalize(feature, p=2, dim=-1)
                feature = torch.clamp(feature, min=-1, max=127 / 128)

                feature_quantized, info = model.vq_layer[j](feature)
                embed_indices = info['embed_indices'].int()
                mapping4x4_indices[j, i_start:i_end] = embed_indices

        assert torch.all((mapping4x4_indices >= 0) & (mapping4x4_indices < num_codes)), \
            f"mapping4x4_indices.min()={mapping4x4_indices.min().item()}, " \
            f"mapping4x4_indices.max()={mapping4x4_indices.max().item()}"
        mapping4x4_indices = mapping4x4_indices.cpu().numpy().astype(np.uint16)
        
        # compute mapping4x4's features
        mapping4x4_features = torch.zeros((num_mappings, num_codes, dim_feature), device=device)
        for j in range(num_mappings):
            features = model.vq_layer[j].codebook
            features = fake_quant(features, scale=128)
            mapping4x4_features[j] = features

        mapping4x4_features = mapping4x4_features.cpu().numpy()
        ascii_hist(f"mapping4x4", mapping4x4_features)
        mapping4x4_features = np.clip(np.around(mapping4x4_features * 128), -128, 127).astype(np.int8)
        
        return mapping4x4_indices, mapping4x4_features
    
    def _export_value(self, model: FlatSquare7x7NNUEv4VQ):
        l1_weight = model.value_linears[0].fc.weight.cpu().numpy()
        l1_bias = model.value_linears[0].fc.bias.cpu().numpy()
        l2_weight = model.value_linears[1].fc.weight.cpu().numpy()
        l2_bias = model.value_linears[1].fc.bias.cpu().numpy()
        l3_weight = model.value_linears[2].fc.weight.cpu().numpy()
        l3_bias = model.value_linears[2].fc.bias.cpu().numpy()

        ascii_hist("value: linear1_weight", l1_weight)
        ascii_hist("value: linear1_bias", l1_bias)
        ascii_hist("value: linear2_weight", l2_weight)
        ascii_hist("value: linear2_bias", l2_bias)
        ascii_hist("value: linear3_weight", l3_weight)
        ascii_hist("value: linear3_bias", l3_bias)

        if model.value_no_draw:
            l3_weight = np.concatenate([l3_weight, np.zeros((3, 32))], axis=0)
            l3_bias = np.concatenate([l3_bias, np.zeros((3, ))], axis=0)
        else:
            l3_weight = np.concatenate([l3_weight, np.zeros((1, 32))], axis=0)
            l3_bias = np.concatenate([l3_bias, np.zeros((1, ))], axis=0)

        return (
            np.clip(np.around(l1_weight * self.s_input), -128, 127).astype(np.int8),
            np.clip(np.around(l1_bias * self.s_input * 128), -2**31, 2**31 - 1).astype(np.int32),
            np.clip(np.around(l2_weight * self.s_weight), -128, 127).astype(np.int8),
            np.clip(np.around(l2_bias * self.s_weight * 128), -2**31, 2**31 - 1).astype(np.int32),
            np.clip(np.around(l3_weight * self.s_weight), -128, 127).astype(np.int8),
            np.clip(np.around(l3_bias * self.s_weight * 128), -2**31, 2**31 - 1).astype(np.int32),
        )
    
    def serialize(self, out: io.IOBase, model: FlatSquare7x7NNUEv4VQ, device: torch.device):
        mapping4x4_indices, mapping4x4_features = self._export_feature_map(model, device)
        linear1_weight, linear1_bias, \
        linear2_weight, linear2_bias, \
        linear3_weight, linear3_bias = self._export_value(model)

        o: io.RawIOBase = out

        # uint16_t mapping4x4_indices[3][43046721];
        o.write(mapping4x4_indices.astype('<u2').tobytes())
        # int8_t   mapping4x4[3][CodebookSize][FeatureDim];
        o.write(mapping4x4_features.astype('<i1').tobytes())  # (3, CodebookSize, F)

        # float l1_weight[32][FeatureDim];
        # float l1_bias[32];
        o.write(linear1_weight.astype('<i1').tobytes())  # (32, F)
        o.write(linear1_bias.astype('<i4').tobytes())  # (32,)

        # float l2_weight[32][32];
        # float l2_bias[32];
        o.write(linear2_weight.astype('<i1').tobytes())  # (32, 32)
        o.write(linear2_bias.astype('<i4').tobytes())  # (32,)

        # float l3_weight[4][32];
        # float l3_bias[4];
        o.write(linear3_weight.astype('<i1').tobytes())  # (4, 32)
        o.write(linear3_bias.astype('<i4').tobytes())  # (4,)


