import torch
import numpy as np
import io
from zlib import crc32
from math import ceil
from utils.misc_utils import ascii_hist
from . import BaseSerializer, SERIALIZERS
from ..mix6 import Mix6Net, Mix6Netv2


def generate_base3_permutation(length):
    x = np.zeros((1, length), dtype=np.int8)
    for i in range(length):
        x1, x2 = x.copy(), x.copy()
        x1[:, i] = 1
        x2[:, i] = 2
        x = np.concatenate((x, x1, x2), axis=0)
    return x


@SERIALIZERS.register('mix6')
class Mix6NetSerializer(BaseSerializer):
    """
    Mix6Net binary serializer.

    The corresponding C-language struct layout: 
    struct Mix6Weight {
        // 1  map=weight.map(board), shape=H*W*4*c
        int16_t map[ShapeNum][FeatureDim];

        // 2  mapsum=map.sum(2), shape=H*W*c
        // 3  mapAfterLR=leakyRelu(mapsum)
        int16_t map_lr_slope_sub1div8[FeatureDim];
        int16_t map_lr_bias[FeatureDim];

        // 4  update policyBeforeConv and valueSumBoard
        // 5  policyAfterConv=depthwise_conv2d(policyBeforeConv)
        int16_t policy_conv_weight[9][PolicyDim];
        int16_t policy_conv_bias[PolicyDim];

        // 6  policy=conv1x1(relu(policyAfterConv))
        int16_t policy_final_conv[PolicyDim];
        // 7  policy=leakyRelu(policyAfterConv)
        float policy_neg_slope, policy_pos_slope;

        // 8  value leakyRelu
        float scale_before_mlp;
        float value_lr_slope_sub1[ValueDim];

        // 9  mlp
        float mlp_w1[ValueDim][ValueDim];  // shape=(inc, outc)
        float mlp_b1[ValueDim];
        float mlp_w2[ValueDim][ValueDim];
        float mlp_b2[ValueDim];
        float mlp_w3[ValueDim][3];
        float _mlp_w3_padding[5];
        float mlp_b3[3];
        float _mlp_b3_padding[5];
    };
    """
    def __init__(self,
                 rule='freestyle',
                 board_size=None,
                 side_to_move=None,
                 feature_map_bound=4500,
                 text_output=False,
                 **kwargs):
        super().__init__(rules=[rule],
                         boardsizes=list(range(5, 23)) if board_size is None else [board_size],
                         **kwargs)
        self.side_to_move = side_to_move
        self.line_length = 11
        self.text_output = text_output
        self.feature_map_bound = feature_map_bound
        self.map_table_export_batch_size = 4096

    @property
    def is_binary(self) -> bool:
        return not self.text_output

    def arch_hash(self, model: Mix6Net) -> int:
        hash = crc32(b'mix6net')
        _, dim_policy, dim_value = model.model_size
        hash ^= crc32(model.input_type.encode('utf-8'))
        hash ^= (dim_policy << 16) | dim_value
        return hash

    def _export_map_table(self, model: Mix6Net, device, line, stm=None):
        """
        Export line -> feature mapping table.

        Args:
            line: shape (N, Length)
        """
        N, L = line.shape
        b, w = line == 1, line == 2

        if stm is not None:
            s = np.ones_like(line) * stm
            line = (b, w, s)
        else:
            line = (b, w)
        line = np.stack(line, axis=0)[np.newaxis]  # [B=1, C=2, N, L]
        line = torch.tensor(line, dtype=torch.float32, device=device)

        batch_size = self.map_table_export_batch_size

        batch_num = 1 + (N - 1) // batch_size
        map_table = []
        for i in range(batch_num):
            start = i * batch_size
            end = min((i + 1) * batch_size, N)

            map = model.mapping(line[:, :, start:end])[0, 0]
            if model.map_max != 0:
                map = model.map_max * torch.tanh(map / model.map_max)

            map_table.append(map.cpu().numpy())

        map_table = np.concatenate(map_table, axis=1)  # [C=PC+VC, N, L]
        return map_table

    def _export_feature_map(self, model: Mix6Net, device, stm=None):
        L = self.line_length
        _, PC, VC = model.model_size
        lines = generate_base3_permutation(L)  # [177147, 11]
        map_table = self._export_map_table(model, device, lines, stm)  # [C=PC+VC, 177147, 11]

        feature_map = np.zeros((4 * 3**L, PC + VC), dtype=np.float32)  # [708588, 48]
        usage_flags = np.zeros(4 * 3**L, dtype=np.int8)  # [708588], track usage of each feature
        pow3 = 3**np.arange(L + 1, dtype=np.int64)[:, np.newaxis]  # [11, 1]

        for r in range((L + 1) // 2):
            idx = np.matmul(lines[:, r:], pow3[:L - r]) + pow3[L + 1 - r:].sum()
            for i in range(idx.shape[0]):
                feature_map[idx[i]] = map_table[:, i, L // 2 + r]
                usage_flags[idx[i]] = True

        for l in range(1, (L + 1) // 2):
            idx = np.matmul(lines[:, :L - l], pow3[l:-1]) + 2 * pow3[-1] + pow3[:l - 1].sum()
            for i in range(idx.shape[0]):
                feature_map[idx[i]] = map_table[:, i, L // 2 - l]
                usage_flags[idx[i]] = True

        for l in range(1, (L + 1) // 2):
            for r in range(1, (L + 1) // 2):
                lines = generate_base3_permutation(L - l - r)
                map_table = self._export_map_table(model, device, lines, stm)
                idx_offset = 3 * pow3[-1] + pow3[:l - 1].sum() + pow3[L + 1 - r:-1].sum()
                idx = np.matmul(lines, pow3[l:L - r]) + idx_offset
                for i in range(idx.shape[0]):
                    feature_map[idx[i]] = map_table[:, i, L // 2 - l]
                    usage_flags[idx[i]] = True

        feature_map_max = np.abs(feature_map).max()  # stores the upper bound of the feature map
        scale = self.feature_map_bound / feature_map_max
        bound = ceil(feature_map_max * scale)
        print(f"feature map: used {usage_flags.sum()} features of {len(usage_flags)}" +
              f", feature_max = {feature_map_max}, scale = {scale}, bound = {bound}")
        assert bound < 32767, "feature map overflow!"

        return feature_map * scale, usage_flags, scale, bound

    def _export_mapping_activation(self, model: Mix6Net, scale, bound):
        slope = model.mapping_activation.neg_slope.cpu().numpy()
        bias = model.mapping_activation.bias.cpu().numpy()
        slope_bound = model.mapping_activation.bound
        if slope_bound != 0:
            slope = np.tanh(slope / slope_bound) * slope_bound

        slope_max = np.abs(slope).max()
        bias_max = np.abs(bias).max()
        act_max = max((slope_max + 1) / 8 * 2**15, bias_max * scale)
        bound_perchannel = bound * (np.abs(slope) + 1) + np.abs(bias) * scale
        print(f"map leakyrelu: slope_max = {slope_max}, bias_max = {bias_max}" +
              f", max = {act_max}, scale = {scale}, bound = {bound_perchannel.max()}")
        assert act_max < 32767, "map leakyrelu overflow!"
        assert bound_perchannel.max() < 32767, "map activation overflow!"

        slope_sub1div8 = (slope - 1) / 8
        return slope_sub1div8 * 2**15, bias * scale, scale, bound_perchannel

    def _export_policy(self, model: Mix6Net, scale, bound_perchannel):
        _, PC, _ = model.model_size
        # policy layer 1: policy dw conv
        conv_weight = model.policy_conv.conv.weight.cpu().numpy()
        conv_bias = model.policy_conv.conv.bias.cpu().numpy()

        conv_weight_max = np.abs(conv_weight).max()
        conv_bias_max = np.abs(conv_bias).max()
        bound_perchannel = np.abs(conv_weight).sum(
            (1, 2, 3)) * bound_perchannel[:PC] + np.abs(conv_bias) * scale
        conv_max = max(conv_weight_max * 2**15, conv_bias_max * scale)
        conv_scale = min(32766 / conv_max, 32766 / bound_perchannel.max())
        scale *= conv_scale
        bound_perchannel *= conv_scale
        conv_max = max(conv_weight_max * conv_scale * 2**15, conv_bias_max * scale)
        print(f"policy conv: weight_max = {conv_weight_max}, bias_max = {conv_bias_max}" +
              f", max = {conv_max}, scale = {scale}, bound = {bound_perchannel.max()}")
        assert conv_max < 32767, "policy conv overflow!"
        assert bound_perchannel.max() < 32767, "policy overflow!"

        conv_weight = (conv_weight * conv_scale * 2**15).squeeze(1).transpose((1, 2, 0))
        conv_bias = conv_bias * scale

        # policy layer 2: policy pw conv
        pw_conv_weight = model.policy_linear.conv.weight.cpu().numpy()[0, :, 0, 0]
        pw_conv_weight_max = np.abs(pw_conv_weight).max()
        pw_conv_scale = min(32766 / (pw_conv_weight_max * 2**15), 0.25)
        scale *= pw_conv_scale
        bound = (np.abs(pw_conv_weight) * bound_perchannel).sum() * pw_conv_scale
        pw_conv_max = pw_conv_weight_max * pw_conv_scale * 2**15
        print(f"policy pw conv: weight_max = {pw_conv_weight_max}" +
              f", max = {pw_conv_max}, scale = {scale}, bound = {bound}")
        assert pw_conv_max < 32767, "policy pw conv overflow!"
        assert bound < 32767, f"policy overflow! ({bound})"

        pw_conv_weight = pw_conv_weight * pw_conv_scale * 2**15

        # policy layer 3: policy activation
        slope = model.policy_activation.neg_slope.cpu().numpy()
        neg_slope = slope.item() / scale
        pos_slope = 1 / scale

        bound = max(abs(bound * neg_slope), abs(bound * pos_slope))
        print(f"policy act: neg_slope = {neg_slope}, pos_slope = {pos_slope}, bound = {bound}")

        return conv_weight, conv_bias, pw_conv_weight, neg_slope, pos_slope

    def _export_value(self, model: Mix6Net, scale):
        # value layer 0: global mean
        scale_before_mlp = 1 / scale  # Note: divide board_size**2 in engine

        # value layer 1: activation after mean
        lr_slope = model.value_activation.neg_slope.cpu().numpy()
        lr_slope_sub1 = lr_slope - 1

        # value layer 2: linear mlp 01
        linear1_weight = model.value_linear[0].fc.weight.cpu().numpy().T
        linear1_bias = model.value_linear[0].fc.bias.cpu().numpy()

        # value layer 3: linear mlp 02
        linear2_weight = model.value_linear[1].fc.weight.cpu().numpy().T
        linear2_bias = model.value_linear[1].fc.bias.cpu().numpy()

        # value layer 4: linear mlp final
        linear_final_weight = model.value_linear_final.fc.weight.cpu().numpy().T
        linear_final_bias = model.value_linear_final.fc.bias.cpu().numpy()

        print(f"value: scale_before_mlp = {scale_before_mlp}")
        ascii_hist("value: lr_slope_sub1", lr_slope_sub1)
        ascii_hist("value: linear1_weight", linear1_weight)
        ascii_hist("value: linear1_bias", linear1_bias)
        ascii_hist("value: linear2_weight", linear2_weight)
        ascii_hist("value: linear2_bias", linear2_bias)
        ascii_hist("value: linear_final_weight", linear_final_weight)
        ascii_hist("value: linear_final_bias", linear_final_bias)

        return (
            scale_before_mlp,
            lr_slope_sub1,
            linear1_weight,
            linear1_bias,
            linear2_weight,
            linear2_bias,
            linear_final_weight,
            linear_final_bias,
        )

    def serialize(self, out: io.IOBase, model: Mix6Net, device):
        feature_map, usage_flags, scale, bound = \
            self._export_feature_map(model, device, stm=self.side_to_move)
        map_lr_slope_sub1div8, map_lr_bias, scale, bound_perchannel = \
            self._export_mapping_activation(model, scale, bound)
        policy_conv_weight, policy_conv_bias, \
        policy_pw_conv_weight, policy_neg_slope, policy_pos_slope = \
            self._export_policy(model, scale, bound_perchannel)
        scale_before_mlp, lr_slope_sub1, \
        linear1_weight, linear1_bias, \
        linear2_weight, linear2_bias, \
        linear_final_weight, linear_final_bias = self._export_value(model, scale)

        if self.text_output:
            print('featuremap', file=out)
            print(usage_flags.sum(), file=out)
            for i, (f, used) in enumerate(zip(feature_map.astype('i2'), usage_flags)):
                if used:
                    print(i, end=' ', file=out)
                    f.tofile(out, sep=' ')
                    print(file=out)

            print('map_lr_slope_sub1div8', file=out)
            map_lr_slope_sub1div8.astype('i2').tofile(out, sep=' ')
            print(file=out)

            print('map_lr_bias', file=out)
            map_lr_bias.astype('i2').tofile(out, sep=' ')
            print(file=out)

            print('policyConvWeight', file=out)
            policy_conv_weight.astype('i2').tofile(out, sep=' ')
            print(file=out)

            print('policyConvBias', file=out)
            policy_conv_bias.astype('i2').tofile(out, sep=' ')
            print(file=out)

            print('policyFinalConv', file=out)
            policy_pw_conv_weight.astype('i2').tofile(out, sep=' ')
            print(file=out)

            print('policy_neg_slope', file=out)
            print(policy_neg_slope, file=out)
            print('policy_pos_slope', file=out)
            print(policy_pos_slope, file=out)

            print('scale_beforemlp', file=out)
            print(scale_before_mlp, file=out)

            print('value_lr_slope_sub1', file=out)
            lr_slope_sub1.astype('f4').tofile(out, sep=' ')
            print(file=out)

            print('mlp_w1', file=out)
            linear1_weight.astype('f4').tofile(out, sep=' ')
            print(file=out)
            print('mlp_b1', file=out)
            linear1_bias.astype('f4').tofile(out, sep=' ')
            print(file=out)

            print('mlp_w2', file=out)
            linear2_weight.astype('f4').tofile(out, sep=' ')
            print(file=out)
            print('mlp_b2', file=out)
            linear2_bias.astype('f4').tofile(out, sep=' ')
            print(file=out)

            print('mlp_w3', file=out)
            linear_final_weight.astype('f4').tofile(out, sep=' ')
            print(file=out)
            print('mlp_b3', file=out)
            linear_final_bias.astype('f4').tofile(out, sep=' ')
            print(file=out)
        else:
            o: io.RawIOBase = out

            # int16_t map[ShapeNum][FeatureDim];
            o.write(feature_map.astype('<i2').tobytes())  # (708588, 48)

            # int16_t map_lr_slope_sub1div8[FeatureDim];
            # int16_t map_lr_bias[FeatureDim];
            o.write(map_lr_slope_sub1div8.astype('<i2').tobytes())  # (48,)
            o.write(map_lr_bias.astype('<i2').tobytes())  # (48,)

            # int16_t policy_conv_weight[9][PolicyDim];
            # int16_t policy_conv_bias[PolicyDim];
            o.write(policy_conv_weight.astype('<i2').tobytes())  # (3, 3, 16)
            o.write(policy_conv_bias.astype('<i2').tobytes())  # (16,)

            # int16_t policy_final_conv[PolicyDim];
            # float policy_neg_slope, policy_pos_slope;
            o.write(policy_pw_conv_weight.astype('<i2').tobytes())  # (16,)
            o.write(np.array([policy_neg_slope, policy_pos_slope], dtype='<f4').tobytes())

            # float scale_before_mlp;
            # float value_lr_slope_sub1[ValueDim];
            o.write(np.array([scale_before_mlp], dtype='<f4').tobytes())
            o.write(lr_slope_sub1.astype('<f4').tobytes())  # (32,)

            # float mlp_w1[ValueDim][ValueDim];  // shape=(inc, outc)
            # float mlp_b1[ValueDim];
            o.write(linear1_weight.astype('<f4').tobytes())  # (32, 32)
            o.write(linear1_bias.astype('<f4').tobytes())  # (32,)

            # float mlp_w2[ValueDim][ValueDim];
            # float mlp_b2[ValueDim];
            o.write(linear2_weight.astype('<f4').tobytes())  # (32, 32)
            o.write(linear2_bias.astype('<f4').tobytes())  # (32,)

            # float mlp_w3[ValueDim][3];
            # float _mlp_w3_padding[5];
            # float mlp_b3[3];
            # float _mlp_b3_padding[5];
            o.write(linear_final_weight.astype('<f4').tobytes())  # (32, 3)
            o.write(np.zeros(5, dtype='<f4').tobytes())  # padding: (5,)
            o.write(linear_final_bias.astype('<f4').tobytes())  # (3,)
            o.write(np.zeros(5, dtype='<f4').tobytes())  # padding: (5,)


@SERIALIZERS.register('mix6v2')
class Mix6Netv2Serializer(BaseSerializer):
    """
    Mix6Netv2 binary serializer.

    The corresponding C-language struct layout: 
    struct Mix6Weight {
        // 1  mapping layer
        int32_t NumMappings;
        int8_t  mappings[NumMappings][ShapeNum][FeatureDim];

        // 2  PReLU after mapping
        int16_t prelu_weight[FeatureDim];

        // 3  Policy depthwise conv
        int8_t  policy_dw_conv_weight[9][PolicyDim];
        int16_t policy_dw_conv_bias[PolicyDim];

        // 4  Policy pointwise conv
        int8_t policy_pw_conv_weight[PolicyDim];

        // 5  Value PReLU
        int16_t value_prelu_weight[ValueDim];

        // 6  Value MLP (layer 1,2,3)
        int8_t  value_linear1_weight[ValueDim][ValueDim];  // shape=(out channel, in channel)
        int32_t value_linear1_bias[ValueDim];
        int8_t  value_linear2_weight[ValueDim][ValueDim];
        int32_t value_linear2_bias[ValueDim];
        int8_t  value_linear3_weight[3+1][ValueDim];  // add one for padding
        int32_t value_linear3_bias[3+1];              // add one for padding

        float policy_output_scale;
        float value_output_scale;
    };
    """
    def __init__(self, rule='freestyle', board_size=None, text_output=False, **kwargs):
        super().__init__(rules=[rule],
                         boardsizes=list(range(5, 23)) if board_size is None else [board_size],
                         **kwargs)
        self.line_length = 11
        self.text_output = text_output
        self.map_table_export_batch_size = 4096

    @property
    def is_binary(self) -> bool:
        return not self.text_output

    def arch_hash(self, model: Mix6Netv2) -> int:
        assert model.input_type == 'basic' or model.input_type == 'basic-nostm'
        _, dim_policy, dim_value = model.model_size
        hash = crc32(b'mix6netv2')
        hash ^= crc32(b'basic')
        hash ^= (dim_policy << 16) | dim_value
        hash ^= hash << 7
        hash += hash >> 3
        hash ^= model.scale_weight
        return hash

    def _export_map_table(self, model: Mix6Netv2, device, line, stm=None):
        """
        Export line -> feature mapping table.

        Args:
            line: shape (N, Length)
        """
        N, L = line.shape
        b, w = line == 1, line == 2

        if stm is not None:
            s = np.ones_like(line) * stm
            line = (b, w, s)
        else:
            line = (b, w)
        line = np.stack(line, axis=0)[np.newaxis]  # [B=1, C=2 or 3, N, L]
        line = torch.tensor(line, dtype=torch.float32, device=device)

        batch_size = self.map_table_export_batch_size

        batch_num = 1 + (N - 1) // batch_size
        map_table = []
        for i in range(batch_num):
            start = i * batch_size
            end = min((i + 1) * batch_size, N)

            map = model.mapping(line[:, :, start:end])[0, 0]
            map = model.maxf_i8_f * torch.tanh(map / model.maxf_i8_f)

            map_table.append(map.cpu().numpy())

        map_table = np.concatenate(map_table, axis=1)  # [C=PC+VC, N, L]
        return map_table

    def _export_feature_map(self, model: Mix6Netv2, device, stm=None):
        L = self.line_length
        _, PC, VC = model.model_size
        lines = generate_base3_permutation(L)  # [177147, 11]
        map_table = self._export_map_table(model, device, lines, stm)  # [C=PC+VC, 177147, 11]

        feature_map = np.zeros((4 * 3**L, PC + VC), dtype=np.float32)  # [708588, 48]
        usage_flags = np.zeros(4 * 3**L, dtype=np.int8)  # [708588], track usage of each feature
        pow3 = 3**np.arange(L + 1, dtype=np.int64)[:, np.newaxis]  # [11, 1]

        for r in range((L + 1) // 2):
            idx = np.matmul(lines[:, r:], pow3[:L - r]) + pow3[L + 1 - r:].sum()
            for i in range(idx.shape[0]):
                feature_map[idx[i]] = map_table[:, i, L // 2 + r]
                usage_flags[idx[i]] = True

        for l in range(1, (L + 1) // 2):
            idx = np.matmul(lines[:, :L - l], pow3[l:-1]) + 2 * pow3[-1] + pow3[:l - 1].sum()
            for i in range(idx.shape[0]):
                feature_map[idx[i]] = map_table[:, i, L // 2 - l]
                usage_flags[idx[i]] = True

        for l in range(1, (L + 1) // 2):
            for r in range(1, (L + 1) // 2):
                lines = generate_base3_permutation(L - l - r)
                map_table = self._export_map_table(model, device, lines, stm)
                idx_offset = 3 * pow3[-1] + pow3[:l - 1].sum() + pow3[L + 1 - r:-1].sum()
                idx = np.matmul(lines, pow3[l:L - r]) + idx_offset
                for i in range(idx.shape[0]):
                    feature_map[idx[i]] = map_table[:, i, L // 2 - l]
                    usage_flags[idx[i]] = True

        feature_map_quant = feature_map * model.scale_feature
        feature_map_max = np.abs(feature_map).max()
        feature_map_quant_max = np.abs(feature_map_quant).max()
        print(f"feature map: used {usage_flags.sum()} features of {len(usage_flags)}" +
              f", feature_max = {feature_map_max}, feature_quant_max = {feature_map_quant_max}")
        assert feature_map_quant_max <= 127, "feature map overflow!"

        return feature_map_quant, usage_flags

    def _export_mapping_activation(self, model: Mix6Netv2):
        weight = model.mapping_activation.weight.cpu().numpy()
        weight_quant = weight * 64

        weight_max = np.abs(weight).max()
        weight_quant_max = np.abs(weight_quant).max()
        ascii_hist("mapping prelu weight", weight)
        print(f"map prelu: weight_max = {weight_max}, weight_quant_max = {weight_quant_max}")
        assert weight_quant_max <= 64, "map prelu overflow!"

        return weight_quant

    def _export_policy(self, model: Mix6Netv2):
        # policy layer 1: policy dw conv
        dw_conv_weight = model.policy_dw_conv.conv.weight.cpu().numpy()
        dw_conv_bias = model.policy_dw_conv.conv.bias.cpu().numpy()
        dw_conv_weight_quant = dw_conv_weight * model.scale_weight
        dw_conv_bias_quant = dw_conv_bias * (model.scale_weight * model.scale_feature / 16)

        # transpose weight to [9][dim_policy]
        dw_conv_weight_quant = dw_conv_weight_quant.squeeze(1).transpose((1, 2, 0))

        dw_conv_weight_max = np.abs(dw_conv_weight).max()
        dw_conv_bias_max = np.abs(dw_conv_bias).max()
        dw_conv_weight_quant_max = np.abs(dw_conv_weight_quant).max()
        dw_conv_bias_quant_max = np.abs(dw_conv_bias_quant).max()
        print(
            f"policy dw conv: weight_max = {dw_conv_weight_max}, bias_max = {dw_conv_bias_max}, " +
            f"weight_quant_max = {dw_conv_weight_quant_max}, bias_quant_max = {dw_conv_bias_quant_max}"
        )
        assert dw_conv_weight_quant_max <= 127, "policy dw conv weight overflow!"
        assert dw_conv_bias_quant_max <= 20000, "policy dw conv bias overflow!"

        # policy layer 2: policy pw conv
        pw_conv_weight = model.policy_pw_conv.conv.weight.cpu().numpy()[0, :, 0, 0]
        pw_conv_weight_quant = pw_conv_weight * model.scale_weight

        pw_conv_weight_max = np.abs(pw_conv_weight).max()
        pw_conv_weight_quant_max = np.abs(pw_conv_weight_quant).max()
        print(f"policy pw conv: weight_max = {pw_conv_weight_max}" +
              f", weight_quant_max = {pw_conv_weight_quant_max}")
        assert pw_conv_weight_quant_max <= 127, "policy pw conv weight overflow!"

        policy_output_scale = model.policy_output_scale.item()
        policy_scale = policy_output_scale / model.scale_feature
        print(f"policy output: float_scale = {policy_output_scale}, scale = {policy_scale}")

        return dw_conv_weight_quant, dw_conv_bias_quant, pw_conv_weight_quant, policy_scale

    def _export_value(self, model: Mix6Netv2):
        # value layer 0: activation after mean
        prelu_weight = model.value_activation.weight.cpu().numpy()
        prelu_weight_quant = prelu_weight * 128

        ascii_hist("value prelu weight", prelu_weight)
        prelu_weight_quant_max = np.abs(prelu_weight_quant).max()
        assert prelu_weight_quant_max <= 127, "value prelu weight overflow!"

        # value layer 1: linear mlp 01
        linear1_weight = model.value_linear1.fc.weight.cpu().numpy()
        linear1_bias = model.value_linear1.fc.bias.cpu().numpy()
        linear1_weight_quant = linear1_weight * model.scale_weight
        linear1_bias_quant = linear1_bias * model.scale_weight * model.scale_feature_after_mean

        ascii_hist("value linear1 weight", linear1_weight)
        ascii_hist("value linear1 bias", linear1_bias)
        linear1_weight_quant_max = np.abs(linear1_weight_quant).max()
        linear1_bias_quant_max = np.abs(linear1_bias_quant).max()
        assert linear1_weight_quant_max <= 127, "value linear1 weight overflow!"
        assert linear1_bias_quant_max <= 2**30, "value linear1 bias overflow!"

        # value layer 2: linear mlp 02
        linear2_weight = model.value_linear2.fc.weight.cpu().numpy()
        linear2_bias = model.value_linear2.fc.bias.cpu().numpy()
        linear2_weight_quant = linear2_weight * model.scale_weight
        linear2_bias_quant = linear2_bias * model.scale_weight * model.scale_feature_after_mean

        ascii_hist("value linear2 weight", linear2_weight)
        ascii_hist("value linear2 bias", linear2_bias)
        linear2_weight_quant_max = np.abs(linear2_weight_quant).max()
        linear2_bias_quant_max = np.abs(linear2_bias_quant).max()
        assert linear2_weight_quant_max <= 127, "value linear2 weight overflow!"
        assert linear2_bias_quant_max <= 2**30, "value linear2 bias overflow!"

        # value layer 3: linear mlp final
        linear3_weight = model.value_linear_final.fc.weight.cpu().numpy()
        linear3_bias = model.value_linear_final.fc.bias.cpu().numpy()

        # padding weight and bias to 4 output channels
        linear3_weight = np.concatenate(
            (linear3_weight, np.zeros_like(linear3_weight[0])[np.newaxis]), axis=0)
        linear3_bias = np.concatenate((linear3_bias, np.zeros_like(linear3_bias[0])[np.newaxis]),
                                      axis=0)

        linear3_weight_quant = linear3_weight * model.scale_weight
        linear3_bias_quant = linear3_bias * model.scale_weight * model.scale_feature_after_mean

        ascii_hist("value linear3 weight", linear3_weight)
        ascii_hist("value linear3 bias", linear3_bias)
        linear3_weight_quant_max = np.abs(linear3_weight_quant).max()
        linear3_bias_quant_max = np.abs(linear3_bias_quant).max()
        assert linear3_weight_quant_max <= 127, "value linear3 weight overflow!"
        assert linear3_bias_quant_max < 2**31, "value linear3 bias overflow!"

        value_output_scale = model.value_output_scale.item()
        value_scale = value_output_scale / model.scale_feature_after_mean
        print(f"value output: float_scale = {value_output_scale}, scale = {value_scale}")

        return (
            prelu_weight_quant,
            linear1_weight_quant,
            linear1_bias_quant,
            linear2_weight_quant,
            linear2_bias_quant,
            linear3_weight_quant,
            linear3_bias_quant,
            value_scale,
        )

    def serialize(self, out: io.IOBase, model: Mix6Net, device):
        mappings = []
        if model.input_type == 'basic':
            mappings.append(self._export_feature_map(model, device, stm=-1))  # Black
            mappings.append(self._export_feature_map(model, device, stm=1))  # White
        else:
            mappings.append(self._export_feature_map(model, device))

        map_prelu_weight = self._export_mapping_activation(model)
        policy_dw_conv_weight, policy_dw_conv_bias, policy_pw_conv_weight, \
            policy_output_scale = self._export_policy(model)
        value_prelu_weight, linear1_weight, linear1_bias, \
        linear2_weight, linear2_bias, linear3_weight, \
        linear3_bias, value_output_scale = self._export_value(model)

        if self.text_output:
            print('num_mappings', file=out)
            print(len(mappings), file=out)
            for mapidx, (feature_map, usage_flags) in enumerate(mappings):
                print(f'feature_map {mapidx}', file=out)
                print(usage_flags.sum(), file=out)
                for i, (f, used) in enumerate(zip(feature_map.astype('i1'), usage_flags)):
                    if used:
                        print(i, end=' ', file=out)
                        f.tofile(out, sep=' ')
                        print(file=out)

            print('map_prelu_weight', file=out)
            map_prelu_weight.astype('i2').tofile(out, sep=' ')
            print(file=out)

            print('policy_dw_conv_weight', file=out)
            policy_dw_conv_weight.astype('i1').tofile(out, sep=' ')
            print(file=out)

            print('policy_dw_conv_bias', file=out)
            policy_dw_conv_bias.astype('i2').tofile(out, sep=' ')
            print(file=out)

            print('policy_pw_conv_weight', file=out)
            policy_pw_conv_weight.astype('i1').tofile(out, sep=' ')
            print(file=out)

            print('value_prelu_weight', file=out)
            value_prelu_weight.astype('i2').tofile(out, sep=' ')
            print(file=out)

            print('linear1_weight', file=out)
            linear1_weight.astype('i1').tofile(out, sep=' ')
            print(file=out)
            print('linear1_bias', file=out)
            linear1_bias.astype('i2').tofile(out, sep=' ')
            print(file=out)

            print('linear2_weight', file=out)
            linear2_weight.astype('i1').tofile(out, sep=' ')
            print(file=out)
            print('linear2_bias', file=out)
            linear2_bias.astype('i2').tofile(out, sep=' ')
            print(file=out)

            print('linear3_weight', file=out)
            linear3_weight.astype('i1').tofile(out, sep=' ')
            print(file=out)
            print('linear3_bias', file=out)
            linear3_bias.astype('i2').tofile(out, sep=' ')
            print(file=out)

            print('policy_output_scale', file=out)
            print(policy_output_scale, file=out)
            print('value_output_scale', file=out)
            print(value_output_scale, file=out)
        else:
            o: io.RawIOBase = out

            # int32_t NumMappings
            # int8_t  mappings[NumMappings][ShapeNum][FeatureDim]
            o.write(np.array([len(mappings)], dtype='<i4').tobytes())
            for mapidx, (feature_map, _) in enumerate(mappings):
                o.write(feature_map.astype('<i1').tobytes())  # (708588, PC+VC)

            # int16_t prelu_weight[FeatureDim]
            o.write(map_prelu_weight.astype('<i2').tobytes())  # (PV+VC,)

            # int8_t  policy_dw_conv_weight[9][PolicyDim]
            # int16_t policy_dw_conv_bias[PolicyDim]
            o.write(policy_dw_conv_weight.astype('<i1').tobytes())  # (3, 3, PC)
            o.write(policy_dw_conv_bias.astype('<i2').tobytes())  # (PC,)

            # int8_t policy_pw_conv_weight[PolicyDim]
            o.write(policy_pw_conv_weight.astype('<i1').tobytes())  # (PC,)

            # int16_t value_prelu_weight[ValueDim]
            o.write(value_prelu_weight.astype('<i2').tobytes())  # (VC,)

            # int8_t  value_linear1_weight[ValueDim][ValueDim]  // shape=(out channel, in channel)
            # int32_t value_linear1_bias[ValueDim]
            o.write(linear1_weight.astype('<i1').tobytes())  # (VC, VC)
            o.write(linear1_bias.astype('<i4').tobytes())  # (VC,)

            # int8_t  value_linear2_weight[ValueDim][ValueDim]
            # int32_t value_linear2_bias[ValueDim]
            o.write(linear2_weight.astype('<i1').tobytes())  # (VC, VC)
            o.write(linear2_bias.astype('<i4').tobytes())  # (VC,)

            # int8_t  value_linear3_weight[3+1][ValueDim]  // add one for padding
            # int32_t value_linear3_bias[3+1]              // add one for padding
            o.write(linear3_weight.astype('<i1').tobytes())  # (VC, VC)
            o.write(linear3_bias.astype('<i4').tobytes())  # (VC,)

            # float policy_output_scale
            # float value_output_scale
            o.write(np.array([policy_output_scale, value_output_scale], dtype='<f4').tobytes())