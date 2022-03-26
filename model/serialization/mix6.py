import torch
import numpy as np
import io
from math import ceil
from . import BaseSerializer, SERIALIZERS
from ..mix6 import Mix6Net


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
    def __init__(self, board_size=15, feature_map_bound=4500, text_output=False):
        super().__init__()
        self.line_length = 11
        self.board_size = board_size
        self.text_output = text_output
        self.feature_map_bound = feature_map_bound
        self.map_table_export_batch_size = 4096

    @property
    def is_binary(self):
        return not self.text_output

    def _export_map_table(self, model: Mix6Net, device, line):
        """
        Export line -> feature mapping table.

        Args:
            line: shape (N, Length)
        """
        N, L = line.shape
        b, w = line == 1, line == 2
        line = np.stack((b, w), axis=0)[np.newaxis]  # [B=1, C=2, N, L]
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

    def _export_feature_map(self, model: Mix6Net, device):
        L = self.line_length
        _, PC, VC = model.model_size
        lines = generate_base3_permutation(L)  # [177147, 11]
        map_table = self._export_map_table(model, device, lines)  # [C=PC+VC, 177147, 11]

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
                map_table = self._export_map_table(model, device, lines)
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
        assert bound < 32768, "feature map overflow!"

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
        assert act_max < 32768, "map activation overflow!"

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
        conv_scale = min(32767 / conv_max, 32767 / bound_perchannel.max())
        scale *= conv_scale
        bound_perchannel *= conv_scale
        conv_max = max(conv_weight_max * conv_scale * 2**15, conv_bias_max * scale)
        print(f"policy conv: weight_max = {conv_weight_max}, bias_max = {conv_bias_max}" +
              f", max = {conv_max}, scale = {scale}, bound = {bound_perchannel.max()}")
        assert conv_max < 32768, "policy conv overflow!"

        conv_weight = (conv_weight * conv_scale * 2**15).squeeze(1).transpose((1, 2, 0))
        conv_bias = conv_bias * scale

        # policy layer 2: policy pw conv
        pw_conv_weight = model.policy_linear.conv.weight.cpu().numpy()[0, :, 0, 0]
        pw_conv_weight_max = np.abs(pw_conv_weight).max()
        pw_conv_scale = min(32767 / (pw_conv_weight_max * 2**15), 0.5)
        scale *= pw_conv_scale
        bound = (np.abs(pw_conv_weight) * bound_perchannel).sum() * pw_conv_scale
        pw_conv_max = pw_conv_weight_max * pw_conv_scale * 2**15
        print(f"policy pw conv: weight_max = {pw_conv_weight_max}" +
              f", max = {pw_conv_max}, scale = {scale}, bound = {bound}")
        assert pw_conv_max < 32768, "policy pw conv overflow!"
        assert bound < 32768, "policy overflow!"

        pw_conv_weight = pw_conv_weight * pw_conv_scale * 2**15

        # policy layer 3: policy activation
        slope = model.policy_activation.neg_slope.cpu().numpy()
        neg_slope = slope.item() / scale
        pos_slope = 1 / scale

        bound = max(abs(bound * neg_slope), abs(bound * pos_slope))
        print(f"policy act: neg_slope = {neg_slope}, pos_slope = {pos_slope}, bound = {bound}")

        return conv_weight, conv_bias, pw_conv_weight, neg_slope, pos_slope

    def _export_value(self, model: Mix6Net, scale):
        _, PC, _ = model.model_size

        # value layer 0: global mean
        scale_before_mlp = 1 / scale / self.board_size**2

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
        feature_map, usage_flags, scale, bound = self._export_feature_map(model, device)
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