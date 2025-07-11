import torch
import numpy as np
import io
from tqdm import tqdm
from zlib import crc32
from math import ceil
from utils.misc_utils import ascii_hist
from . import BaseSerializer, SERIALIZERS
from ..mixnet import Mix6Net, Mix7Net, Mix8Net, Mix9Net, Mix9sVQNet, Mix10Net


def generate_base3_permutation(length):
    x = np.zeros((1, length), dtype=np.int8)
    for i in range(length):
        x1, x2 = x.copy(), x.copy()
        x1[:, i] = 1
        x2[:, i] = 2
        x = np.concatenate((x, x1, x2), axis=0)
    return x


@SERIALIZERS.register("mix6")
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

    def __init__(
        self,
        rules,
        boardsizes,
        side_to_move=None,
        feature_map_bound=4500,
        text_output=False,
        **kwargs,
    ):
        super().__init__(rules, boardsizes, **kwargs)
        self.side_to_move = side_to_move
        self.line_length = 11
        self.text_output = text_output
        self.feature_map_bound = feature_map_bound
        self.map_table_export_batch_size = 4096

    @property
    def is_binary(self) -> bool:
        return not self.text_output

    def arch_hash(self, model: Mix6Net) -> int:
        hash = crc32(b"mix6net")
        _, dim_policy, dim_value = model.model_size
        hash ^= crc32(model.input_type.encode("utf-8"))
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
        pow3 = 3 ** np.arange(L + 1, dtype=np.int64)[:, np.newaxis]  # [11, 1]

        for r in range((L + 1) // 2):
            idx = np.matmul(lines[:, r:], pow3[: L - r]) + pow3[L + 1 - r :].sum()
            for i in range(idx.shape[0]):
                feature_map[idx[i]] = map_table[:, i, L // 2 + r]
                usage_flags[idx[i]] = True

        for l in range(1, (L + 1) // 2):
            idx = np.matmul(lines[:, : L - l], pow3[l:-1]) + 2 * pow3[-1] + pow3[: l - 1].sum()
            for i in range(idx.shape[0]):
                feature_map[idx[i]] = map_table[:, i, L // 2 - l]
                usage_flags[idx[i]] = True

        for l in range(1, (L + 1) // 2):
            for r in range(1, (L + 1) // 2):
                lines = generate_base3_permutation(L - l - r)
                map_table = self._export_map_table(model, device, lines, stm)
                idx_offset = 3 * pow3[-1] + pow3[: l - 1].sum() + pow3[L + 1 - r : -1].sum()
                idx = np.matmul(lines, pow3[l : L - r]) + idx_offset
                for i in range(idx.shape[0]):
                    feature_map[idx[i]] = map_table[:, i, L // 2 - l]
                    usage_flags[idx[i]] = True

        feature_map_max = np.abs(feature_map).max()  # stores the upper bound of the feature map
        scale = self.feature_map_bound / feature_map_max
        bound = ceil(feature_map_max * scale)
        print(
            f"feature map: used {usage_flags.sum()} features of {len(usage_flags)}"
            + f", feature_max = {feature_map_max}, scale = {scale}, bound = {bound}"
        )
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
        print(
            f"map leakyrelu: slope_max = {slope_max}, bias_max = {bias_max}"
            + f", max = {act_max}, scale = {scale}, bound = {bound_perchannel.max()}"
        )
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
        bound_perchannel = (
            np.abs(conv_weight).sum((1, 2, 3)) * bound_perchannel[:PC] + np.abs(conv_bias) * scale
        )
        conv_max = max(conv_weight_max * 2**15, conv_bias_max * scale)
        conv_scale = min(32766 / conv_max, 32766 / bound_perchannel.max())
        scale *= conv_scale
        bound_perchannel *= conv_scale
        conv_max = max(conv_weight_max * conv_scale * 2**15, conv_bias_max * scale)
        print(
            f"policy conv: weight_max = {conv_weight_max}, bias_max = {conv_bias_max}"
            + f", max = {conv_max}, scale = {scale}, bound = {bound_perchannel.max()}"
        )
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
        print(
            f"policy pw conv: weight_max = {pw_conv_weight_max}"
            + f", max = {pw_conv_max}, scale = {scale}, bound = {bound}"
        )
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
        feature_map, usage_flags, scale, bound = self._export_feature_map(
            model, device, stm=self.side_to_move
        )
        map_lr_slope_sub1div8, map_lr_bias, scale, bound_perchannel = self._export_mapping_activation(
            model, scale, bound
        )
        policy_conv_weight, policy_conv_bias, policy_pw_conv_weight, policy_neg_slope, policy_pos_slope = (
            self._export_policy(model, scale, bound_perchannel)
        )
        (
            scale_before_mlp,
            lr_slope_sub1,
            linear1_weight,
            linear1_bias,
            linear2_weight,
            linear2_bias,
            linear_final_weight,
            linear_final_bias,
        ) = self._export_value(model, scale)

        if self.text_output:
            print("featuremap", file=out)
            print(usage_flags.sum(), file=out)
            for i, (f, used) in enumerate(zip(feature_map.astype("i2"), usage_flags)):
                if used:
                    print(i, end=" ", file=out)
                    f.tofile(out, sep=" ")
                    print(file=out)

            print("map_lr_slope_sub1div8", file=out)
            map_lr_slope_sub1div8.astype("i2").tofile(out, sep=" ")
            print(file=out)

            print("map_lr_bias", file=out)
            map_lr_bias.astype("i2").tofile(out, sep=" ")
            print(file=out)

            print("policyConvWeight", file=out)
            policy_conv_weight.astype("i2").tofile(out, sep=" ")
            print(file=out)

            print("policyConvBias", file=out)
            policy_conv_bias.astype("i2").tofile(out, sep=" ")
            print(file=out)

            print("policyFinalConv", file=out)
            policy_pw_conv_weight.astype("i2").tofile(out, sep=" ")
            print(file=out)

            print("policy_neg_slope", file=out)
            print(policy_neg_slope, file=out)
            print("policy_pos_slope", file=out)
            print(policy_pos_slope, file=out)

            print("scale_beforemlp", file=out)
            print(scale_before_mlp, file=out)

            print("value_lr_slope_sub1", file=out)
            lr_slope_sub1.astype("f4").tofile(out, sep=" ")
            print(file=out)

            print("mlp_w1", file=out)
            linear1_weight.astype("f4").tofile(out, sep=" ")
            print(file=out)
            print("mlp_b1", file=out)
            linear1_bias.astype("f4").tofile(out, sep=" ")
            print(file=out)

            print("mlp_w2", file=out)
            linear2_weight.astype("f4").tofile(out, sep=" ")
            print(file=out)
            print("mlp_b2", file=out)
            linear2_bias.astype("f4").tofile(out, sep=" ")
            print(file=out)

            print("mlp_w3", file=out)
            linear_final_weight.astype("f4").tofile(out, sep=" ")
            print(file=out)
            print("mlp_b3", file=out)
            linear_final_bias.astype("f4").tofile(out, sep=" ")
            print(file=out)
        else:
            o: io.RawIOBase = out

            # int16_t map[ShapeNum][FeatureDim];
            o.write(feature_map.astype("<i2").tobytes())  # (708588, 48)

            # int16_t map_lr_slope_sub1div8[FeatureDim];
            # int16_t map_lr_bias[FeatureDim];
            o.write(map_lr_slope_sub1div8.astype("<i2").tobytes())  # (48,)
            o.write(map_lr_bias.astype("<i2").tobytes())  # (48,)

            # int16_t policy_conv_weight[9][PolicyDim];
            # int16_t policy_conv_bias[PolicyDim];
            o.write(policy_conv_weight.astype("<i2").tobytes())  # (3, 3, 16)
            o.write(policy_conv_bias.astype("<i2").tobytes())  # (16,)

            # int16_t policy_final_conv[PolicyDim];
            # float policy_neg_slope, policy_pos_slope;
            o.write(policy_pw_conv_weight.astype("<i2").tobytes())  # (16,)
            o.write(np.array([policy_neg_slope, policy_pos_slope], dtype="<f4").tobytes())

            # float scale_before_mlp;
            # float value_lr_slope_sub1[ValueDim];
            o.write(np.array([scale_before_mlp], dtype="<f4").tobytes())
            o.write(lr_slope_sub1.astype("<f4").tobytes())  # (32,)

            # float mlp_w1[ValueDim][ValueDim];  // shape=(inc, outc)
            # float mlp_b1[ValueDim];
            o.write(linear1_weight.astype("<f4").tobytes())  # (32, 32)
            o.write(linear1_bias.astype("<f4").tobytes())  # (32,)

            # float mlp_w2[ValueDim][ValueDim];
            # float mlp_b2[ValueDim];
            o.write(linear2_weight.astype("<f4").tobytes())  # (32, 32)
            o.write(linear2_bias.astype("<f4").tobytes())  # (32,)

            # float mlp_w3[ValueDim][3];
            # float _mlp_w3_padding[5];
            # float mlp_b3[3];
            # float _mlp_b3_padding[5];
            o.write(linear_final_weight.astype("<f4").tobytes())  # (32, 3)
            o.write(np.zeros(5, dtype="<f4").tobytes())  # padding: (5,)
            o.write(linear_final_bias.astype("<f4").tobytes())  # (3,)
            o.write(np.zeros(5, dtype="<f4").tobytes())  # padding: (5,)


@SERIALIZERS.register("mix7")
class Mix7NetSerializer(BaseSerializer):
    """
    Mix7Net binary serializer.

    The corresponding C-language struct layout:
    struct Mix7Weight {
        // 1  mapping layer
        int16_t mapping[ShapeNum][FeatureDim];

        // 2  PReLU after mapping
        int16_t map_prelu_weight[FeatureDim];

        // 3  Depthwise conv
        int16_t dw_conv_weight[9][FeatureDim];
        int16_t dw_conv_bias[FeatureDim];

        // 4  Policy pointwise conv
        int16_t policy_pw_conv_weight[PolicyDim];

        // 7  Value MLP (layer 1,2,3)
        float   value_l1_weight[ValueDim][ValueDim];  // shape=(in, out)
        float   value_l1_bias[ValueDim];
        float   value_l2_weight[ValueDim][ValueDim];
        float   value_l2_bias[ValueDim];
        float   value_l3_weight[ValueDim][3];
        float   value_l3_bias[3];

        // 5  Policy PReLU
        float   policy_neg_weight;
        float   policy_pos_weight;

        // 6  Value sum scale
        float   value_sum_scale_after_conv;
        float   value_sum_scale_direct;

        char    __padding_to_32bytes[4];
    };
    """

    def __init__(
        self,
        rules,
        boardsizes,
        text_output=False,
        feature_map_bound=6000,
        feature_bound_scale=1.0,
        **kwargs,
    ):
        super().__init__(rules, boardsizes, **kwargs)
        self.line_length = 11
        self.text_output = text_output
        self.feature_map_bound = feature_map_bound
        self.feature_bound_scale = feature_bound_scale
        self.map_table_export_batch_size = 4096

    @property
    def is_binary(self) -> bool:
        return not self.text_output

    def arch_hash(self, model: Mix7Net) -> int:
        assert model.input_type == "basic-nostm"
        _, dim_policy, dim_value = model.model_size
        hash = crc32(b"Mix7Net")
        hash ^= crc32(model.input_type.encode("utf-8"))
        hash ^= crc32(model.dwconv_kernel_size.to_bytes(4, "little"))
        hash ^= (model.dim_dwconv << 24) | (dim_policy << 16) | dim_value
        return hash

    def _export_map_table(self, model: Mix7Net, device, line, stm=None):
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
            if model.map_max != 0:
                map = model.map_max * torch.tanh(map / model.map_max)

            map_table.append(map.cpu().numpy())

        map_table = np.concatenate(map_table, axis=1)  # [C=max(PC,VC), N, L]
        return map_table

    def _export_feature_map(self, model: Mix7Net, device, stm=None):
        L = self.line_length
        _, PC, VC = model.model_size
        lines = generate_base3_permutation(L)  # [177147, 11]
        map_table = self._export_map_table(model, device, lines, stm)  # [C=PC+VC, 177147, 11]

        feature_map = np.zeros((4 * 3**L, max(PC, VC)), dtype=np.float32)  # [708588, 64]
        usage_flags = np.zeros(4 * 3**L, dtype=np.int8)  # [708588], track usage of each feature
        pow3 = 3 ** np.arange(L + 1, dtype=np.int64)[:, np.newaxis]  # [11, 1]

        for r in range((L + 1) // 2):
            idx = np.matmul(lines[:, r:], pow3[: L - r]) + pow3[L + 1 - r :].sum()
            for i in range(idx.shape[0]):
                feature_map[idx[i]] = map_table[:, i, L // 2 + r]
                usage_flags[idx[i]] = True

        for l in range(1, (L + 1) // 2):
            idx = np.matmul(lines[:, : L - l], pow3[l:-1]) + 2 * pow3[-1] + pow3[: l - 1].sum()
            for i in range(idx.shape[0]):
                feature_map[idx[i]] = map_table[:, i, L // 2 - l]
                usage_flags[idx[i]] = True

        for l in range(1, (L + 1) // 2):
            for r in range(1, (L + 1) // 2):
                lines = generate_base3_permutation(L - l - r)
                map_table = self._export_map_table(model, device, lines, stm)
                idx_offset = 3 * pow3[-1] + pow3[: l - 1].sum() + pow3[L + 1 - r : -1].sum()
                idx = np.matmul(lines, pow3[l : L - r]) + idx_offset
                for i in range(idx.shape[0]):
                    feature_map[idx[i]] = map_table[:, i, L // 2 - l]
                    usage_flags[idx[i]] = True

        feature_map_max = np.abs(feature_map).max()
        quant_scale = self.feature_map_bound / feature_map_max
        quant_bound = ceil(feature_map_max * quant_scale)
        print(
            f"feature map: used {usage_flags.sum()} features of {len(usage_flags)}, "
            + f"feature_max = {feature_map_max}, scale = {quant_scale}*4, bound = {quant_bound}*4"
        )
        assert quant_bound * 4 < 32767, "feature map overflow!"

        # fuse mean operation into map by multiply 4
        return feature_map * quant_scale, usage_flags, quant_scale * 4, quant_bound * 4

    def _export_mapping_activation(self, model: Mix7Net):
        weight = model.mapping_activation.weight.cpu().numpy()
        weight_max = np.abs(weight).max()
        weight_clipped = np.clip(weight, a_min=-1.0, a_max=1.0)
        num_params_clipped = np.sum(weight != weight_clipped)
        weight_quant = np.clip(weight_clipped * 2**15, a_min=-32768, a_max=32767)
        weight_quant_max = np.abs(weight_quant).max()
        ascii_hist("map prelu weight", weight)
        print(
            f"map prelu: clipped {num_params_clipped}/{len(weight)}, "
            + f"weight_max = {weight_max}, weight_quant_max = {weight_quant_max}"
        )

        return weight_quant

    def _export_feature_dwconv(self, model: Mix7Net, quant_scale, quant_bound):
        conv_weight = model.feature_dwconv.conv.weight.cpu().numpy()  # [max(PC,VC), 1, 3, 3]
        conv_bias = model.feature_dwconv.conv.bias.cpu().numpy()
        ascii_hist("feature dwconv weight", conv_weight)
        ascii_hist("feature dwconv bias", conv_bias)

        conv_weight_max = np.abs(conv_weight).max()
        conv_bias_max = np.abs(conv_bias).max()
        quant_bound_perchannel = self.feature_bound_scale * np.abs(
            np.abs(conv_weight).sum((1, 2, 3)) * quant_bound + conv_bias * quant_scale
        )

        conv_quant_max = max(conv_weight_max * 2**15, conv_bias_max * quant_scale)
        conv_quant_scale = min(32766 / conv_quant_max, 32766 / quant_bound_perchannel.max())
        quant_scale *= conv_quant_scale
        quant_bound_perchannel *= conv_quant_scale

        conv_weight_quant = conv_weight * conv_quant_scale * 2**15
        conv_bias_quant = conv_bias * quant_scale
        conv_quant_max = max(np.abs(conv_weight_quant).max(), np.abs(conv_bias_quant).max())
        print(
            f"feature dwconv: weight_max = {conv_weight_max}, bias_max = {conv_bias_max}"
            + f", quant_max = {conv_quant_max}, scale = {quant_scale}"
            + f", bound = {quant_bound_perchannel.max()}"
        )
        assert conv_quant_max < 32767, f"feature dwconv weight overflow! ({conv_quant_max})"
        assert (
            quant_bound_perchannel.max() < 32767
        ), f"feature dwconv overflow! ({quant_bound_perchannel.max()})"

        # transpose weight to [9][dim_feature]
        conv_weight_quant = conv_weight_quant.squeeze(1).transpose((1, 2, 0))

        return conv_weight_quant, conv_bias_quant, quant_scale, quant_bound_perchannel

    def _export_policy(self, model: Mix7Net, quant_scale, quant_bound_perchannel):
        _, PC, _ = model.model_size

        # policy pw conv
        pwconv_weight = model.policy_pwconv.conv.weight.cpu().numpy()[0, :, 0, 0]
        pwconv_weight_max = np.abs(pwconv_weight).max()
        quant_bound = (np.abs(pwconv_weight) * quant_bound_perchannel[:PC]).sum()
        pwconv_quant_scale = min(32766 / (pwconv_weight_max * 2**15), 32766 / quant_bound)
        quant_scale *= pwconv_quant_scale
        quant_bound *= pwconv_quant_scale
        pwconv_weight_quant = pwconv_weight * pwconv_quant_scale * 2**15

        pwconv_weight_max = np.abs(pwconv_weight).max()
        pwconv_weight_quant_max = np.abs(pwconv_weight_quant).max()
        print(
            f"policy pwconv: weight_max = {pwconv_weight_max}, weight_quant_max = "
            + f"{pwconv_weight_quant_max}, scale = {quant_scale}, bound = {quant_bound}"
        )
        assert pwconv_weight_quant_max < 32767, "policy pw conv weight overflow!"
        assert quant_bound < 32767, f"policy pw conv overflow! ({quant_bound})"

        # policy PReLU activation
        policy_prelu_weight = model.policy_activation.weight.cpu().numpy().item()
        policy_neg_weight = policy_prelu_weight / quant_scale
        policy_pos_weight = 1 / quant_scale
        print(
            f"policy prelu: weight = {policy_prelu_weight}, "
            + f"neg_weight = {policy_neg_weight}, pos_weight = {policy_pos_weight}"
        )

        return pwconv_weight_quant, policy_neg_weight, policy_pos_weight

    def _export_value(self, model: Mix7Net, quant_scale_after_conv, quant_scale_direct):
        # value layer 0: global mean
        scale_after_conv = 1 / quant_scale_after_conv  # Note: divide board_size**2 in engine
        scale_direct = 1 / quant_scale_direct  # Note: divide board_size**2 in engine

        # value layer 1: linear mlp 01
        linear1_weight = model.value_linear[0].fc.weight.cpu().numpy().T
        linear1_bias = model.value_linear[0].fc.bias.cpu().numpy()

        # value layer 2: linear mlp 02
        linear2_weight = model.value_linear[1].fc.weight.cpu().numpy().T
        linear2_bias = model.value_linear[1].fc.bias.cpu().numpy()

        # value layer 3: linear mlp final
        linear3_weight = model.value_linear[2].fc.weight.cpu().numpy().T
        linear3_bias = model.value_linear[2].fc.bias.cpu().numpy()

        print(f"value: scale_after_conv = {scale_after_conv}, scale_direct = {scale_direct}")
        ascii_hist("value: linear1 weight", linear1_weight)
        ascii_hist("value: linear1 bias", linear1_bias)
        ascii_hist("value: linear2 weight", linear2_weight)
        ascii_hist("value: linear2 bias", linear2_bias)
        ascii_hist("value: linear3 weight", linear3_weight)
        ascii_hist("value: linear3 bias", linear3_bias)

        return (
            scale_after_conv,
            scale_direct,
            linear1_weight,
            linear1_bias,
            linear2_weight,
            linear2_bias,
            linear3_weight,
            linear3_bias,
        )

    def serialize(self, out: io.IOBase, model: Mix7Net, device):
        feature_map, usage_flags, quant_scale_direct, quant_bound = self._export_feature_map(model, device)
        map_prelu_weight = self._export_mapping_activation(model)
        feat_dwconv_weight, feat_dwconv_bias, quant_scale_after_conv, quant_bound_perchannel = (
            self._export_feature_dwconv(model, quant_scale_direct, quant_bound)
        )
        policy_pwconv_weight, policy_neg_weight, policy_pos_weight = self._export_policy(
            model, quant_scale_after_conv, quant_bound_perchannel
        )
        (
            scale_after_conv,
            scale_direct,
            linear1_weight,
            linear1_bias,
            linear2_weight,
            linear2_bias,
            linear3_weight,
            linear3_bias,
        ) = self._export_value(model, quant_scale_after_conv, quant_scale_direct)

        if self.text_output:
            print("featuremap", file=out)
            print(usage_flags.sum(), file=out)
            for i, (f, used) in enumerate(zip(feature_map.astype("i2"), usage_flags)):
                if used:
                    print(i, end=" ", file=out)
                    f.tofile(out, sep=" ")
                    print(file=out)

            print("map_prelu_weight", file=out)
            map_prelu_weight.astype("i2").tofile(out, sep=" ")
            print(file=out)

            print("dw_conv_weight", file=out)
            feat_dwconv_weight.astype("i2").tofile(out, sep=" ")
            print(file=out)

            print("dw_conv_bias", file=out)
            feat_dwconv_bias.astype("i2").tofile(out, sep=" ")
            print(file=out)

            print("policy_pwconv_weight", file=out)
            policy_pwconv_weight.astype("i2").tofile(out, sep=" ")
            print(file=out)

            print("value_l1_weight", file=out)
            linear1_weight.astype("f4").tofile(out, sep=" ")
            print(file=out)
            print("value_l1_bias", file=out)
            linear1_bias.astype("f4").tofile(out, sep=" ")
            print(file=out)

            print("value_l2_weight", file=out)
            linear2_weight.astype("f4").tofile(out, sep=" ")
            print(file=out)
            print("value_l2_bias", file=out)
            linear2_bias.astype("f4").tofile(out, sep=" ")
            print(file=out)

            print("value_l3_weight", file=out)
            linear3_weight.astype("f4").tofile(out, sep=" ")
            print(file=out)
            print("value_l3_bias", file=out)
            linear3_bias.astype("f4").tofile(out, sep=" ")
            print(file=out)

            print("policy_neg_weight", file=out)
            print(policy_neg_weight, file=out)
            print("policy_pos_weight", file=out)
            print(policy_pos_weight, file=out)
            print("value_sum_scale_after_conv", file=out)
            print(scale_after_conv, file=out)
            print("value_sum_scale_direct", file=out)
            print(scale_direct, file=out)
        else:
            o: io.RawIOBase = out

            # int16_t mapping[ShapeNum][FeatureDim];
            o.write(feature_map.astype("<i2").tobytes())  # (708588, max(PC, VC))

            # int16_t map_prelu_weight[FeatureDim];
            o.write(map_prelu_weight.astype("<i2").tobytes())  # (max(PC, VC),)

            # int16_t dw_conv_weight[9][FeatureDim];
            # int16_t dw_conv_bias[FeatureDim];
            o.write(feat_dwconv_weight.astype("<i2").tobytes())  # (3, 3, max(PC, VC))
            o.write(feat_dwconv_bias.astype("<i2").tobytes())  # (max(PC, VC),)

            # int16_t policy_pw_conv_weight[PolicyDim];
            o.write(policy_pwconv_weight.astype("<i2").tobytes())  # (PC,)

            # float   value_l1_weight[ValueDim][ValueDim];  // shape=(in, out)
            # float   value_l1_bias[ValueDim];
            o.write(linear1_weight.astype("<f4").tobytes())  # (VC, VC)
            o.write(linear1_bias.astype("<f4").tobytes())  # (VC,)
            # float   value_l2_weight[ValueDim][ValueDim];
            # float   value_l2_bias[ValueDim];
            o.write(linear2_weight.astype("<f4").tobytes())  # (VC, VC)
            o.write(linear2_bias.astype("<f4").tobytes())  # (VC,)
            # float   value_l3_weight[ValueDim][3];
            # float   value_l3_bias[3];
            o.write(linear3_weight.astype("<f4").tobytes())  # (VC, 3)
            o.write(linear3_bias.astype("<f4").tobytes())  # (3,)

            # float   policy_neg_weight;
            # float   policy_pos_weight;
            o.write(np.array([policy_neg_weight, policy_pos_weight], dtype="<f4").tobytes())
            # float   value_sum_scale_after_conv;
            # float   value_sum_scale_direct;
            o.write(np.array([scale_after_conv, scale_direct], dtype="<f4").tobytes())

            # char    __padding_to_32bytes[4];
            o.write(np.zeros(4, dtype="<i1").tobytes())


@SERIALIZERS.register("mix8")
class Mix8NetSerializer(BaseSerializer):
    """
    Mix8Net binary serializer.

    The corresponding C-language struct layout:
    struct Mix8Weight {
        // 1  mapping layer
        int16_t mapping[ShapeNum][FeatureDim];

        // 2  PReLU after mapping
        int16_t map_prelu_weight[FeatureDim];

        // 3  Depthwise conv
        int16_t feature_dwconv_weight[9][FeatureDWConvDim];
        int16_t feature_dwconv_bias[FeatureDWConvDim];

        // 4  Value sum scale
        float   value_sum_scale_after_conv;
        float   value_sum_scale_direct;

        int32_t num_head_buckets;  // used to validate the number of head buckets
        char    __padding_to_64bytes_0[52];

        struct HeadBucket {
            // 5  Policy dynamic pointwise conv
            float   policy_pwconv_layer_l1_weight[FeatureDim][PolicyDim];
            float   policy_pwconv_layer_l1_bias[PolicyDim];
            float   policy_pwconv_layer_l1_prelu[PolicyDim];
            float   policy_pwconv_layer_l2_weight[PolicyDim][4*PolicyDim];
            float   policy_pwconv_layer_l2_bias[4*PolicyDim];

            // 6  Group Value MLP (layer 1,2)
            float   value_corner_weight[FeatureDim][VGroupDim];
            float   value_corner_bias[VGroupDim];
            float   value_corner_prelu[VGroupDim];
            float   value_edge_weight[FeatureDim][VGroupDim];
            float   value_edge_bias[VGroupDim];
            float   value_edge_prelu[VGroupDim];
            float   value_center_weight[FeatureDim][VGroupDim];
            float   value_center_bias[VGroupDim];
            float   value_center_prelu[VGroupDim];
            float   value_quad_weight[VGroupDim][VGroupDim];
            float   value_quad_bias[VGroupDim];
            float   value_quad_prelu[VGroupDim];

            // 7  Value MLP (layer 1,2,3)
            float   value_l1_weight[FeatureDim+4*VGroupDim][ValueDim]; // shape=(in, out)
            float   value_l1_bias[ValueDim];
            float   value_l2_weight[ValueDim][ValueDim];
            float   value_l2_bias[ValueDim];
            float   value_l3_weight[ValueDim][3];
            float   value_l3_bias[3];

            // 8  Policy Output
            float   policy_output_pos_weight[4];
            float   policy_output_new_weight[4];
            float   policy_output_bias;
            char    __padding_to_64bytes_1[16];
        } buckets[num_head_buckets];
    };
    """

    def __init__(
        self,
        rules,
        boardsizes,
        text_output=False,
        feature_bound_scale=1.0,
        **kwargs,
    ):
        super().__init__(rules=rules, boardsizes=boardsizes, **kwargs)
        self.line_length = 11
        self.text_output = text_output
        self.feature_bound_scale = feature_bound_scale
        self.map_table_export_batch_size = 4096

    @property
    def is_binary(self) -> bool:
        return not self.text_output

    def arch_hash(self, model: Mix8Net) -> int:
        assert model.input_plane.dim_plane == 2
        _, dim_feature, dim_policy, dim_value, dim_value_group = model.model_size
        hash = crc32(b"Mix8Net")
        hash ^= crc32(model.input_type.encode("utf-8"))
        print(f"Mix8 ArchHashBase: {hex(hash)}")

        assert dim_feature % 8 == 0, f"dim_feature must be a multiply of 8"
        assert dim_policy % 8 == 0, f"dim_policy must be a multiply of 8"
        assert dim_value % 8 == 0, f"dim_value must be a multiply of 8"
        assert dim_value_group % 8 == 0, f"dim_value_group must be a multiply of 8"
        assert model.dim_dwconv % 8 == 0, f"dim_dwconv must be a multiply of 8"
        hash ^= (
            (dim_feature // 8)
            | ((dim_policy // 8) << 8)
            | ((dim_value // 8) << 14)
            | ((dim_value_group // 8) << 20)
            | ((model.dim_dwconv // 8) << 26)
        )
        return hash

    def _export_map_table(self, model: Mix8Net, device, line):
        """
        Export line -> feature mapping table.

        Args:
            line: shape (N, Length)
        """
        N, L = line.shape
        b, w = line == 1, line == 2

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
            map = torch.clamp(map, min=-32, max=32)
            map_table.append(map.cpu().numpy())

        map_table = np.concatenate(map_table, axis=1)  # [dim_feature, N, L]
        return map_table

    def _export_feature_map(self, model: Mix8Net, device):
        L = self.line_length
        dim_feature = model.model_size[1]
        lines = generate_base3_permutation(L)  # [177147, 11]
        map_table = self._export_map_table(model, device, lines)  # [dim_feature, 177147, 11]

        feature_map = np.zeros((4 * 3**L, dim_feature), dtype=np.float32)  # [708588, dim_feature]
        usage_flags = np.zeros(4 * 3**L, dtype=np.int8)  # [708588], track usage of each feature
        pow3 = 3 ** np.arange(L + 1, dtype=np.int64)[:, np.newaxis]  # [11, 1]

        for r in range((L + 1) // 2):
            idx = np.matmul(lines[:, r:], pow3[: L - r]) + pow3[L + 1 - r :].sum()
            for i in range(idx.shape[0]):
                feature_map[idx[i]] = map_table[:, i, L // 2 + r]
                usage_flags[idx[i]] = True

        for l in range(1, (L + 1) // 2):
            idx = np.matmul(lines[:, : L - l], pow3[l:-1]) + 2 * pow3[-1] + pow3[: l - 1].sum()
            for i in range(idx.shape[0]):
                feature_map[idx[i]] = map_table[:, i, L // 2 - l]
                usage_flags[idx[i]] = True

        for l in range(1, (L + 1) // 2):
            for r in range(1, (L + 1) // 2):
                lines = generate_base3_permutation(L - l - r)
                map_table = self._export_map_table(model, device, lines)
                idx_offset = 3 * pow3[-1] + pow3[: l - 1].sum() + pow3[L + 1 - r : -1].sum()
                idx = np.matmul(lines, pow3[l : L - r]) + idx_offset
                for i in range(idx.shape[0]):
                    feature_map[idx[i]] = map_table[:, i, L // 2 - l]
                    usage_flags[idx[i]] = True

        feature_map_max = np.abs(feature_map).max()
        quant_scale = (32 * 255) / feature_map_max
        quant_bound = ceil(feature_map_max * quant_scale)
        print(
            f"feature map: used {usage_flags.sum()} features of {len(usage_flags)}, "
            + f"feature_max = {feature_map_max}, scale = {quant_scale}*4, bound = {quant_bound}*4"
        )
        assert quant_bound * 4 <= 32767, f"feature map overflow! (quant_bound = {quant_bound})"
        feature_map_quant = np.around(feature_map * quant_scale)

        # fuse mean operation into map by multiply 4
        return feature_map_quant, usage_flags, quant_scale * 4, quant_bound * 4

    def _export_mapping_activation(self, model: Mix8Net):
        weight = model.mapping_activation.weight.cpu().numpy()
        weight_clipped = np.clip(weight, a_min=-1.0, a_max=1.0)
        num_params_clipped = np.sum(weight != weight_clipped)
        weight_max = np.abs(weight_clipped).max()
        weight_quant = np.clip(np.around(weight_clipped * 2**15), -32768, 32767)
        weight_quant_max = np.abs(weight_quant).max()
        ascii_hist("map prelu weight", weight)
        print(
            f"map prelu: clipped {num_params_clipped}/{len(weight)}, "
            + f"weight_max = {weight_max}, weight_quant_max = {weight_quant_max}"
        )

        return weight_quant

    def _export_feature_dwconv(self, model: Mix8Net, quant_scale, quant_bound):
        conv_weight = model.feature_dwconv.conv.weight.cpu().numpy()  # [FeatureDWConvDim,1,3,3]
        conv_bias = model.feature_dwconv.conv.bias.cpu().numpy()  # [FeatureDWConvDim]
        conv_weight = conv_weight.reshape(conv_weight.shape[0], -1)  # [FeatureDWConvDim, 9]
        ascii_hist("feature dwconv weight", conv_weight)
        ascii_hist("feature dwconv bias", conv_bias)

        conv_weight_clipped = np.clip(conv_weight, a_min=-1.5, a_max=1.5)
        conv_bias_clipped = np.clip(conv_bias, a_min=-4.0, a_max=4.0)
        weight_num_params_clipped = np.sum(conv_weight != conv_weight_clipped)
        bias_num_params_clipped = np.sum(conv_bias != conv_bias_clipped)
        conv_weight_max = np.abs(conv_weight_clipped).max()
        conv_bias_max = np.abs(conv_bias_clipped).max()
        quant_bound = np.abs(
            conv_bias_clipped * quant_scale + np.abs(conv_weight_clipped).sum(1) * quant_bound
        )
        quant_bound = self.feature_bound_scale * quant_bound

        conv_quant_max = max(conv_weight_max * 2**15, conv_bias_max * quant_scale)
        conv_quant_scale = min(32767 / conv_quant_max, 32767 / quant_bound.max())
        quant_scale *= conv_quant_scale
        quant_bound *= conv_quant_scale

        conv_weight_quant = np.around(conv_weight_clipped * conv_quant_scale * 2**15)
        conv_bias_quant = np.around(conv_bias_clipped * quant_scale)
        conv_quant_max = max(np.abs(conv_weight_quant).max(), np.abs(conv_bias_quant).max())
        print(
            f"feature dwconv: weight clipped {weight_num_params_clipped}/{conv_weight.size}"
            + f", bias clipped {bias_num_params_clipped}/{conv_bias.size}"
            + f", weight_max = {conv_weight_max}, bias_max = {conv_bias_max}"
            + f", quant_max = {conv_quant_max}, scale = {quant_scale}"
            + f", bound = {quant_bound.max()} {quant_bound.shape}"
        )
        assert conv_quant_max < 32767.5, f"feature dwconv weight overflow! ({conv_quant_max})"
        assert quant_bound.max() < 32767.5, f"feature dwconv overflow! ({quant_bound.max()})"

        # transpose weight to [9][FeatureDWConvDim]
        conv_weight_quant = conv_weight_quant.transpose()

        return conv_weight_quant, conv_bias_quant, quant_scale, quant_bound

    def _export_policy_pwconv(self, model: Mix8Net, quant_scale):
        # policy pw conv dynamic weight layer 1
        pwconv_layer_l1_weight = model.policy_pwconv_weight_linear[0].fc.weight.cpu().numpy().T
        pwconv_layer_l1_bias = model.policy_pwconv_weight_linear[0].fc.bias.cpu().numpy()
        pwconv_layer_l1_prelu = model.policy_pwconv_weight_linear[1].weight.cpu().numpy()
        ascii_hist("policy pwconv layer l1 weight", pwconv_layer_l1_weight)
        ascii_hist("policy pwconv layer l1 bias", pwconv_layer_l1_bias)
        ascii_hist("policy pwconv layer l1 prelu", pwconv_layer_l1_prelu)

        # policy pw conv dynamic weight layer 2
        pwconv_layer_l2_weight = model.policy_pwconv_weight_linear[2].fc.weight.cpu().numpy().T
        pwconv_layer_l2_bias = model.policy_pwconv_weight_linear[2].fc.bias.cpu().numpy()
        ascii_hist("policy pwconv layer l2 weight", pwconv_layer_l2_weight)
        ascii_hist("policy pwconv layer l2 bias", pwconv_layer_l2_bias)

        # policy PReLU activation
        policy_output_prelu = model.policy_output[0].weight.cpu().numpy()
        policy_output_weight = model.policy_output[1].weight.cpu().squeeze().numpy()
        policy_output_bias = model.policy_output[1].bias.cpu().numpy()
        ascii_hist("policy output prelu", policy_output_prelu)
        ascii_hist("policy output weight", policy_output_weight)
        ascii_hist("policy output bias", policy_output_bias)
        policy_output_pos_weight = policy_output_weight / quant_scale
        policy_output_neg_weight = policy_output_pos_weight * policy_output_prelu
        ascii_hist("policy output pos weight", policy_output_pos_weight)
        ascii_hist("policy output neg weight", policy_output_neg_weight)

        return (
            pwconv_layer_l1_weight,
            pwconv_layer_l1_bias,
            pwconv_layer_l1_prelu,
            pwconv_layer_l2_weight,
            pwconv_layer_l2_bias,
            policy_output_pos_weight,
            policy_output_neg_weight,
            policy_output_bias,
        )

    def _export_value(self, model: Mix8Net, quant_scale_after_conv, quant_scale_direct):
        # feature accumulator mean
        scale_after_conv = 1 / quant_scale_after_conv  # Note: divide board_size**2 in engine
        scale_direct = 1 / quant_scale_direct  # Note: divide board_size**2 in engine
        print(f"value: scale_after_conv = {scale_after_conv}, scale_direct = {scale_direct}")

        # group value layer 1: 3x3 group
        corner_weight = model.value_corner_linear.fc.weight.cpu().numpy().T
        corner_bias = model.value_corner_linear.fc.bias.cpu().numpy()
        corner_prelu = model.value_corner_act.weight.cpu().numpy()
        edge_weight = model.value_edge_linear.fc.weight.cpu().numpy().T
        edge_bias = model.value_edge_linear.fc.bias.cpu().numpy()
        edge_prelu = model.value_edge_act.weight.cpu().numpy()
        center_weight = model.value_center_linear.fc.weight.cpu().numpy().T
        center_bias = model.value_center_linear.fc.bias.cpu().numpy()
        center_prelu = model.value_center_act.weight.cpu().numpy()

        ascii_hist("value: corner linear weight", corner_weight)
        ascii_hist("value: corner linear bias", corner_bias)
        ascii_hist("value: corner linear prelu", corner_prelu)
        ascii_hist("value: edge linear weight", edge_weight)
        ascii_hist("value: edge linear bias", edge_bias)
        ascii_hist("value: edge linear prelu", edge_prelu)
        ascii_hist("value: center linear weight", center_weight)
        ascii_hist("value: center linear bias", center_bias)
        ascii_hist("value: center linear prelu", center_prelu)

        # group value layer 2: quadrant group
        quad_weight = model.value_quad_linear.fc.weight.cpu().numpy().T
        quad_bias = model.value_quad_linear.fc.bias.cpu().numpy()
        quad_prelu = model.value_quad_act.weight.cpu().numpy()

        ascii_hist("value: quad linear weight", quad_weight)
        ascii_hist("value: quad linear bias", quad_bias)
        ascii_hist("value: quad linear prelu", quad_prelu)

        # value layer 1: linear mlp 01
        l1_weight = model.value_linear[0].fc.weight.cpu().numpy().T
        l1_bias = model.value_linear[0].fc.bias.cpu().numpy()

        # value layer 2: linear mlp 02
        l2_weight = model.value_linear[1].fc.weight.cpu().numpy().T
        l2_bias = model.value_linear[1].fc.bias.cpu().numpy()

        # value layer 3: linear mlp final
        l3_weight = model.value_linear[2].fc.weight.cpu().numpy().T
        l3_bias = model.value_linear[2].fc.bias.cpu().numpy()

        ascii_hist("value: linear1 weight", l1_weight)
        ascii_hist("value: linear1 bias", l1_bias)
        ascii_hist("value: linear2 weight", l2_weight)
        ascii_hist("value: linear2 bias", l2_bias)
        ascii_hist("value: linear3 weight", l3_weight)
        ascii_hist("value: linear3 bias", l3_bias)

        return (
            scale_after_conv,
            scale_direct,
            corner_weight,
            corner_bias,
            corner_prelu,
            edge_weight,
            edge_bias,
            edge_prelu,
            center_weight,
            center_bias,
            center_prelu,
            quad_weight,
            quad_bias,
            quad_prelu,
            l1_weight,
            l1_bias,
            l2_weight,
            l2_bias,
            l3_weight,
            l3_bias,
        )

    def serialize(self, out: io.IOBase, model: Mix8Net, device):
        feature_map, usage_flags, quant_scale_direct, quant_bound = self._export_feature_map(model, device)
        map_prelu_weight = self._export_mapping_activation(model)
        feat_dwconv_weight, feat_dwconv_bias, quant_scale_ftconv, quant_bound_perchannel = (
            self._export_feature_dwconv(model, quant_scale_direct, quant_bound)
        )
        (
            policy_pwconv_layer_l1_weight,
            policy_pwconv_layer_l1_bias,
            policy_pwconv_layer_l1_prelu,
            policy_pwconv_layer_l2_weight,
            policy_pwconv_layer_l2_bias,
            policy_output_pos_weight,
            policy_output_neg_weight,
            policy_output_bias,
        ) = self._export_policy_pwconv(model, quant_scale_ftconv)
        (
            scale_after_conv,
            scale_direct,
            corner_weight,
            corner_bias,
            corner_prelu,
            edge_weight,
            edge_bias,
            edge_prelu,
            center_weight,
            center_bias,
            center_prelu,
            quad_weight,
            quad_bias,
            quad_prelu,
            l1_weight,
            l1_bias,
            l2_weight,
            l2_bias,
            l3_weight,
            l3_bias,
        ) = self._export_value(model, quant_scale_ftconv, quant_scale_direct)

        if self.text_output:
            print("featuremap", file=out)
            print(usage_flags.sum(), file=out)
            for i, (f, used) in enumerate(zip(feature_map.astype("i2"), usage_flags)):
                if used:
                    print(i, end=" ", file=out)
                    f.tofile(out, sep=" ")
                    print(file=out)

            print("map_prelu_weight", file=out)
            map_prelu_weight.astype("i2").tofile(out, sep=" ")
            print(file=out)

            print("feature_dwconv_weight", file=out)
            feat_dwconv_weight.astype("i2").tofile(out, sep=" ")
            print(file=out)

            print("feature_dwconv_bias", file=out)
            feat_dwconv_bias.astype("i2").tofile(out, sep=" ")
            print(file=out)

            print("value_sum_scale_after_conv", file=out)
            print(scale_after_conv, file=out)
            print("value_sum_scale_direct", file=out)
            print(scale_direct, file=out)
            print("num_head_buckets", file=out)
            print(1, file=out)

            print("policy_pwconv_layer_l1_weight", file=out)
            policy_pwconv_layer_l1_weight.astype("f4").tofile(out, sep=" ")
            print(file=out)
            print("policy_pwconv_layer_l1_bias", file=out)
            policy_pwconv_layer_l1_bias.astype("f4").tofile(out, sep=" ")
            print(file=out)
            print("policy_pwconv_layer_l1_prelu", file=out)
            policy_pwconv_layer_l1_prelu.astype("f4").tofile(out, sep=" ")
            print(file=out)
            print("policy_pwconv_layer_l2_weight", file=out)
            policy_pwconv_layer_l2_weight.astype("f4").tofile(out, sep=" ")
            print(file=out)
            print("policy_pwconv_layer_l2_bias", file=out)
            policy_pwconv_layer_l2_bias.astype("f4").tofile(out, sep=" ")
            print(file=out)

            print("value_corner_weight", file=out)
            corner_weight.astype("f4").tofile(out, sep=" ")
            print(file=out)
            print("value_corner_bias", file=out)
            corner_bias.astype("f4").tofile(out, sep=" ")
            print(file=out)
            print("value_corner_prelu", file=out)
            corner_prelu.astype("f4").tofile(out, sep=" ")
            print(file=out)

            print("value_edge_weight", file=out)
            edge_weight.astype("f4").tofile(out, sep=" ")
            print(file=out)
            print("value_edge_bias", file=out)
            edge_bias.astype("f4").tofile(out, sep=" ")
            print(file=out)
            print("value_edge_prelu", file=out)
            edge_prelu.astype("f4").tofile(out, sep=" ")
            print(file=out)

            print("value_center_weight", file=out)
            center_weight.astype("f4").tofile(out, sep=" ")
            print(file=out)
            print("value_center_bias", file=out)
            center_bias.astype("f4").tofile(out, sep=" ")
            print(file=out)
            print("value_center_prelu", file=out)
            center_prelu.astype("f4").tofile(out, sep=" ")
            print(file=out)

            print("value_quad_weight", file=out)
            quad_weight.astype("f4").tofile(out, sep=" ")
            print(file=out)
            print("value_quad_bias", file=out)
            quad_bias.astype("f4").tofile(out, sep=" ")
            print(file=out)
            print("value_quad_prelu", file=out)
            quad_prelu.astype("f4").tofile(out, sep=" ")
            print(file=out)

            print("value_l1_weight", file=out)
            l1_weight.astype("f4").tofile(out, sep=" ")
            print(file=out)
            print("value_l1_bias", file=out)
            l1_bias.astype("f4").tofile(out, sep=" ")
            print(file=out)

            print("value_l2_weight", file=out)
            l2_weight.astype("f4").tofile(out, sep=" ")
            print(file=out)
            print("value_l2_bias", file=out)
            l2_bias.astype("f4").tofile(out, sep=" ")
            print(file=out)

            print("value_l3_weight", file=out)
            l3_weight.astype("f4").tofile(out, sep=" ")
            print(file=out)
            print("value_l3_bias", file=out)
            l3_bias.astype("f4").tofile(out, sep=" ")
            print(file=out)

            print("policy_output_pos_weight", file=out)
            policy_output_pos_weight.astype("f4").tofile(out, sep=" ")
            print("policy_output_neg_weight", file=out)
            policy_output_neg_weight.astype("f4").tofile(out, sep=" ")
            print("policy_output_bias", file=out)
            policy_output_bias.astype("f4").tofile(out, sep=" ")
        else:
            o: io.RawIOBase = out

            # int16_t mapping[ShapeNum][FeatureDim];
            o.write(feature_map.astype("<i2").tobytes())  # (708588, FC)

            # int16_t map_prelu_weight[FeatureDim];
            o.write(map_prelu_weight.astype("<i2").tobytes())  # (FC,)

            # int16_t feature_dwconv_weight[9][FeatureDWConvDim];
            # int16_t feature_dwconv_bias[FeatureDWConvDim];
            o.write(feat_dwconv_weight.astype("<i2").tobytes())  # (9, DC)
            o.write(feat_dwconv_bias.astype("<i2").tobytes())  # (DC,)

            # float   value_sum_scale_after_conv;
            # float   value_sum_scale_direct;
            o.write(np.array([scale_after_conv, scale_direct], dtype="<f4").tobytes())
            # int32_t numHeadBuckets;
            o.write(np.array([1], dtype="<i4").tobytes())
            # char    __padding_to_64bytes_0[52];
            o.write(np.zeros(52, dtype="<i1").tobytes())

            # float   policy_pwconv_layer_l1_weight[FeatureDim][PolicyDim];
            # float   policy_pwconv_layer_l1_bias[PolicyDim];
            # float   policy_pwconv_layer_l1_prelu[PolicyDim];
            # float   policy_pwconv_layer_l2_weight[PolicyDim][4*PolicyDim];
            # float   policy_pwconv_layer_l2_bias[4*PolicyDim];
            o.write(policy_pwconv_layer_l1_weight.astype("<f4").tobytes())  # (VSum, PC)
            o.write(policy_pwconv_layer_l1_bias.astype("<f4").tobytes())  # (PC,)
            o.write(policy_pwconv_layer_l1_prelu.astype("<f4").tobytes())  # (PC,)
            o.write(policy_pwconv_layer_l2_weight.astype("<f4").tobytes())  # (4*PC,PC)
            o.write(policy_pwconv_layer_l2_bias.astype("<f4").tobytes())  # (4*PC)

            # float   value_corner_weight[FeatureDim][VGroupDim];
            # float   value_corner_bias[VGroupDim];
            # float   value_corner_prelu[VGroupDim];
            # float   value_edge_weight[FeatureDim][VGroupDim];
            # float   value_edge_bias[VGroupDim];
            # float   value_edge_prelu[VGroupDim];
            # float   value_center_weight[FeatureDim][VGroupDim];
            # float   value_center_bias[VGroupDim];
            # float   value_center_prelu[VGroupDim];
            # float   value_quad_weight[VGroupDim][VGroupDim];
            # float   value_quad_bias[VGroupDim];
            # float   value_quad_prelu[VGroupDim];
            o.write(corner_weight.astype("<f4").tobytes())  # (FeatureDim, VGroupDim)
            o.write(corner_bias.astype("<f4").tobytes())  # (VGroupDim,)
            o.write(corner_prelu.astype("<f4").tobytes())  # (VGroupDim,)
            o.write(edge_weight.astype("<f4").tobytes())  # (FeatureDim, VGroupDim)
            o.write(edge_bias.astype("<f4").tobytes())  # (VGroupDim,)
            o.write(edge_prelu.astype("<f4").tobytes())  # (VGroupDim,)
            o.write(center_weight.astype("<f4").tobytes())  # (FeatureDim, VGroupDim)
            o.write(center_bias.astype("<f4").tobytes())  # (VGroupDim,)
            o.write(center_prelu.astype("<f4").tobytes())  # (VGroupDim,)
            o.write(quad_weight.astype("<f4").tobytes())  # (VGroupDim, VGroupDim)
            o.write(quad_bias.astype("<f4").tobytes())  # (VGroupDim,)
            o.write(quad_prelu.astype("<f4").tobytes())  # (VGroupDim,)

            # float   value_l1_weight[FeatureDim+4*VGroupDim][ValueDim];
            # float   value_l1_bias[ValueDim];
            o.write(l1_weight.astype("<f4").tobytes())  # (FeatureDim+4*VGroupDim, VC)
            o.write(l1_bias.astype("<f4").tobytes())  # (VC,)
            # float   value_l2_weight[ValueDim][ValueDim];
            # float   value_l2_bias[ValueDim];
            o.write(l2_weight.astype("<f4").tobytes())  # (VC, VC)
            o.write(l2_bias.astype("<f4").tobytes())  # (VC,)
            # float   value_l3_weight[ValueDim][3];
            # float   value_l3_bias[3];
            o.write(l3_weight.astype("<f4").tobytes())  # (VC, 3)
            o.write(l3_bias.astype("<f4").tobytes())  # (3,)

            # float   policy_output_pos_weight[4];
            # float   policy_output_neg_weight[4];
            # float   policy_output_bias;
            o.write(policy_output_pos_weight.astype("<f4").tobytes())  # (4,)
            o.write(policy_output_neg_weight.astype("<f4").tobytes())  # (4,)
            o.write(policy_output_bias.astype("<f4").tobytes())  # (1,)
            # char    __padding_to_64bytes_1[16];
            o.write(np.zeros(16, dtype="<i1").tobytes())


@SERIALIZERS.register("mix9")
class Mix9NetSerializer(BaseSerializer):
    """
    Mix9Net binary serializer.

    The corresponding C++ language struct layout:
    template <int OutSize, int InSize>
    struct StarBlockWeight
    {
        int8_t  value_corner_up1_weight[(OutSize * 2) * InSize];
        int32_t value_corner_up1_bias[(OutSize * 2)];
        int8_t  value_corner_up2_weight[(OutSize * 2) * InSize];
        int32_t value_corner_up2_bias[(OutSize * 2)];
        int8_t  value_corner_down_weight[OutSize * OutSize];
        int32_t value_corner_down_bias[OutSize];
    };

    struct Mix9Weight {
        // 1  mapping layer
        int16_t mapping[2][ShapeNum][FeatureDim];

        // 2  Depthwise conv
        int16_t feature_dwconv_weight[9][FeatDWConvDim];
        int16_t feature_dwconv_bias[FeatDWConvDim];

        struct HeadBucket {
            // 3  Policy dynamic pointwise conv
            int8_t  policy_pwconv_layer_l1_weight[(PolicyDim * 2) * FeatureDim];
            int32_t policy_pwconv_layer_l1_bias[PolicyDim * 2];
            int8_t  policy_pwconv_layer_l2_weight[(PolicyPWConvDim * PolicyDim + PolicyPWConvDim) * (PolicyDim * 2)];
            int32_t policy_pwconv_layer_l2_bias[(PolicyPWConvDim * PolicyDim + PolicyPWConvDim)];

            // 4  Value Group MLP (layer 1,2)
            StarBlockWeight<ValueDim, FeatureDim> value_corner;
            StarBlockWeight<ValueDim, FeatureDim> value_edge;
            StarBlockWeight<ValueDim, FeatureDim> value_center;
            StarBlockWeight<ValueDim, ValueDim>   value_quad;

            // 5  Value MLP (layer 1,2,3)
            int8_t  value_l1_weight[ValueDim * (FeatureDim + ValueDim * 4)];
            int32_t value_l1_bias[ValueDim];
            int8_t  value_l2_weight[ValueDim * ValueDim];
            int32_t value_l2_bias[ValueDim];
            int8_t  value_l3_weight[4 * ValueDim];
            int32_t value_l3_bias[4];

            // 6  Policy output linear
            float policy_output_weight[16];
            float policy_output_bias;
            char  __padding_to_64bytes_1[44];
        } buckets[NumHeadBucket];
    };

    If no_star_block is enabled, then the struct layout will be:
    struct Mix9Weight {
        // 1  mapping layer
        int16_t mapping[2][ShapeNum][FeatureDim];

        // 2  Depthwise conv
        int16_t feature_dwconv_weight[9][FeatDWConvDim];
        int16_t feature_dwconv_bias[FeatDWConvDim];

        struct HeadBucket {
            // 3  Policy dynamic pointwise conv
            int8_t  policy_pwconv_layer_l1_weight[(PolicyDim * 2) * FeatureDim];
            int32_t policy_pwconv_layer_l1_bias[PolicyDim * 2];
            int8_t  policy_pwconv_layer_l2_weight[(PolicyPWConvDim * PolicyDim + PolicyPWConvDim) * (PolicyDim * 2)];
            int32_t policy_pwconv_layer_l2_bias[(PolicyPWConvDim * PolicyDim + PolicyPWConvDim)];

            // 4  Value Group MLP (layer 1,2)
            int8_t  value_corner_weight[ValueDim * FeatureDim];
            int32_t value_corner_bias[ValueDim];
            int8_t  value_edge_weight[ValueDim * FeatureDim];
            int32_t value_edge_bias[ValueDim];
            int8_t  value_center_weight[ValueDim * FeatureDim];
            int32_t value_center_bias[ValueDim];
            int8_t  value_quad_weight[ValueDim * ValueDim];
            int32_t value_quad_bias[ValueDim];

            // 5  Value MLP (layer 1,2,3)
            int8_t  value_l1_weight[ValueDim * (FeatureDim + ValueDim * 4)];
            int32_t value_l1_bias[ValueDim];
            int8_t  value_l2_weight[ValueDim * ValueDim];
            int32_t value_l2_bias[ValueDim];
            int8_t  value_l3_weight[4 * ValueDim];
            int32_t value_l3_bias[4];

            // 6  Policy output linear
            float policy_output_weight[16];
            float policy_output_bias;
            char  __padding_to_64bytes_1[44];
        } buckets[NumHeadBucket];
    };

    If no_dynamic_pwconv is enabled, then the struct layout will be:
    struct Mix9Weight {
        // 1  mapping layer
        int16_t mapping[2][ShapeNum][FeatureDim];

        // 2  Depthwise conv
        int16_t feature_dwconv_weight[9][FeatDWConvDim];
        int16_t feature_dwconv_bias[FeatDWConvDim];

        struct HeadBucket {
            // 3  Policy pointwise conv
            int16_t policy_pwconv_weight[PolicyPWConvDim * PolicyDim];
            int32_t policy_pwconv_bias[PolicyPWConvDim];

            // 4  Value Group MLP (layer 1,2)
            StarBlockWeight<ValueDim, FeatureDim> value_corner;
            StarBlockWeight<ValueDim, FeatureDim> value_edge;
            StarBlockWeight<ValueDim, FeatureDim> value_center;
            StarBlockWeight<ValueDim, ValueDim>   value_quad;

            // 5  Value MLP (layer 1,2,3)
            int8_t  value_l1_weight[ValueDim * (FeatureDim + ValueDim * 4)];
            int32_t value_l1_bias[ValueDim];
            int8_t  value_l2_weight[ValueDim * ValueDim];
            int32_t value_l2_bias[ValueDim];
            int8_t  value_l3_weight[4 * ValueDim];
            int32_t value_l3_bias[4];

            // 6  Policy output linear
            float policy_output_weight[16];
            float policy_output_bias;
            char  __padding_to_64bytes_1[44];
        } buckets[NumHeadBucket];
    };

    If no_value_group is enabled, then the struct layout will be:
    struct Mix9Weight {
        // 1  mapping layer
        int16_t mapping[2][ShapeNum][FeatureDim];

        // 2  Depthwise conv
        int16_t feature_dwconv_weight[9][FeatDWConvDim];
        int16_t feature_dwconv_bias[FeatDWConvDim];

        struct HeadBucket {
            // 3  Policy dynamic pointwise conv
            int8_t  policy_pwconv_layer_l1_weight[(PolicyDim * 2) * FeatureDim];
            int32_t policy_pwconv_layer_l1_bias[PolicyDim * 2];
            int8_t  policy_pwconv_layer_l2_weight[(PolicyPWConvDim * PolicyDim + PolicyPWConvDim) * (PolicyDim * 2)];
            int32_t policy_pwconv_layer_l2_bias[(PolicyPWConvDim * PolicyDim + PolicyPWConvDim)];

            // 4  Value MLP (layer 1,2,3)
            int8_t  value_l1_weight[ValueDim * FeatureDim];
            int32_t value_l1_bias[ValueDim];
            int8_t  value_l2_weight[ValueDim * ValueDim];
            int32_t value_l2_bias[ValueDim];
            int8_t  value_l3_weight[4 * ValueDim];
            int32_t value_l3_bias[4];

            // 5  Policy output linear
            float policy_output_weight[16];
            float policy_output_bias;
            char  __padding_to_64bytes_1[44];
        } buckets[NumHeadBucket];
    };
    """

    def __init__(self, rules, boardsizes, text_output=False, **kwargs):
        super().__init__(rules, boardsizes, **kwargs)
        self.line_length = 11
        self.text_output = text_output
        self.map_table_export_batch_size = 4096

    @property
    def is_binary(self) -> bool:
        return not self.text_output

    def arch_hash(self, model: Mix9Net) -> int:
        assert model.input_plane.dim_plane == 2
        _, dim_feature, dim_policy, dim_value, dim_dwconv = model.model_size
        hash = crc32(b"Mix9Net")
        hash ^= crc32(model.input_type.encode("utf-8"))
        print(f"Mix9 ArchHashBase: {hex(hash)}")

        assert dim_feature % 8 == 0, f"dim_feature must be a multiply of 8"
        assert dim_policy % 8 == 0, f"dim_policy must be a multiply of 8"
        assert dim_value % 8 == 0, f"dim_value must be a multiply of 8"
        assert dim_dwconv % 8 == 0, f"dim_dwconv must be a multiply of 8"
        hash ^= (
            (dim_feature // 8)
            | ((dim_policy // 8) << 8)
            | ((dim_value // 8) << 14)
            | ((dim_dwconv // 8) << 20)
        )
        return hash

    def _export_map_table(self, model: Mix9Net, device, line, mapping_idx):
        """
        Export line -> feature mapping table.

        Args:
            line: shape (N, Length)
        """
        N, L = line.shape
        us, opponent = line == 1, line == 2
        line = np.stack((us, opponent), axis=0)[np.newaxis]  # [1, C=2, N, L]
        line = torch.tensor(line, dtype=torch.float32, device=device)

        batch_size = self.map_table_export_batch_size
        batch_num = 1 + (N - 1) // batch_size
        map_table = []
        for i in range(batch_num):
            start = i * batch_size
            end = min((i + 1) * batch_size, N)

            mapping = getattr(model, f"mapping{mapping_idx}")
            feature = mapping(line[:, :, start:end], dirs=[0])[0, 0]  # [dim_feature, batch, L]
            feature = torch.permute(feature, (1, 2, 0)).cpu().numpy()  # [batch, L, dim_feature]
            map_table.append(feature)

        map_table = np.concatenate(map_table, axis=0)  # [N, L, dim_feature]
        return map_table

    def _export_feature_map(self, model: Mix9Net, device, mapping_idx):
        # use line encoding to generate feature map
        from dataset.pipeline.line_encoding import (
            get_total_num_encoding,
            get_encoding_usage_flags,
            transform_lines_to_line_encoding,
        )

        L = self.line_length
        dim_feature = model.model_size[1]
        num_encoding = get_total_num_encoding(L)
        usage_flags = get_encoding_usage_flags(L)  # [N] bool, track usage of each feature
        feature_map = np.zeros((num_encoding, dim_feature), dtype=np.float32)  # [N, dim_feature]

        # generate line features
        for l in reversed(range(1, L + 1)):
            lines = generate_base3_permutation(l)
            idxs = transform_lines_to_line_encoding(lines, L)  # [n, 11]
            map_table = self._export_map_table(model, device, lines, mapping_idx)
            rows = np.arange(idxs.shape[0])[:, None]  # [n, 1]
            feature_map[idxs, :] = map_table[rows, np.arange(l), :]

        ascii_hist(f"feature map {mapping_idx}", feature_map)
        feature_map_clipped = np.clip(feature_map, a_min=-16, a_max=511 / 32)
        num_params_clipped = np.sum(feature_map != feature_map_clipped)
        feature_map_quant = np.around(feature_map_clipped * 32).astype(np.int16)  # [-512, 511]
        print(
            f"feature map: used {usage_flags.sum()} features of {len(usage_flags)}, "
            + f": clipped {num_params_clipped}/{feature_map.size}"
            + f", quant_range = {(feature_map_quant.min(), feature_map_quant.max())}"
        )
        return feature_map_quant, usage_flags

    def _export_feature_dwconv(self, model: Mix9Net):
        conv_weight = model.feature_dwconv.conv.weight.cpu().numpy()  # [FeatureDWConvDim,1,3,3]
        conv_bias = model.feature_dwconv.conv.bias.cpu().numpy()  # [FeatureDWConvDim]
        conv_weight = conv_weight.reshape(conv_weight.shape[0], -1).transpose()  # [9, FeatureDWConvDim]
        ascii_hist("feature dwconv weight", conv_weight)
        ascii_hist("feature dwconv bias", conv_bias)

        conv_weight_clipped = np.clip(conv_weight, a_min=-32768 / 65536, a_max=32767 / 65536)
        conv_bias_clipped = np.clip(
            conv_bias, a_min=-64, a_max=64
        )  # not too large, otherwise it may overflow
        weight_num_params_clipped = np.sum(conv_weight != conv_weight_clipped)
        bias_num_params_clipped = np.sum(conv_bias != conv_bias_clipped)
        conv_weight_quant = np.clip(np.around(conv_weight_clipped * 65536), -32768, 32767).astype(np.int16)
        conv_bias_quant = np.clip(np.around(conv_bias_clipped * 128), -32768, 32767).astype(np.int16)
        print(
            f"feature dwconv: weight clipped {weight_num_params_clipped}/{conv_weight.size}"
            + f", bias clipped {bias_num_params_clipped}/{conv_bias.size}"
            + f", weight_quant_range = {(conv_weight_quant.min(), conv_weight_quant.max())}"
            + f", bias_quant_range = {(conv_bias_quant.min(), conv_bias_quant.max())}"
        )

        # Make sure that the dwconv will not overflow
        assert np.all(
            np.abs(conv_weight_clipped).sum(0) / 2 * 16 * 4 * 128 < 32767
        ), f"feature dwconv would overflow! (maxsum={np.abs(conv_weight_clipped).sum(0).max()})"

        return conv_weight_quant, conv_bias_quant

    def _export_policy_pwconv(self, model: Mix9Net):
        if model.no_dynamic_pwconv:
            dwconv_weight = (
                model.policy_pwconv.conv.weight.cpu().squeeze().numpy()
            )  # [PolicyPWConvDim, FeatureDim]
            dwconv_bias = model.policy_pwconv.conv.bias.cpu().numpy()  # [PolicyPWConvDim]
        else:
            # policy pw conv dynamic weight layer 1
            l1_weight = model.policy_pwconv_weight_linear[0].fc.weight.cpu().numpy()
            l1_bias = model.policy_pwconv_weight_linear[0].fc.bias.cpu().numpy()
            ascii_hist("policy pwconv layer l1 weight", l1_weight)
            ascii_hist("policy pwconv layer l1 bias", l1_bias)

            # policy pw conv dynamic weight layer 2
            l2_weight = model.policy_pwconv_weight_linear[1].fc.weight.cpu().numpy()
            l2_bias = model.policy_pwconv_weight_linear[1].fc.bias.cpu().numpy()
            ascii_hist("policy pwconv layer l2 weight", l2_weight)
            ascii_hist("policy pwconv layer l2 bias", l2_bias)

        # policy PReLU activation
        policy_output_weight = model.policy_output.weight.cpu().squeeze().numpy()
        policy_output_bias = model.policy_output.bias.cpu().numpy()
        ascii_hist("policy output weight", policy_output_weight)
        ascii_hist("policy output bias", policy_output_bias)

        if model.no_dynamic_pwconv:
            return (
                np.clip(np.around(dwconv_weight * (128 * 128)), -32768, 32767).astype(np.int16),
                np.clip(np.around(dwconv_bias * (128 * 128 * 128)), -(2**31), 2**31 - 1).astype(np.int32),
                policy_output_weight / (128 * 128 * 128),
                policy_output_bias,
            )
        else:
            return (
                np.clip(np.around(l1_weight * 128), -128, 127).astype(np.int8),
                np.clip(np.around(l1_bias * 128 * 128), -(2**31), 2**31 - 1).astype(np.int32),
                np.clip(np.around(l2_weight * 128), -128, 127).astype(np.int8),
                np.clip(np.around(l2_bias * 128 * 128), -(2**31), 2**31 - 1).astype(np.int32),
                policy_output_weight / (128 * 128 * 128),
                policy_output_bias,
            )

    def _export_star_block(self, model: Mix9Net, prefix: str):
        block = getattr(model, prefix)
        up1_weight = block.up1.fc.weight.cpu().numpy()
        up1_bias = block.up1.fc.bias.cpu().numpy()
        up2_weight = block.up2.fc.weight.cpu().numpy()
        up2_bias = block.up2.fc.bias.cpu().numpy()
        down_weight = block.down.fc.weight.cpu().numpy()
        down_bias = block.down.fc.bias.cpu().numpy()
        ascii_hist(f"{prefix}: up1 weight", up1_weight)
        ascii_hist(f"{prefix}: up1 bias", up1_bias)
        ascii_hist(f"{prefix}: up2 weight", up2_weight)
        ascii_hist(f"{prefix}: up2 bias", up2_bias)
        ascii_hist(f"{prefix}: down weight", down_weight)
        ascii_hist(f"{prefix}: down bias", down_bias)

        return (
            np.clip(np.around(up1_weight * 128), -128, 127).astype(np.int8),
            np.clip(np.around(up1_bias * 128 * 128), -(2**31), 2**31 - 1).astype(np.int32),
            np.clip(np.around(up2_weight * 128), -128, 127).astype(np.int8),
            np.clip(np.around(up2_bias * 128 * 128), -(2**31), 2**31 - 1).astype(np.int32),
            np.clip(np.around(down_weight * 128), -128, 127).astype(np.int8),
            np.clip(np.around(down_bias * 128 * 128), -(2**31), 2**31 - 1).astype(np.int32),
        )

    def _export_linear(self, model: Mix9Net, prefix: str):
        block = getattr(model, prefix)
        weight = block.fc.weight.cpu().numpy()
        bias = block.fc.bias.cpu().numpy()
        ascii_hist(f"{prefix}: weight", weight)
        ascii_hist(f"{prefix}: bias", bias)

        return (
            np.clip(np.around(weight * 128), -128, 127).astype(np.int8),
            np.clip(np.around(bias * 128 * 128), -(2**31), 2**31 - 1).astype(np.int32),
        )

    def _export_value(self, model: Mix9Net):
        # value layers
        l1_weight = model.value_linear[0].fc.weight.cpu().numpy()
        l1_bias = model.value_linear[0].fc.bias.cpu().numpy()
        l2_weight = model.value_linear[1].fc.weight.cpu().numpy()
        l2_bias = model.value_linear[1].fc.bias.cpu().numpy()
        l3_weight = model.value_linear[2].fc.weight.cpu().numpy()
        l3_bias = model.value_linear[2].fc.bias.cpu().numpy()
        ascii_hist("value: linear1 weight", l1_weight)
        ascii_hist("value: linear1 bias", l1_bias)
        ascii_hist("value: linear2 weight", l2_weight)
        ascii_hist("value: linear2 bias", l2_bias)
        ascii_hist("value: linear3 weight", l3_weight)
        ascii_hist("value: linear3 bias", l3_bias)

        return (
            np.clip(np.around(l1_weight * 128), -128, 127).astype(np.int8),
            np.clip(np.around(l1_bias * 128 * 128), -(2**31), 2**31 - 1).astype(np.int32),
            np.clip(np.around(l2_weight * 128), -128, 127).astype(np.int8),
            np.clip(np.around(l2_bias * 128 * 128), -(2**31), 2**31 - 1).astype(np.int32),
            np.clip(np.around(l3_weight * 128), -128, 127).astype(np.int8),
            np.clip(np.around(l3_bias * 128 * 128), -(2**31), 2**31 - 1).astype(np.int32),
        )

    def serialize(self, out: io.IOBase, model: Mix9Net, device):
        if model.one_mapping:
            feature_map0, usage_flags0 = self._export_feature_map(model, device, 0)
        else:
            feature_map1, usage_flags1 = self._export_feature_map(model, device, 1)
            feature_map2, usage_flags2 = self._export_feature_map(model, device, 2)
        feat_dwconv_weight, feat_dwconv_bias = self._export_feature_dwconv(model)
        if model.no_dynamic_pwconv:
            policy_pwconv_weight, policy_pwconv_bias, policy_output_weight, policy_output_bias = (
                self._export_policy_pwconv(model)
            )
        else:
            (
                policy_pwconv_layer_l1_weight,
                policy_pwconv_layer_l1_bias,
                policy_pwconv_layer_l2_weight,
                policy_pwconv_layer_l2_bias,
                policy_output_weight,
                policy_output_bias,
            ) = self._export_policy_pwconv(model)
        if model.no_value_group:
            pass
        elif model.no_star_block:
            corner_weight, corner_bias = self._export_linear(model, "value_corner")
            edge_weight, edge_bias = self._export_linear(model, "value_edge")
            center_weight, center_bias = self._export_linear(model, "value_center")
            quad_weight, quad_bias = self._export_linear(model, "value_quad")
        else:
            corner_weights = self._export_star_block(model, "value_corner")
            edge_weights = self._export_star_block(model, "value_edge")
            center_weights = self._export_star_block(model, "value_center")
            quad_weights = self._export_star_block(model, "value_quad")
        l1_weight, l1_bias, l2_weight, l2_bias, l3_weight, l3_bias = self._export_value(model)

        if self.text_output:
            if model.one_mapping:
                print("featuremap0", file=out)
                print(usage_flags0.sum(), file=out)
                for i, (f, used) in enumerate(zip(feature_map0.astype("i2"), usage_flags0)):
                    if used:
                        print(i, end=" ", file=out)
                        f.tofile(out, sep=" ")
                        print(file=out)
            else:
                print("featuremap1", file=out)
                print(usage_flags1.sum(), file=out)
                for i, (f, used) in enumerate(zip(feature_map1.astype("i2"), usage_flags1)):
                    if used:
                        print(i, end=" ", file=out)
                        f.tofile(out, sep=" ")
                        print(file=out)

                print("featuremap2", file=out)
                print(usage_flags2.sum(), file=out)
                for i, (f, used) in enumerate(zip(feature_map2.astype("i2"), usage_flags2)):
                    if used:
                        print(i, end=" ", file=out)
                        f.tofile(out, sep=" ")
                        print(file=out)

            print("feature_dwconv_weight", file=out)
            feat_dwconv_weight.astype("i2").tofile(out, sep=" ")
            print(file=out)
            print("feature_dwconv_bias", file=out)
            feat_dwconv_bias.astype("i2").tofile(out, sep=" ")
            print(file=out)

            def print_linear_block(name, weight, bias):
                print(f"{name}_weight", file=out)
                weight.astype("i1").tofile(out, sep=" ")
                print(file=out)
                print(f"{name}_bias", file=out)
                bias.astype("i4").tofile(out, sep=" ")
                print(file=out)

            if model.no_dynamic_pwconv:
                print_linear_block("policy_pwconv", policy_pwconv_weight, policy_pwconv_bias)
            else:
                print_linear_block(
                    "policy_pwconv_layer_l1", policy_pwconv_layer_l1_weight, policy_pwconv_layer_l1_bias
                )
                print_linear_block(
                    "policy_pwconv_layer_l2", policy_pwconv_layer_l2_weight, policy_pwconv_layer_l2_bias
                )

            if model.no_value_group:
                pass
            elif model.no_star_block:
                print_linear_block("value_corner", corner_weight, corner_bias)
                print_linear_block("value_edge", edge_weight, edge_bias)
                print_linear_block("value_center", center_weight, center_bias)
                print_linear_block("value_quad", quad_weight, quad_bias)
            else:

                def print_star_block(name, weights):
                    print(f"{name}_up1_weight", file=out)
                    weights[0].astype("i1").tofile(out, sep=" ")
                    print(file=out)
                    print(f"{name}_up1_bias", file=out)
                    weights[1].astype("i4").tofile(out, sep=" ")
                    print(file=out)
                    print(f"{name}_up2_weight", file=out)
                    weights[2].astype("i1").tofile(out, sep=" ")
                    print(file=out)
                    print(f"{name}_up2_bias", file=out)
                    weights[3].astype("i4").tofile(out, sep=" ")
                    print(file=out)
                    print(f"{name}_down_weight", file=out)
                    weights[4].astype("i1").tofile(out, sep=" ")
                    print(file=out)
                    print(f"{name}_down_bias", file=out)
                    weights[5].astype("i4").tofile(out, sep=" ")
                    print(file=out)

                print_star_block("value_corner", corner_weights)
                print_star_block("value_edge", edge_weights)
                print_star_block("value_center", center_weights)
                print_star_block("value_quad", quad_weights)

            print_linear_block("value_l1", l1_weight, l1_bias)
            print_linear_block("value_l2", l2_weight, l2_bias)
            print_linear_block("value_l3", l3_weight, l3_bias)

            print("policy_output_weight", file=out)
            policy_output_weight.astype("f4").tofile(out, sep=" ")
            print(file=out)
            print("policy_output_bias", file=out)
            policy_output_bias.astype("f4").tofile(out, sep=" ")
            print(file=out)
        else:
            o: io.RawIOBase = out

            # Since each quantized feature is in [-512, 511] range, they only uses 10 bits,
            # here we write compressed uint64 streams to save space.
            def write_feature_map_compressed(feature_map, feature_bits=10):
                feature_map_i16 = feature_map.astype("<i2")  # (442503, FC)
                feature_map_i16 = feature_map_i16.reshape(-1)  # (442503*FC,)
                bitmask = np.uint64(2**feature_bits - 1)
                uint64 = np.uint64(0)
                bits_used = 0
                uint64_written = 0
                for i in range(feature_map_i16.shape[0]):
                    v = feature_map_i16[i].astype(np.uint64) & bitmask
                    uint64 |= v << np.uint64(bits_used)
                    if 64 - bits_used >= feature_bits:
                        bits_used += feature_bits
                    else:
                        o.write(uint64.tobytes())
                        uint64_written += 1
                        uint64 = v >> np.uint64(64 - bits_used)
                        bits_used = feature_bits - (64 - bits_used)
                if bits_used > 0:
                    o.write(uint64.tobytes())
                    uint64_written += 1
                print(f"write_feature_map_compressed: {feature_map_i16.shape[0]} -> {uint64_written} uint64")

            if model.one_mapping:
                # int16_t mapping[ShapeNum][FeatureDim];
                write_feature_map_compressed(feature_map0)
            else:
                # int16_t mapping[2][ShapeNum][FeatureDim];
                write_feature_map_compressed(feature_map1)
                write_feature_map_compressed(feature_map2)

            # int16_t feature_dwconv_weight[9][FeatureDWConvDim];
            # int16_t feature_dwconv_bias[FeatureDWConvDim];
            o.write(feat_dwconv_weight.astype("<i2").tobytes())
            o.write(feat_dwconv_bias.astype("<i2").tobytes())

            if model.no_dynamic_pwconv:
                # int16_t policy_pwconv_weight[PolicyPWConvDim * PolicyDim];
                # int32_t policy_pwconv_bias[PolicyPWConvDim];
                o.write(policy_pwconv_weight.astype("<i2").tobytes())
                o.write(policy_pwconv_bias.astype("<i4").tobytes())
            else:
                # int8_t  policy_pwconv_layer_l1_weight[(PolicyDim * 2) * FeatureDim];
                # int32_t policy_pwconv_layer_l1_bias[PolicyDim * 2];
                # int8_t  policy_pwconv_layer_l2_weight[(PolicyPWConvDim * PolicyDim + PolicyPWConvDim) * (PolicyDim * 2)];
                # int32_t policy_pwconv_layer_l2_bias[(PolicyPWConvDim * PolicyDim + PolicyPWConvDim)];
                o.write(policy_pwconv_layer_l1_weight.astype("<i1").tobytes())
                o.write(policy_pwconv_layer_l1_bias.astype("<i4").tobytes())
                o.write(policy_pwconv_layer_l2_weight.astype("<i1").tobytes())
                o.write(policy_pwconv_layer_l2_bias.astype("<i4").tobytes())

            if model.no_value_group:
                pass
            elif model.no_star_block:
                # int8_t  value_corner_weight[ValueDim * FeatureDim];
                # int32_t value_corner_bias[ValueDim];
                # int8_t  value_edge_weight[ValueDim * FeatureDim];
                # int32_t value_edge_bias[ValueDim];
                # int8_t  value_center_weight[ValueDim * FeatureDim];
                # int32_t value_center_bias[ValueDim];
                # int8_t  value_quad_weight[ValueDim * ValueDim];
                # int32_t value_quad_bias[ValueDim];
                o.write(corner_weight.astype("<i1").tobytes())
                o.write(corner_bias.astype("<i4").tobytes())
                o.write(edge_weight.astype("<i1").tobytes())
                o.write(edge_bias.astype("<i4").tobytes())
                o.write(center_weight.astype("<i1").tobytes())
                o.write(center_bias.astype("<i4").tobytes())
                o.write(quad_weight.astype("<i1").tobytes())
                o.write(quad_bias.astype("<i4").tobytes())
            else:
                # StarBlockWeight<ValueDim, FeatureDim> value_corner;
                # StarBlockWeight<ValueDim, FeatureDim> value_edge;
                # StarBlockWeight<ValueDim, FeatureDim> value_center;
                # StarBlockWeight<ValueDim, ValueDim>   value_quad;
                def write_star_block(weights):
                    o.write(weights[0].astype("<i1").tobytes())
                    o.write(weights[1].astype("<i4").tobytes())
                    o.write(weights[2].astype("<i1").tobytes())
                    o.write(weights[3].astype("<i4").tobytes())
                    o.write(weights[4].astype("<i1").tobytes())
                    o.write(weights[5].astype("<i4").tobytes())

                write_star_block(corner_weights)
                write_star_block(edge_weights)
                write_star_block(center_weights)
                write_star_block(quad_weights)

            # int8_t  value_l1_weight[ValueDim * (FeatureDim + ValueDim * 4)];
            # int32_t value_l1_bias[ValueDim];
            # int8_t  value_l2_weight[ValueDim * ValueDim];
            # int32_t value_l2_bias[ValueDim];
            # int8_t  value_l3_weight[4 * ValueDim];
            # int32_t value_l3_bias[4];
            o.write(l1_weight.astype("<i1").tobytes())
            o.write(l1_bias.astype("<i4").tobytes())
            o.write(l2_weight.astype("<i1").tobytes())
            o.write(l2_bias.astype("<i4").tobytes())
            l3_weight = np.concatenate([l3_weight, np.zeros((1, l3_weight.shape[1]), dtype=np.int8)], axis=0)
            l3_bias = np.concatenate([l3_bias, np.zeros((1,), dtype=np.int32)], axis=0)
            o.write(l3_weight.astype("<i1").tobytes())
            o.write(l3_bias.astype("<i4").tobytes())

            # float policy_output_weight[16];
            # float policy_output_bias;
            o.write(policy_output_weight.astype("<f4").tobytes())
            o.write(policy_output_bias.astype("<f4").tobytes())
            # char  __padding_to_64bytes_1[44];
            o.write(np.zeros(44, dtype="<i1").tobytes())


@SERIALIZERS.register("mix9svq")
class Mix9svqNetSerializer(BaseSerializer):
    """
    Mix9svqNet binary serializer.

    The corresponding C++ language struct layout:

    template <int OutSize, int InSize>
    struct FCWeight
    {
        int8_t  weight[OutSize * InSize];
        int32_t bias[OutSize];
    };

    template <int OutSize, int InSize>
    struct StarBlockWeight
    {
        FCWeight<OutSize * 2, InSize> value_corner_up1;
        FCWeight<OutSize * 2, InSize> value_corner_up2;
        FCWeight<OutSize, OutSize>    value_corner_down;
    };

    struct alignas(simd::NativeAlignment) Weight
    {
        // 1  mapping layer
        int16_t codebook[2][65536][FeatureDim];
        uint16_t mapping_index[2][ShapeNum];

        // 2  Depthwise conv
        int16_t feature_dwconv_weight[9][FeatDWConvDim];
        int16_t feature_dwconv_bias[FeatDWConvDim];

        struct HeadBucket
        {
            // 3  Policy dynamic pointwise conv
            FCWeight<PolicyDim * 2, FeatureDim> policy_pwconv_layer_l1;
            FCWeight<PolicyPWConvDim * PolicyDim + PolicyPWConvDim, PolicyDim * 2>
                policy_pwconv_layer_l2;

            // 4  Value Group MLP (layer 1,2)
            StarBlockWeight<ValueDim, FeatureDim> value_corner;
            StarBlockWeight<ValueDim, FeatureDim> value_edge;
            StarBlockWeight<ValueDim, FeatureDim> value_center;
            StarBlockWeight<ValueDim, ValueDim>   value_quad;

            // 5  Value MLP (layer 1,2,3)
            FCWeight<ValueDim, FeatureDim + ValueDim * 4> value_l1;
            FCWeight<ValueDim, ValueDim>                  value_l2;
            FCWeight<4, ValueDim>                         value_l3;

            // 6  Policy output linear
            float policy_output_weight[16];
            float policy_output_bias;
            char  __padding_to_64bytes_1[44];
        } buckets[NumHeadBucket];
    };
    """

    def __init__(self, rules, boardsizes, **kwargs):
        super().__init__(rules, boardsizes, **kwargs)
        self.line_length = 11
        self.map_table_export_batch_size = 4096

    @property
    def is_binary(self) -> bool:
        return True

    def arch_hash(self, model: Mix9sVQNet) -> int:
        assert model.input_plane.dim_plane == 2
        _, dim_feature, dim_policy, dim_value, dim_dwconv = model.model_size
        hash = crc32(b"Mix9sVQNet")
        hash ^= crc32(model.input_type.encode("utf-8"))
        print(f"Mix9sVQ ArchHashBase: {hex(hash)}")

        assert dim_feature % 8 == 0, f"dim_feature must be a multiply of 8"
        assert dim_policy % 8 == 0, f"dim_policy must be a multiply of 8"
        assert dim_value % 8 == 0, f"dim_value must be a multiply of 8"
        assert dim_dwconv % 8 == 0, f"dim_dwconv must be a multiply of 8"
        hash ^= (
            (dim_feature // 8)
            | ((dim_policy // 8) << 8)
            | ((dim_value // 8) << 14)
            | ((dim_dwconv // 8) << 20)
        )
        return hash

    def _export_map_table(self, model: Mix9sVQNet, device, line, mapping_idx):
        """
        Export line -> feature mapping table.
        Args:
            line: shape (N, Length)
        Returns:
            features: shape (N, Length, dim_feature)
        """
        N, L = line.shape
        us, opponent = line == 1, line == 2
        line = np.stack((us, opponent), axis=0)[np.newaxis]  # (1, C=2, N, L)
        line = torch.tensor(line, dtype=torch.float32, device=device)

        batch_size = self.map_table_export_batch_size
        batch_num = 1 + (N - 1) // batch_size
        map_table = []
        for i in range(batch_num):
            start = i * batch_size
            end = min((i + 1) * batch_size, N)

            mapping = getattr(model, f"mapping{mapping_idx}")
            feature = mapping(line[:, :, start:end], dirs=[0])[0, 0]  # (dim_feature, batch, L)
            feature = torch.clamp(feature, min=-511 / 32, max=511 / 32)
            feature = torch.permute(feature, (1, 2, 0))  # (batch, L, dim_feature)
            feature = feature.cpu().numpy()
            map_table.append(feature)

        map_table = np.concatenate(map_table, axis=0)  # (N, L, dim_feature)
        return map_table

    def _export_feature_codebook(self, model: Mix9sVQNet, device, mapping_idx):
        # export feature codebook from vq layer
        vq_layer = model.vq_layers[mapping_idx - 1]
        feature_codebook = vq_layer.codebook.cpu().numpy()  # (num_codebooks, codebook_size, dim_feature)
        assert feature_codebook.shape[:2] == (model.num_codebooks, model.codebook_size)
        ascii_hist(f"mapping {mapping_idx} codebook", feature_codebook)
        feature_codebook_clipped = np.clip(feature_codebook, a_min=-16, a_max=511 / 32)
        num_params_clipped = np.sum(feature_codebook != feature_codebook_clipped)
        feature_codebook_quant = np.around(feature_codebook_clipped * 32).astype(np.int16)  # [-512, 511]
        print(
            f"feature codebook: clipped {num_params_clipped}/{feature_codebook.size}"
            + f", quant_range = {(feature_codebook_quant.min(), feature_codebook_quant.max())}"
        )

        # use line encoding to generate feature map
        from dataset.pipeline.line_encoding import (
            get_total_num_encoding,
            get_encoding_usage_flags,
            transform_lines_to_line_encoding,
        )

        # generate line features
        num_encoding = get_total_num_encoding(self.line_length)
        usage_flags = get_encoding_usage_flags(self.line_length)  # (N,) bool
        feature_map = np.zeros((num_encoding, model.model_size[1]), dtype=np.float32)
        for l in reversed(range(1, self.line_length + 1)):
            lines = generate_base3_permutation(l)
            idxs = transform_lines_to_line_encoding(lines, self.line_length)  # (n, 11)
            map_table = self._export_map_table(model, device, lines, mapping_idx)
            rows = np.arange(idxs.shape[0])[:, None]  # (n, 1)
            feature_map[idxs, :] = map_table[rows, np.arange(l), :]

        feature_map = torch.from_numpy(feature_map).to(device)  # (N, dim_feature)
        _, info, _ = vq_layer(feature_map)
        feature_index = info["embed_indices"]  # (N, num_codebooks)
        feature_index = feature_index.cpu().numpy()
        feature_index[~usage_flags] = 0  # unused codebook index is set to 0
        assert model.codebook_size <= 65536, "assume uint16 for codebook index now"
        assert feature_index.min() >= 0, "feature index should be >= 0"
        assert feature_index.max() < model.codebook_size, "feature index should be < codebook_size"
        feature_index = feature_index.astype(np.uint16)

        for codebook_idx in range(model.num_codebooks):
            index_bincount = np.bincount(feature_index[:, codebook_idx], minlength=model.codebook_size)
            usecount_bincount = np.bincount(index_bincount)

            # print the distribution of feature usage count
            print(f"mapping {mapping_idx} codebook {codebook_idx} usecount distribution:")
            print(f"  0: {usecount_bincount[0]}")
            print(f"  1-9: {usecount_bincount[1:10].sum()}")
            print(f"  10-99: {usecount_bincount[10:100].sum()}")
            print(f"  100-999: {usecount_bincount[100:1000].sum()}")
            print(f"  1000-4999: {usecount_bincount[1000:5000].sum()}")
            print(f"  5000-9999: {usecount_bincount[5000:10000].sum()}")
            for i in np.nonzero(usecount_bincount[10000:])[0]:
                print(f"  {10000 + i}: {usecount_bincount[10000 + i]}")

            # reset unused codebook to zeros
            print(f"Reset {usecount_bincount[0]} unused codebook to zeros")
            feature_codebook_quant[codebook_idx, index_bincount == 0] = 0

        return feature_codebook_quant, feature_index

    def _export_feature_dwconv(self, model: Mix9sVQNet):
        conv_weight = model.feature_dwconv.conv.weight.cpu().numpy()  # [FeatureDWConvDim,1,3,3]
        conv_bias = model.feature_dwconv.conv.bias.cpu().numpy()  # [FeatureDWConvDim]
        conv_weight = conv_weight.reshape(conv_weight.shape[0], -1).transpose()  # [9, FeatureDWConvDim]
        ascii_hist("feature dwconv weight", conv_weight)
        ascii_hist("feature dwconv bias", conv_bias)

        conv_weight_clipped = np.clip(conv_weight, a_min=-32768 / 65536, a_max=32767 / 65536)
        conv_bias_clipped = np.clip(
            conv_bias, a_min=-64, a_max=64
        )  # not too large, otherwise it may overflow
        weight_num_params_clipped = np.sum(conv_weight != conv_weight_clipped)
        bias_num_params_clipped = np.sum(conv_bias != conv_bias_clipped)
        conv_weight_quant = np.clip(np.around(conv_weight_clipped * 65536), -32768, 32767).astype(np.int16)
        conv_bias_quant = np.clip(np.around(conv_bias_clipped * 128), -32768, 32767).astype(np.int16)
        print(
            f"feature dwconv: weight clipped {weight_num_params_clipped}/{conv_weight.size}"
            + f", bias clipped {bias_num_params_clipped}/{conv_bias.size}"
            + f", weight_quant_range = {(conv_weight_quant.min(), conv_weight_quant.max())}"
            + f", bias_quant_range = {(conv_bias_quant.min(), conv_bias_quant.max())}"
        )

        # Make sure that the dwconv will not overflow
        assert np.all(
            np.abs(conv_weight_clipped).sum(0) / 2 * 16 * 4 * 128 < 32767
        ), f"feature dwconv would overflow! (maxsum={np.abs(conv_weight_clipped).sum(0).max()})"

        return conv_weight_quant, conv_bias_quant

    def _export_policy_pwconv(self, model: Mix9sVQNet):
        # policy pw conv dynamic weight layer 1
        l1_weight = model.policy_pwconv_weight_linear[0].fc.weight.cpu().numpy()
        l1_bias = model.policy_pwconv_weight_linear[0].fc.bias.cpu().numpy()
        ascii_hist("policy pwconv layer l1 weight", l1_weight)
        ascii_hist("policy pwconv layer l1 bias", l1_bias)

        # policy pw conv dynamic weight layer 2
        l2_weight = model.policy_pwconv_weight_linear[1].fc.weight.cpu().numpy()
        l2_bias = model.policy_pwconv_weight_linear[1].fc.bias.cpu().numpy()
        ascii_hist("policy pwconv layer l2 weight", l2_weight)
        ascii_hist("policy pwconv layer l2 bias", l2_bias)

        # policy output layer
        policy_output_weight = model.policy_output.weight.cpu().squeeze().numpy()
        policy_output_bias = model.policy_output.bias.cpu().numpy()
        ascii_hist("policy output weight", policy_output_weight)
        ascii_hist("policy output bias", policy_output_bias)

        return (
            np.clip(np.around(l1_weight * 128), -128, 127).astype(np.int8),
            np.clip(np.around(l1_bias * 128 * 128), -(2**31), 2**31 - 1).astype(np.int32),
            np.clip(np.around(l2_weight * 128), -128, 127).astype(np.int8),
            np.clip(np.around(l2_bias * 128 * 128), -(2**31), 2**31 - 1).astype(np.int32),
            policy_output_weight / (128 * 128 * 128),
            policy_output_bias,
        )

    def _export_star_block(self, model: Mix9sVQNet, prefix: str):
        block = getattr(model, prefix)
        up1_weight = block.up1.fc.weight.cpu().numpy()
        up1_bias = block.up1.fc.bias.cpu().numpy()
        up2_weight = block.up2.fc.weight.cpu().numpy()
        up2_bias = block.up2.fc.bias.cpu().numpy()
        down_weight = block.down.fc.weight.cpu().numpy()
        down_bias = block.down.fc.bias.cpu().numpy()
        ascii_hist(f"{prefix}: up1 weight", up1_weight)
        ascii_hist(f"{prefix}: up1 bias", up1_bias)
        ascii_hist(f"{prefix}: up2 weight", up2_weight)
        ascii_hist(f"{prefix}: up2 bias", up2_bias)
        ascii_hist(f"{prefix}: down weight", down_weight)
        ascii_hist(f"{prefix}: down bias", down_bias)

        return (
            np.clip(np.around(up1_weight * 128), -128, 127).astype(np.int8),
            np.clip(np.around(up1_bias * 128 * 128), -(2**31), 2**31 - 1).astype(np.int32),
            np.clip(np.around(up2_weight * 128), -128, 127).astype(np.int8),
            np.clip(np.around(up2_bias * 128 * 128), -(2**31), 2**31 - 1).astype(np.int32),
            np.clip(np.around(down_weight * 128), -128, 127).astype(np.int8),
            np.clip(np.around(down_bias * 128 * 128), -(2**31), 2**31 - 1).astype(np.int32),
        )

    def _export_linear(self, model: Mix9sVQNet, prefix: str):
        block = getattr(model, prefix)
        weight = block.fc.weight.cpu().numpy()
        bias = block.fc.bias.cpu().numpy()
        ascii_hist(f"{prefix}: weight", weight)
        ascii_hist(f"{prefix}: bias", bias)

        return (
            np.clip(np.around(weight * 128), -128, 127).astype(np.int8),
            np.clip(np.around(bias * 128 * 128), -(2**31), 2**31 - 1).astype(np.int32),
        )

    def _export_value(self, model: Mix9sVQNet):
        # value layers
        l1_weight = model.value_linear[0].fc.weight.cpu().numpy()
        l1_bias = model.value_linear[0].fc.bias.cpu().numpy()
        l2_weight = model.value_linear[1].fc.weight.cpu().numpy()
        l2_bias = model.value_linear[1].fc.bias.cpu().numpy()
        l3_weight = model.value_linear[2].fc.weight.cpu().numpy()
        l3_bias = model.value_linear[2].fc.bias.cpu().numpy()
        ascii_hist("value: linear1 weight", l1_weight)
        ascii_hist("value: linear1 bias", l1_bias)
        ascii_hist("value: linear2 weight", l2_weight)
        ascii_hist("value: linear2 bias", l2_bias)
        ascii_hist("value: linear3 weight", l3_weight)
        ascii_hist("value: linear3 bias", l3_bias)

        return (
            np.clip(np.around(l1_weight * 128), -128, 127).astype(np.int8),
            np.clip(np.around(l1_bias * 128 * 128), -(2**31), 2**31 - 1).astype(np.int32),
            np.clip(np.around(l2_weight * 128), -128, 127).astype(np.int8),
            np.clip(np.around(l2_bias * 128 * 128), -(2**31), 2**31 - 1).astype(np.int32),
            np.clip(np.around(l3_weight * 128), -128, 127).astype(np.int8),
            np.clip(np.around(l3_bias * 128 * 128), -(2**31), 2**31 - 1).astype(np.int32),
        )

    def serialize(self, out: io.RawIOBase, model: Mix9sVQNet, device):
        feature_codebook1, feature_index1 = self._export_feature_codebook(model, device, 1)
        feature_codebook2, feature_index2 = self._export_feature_codebook(model, device, 2)
        feat_dwconv_weight, feat_dwconv_bias = self._export_feature_dwconv(model)
        (
            policy_pwconv_layer_l1_weight,
            policy_pwconv_layer_l1_bias,
            policy_pwconv_layer_l2_weight,
            policy_pwconv_layer_l2_bias,
            policy_output_weight,
            policy_output_bias,
        ) = self._export_policy_pwconv(model)
        corner_weights = self._export_star_block(model, "value_corner")
        edge_weights = self._export_star_block(model, "value_edge")
        center_weights = self._export_star_block(model, "value_center")
        quad_weights = self._export_star_block(model, "value_quad")
        l1_weight, l1_bias, l2_weight, l2_bias, l3_weight, l3_bias = self._export_value(model)

        # Since each quantized feature is in [-512, 511] range, they only uses 10 bits,
        # here we write compressed uint64 streams to save space.
        def write_features_compressed(feature_codebook, feature_bits=10):
            features_i16 = feature_codebook.reshape(-1).astype("<i2")  # (num_codes*FC,)
            bitmask = np.uint64(2**feature_bits - 1)
            uint64 = np.uint64(0)
            bits_used = 0
            uint64_written = 0
            for i in range(features_i16.shape[0]):
                v = features_i16[i].astype(np.uint64) & bitmask
                uint64 |= v << np.uint64(bits_used)
                if 64 - bits_used >= feature_bits:
                    bits_used += feature_bits
                else:
                    out.write(uint64.tobytes())
                    uint64_written += 1
                    uint64 = v >> np.uint64(64 - bits_used)
                    bits_used = feature_bits - (64 - bits_used)
            if bits_used > 0:
                out.write(uint64.tobytes())
                uint64_written += 1
            original_num_bytes = features_i16.size * 2
            compressed_num_bytes = uint64_written * 8
            compression_rate = 100 * (compressed_num_bytes / original_num_bytes)
            print(
                f"write feature compressed: {original_num_bytes} -> {compressed_num_bytes}"
                f" bytes ({compression_rate:.2f}%)"
            )

        # int16_t codebook[2][65536][FeatureDim];
        write_features_compressed(feature_codebook1)
        write_features_compressed(feature_codebook2)

        # Write feature index as uint16
        def write_feature_index(feature_index):
            feature_index = feature_index.reshape(-1).astype("<u2")
            out.write(feature_index.tobytes())

        # int16_t mapping_index[2][ShapeNum];
        write_feature_index(feature_index1)
        write_feature_index(feature_index2)

        # int16_t feature_dwconv_weight[9][FeatureDWConvDim];
        # int16_t feature_dwconv_bias[FeatureDWConvDim];
        out.write(feat_dwconv_weight.astype("<i2").tobytes())
        out.write(feat_dwconv_bias.astype("<i2").tobytes())

        # FCWeight<PolicyDim * 2, FeatureDim> policy_pwconv_layer_l1;
        # FCWeight<PolicyPWConvDim * PolicyDim + PolicyPWConvDim, PolicyDim * 2> policy_pwconv_layer_l2;
        out.write(policy_pwconv_layer_l1_weight.astype("<i1").tobytes())
        out.write(policy_pwconv_layer_l1_bias.astype("<i4").tobytes())
        out.write(policy_pwconv_layer_l2_weight.astype("<i1").tobytes())
        out.write(policy_pwconv_layer_l2_bias.astype("<i4").tobytes())

        # StarBlockWeight<ValueDim, FeatureDim> value_corner;
        # StarBlockWeight<ValueDim, FeatureDim> value_edge;
        # StarBlockWeight<ValueDim, FeatureDim> value_center;
        # StarBlockWeight<ValueDim, ValueDim>   value_quad;
        def write_star_block(weights):
            out.write(weights[0].astype("<i1").tobytes())
            out.write(weights[1].astype("<i4").tobytes())
            out.write(weights[2].astype("<i1").tobytes())
            out.write(weights[3].astype("<i4").tobytes())
            out.write(weights[4].astype("<i1").tobytes())
            out.write(weights[5].astype("<i4").tobytes())

        write_star_block(corner_weights)
        write_star_block(edge_weights)
        write_star_block(center_weights)
        write_star_block(quad_weights)

        # FCWeight<ValueDim, FeatureDim + ValueDim * 4> value_l1;
        # FCWeight<ValueDim, ValueDim>                  value_l2;
        # FCWeight<4, ValueDim>                         value_l3;
        out.write(l1_weight.astype("<i1").tobytes())
        out.write(l1_bias.astype("<i4").tobytes())
        out.write(l2_weight.astype("<i1").tobytes())
        out.write(l2_bias.astype("<i4").tobytes())
        # pad l3 weights to have at least output dim = 4
        l3_weight = np.concatenate([l3_weight, np.zeros_like(l3_weight[:1])], axis=0)
        l3_bias = np.concatenate([l3_bias, np.zeros_like(l3_bias[:1])], axis=0)
        out.write(l3_weight.astype("<i1").tobytes())
        out.write(l3_bias.astype("<i4").tobytes())

        # float policy_output_weight[16];
        # float policy_output_bias;
        out.write(policy_output_weight.astype("<f4").tobytes())
        out.write(policy_output_bias.astype("<f4").tobytes())
        # char  __padding_to_64bytes_1[44];
        out.write(np.zeros(44, dtype="<i1").tobytes())


@SERIALIZERS.register("mix10")
class Mix10NetSerializer(BaseSerializer):
    """
    Mix10Net binary serializer.

    The corresponding C++ language struct layout:
    template <int OutSize, int InSize>
    struct FCWeight
    {
        int8_t  weight[OutSize * InSize];
        int32_t bias[OutSize];
    };

    struct alignas(64) Weight
    {
        // 1  mapping layer
        int16_t mapping[2][ShapeNum][FeatureDim];

        // 2  Depthwise conv
        int16_t feature_dwconv_weight[9][FeatDWConvDim];
        int16_t feature_dwconv_bias[FeatDWConvDim];

        struct HeadBucket
        {
            // 3  Small Policy dynamic pointwise conv
            FCWeight<ValueDim, ValueDim>                          policy_small_pwconv_weight_1;
            FCWeight<PolicySOutDim *(PolicySInDim + 1), ValueDim> policy_small_pwconv_weight_2;

            // 4  Large Policy dynamic pointwise conv
            FCWeight<ValueDim, ValueDim>                           policy_large_pwconv_weight_0;
            FCWeight<PolicyLMidDim *(PolicyLInDim + 1), ValueDim>  policy_large_pwconv_weight_1;
            FCWeight<PolicyLOutDim *(PolicyLMidDim + 1), ValueDim> policy_large_pwconv_weight_2;

            // 5  Small Value Head MLP (layer 1,2,3)
            FCWeight<ValueDim, FeatureDim> value_small_l1;
            FCWeight<ValueDim, ValueDim>   value_small_l2;
            FCWeight<4, ValueDim>          value_small_l3;

            // 6  Large Value Gate & Group MLP
            char                               __padding_to_64bytes_0[48];
            FCWeight<FeatureDim * 2, ValueDim> value_gate;
            FCWeight<ValueDim, FeatureDim>     value_corner;
            FCWeight<ValueDim, FeatureDim>     value_edge;
            FCWeight<ValueDim, FeatureDim>     value_center;
            FCWeight<ValueDim, ValueDim>       value_quad;

            // 7  Large Value Head MLP (layer 1,2,3)
            FCWeight<ValueDim, ValueDim * 5> value_l1;
            FCWeight<ValueDim, ValueDim>     value_l2;
            FCWeight<4, ValueDim>            value_l3;

            // 8  Policy output linear
            char  __padding_to_64bytes_1[48];
            float policy_small_output_weight[PolicySOutDim];
            float policy_large_output_weight[PolicyLOutDim];
            float policy_small_output_bias;
            float policy_large_output_bias;

            char __padding_to_64bytes_2[56];
        } buckets[NumHeadBucket];
    };
    """

    def __init__(self, rules, boardsizes, **kwargs):
        super().__init__(rules, boardsizes, **kwargs)
        self.line_length = 11
        self.map_table_export_batch_size = 4096

    @property
    def is_binary(self) -> bool:
        return True

    def arch_hash(self, model: Mix10Net) -> int:
        assert model.input_plane.dim_plane == 2
        _, dim_feature, dim_dwconv, dim_value = model.model_size
        hash = crc32(b"Mix10Net")
        hash ^= crc32(model.input_type.encode("utf-8"))
        print(f"Mix10 ArchHashBase: {hex(hash)}")

        assert dim_feature % 8 == 0, f"dim_feature must be a multiply of 8"
        assert dim_dwconv % 8 == 0, f"dim_dwconv must be a multiply of 8"
        assert dim_value % 8 == 0, f"dim_value must be a multiply of 8"
        hash ^= (
            (dim_feature // 8)
            | ((dim_dwconv // 8) << 8)
            | ((dim_value // 8) << 16)
        )
        return hash

    def _export_map_table(self, model: Mix10Net, device, line, mapping_idx):
        """
        Export line -> feature mapping table.

        Args:
            line: shape (N, Length)
        """
        N, L = line.shape
        us, opponent = line == 1, line == 2
        line = np.stack((us, opponent), axis=0)[np.newaxis]  # [1, C=2, N, L]
        line = torch.tensor(line, dtype=torch.float32, device=device)

        batch_size = self.map_table_export_batch_size
        batch_num = 1 + (N - 1) // batch_size
        map_table = []
        for i in range(batch_num):
            start = i * batch_size
            end = min((i + 1) * batch_size, N)

            mapping = getattr(model, f"mapping{mapping_idx}")
            feature = mapping(line[:, :, start:end], dirs=[0])[0, 0]  # [dim_feature, batch, L]
            feature = torch.permute(feature, (1, 2, 0)).cpu().numpy()  # [batch, L, dim_feature]
            map_table.append(feature)

        map_table = np.concatenate(map_table, axis=0)  # [N, L, dim_feature]
        return map_table

    def _export_feature_map(self, model: Mix10Net, device, mapping_idx):
        # use line encoding to generate feature map
        from dataset.pipeline.line_encoding import (
            get_total_num_encoding,
            get_encoding_usage_flags,
            transform_lines_to_line_encoding,
        )

        L = self.line_length
        dim_feature = model.model_size[1]
        num_encoding = get_total_num_encoding(L)
        usage_flags = get_encoding_usage_flags(L)  # [N] bool, track usage of each feature
        feature_map = np.zeros((num_encoding, dim_feature), dtype=np.float32)  # [N, dim_feature]

        # generate line features
        for l in reversed(range(1, L + 1)):
            lines = generate_base3_permutation(l)
            idxs = transform_lines_to_line_encoding(lines, L)  # [n, 11]
            map_table = self._export_map_table(model, device, lines, mapping_idx)
            rows = np.arange(idxs.shape[0])[:, None]  # [n, 1]
            feature_map[idxs, :] = map_table[rows, np.arange(l), :]

        ascii_hist(f"feature map {mapping_idx}", feature_map)
        feature_map_clipped = np.clip(feature_map, a_min=-16, a_max=511 / 32)
        num_params_clipped = np.sum(feature_map != feature_map_clipped)
        feature_map_quant = np.around(feature_map_clipped * 32).astype(np.int16)  # [-512, 511]
        print(
            f"feature map: used {usage_flags.sum()} features of {len(usage_flags)}, "
            + f": clipped {num_params_clipped}/{feature_map.size}"
            + f", quant_range = {(feature_map_quant.min(), feature_map_quant.max())}"
        )
        return feature_map_quant, usage_flags

    def _export_feature_dwconv(self, model: Mix10Net):
        conv_weight = model.feature_dwconv.conv.weight.cpu().numpy()  # [FeatureDWConvDim,1,3,3]
        conv_bias = model.feature_dwconv.conv.bias.cpu().numpy()  # [FeatureDWConvDim]
        conv_weight = conv_weight.reshape(conv_weight.shape[0], -1).transpose()  # [9, FeatureDWConvDim]
        ascii_hist("feature dwconv weight", conv_weight)
        ascii_hist("feature dwconv bias", conv_bias)

        conv_weight_clipped = np.clip(conv_weight, a_min=-32768 / 65536, a_max=32767 / 65536)
        conv_bias_clipped = np.clip(
            conv_bias, a_min=-64, a_max=64
        )  # not too large, otherwise it may overflow
        weight_num_params_clipped = np.sum(conv_weight != conv_weight_clipped)
        bias_num_params_clipped = np.sum(conv_bias != conv_bias_clipped)
        conv_weight_quant = np.clip(np.around(conv_weight_clipped * 65536), -32768, 32767).astype(np.int16)
        conv_bias_quant = np.clip(np.around(conv_bias_clipped * 128), -32768, 32767).astype(np.int16)
        print(
            f"feature dwconv: weight clipped {weight_num_params_clipped}/{conv_weight.size}"
            + f", bias clipped {bias_num_params_clipped}/{conv_bias.size}"
            + f", weight_quant_range = {(conv_weight_quant.min(), conv_weight_quant.max())}"
            + f", bias_quant_range = {(conv_bias_quant.min(), conv_bias_quant.max())}"
        )

        # Make sure that the dwconv will not overflow
        assert np.all(
            np.abs(conv_weight_clipped).sum(0) / 2 * 16 * 4 * 128 < 32767
        ), f"feature dwconv would overflow! (maxsum={np.abs(conv_weight_clipped).sum(0).max()})"

        return conv_weight_quant, conv_bias_quant

    def _export_policy_small_pwconv(self, model: Mix10Net):
        # policy pw conv dynamic weight layer 1
        l1_weight = model.policy_small_pwconv_weight[0].fc.weight.cpu().numpy()
        l1_bias = model.policy_small_pwconv_weight[0].fc.bias.cpu().numpy()
        ascii_hist("policy small pwconv l1 weight", l1_weight)
        ascii_hist("policy small pwconv l1 bias", l1_bias)

        # policy pw conv dynamic weight layer 2
        l2_weight = model.policy_small_pwconv_weight[1].fc.weight.cpu().numpy()
        l2_bias = model.policy_small_pwconv_weight[1].fc.bias.cpu().numpy()
        ascii_hist("policy small pwconv l2 weight", l2_weight)
        ascii_hist("policy small pwconv l2 bias", l2_bias)

        # policy PReLU activation
        policy_output_weight = model.policy_small_output.weight.cpu().squeeze().numpy()
        policy_output_bias = model.policy_small_output.bias.cpu().numpy()
        ascii_hist("policy small output weight", policy_output_weight)
        ascii_hist("policy small output bias", policy_output_bias)

        return (
            (np.clip(np.around(l1_weight * 128), -128, 127).astype(np.int8),
             np.clip(np.around(l1_bias * 128 * 128), -(2**31), 2**31 - 1).astype(np.int32)),
            (np.clip(np.around(l2_weight * 128), -128, 127).astype(np.int8),
             np.clip(np.around(l2_bias * 128 * 128), -(2**31), 2**31 - 1).astype(np.int32)),
            policy_output_weight / (128 * 128 * 128),
            policy_output_bias,
        )

    def _export_policy_large_pwconv(self, model: Mix10Net):
        # policy pw conv dynamic weight layer 0
        l0_weight = model.policy_large_pwconv_weight_0.fc.weight.cpu().numpy()
        l0_bias = model.policy_large_pwconv_weight_0.fc.bias.cpu().numpy()
        ascii_hist("policy large pwconv layer 0 weight", l0_weight)
        ascii_hist("policy large pwconv layer 0 bias", l0_bias)

        # policy pw conv dynamic weight 1
        l1_weight = model.policy_large_pwconv_weight_1.fc.weight.cpu().numpy()
        l1_bias = model.policy_large_pwconv_weight_1.fc.bias.cpu().numpy()
        ascii_hist("policy large pwconv 1 weight", l1_weight)
        ascii_hist("policy large pwconv 1 bias", l1_bias)

        # policy pw conv dynamic weight 2
        l2_weight = model.policy_large_pwconv_weight_2.fc.weight.cpu().numpy()
        l2_bias = model.policy_large_pwconv_weight_2.fc.bias.cpu().numpy()
        ascii_hist("policy large pwconv 2 weight", l2_weight)
        ascii_hist("policy large pwconv 2 bias", l2_bias)

        # policy PReLU activation
        policy_output_weight = model.policy_large_output.weight.cpu().squeeze().numpy()
        policy_output_bias = model.policy_large_output.bias.cpu().numpy()
        ascii_hist("policy large output weight", policy_output_weight)
        ascii_hist("policy large output bias", policy_output_bias)

        return (
            (np.clip(np.around(l0_weight * 128), -128, 127).astype(np.int8),
             np.clip(np.around(l0_bias * 128 * 128), -(2**31), 2**31 - 1).astype(np.int32)),
            (np.clip(np.around(l1_weight * 128), -128, 127).astype(np.int8),
             np.clip(np.around(l1_bias * 128 * 128), -(2**31), 2**31 - 1).astype(np.int32)),
            (np.clip(np.around(l2_weight * 128), -128, 127).astype(np.int8),
             np.clip(np.around(l2_bias * 128 * 128), -(2**31), 2**31 - 1).astype(np.int32)),
            policy_output_weight / (128 * 128 * 128),
            policy_output_bias,
        )

    def _export_linear(self, model: Mix10Net, prefix: str):
        block = eval(f"model.{prefix}", {"model": model})
        weight = block.fc.weight.cpu().numpy()
        bias = block.fc.bias.cpu().numpy()
        ascii_hist(f"{prefix}: weight", weight)
        ascii_hist(f"{prefix}: bias", bias)

        return (
            np.clip(np.around(weight * 128), -128, 127).astype(np.int8),
            np.clip(np.around(bias * 128 * 128), -(2**31), 2**31 - 1).astype(np.int32),
        )

    def serialize(self, out: io.IOBase, model: Mix10Net, device):
        feature_map1, usage_flags1 = self._export_feature_map(model, device, 1)
        feature_map2, usage_flags2 = self._export_feature_map(model, device, 2)
        feat_dwconv_weight, feat_dwconv_bias = self._export_feature_dwconv(model)

        (
            policy_small_pwconv_weight_l1,
            policy_small_pwconv_weight_l2,
            policy_small_output_weight,
            policy_small_output_bias,
        ) = self._export_policy_small_pwconv(model)

        (
            policy_large_pwconv_weight_0,
            policy_large_pwconv_weight_1,
            policy_large_pwconv_weight_2,
            policy_large_output_weight,
            policy_large_output_bias,
        ) = self._export_policy_large_pwconv(model)

        small_l1_weights = self._export_linear(model, "value_linear_small[0]")
        small_l2_weights = self._export_linear(model, "value_linear_small[1]")
        small_l3_weights = self._export_linear(model, "value_small_output")

        gate_weights = self._export_linear(model, "value_gate")
        corner_weights = self._export_linear(model, "value_corner")
        edge_weights = self._export_linear(model, "value_edge")
        center_weights = self._export_linear(model, "value_center")
        quad_weights = self._export_linear(model, "value_quad")

        large_l1_weights = self._export_linear(model, "value_linear_large[0]")
        large_l2_weights = self._export_linear(model, "value_linear_large[1]")
        large_l3_weights = self._export_linear(model, "value_large_output")

        o: io.RawIOBase = out  # always use binary output stream

        # Since each quantized feature is in [-512, 511] range, they only uses 10 bits,
        # here we write compressed uint64 streams to save space.
        def write_feature_map_compressed(feature_map, feature_bits=10):
            feature_map_i16 = feature_map.astype("<i2")  # (442503, FC)
            feature_map_i16 = feature_map_i16.reshape(-1)  # (442503*FC,)
            bitmask = np.uint64(2**feature_bits - 1)
            uint64 = np.uint64(0)
            bits_used = 0
            uint64_written = 0
            for i in tqdm(range(feature_map_i16.shape[0])):
                v = feature_map_i16[i].astype(np.uint64) & bitmask
                uint64 |= v << np.uint64(bits_used)
                if 64 - bits_used >= feature_bits:
                    bits_used += feature_bits
                else:
                    o.write(uint64.tobytes())
                    uint64_written += 1
                    uint64 = v >> np.uint64(64 - bits_used)
                    bits_used = feature_bits - (64 - bits_used)
            if bits_used > 0:
                o.write(uint64.tobytes())
                uint64_written += 1
            print(f"write_feature_map_compressed: {feature_map_i16.shape[0]} -> {uint64_written} uint64")

        # int16_t mapping[2][ShapeNum][FeatureDim];
        write_feature_map_compressed(feature_map1)
        write_feature_map_compressed(feature_map2)

        # int16_t feature_dwconv_weight[9][FeatureDWConvDim];
        # int16_t feature_dwconv_bias[FeatureDWConvDim];
        o.write(feat_dwconv_weight.astype("<i2").tobytes())
        o.write(feat_dwconv_bias.astype("<i2").tobytes())

        def write_linear(weight_and_bias, zero_padding=0):
            weight, bias = weight_and_bias
            if zero_padding > 0:
                weight = np.concatenate(
                    [weight, np.zeros((zero_padding, weight.shape[1]), dtype=np.int8)], axis=0
                )
                bias = np.concatenate([bias, np.zeros((zero_padding,), dtype=np.int32)], axis=0)
            o.write(weight.astype("<i1").tobytes())
            o.write(bias.astype("<i4").tobytes())

        # FCWeight<ValueDim, ValueDim>                          policy_small_pwconv_weight_1;
        # FCWeight<PolicySOutDim *(PolicySInDim + 1), ValueDim> policy_small_pwconv_weight_2;
        write_linear(policy_small_pwconv_weight_l1)
        write_linear(policy_small_pwconv_weight_l2)

        # FCWeight<ValueDim, ValueDim>                           policy_large_pwconv_weight_0;
        # FCWeight<PolicyLMidDim *(PolicyLInDim + 1), ValueDim>  policy_large_pwconv_weight_1;
        # FCWeight<PolicyLOutDim *(PolicyLMidDim + 1), ValueDim> policy_large_pwconv_weight_2;
        write_linear(policy_large_pwconv_weight_0)
        write_linear(policy_large_pwconv_weight_1)
        write_linear(policy_large_pwconv_weight_2)

        # FCWeight<ValueDim, FeatureDim> value_small_l1;
        # FCWeight<ValueDim, ValueDim>   value_small_l2;
        # FCWeight<4, ValueDim>          value_small_l3;
        write_linear(small_l1_weights)
        write_linear(small_l2_weights)
        write_linear(small_l3_weights, 1)

        # char                               __padding_to_64bytes_0[48];
        # FCWeight<FeatureDim * 2, ValueDim> value_gate;
        # FCWeight<ValueDim, FeatureDim>     value_corner;
        # FCWeight<ValueDim, FeatureDim>     value_edge;
        # FCWeight<ValueDim, FeatureDim>     value_center;
        # FCWeight<ValueDim, ValueDim>       value_quad;
        o.write(np.zeros(48, dtype="<i1").tobytes())
        write_linear(gate_weights)
        write_linear(corner_weights)
        write_linear(edge_weights)
        write_linear(center_weights)
        write_linear(quad_weights)

        # FCWeight<ValueDim, ValueDim * 5> value_l1;
        # FCWeight<ValueDim, ValueDim>     value_l2;
        # FCWeight<4, ValueDim>            value_l3;
        write_linear(large_l1_weights)
        write_linear(large_l2_weights)
        write_linear(large_l3_weights, 1)

        # char  __padding_to_64bytes_1[48];
        # float policy_small_output_weight[PolicySOutDim];
        # float policy_large_output_weight[PolicyLOutDim];
        # float policy_small_output_bias;
        # float policy_large_output_bias;
        o.write(np.zeros(48, dtype="<i1").tobytes())
        o.write(policy_small_output_weight.astype("<f4").tobytes())
        o.write(policy_large_output_weight.astype("<f4").tobytes())
        o.write(policy_small_output_bias.astype("<f4").tobytes())
        o.write(policy_large_output_bias.astype("<f4").tobytes())

        # char __padding_to_64bytes_2[56];
        o.write(np.zeros(56, dtype="<i1").tobytes())
