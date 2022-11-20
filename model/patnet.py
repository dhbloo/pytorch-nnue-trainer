import torch
import torch.nn as nn
import torch.nn.functional as F

from . import MODELS
from .blocks import Conv2dBlock, LinearBlock, ChannelWiseLeakyReLU


def build_embedding(type, feature_dim, board_size=15, pcode_dim=2380, **kwargs):
    if type == 'pcode':
        return PatternCodeEmbedding(feature_dim, pcode_dim)
    elif type == 'pcode-twoside':
        return PatternCodeTwoSideEmbedding(feature_dim, pcode_dim)
    elif type == 'pcode-board':
        return PatternCodeBoardEmbedding(feature_dim, board_size, pcode_dim)
    elif type == 'pcode-symboard':
        return PatternCodeSymBoardEmbedding(feature_dim, board_size, pcode_dim, **kwargs)
    elif type == 'pcode-symouterboard':
        return PatternCodeSymOuterBoardEmbedding(feature_dim, board_size, pcode_dim, **kwargs)
    else:
        assert 0, f"Unsupported embedding: {type}"


class PatternCodeEmbedding(nn.Module):
    def __init__(self, feature_dim, pcode_dim=2380):
        super().__init__()
        self.feature_dim = feature_dim
        self.pcode_dim = pcode_dim
        self.pcode_embedding = nn.Embedding(num_embeddings=2 * (pcode_dim + 1),
                                            embedding_dim=feature_dim)

    def forward(self, data):
        assert torch.all(self.pcode_dim == data['sparse_feature_dim'][:, 10:12])
        pcode_sparse_input = data['sparse_feature_input'][:, [10, 11]].int()  # [B, 2, H, W]

        # set sparse input at non-empty cell
        board_input = data['board_input']  # [B, 2, H, W]
        non_empty_mask = board_input[:, 0] + board_input[:, 1] > 0  # [B, H, W]
        non_empty_mask = torch.unsqueeze(non_empty_mask, dim=1)  # [B, 1, H, W]
        pcode_sparse_input.masked_fill_(non_empty_mask > 0, self.pcode_dim)

        # add index offset for opponent side
        pcode_sparse_input[:, 1] += self.pcode_dim + 1

        # convert sparse input to dense feature through embedding
        pcode_feature = self.pcode_embedding(pcode_sparse_input)  # [B, 2, H, W, feature_dim]
        pcode_feature = torch.sum(pcode_feature, dim=1, keepdim=False)  # [B, H, W, feature_dim]
        pcode_feature = torch.permute(pcode_feature, (0, 3, 1, 2))  # [B, feature_dim, H, W]
        return pcode_feature.contiguous()


class PatternCodeTwoSideEmbedding(nn.Module):
    def __init__(self, feature_dim, pcode_dim=2380):
        super().__init__()
        self.feature_dim = feature_dim
        self.pcode_dim = pcode_dim
        self.pcode_embedding = nn.Embedding(num_embeddings=(pcode_dim + 1)**2,
                                            embedding_dim=feature_dim)

    def forward(self, data):
        assert torch.all(self.pcode_dim == data['sparse_feature_dim'][:, 10:12])
        pcode_sparse_input = data['sparse_feature_input'][:, [10, 11]].int()  # [B, 2, H, W]

        # set sparse input at non-empty cell
        board_input = data['board_input']  # [B, 2, H, W]
        non_empty_mask = board_input[:, 0] + board_input[:, 1] > 0  # [B, H, W]
        non_empty_mask = torch.unsqueeze(non_empty_mask, dim=1)  # [B, 1, H, W]
        pcode_sparse_input.masked_fill_(non_empty_mask > 0, self.pcode_dim)

        # get sparse index for two side
        pcode_two_side_sparse_input = (pcode_sparse_input[:, 1] * (self.pcode_dim + 1) +
                                       pcode_sparse_input[:, 0])  # [B, H, W]

        # convert sparse input to dense feature through embedding
        pcode_feature = self.pcode_embedding(pcode_two_side_sparse_input)  # [B, H, W, feature_dim]
        pcode_feature = torch.permute(pcode_feature, (0, 3, 1, 2))  # [B, feature_dim, H, W]
        return pcode_feature.contiguous()


class PatternCodeBoardEmbedding(nn.Module):
    def __init__(self, feature_dim, board_size, pcode_dim=2380):
        super().__init__()
        self.feature_dim = feature_dim
        self.pcode_dim = pcode_dim
        self.board_size = board_size
        self.cell_dim = board_size * board_size

        embed_dim = 2 * (pcode_dim + 1)
        self.pcode_embedding = nn.Embedding(num_embeddings=embed_dim, embedding_dim=feature_dim)
        self.pcode_board_embedding = nn.Embedding(num_embeddings=self.cell_dim * embed_dim,
                                                  embedding_dim=feature_dim)
        board_cell_index = torch.arange(0, self.cell_dim, 1,
                                        dtype=torch.int32).view(1, 1, board_size, board_size)
        self.board_offset = nn.parameter.Parameter(board_cell_index * embed_dim, False)

    def forward(self, data):
        assert torch.all(self.pcode_dim == data['sparse_feature_dim'][:, 10:12])
        pcode_sparse_input = data['sparse_feature_input'][:, [10, 11]].int()  # [B, 2, H, W]

        # set sparse input at non-empty cell
        board_input = data['board_input']  # [B, 2, H, W]
        non_empty_mask = board_input[:, 0] + board_input[:, 1] > 0  # [B, H, W]
        non_empty_mask = torch.unsqueeze(non_empty_mask, dim=1)  # [B, 1, H, W]
        pcode_sparse_input.masked_fill_(non_empty_mask > 0, self.pcode_dim)

        # add index offset for opponent side
        pcode_sparse_input[:, 1] += self.pcode_dim + 1

        # add index offset for board
        pcode_board_sparse_input = pcode_sparse_input + self.board_offset

        # convert sparse input to dense feature through embedding
        pcode_feature = self.pcode_embedding(pcode_sparse_input)  # [B, 2, H, W, feature_dim]
        pcode_board_feature = self.pcode_board_embedding(pcode_board_sparse_input)
        pcode_feature = pcode_feature + pcode_board_feature

        pcode_feature = torch.sum(pcode_feature, dim=1, keepdim=False)  # [B, H, W, feature_dim]
        pcode_feature = torch.permute(pcode_feature, (0, 3, 1, 2))  # [B, feature_dim, H, W]
        return pcode_feature.contiguous()


class PatternCodeSymBoardEmbedding(nn.Module):
    def __init__(self, feature_dim, board_size, pcode_dim=2380, max_offset=None):
        super().__init__()
        self.feature_dim = feature_dim
        self.pcode_dim = pcode_dim
        self.board_size = board_size
        self.cell_dim = board_size * board_size

        embed_dim = 2 * (pcode_dim + 1)
        self.pcode_embedding = nn.Embedding(num_embeddings=embed_dim, embedding_dim=feature_dim)

        map, max_offset = self._make_symmetry_board_map(max_offset)
        self.pcode_symboard_embedding = nn.Embedding(num_embeddings=(max_offset + 1) * embed_dim,
                                                     embedding_dim=feature_dim)
        self.offset_map = nn.parameter.Parameter(map * embed_dim, False)

    def _make_symmetry_board_map(self, max_offset=None):
        map = torch.zeros(self.board_size, self.board_size, dtype=torch.int32)
        cnt = 0
        for y in range((self.board_size + 1) // 2):
            for x in range((self.board_size + 1) // 2):
                if y <= x:
                    map[y, x] = cnt
                    cnt += 1
                    if max_offset:
                        cnt = min(cnt, max_offset)
        map = torch.max(map, torch.fliplr(map))
        map = torch.max(map, torch.flipud(map))
        map = torch.max(map, torch.transpose(map, 0, 1))
        return map, torch.max(map)

    def forward(self, data):
        assert torch.all(self.pcode_dim == data['sparse_feature_dim'][:, 10:12])
        pcode_sparse_input = data['sparse_feature_input'][:, [10, 11]].int()  # [B, 2, H, W]

        # set sparse input at non-empty cell
        board_input = data['board_input']  # [B, 2, H, W]
        non_empty_mask = board_input[:, 0] + board_input[:, 1] > 0  # [B, H, W]
        non_empty_mask = torch.unsqueeze(non_empty_mask, dim=1)  # [B, 1, H, W]
        pcode_sparse_input.masked_fill_(non_empty_mask > 0, self.pcode_dim)

        # add index offset for opponent side
        pcode_sparse_input[:, 1] += self.pcode_dim + 1

        # add index offset for board
        pcode_symboard_sparse_input = pcode_sparse_input + self.offset_map

        # convert sparse input to dense feature through embedding
        pcode_feature = self.pcode_embedding(pcode_sparse_input)  # [B, 2, H, W, feature_dim]
        pcode_board_feature = self.pcode_symboard_embedding(pcode_symboard_sparse_input)
        pcode_feature = pcode_feature + pcode_board_feature

        pcode_feature = torch.sum(pcode_feature, dim=1, keepdim=False)  # [B, H, W, feature_dim]
        pcode_feature = torch.permute(pcode_feature, (0, 3, 1, 2))  # [B, feature_dim, H, W]
        return pcode_feature.contiguous()


class PatternCodeSymOuterBoardEmbedding(nn.Module):
    def __init__(self, feature_dim, board_size, pcode_dim=2380, outer_size=5):
        super().__init__()
        self.feature_dim = feature_dim
        self.pcode_dim = pcode_dim
        self.board_size = board_size
        self.cell_dim = board_size * board_size

        embed_dim = 2 * (pcode_dim + 1)
        self.pcode_embedding = nn.Embedding(num_embeddings=embed_dim, embedding_dim=feature_dim)

        map, max_offset = self._make_outer_board_map(outer_size)
        self.pcode_outerboard_embedding = nn.Embedding(num_embeddings=(max_offset + 1) * embed_dim,
                                                       embedding_dim=feature_dim)
        self.offset_map = nn.parameter.Parameter(map * embed_dim, False)

    def _make_outer_board_map(self, outer_size):
        assert outer_size <= self.board_size // 2
        map = torch.zeros(self.board_size, self.board_size, dtype=torch.int32)
        cnt = 1
        for i in reversed(range(outer_size)):
            for j in range(outer_size, (self.board_size + 1) // 2):
                map[i, j] = cnt
            cnt += 1
        for y in reversed(range(outer_size)):
            for x in reversed(range(y, outer_size)):
                map[y, x] = cnt
                cnt += 1
        map = torch.max(map, torch.fliplr(map))
        map = torch.max(map, torch.flipud(map))
        map = torch.max(map, torch.transpose(map, 0, 1))
        return map, torch.max(map)

    def forward(self, data):
        assert torch.all(self.pcode_dim == data['sparse_feature_dim'][:, 10:12])
        pcode_sparse_input = data['sparse_feature_input'][:, [10, 11]].int()  # [B, 2, H, W]

        # set sparse input at non-empty cell
        board_input = data['board_input']  # [B, 2, H, W]
        non_empty_mask = board_input[:, 0] + board_input[:, 1] > 0  # [B, H, W]
        non_empty_mask = torch.unsqueeze(non_empty_mask, dim=1)  # [B, 1, H, W]
        pcode_sparse_input.masked_fill_(non_empty_mask > 0, self.pcode_dim)

        # add index offset for opponent side
        pcode_sparse_input[:, 1] += self.pcode_dim + 1

        # add index offset for board
        pcode_outerboard_sparse_input = pcode_sparse_input + self.offset_map

        # convert sparse input to dense feature through embedding
        pcode_feature = self.pcode_embedding(pcode_sparse_input)  # [B, 2, H, W, feature_dim]
        pcode_board_feature = self.pcode_outerboard_embedding(pcode_outerboard_sparse_input)
        pcode_feature = pcode_feature + pcode_board_feature

        pcode_feature = torch.sum(pcode_feature, dim=1, keepdim=False)  # [B, H, W, feature_dim]
        pcode_feature = torch.permute(pcode_feature, (0, 3, 1, 2))  # [B, feature_dim, H, W]
        return pcode_feature.contiguous()


@MODELS.register('patnetbaseline')
class PatNetBaseline(nn.Module):
    def __init__(self,
                 dim_policy=16,
                 dim_value=32,
                 board_size=15,
                 embedding_type='pcode-symboard',
                 map_max=30,
                 **kwargs):
        super().__init__()
        self.model_size = (dim_policy, dim_value)
        self.board_size = board_size
        self.embedding_type = embedding_type
        self.map_max = map_max
        dim_out = dim_policy + dim_value

        self.embedding = build_embedding(embedding_type, dim_out, board_size, **kwargs)

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
        dim_policy, _ = self.model_size

        feature = self.embedding(data)
        # resize feature to range [-map_max, map_max]
        if self.map_max != 0:
            feature = self.map_max * torch.tanh(feature / self.map_max)

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
        p, v = self.model_size
        return f"patnetbaseline_{self.embedding_type}_{p}p{v}v-b{self.board_size}" + (
            f"-{self.map_max}mm" if self.map_max != 0 else "")


@MODELS.register('patnetv1')
class PatNetv1(nn.Module):
    def __init__(self,
                 dim_policy=16,
                 dim_value=16,
                 board_size=15,
                 embedding_type='pcode-symboard',
                 **kwargs):
        super().__init__()
        self.model_size = (dim_policy, dim_value)
        self.board_size = board_size
        self.embedding_type = embedding_type

        self.embedding = build_embedding(embedding_type, dim_policy + dim_value, board_size,
                                         **kwargs)

        # policy net
        assert dim_policy >= 4
        self.policy_dw_conv = Conv2dBlock(dim_policy,
                                          dim_policy,
                                          ks=3,
                                          st=1,
                                          padding=1,
                                          groups=dim_policy,
                                          norm='bn',
                                          activation='relu',
                                          pad_type='zeros')
        self.policy_pw_conv = Conv2dBlock(dim_policy,
                                          dim_policy // 2,
                                          ks=1,
                                          st=1,
                                          norm='bn',
                                          activation='relu')
        self.policy_final_conv = Conv2dBlock(dim_policy // 2,
                                             1,
                                             ks=1,
                                             st=1,
                                             activation='none',
                                             bias=False)

        # value net
        self.value_linear = nn.Sequential(
            LinearBlock(dim_value, dim_value, norm='bn', activation='lrelu/16'),
            LinearBlock(dim_value, dim_value, norm='bn', activation='lrelu/16'),
        )
        self.value_final_linear = LinearBlock(dim_value, 3, activation='none')

    def forward(self, data):
        dim_policy, _ = self.model_size

        feature = self.embedding(data)

        # policy head
        policy = feature[:, :dim_policy]
        policy = self.policy_dw_conv(policy)
        policy = self.policy_pw_conv(policy)
        policy = self.policy_final_conv(policy)

        # value head
        value = feature[:, dim_policy:]
        value = torch.mean(value, dim=(2, 3))
        value = self.value_linear(value)
        value = self.value_final_linear(value)

        return value, policy

    @property
    def name(self):
        p, v = self.model_size
        return f"patnetv1_{self.embedding_type}_{p}p{v}v-b{self.board_size}"


@MODELS.register('patnetv2')
class PatNetv2(nn.Module):
    def __init__(self,
                 dim_policy=16,
                 dim_value=32,
                 board_size=15,
                 embedding_type='pcode-symboard',
                 **kwargs):
        super().__init__()
        self.model_size = (dim_policy, dim_value)
        self.board_size = board_size
        self.embedding_type = embedding_type
        in_dim = dim_policy + dim_value

        self.embedding = build_embedding(embedding_type, in_dim, board_size, **kwargs)
        self.conv = Conv2dBlock(in_dim,
                                in_dim,
                                ks=3,
                                st=1,
                                padding=1,
                                groups=in_dim,
                                activation='relu',
                                pad_type='zeros')

        # policy net
        self.policy_pw_conv = Conv2dBlock(dim_policy, dim_policy, ks=1, st=1, activation='relu')
        self.policy_dw_conv = Conv2dBlock(dim_policy,
                                          dim_policy,
                                          ks=3,
                                          st=1,
                                          padding=1,
                                          groups=dim_policy,
                                          activation='relu',
                                          pad_type='zeros')
        self.policy_final_conv = Conv2dBlock(dim_policy,
                                             1,
                                             ks=1,
                                             st=1,
                                             activation='none',
                                             bias=False)

        # value net
        self.value_pw_conv = Conv2dBlock(dim_value, dim_value, ks=1, st=1, activation='relu')
        self.value_linear1 = LinearBlock(dim_value, dim_value, activation='relu')
        self.value_linear2 = LinearBlock(dim_value, dim_value, activation='relu')
        self.value_final_linear = LinearBlock(dim_value, 3, activation='none')

    def forward(self, data):
        dim_policy, _ = self.model_size

        feature = self.embedding(data)
        # 1-layer 3x3 conv after embedding
        feature = self.conv(feature)

        # policy head
        policy = feature[:, :dim_policy]
        policy = self.policy_pw_conv(policy)
        policy = self.policy_dw_conv(policy)
        policy = self.policy_final_conv(policy)

        # value head
        value = feature[:, dim_policy:]
        value = self.value_pw_conv(value)
        value = torch.mean(value, dim=(2, 3))
        value = self.value_linear1(value)
        value = self.value_linear2(value)
        value = self.value_final_linear(value)

        return value, policy

    @property
    def name(self):
        p, v = self.model_size
        return f"patnetv2_{self.embedding_type}_{p}p{v}v-b{self.board_size}"


@MODELS.register('patnnuev1')
class PatNNUEv1(nn.Module):
    def __init__(self, dim_policy=4, dim_value=4) -> None:
        super().__init__()
        self.model_size = (dim_policy, dim_value)

        self.embedding = PatternCodeEmbedding(dim_policy + dim_value)

        # policy net
        self.policy_dw_conv = Conv2dBlock(dim_policy,
                                          dim_policy,
                                          ks=3,
                                          st=1,
                                          padding=1,
                                          groups=dim_policy,
                                          activation='relu',
                                          pad_type='zeros')
        self.policy_final_conv = Conv2dBlock(dim_policy,
                                             1,
                                             ks=1,
                                             st=1,
                                             activation='none',
                                             bias=False)

        # value net
        self.value_linear = nn.Sequential(
            LinearBlock(dim_value, dim_value, activation='lrelu/16'),
            LinearBlock(dim_value, dim_value, activation='lrelu/16'),
        )
        self.value_final_linear = LinearBlock(dim_value, 3, activation='none')

    def forward(self, data):
        dim_policy, _ = self.model_size

        feature = self.embedding(data)

        # policy head
        policy = feature[:, :dim_policy]
        policy = self.policy_dw_conv(policy)
        policy = self.policy_final_conv(policy)

        # value head
        value = feature[:, dim_policy:]
        value = torch.sum(value, dim=(2, 3))
        value = self.value_linear(value)
        value = self.value_final_linear(value)

        return value, policy

    @property
    def name(self):
        p, v = self.model_size
        return f"patnnuev1_{p}p{v}v"


@MODELS.register('patnnuev2')
class PatNNUEv2(nn.Module):
    def __init__(self, dim_policy=4, dim_value=4) -> None:
        super().__init__()
        self.model_size = (dim_policy, dim_value)
        dim_in = dim_policy + dim_value

        self.embedding = PatternCodeEmbedding(dim_in)

        # policy net
        self.policy_dw_conv = Conv2dBlock(dim_policy,
                                          dim_policy,
                                          ks=3,
                                          st=1,
                                          padding=1,
                                          groups=dim_policy,
                                          norm='bn',
                                          activation='relu',
                                          pad_type='zeros')
        self.policy_final_conv = Conv2dBlock(dim_policy,
                                             1,
                                             ks=1,
                                             st=1,
                                             activation='none',
                                             bias=False)

        # value net
        self.value_linear = nn.Sequential(
            LinearBlock(dim_in, dim_in, norm='bn', activation='lrelu/16'),
            LinearBlock(dim_in, dim_in, norm='bn', activation='lrelu/16'),
        )
        self.value_final_linear = LinearBlock(dim_in, 3, activation='none')

    def forward(self, data):
        dim_policy, _ = self.model_size

        feature = self.embedding(data)

        # policy head
        policy = feature[:, :dim_policy]
        policy = self.policy_dw_conv(policy)
        policy = self.policy_final_conv(policy)

        # value head
        value = torch.sum(feature, dim=(2, 3))
        value = self.value_linear(value)
        value = self.value_final_linear(value)

        return value, policy

    @property
    def name(self):
        p, v = self.model_size
        return f"patnnuev2_{p}p{v}v"
