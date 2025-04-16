import torch
import torch.nn as nn


def build_input_plane(input_type):
    if input_type == "basic":
        return BasicInputPlane(with_stm=True)
    elif input_type == "basic-nostm" or input_type == "basicns":
        return BasicInputPlane(with_stm=False)
    elif input_type == "mask":
        return MaskedInputPlane(with_stm=True)
    elif input_type == "mask-nostm" or input_type == "maskns":
        return MaskedInputPlane(with_stm=False)
    elif input_type.startswith("pcode"):
        input_type = input_type[5:]
        if input_type.endswith("-nostm"):
            with_stm = False
            input_type = input_type[:-6]
        else:
            with_stm = True
        feature_dim = int(input_type)
        return PatternCodeEmbeddingInputPlane(feature_dim, with_stm=with_stm)
    elif input_type.startswith("combpat"):
        input_type = input_type[7:]
        if input_type.endswith("-nostm"):
            with_stm = False
            input_type = input_type[:-6]
        else:
            with_stm = True
        feature_dim = int(input_type)
        return CombPatEmbeddingInputPlane(feature_dim, with_stm=with_stm)
    elif input_type.startswith("linepat"):
        input_type = input_type[7:]
        if input_type.endswith("-nostm"):
            with_stm = False
            input_type = input_type[:-6]
        else:
            with_stm = True
        feature_dim = int(input_type)
        return LinePatEmbeddingInputPlane(feature_dim, with_stm=with_stm)
    else:
        raise ValueError(f"Unsupported input: {input_type}")


class BasicInputPlane(nn.Module):
    def __init__(self, with_stm=True):
        super().__init__()
        self.with_stm = with_stm

    def forward(self, data):
        board_input = data["board_input"].float()
        stm_input = data["stm_input"]
        assert stm_input.dtype == torch.float32

        if self.with_stm:
            B, C, H, W = board_input.shape
            stm_input = stm_input.reshape(B, 1, 1, 1).expand(B, 1, H, W)
            input_plane = torch.cat([board_input, stm_input], dim=1)
        else:
            input_plane = board_input

        return input_plane

    @property
    def dim_plane(self):
        return 2 + self.with_stm


class MaskedInputPlane(BasicInputPlane):
    def __init__(self, with_stm=True):
        super().__init__(with_stm)

    def forward(self, data):
        input_plane = super().forward(data)

        board_size = data["board_size"]
        B, C, H, W = data["board_input"].shape

        rows = torch.arange(H, device=input_plane.device).view(1, H, 1)  # [1, H, 1]
        cols = torch.arange(W, device=input_plane.device).view(1, 1, W)  # [1, 1, W]

        mask_rows = rows < board_size[:, 0].view(B, 1, 1)  # [B, H, 1]
        mask_cols = cols < board_size[:, 1].view(B, 1, 1)  # [B, 1, W]
        mask_plane = (mask_rows & mask_cols).unsqueeze(1)  # [B, 1, H, W]

        return input_plane, mask_plane.to(input_plane.dtype)


class PatternCodeEmbeddingInputPlane(BasicInputPlane):
    def __init__(self, feature_dim, pcode_dim=2380, with_basic=True, with_stm=True):
        super().__init__(with_stm)
        self.feature_dim = feature_dim
        self.pcode_dim = pcode_dim
        self.with_basic = with_basic
        self.pcode_embedding = nn.Embedding(num_embeddings=pcode_dim, embedding_dim=feature_dim)

    def forward(self, data):
        assert torch.all(self.pcode_dim == data["sparse_feature_dim"][:, 10:12])

        # convert sparse input to dense feature through embedding
        pcode_sparse_input = data["sparse_feature_input"][:, [10, 11]].int()  # [B, 2, H, W]
        pcode_feature = self.pcode_embedding(pcode_sparse_input)  # [B, 2, H, W, feature_dim]
        pcode_feature = torch.sum(pcode_feature, dim=1, keepdim=False)  # [B, H, W, feature_dim]
        pcode_feature = torch.permute(pcode_feature, (0, 3, 1, 2))  # [B, feature_dim, H, W]

        # mask out feature at non-empty cell
        board_input = data["board_input"]  # [B, 2, H, W]
        non_empty_mask = board_input[:, 0] + board_input[:, 1] > 0  # [B, H, W]
        non_empty_mask = torch.unsqueeze(non_empty_mask, dim=1)  # [B, 1, H, W]
        pcode_feature = torch.masked_fill(pcode_feature, non_empty_mask, 0)

        if self.with_basic:
            input_plane = super().forward(data)
            input_plane = torch.cat([input_plane, pcode_feature], dim=1)  # [B, dim_plane, H, W]
        else:
            input_plane = pcode_feature
        return input_plane

    @property
    def dim_plane(self):
        base_dim = super().dim_plane if self.with_basic else 0
        return base_dim + self.feature_dim


class CombPatEmbeddingInputPlane(BasicInputPlane):
    def __init__(self, feature_dim, p4_dim=14, with_basic=True, with_stm=True):
        super().__init__(with_stm)
        self.feature_dim = feature_dim
        self.p4_dim = p4_dim
        self.with_basic = with_basic
        self.p4_embedding = nn.Embedding(num_embeddings=p4_dim, embedding_dim=feature_dim)

    def forward(self, data):
        assert torch.all(self.p4_dim == data["sparse_feature_dim"][:, 8:10])

        # convert sparse input to dense feature through embedding
        p4_sparse_input = data["sparse_feature_input"][:, [8, 9]].int()  # [B, 2, H, W]
        p4_feature = self.p4_embedding(p4_sparse_input)  # [B, 2, H, W, feature_dim]
        p4_feature = torch.sum(p4_feature, dim=1, keepdim=False)  # [B, H, W, feature_dim]
        p4_feature = torch.permute(p4_feature, (0, 3, 1, 2))  # [B, feature_dim, H, W]

        # mask out feature at non-empty cell
        board_input = data["board_input"]  # [B, 2, H, W]
        non_empty_mask = board_input[:, 0] + board_input[:, 1] > 0  # [B, H, W]
        non_empty_mask = torch.unsqueeze(non_empty_mask, dim=1)  # [B, 1, H, W]
        p4_feature = torch.masked_fill(p4_feature, non_empty_mask, 0)

        if self.with_basic:
            input_plane = super().forward(data)
            input_plane = torch.cat([input_plane, p4_feature], dim=1)  # [B, dim_plane, H, W]
        else:
            input_plane = p4_feature
        return input_plane

    @property
    def dim_plane(self):
        base_dim = super().dim_plane if self.with_basic else 0
        return base_dim + self.feature_dim


class LinePatEmbeddingInputPlane(BasicInputPlane):
    def __init__(self, feature_dim, p_dim=14, with_basic=True, with_stm=True):
        super().__init__(with_stm)
        self.feature_dim = feature_dim
        self.p_dim = p_dim
        self.with_basic = with_basic
        self.p_embedding = nn.Embedding(num_embeddings=p_dim, embedding_dim=feature_dim)

    def forward(self, data):
        assert torch.all(self.p_dim == data["sparse_feature_dim"][:, 0:8])

        # convert sparse input to dense feature through embedding
        p_sparse_input = data["sparse_feature_input"][:, 0:8].int()  # [B, 8, H, W]
        p_feature = self.p_embedding(p_sparse_input)  # [B, 8, H, W, feature_dim]
        p_feature_self = torch.permute(p_feature[:, 0:4], (0, 2, 3, 1, 4)).flatten(3)  # [B, H, W, 4*feature_dim]
        p_feature_oppo = torch.permute(p_feature[:, 4:8], (0, 2, 3, 1, 4)).flatten(3)  # [B, H, W, 4*feature_dim]
        p_feature = p_feature_self + p_feature_oppo  # [B, H, W, 4*feature_dim]
        p_feature = torch.permute(p_feature, (0, 3, 1, 2))  # [B, 4*feature_dim, H, W]

        # mask out feature at non-empty cell
        board_input = data["board_input"]  # [B, 2, H, W]
        non_empty_mask = board_input[:, 0] + board_input[:, 1] > 0  # [B, H, W]
        non_empty_mask = torch.unsqueeze(non_empty_mask, dim=1)  # [B, 1, H, W]
        p_feature = torch.masked_fill(p_feature, non_empty_mask, 0)

        if self.with_basic:
            input_plane = super().forward(data)
            input_plane = torch.cat([input_plane, p_feature], dim=1)  # [B, dim_plane, H, W]
        else:
            input_plane = p_feature
        return input_plane

    @property
    def dim_plane(self):
        base_dim = super().dim_plane if self.with_basic else 0
        return base_dim + self.feature_dim * 4
