"""Hash-based feature embedding layers."""

import math

import torch
import torch.nn as nn

from .linear import Linear
from utils.quant_utils import fake_quant


class HashLayer(nn.Module):
    """Maps an input space of size=(input_level**input_size) to a hashed feature space of size=(2**hash_logsize)."""

    def __init__(
        self,
        input_size,
        input_level=2,
        hash_logsize=20,
        dim_feature=32,
        quant_int8=True,
        sub_features=1,
        sub_divisor=2,
        scale_grad_by_freq=False,
    ):
        super().__init__()
        self.input_size = input_size
        self.input_level = input_level
        self.hash_logsize = hash_logsize
        self.dim_feature = dim_feature
        self.quant_int8 = quant_int8
        self.sub_features = sub_features
        self.sub_divisor = sub_divisor

        self.perfect_hash = 2**hash_logsize >= input_level**input_size
        if self.perfect_hash:  # Do perfect hashing
            n_features = input_level**input_size
            self.register_buffer("idx_stride", input_level ** torch.arange(input_size, dtype=torch.int64))
        else:
            n_features = 2**hash_logsize
            self.hashs = nn.Parameter(
                torch.empty((input_size, input_level), dtype=torch.int32),
                requires_grad=False,
            )

        n_features = [math.ceil(n_features / sub_divisor**i) for i in range(sub_features)]
        self.offsets = [sum(n_features[:i]) for i in range(sub_features + 1)]
        level_dim = max(dim_feature // sub_features, 1)
        feature_weight = torch.empty(sum(n_features), level_dim)
        self.features = nn.Embedding(
            sum(n_features),
            level_dim,
            scale_grad_by_freq=scale_grad_by_freq,
            _weight=feature_weight,
        )
        self.reset_parameters()
        if self.sub_features > 1:
            self.feature_mapping = Linear(level_dim * sub_features, dim_feature)

    def reset_parameters(self):
        with torch.no_grad():
            if not self.perfect_hash:
                ii32 = torch.iinfo(torch.int32)
                self.hashs.random_(ii32.min, ii32.max)
            if self.quant_int8:
                nn.init.trunc_normal_(self.features.weight, std=0.5, a=-1, b=127 / 128)
            else:
                nn.init.normal_(self.features.weight)

    def forward(self, x, x_long=None):
        """
        Args:
            x: float tensor of (batch_size, input_size), in range [0, 1].
            x_long: long tensor of (batch_size, input_size), in range [0, input_level-1].
                If not None, x_long will be used and x will be ignored.
        Returns:
            x_features: float tensor of (batch_size, dim_feature).
        """
        # Quantize input to [0, input_level-1] level
        if x_long is None:
            invalid = torch.any(~((x >= 0) & (x <= 1)), dim=1)
            x_long = torch.round((self.input_level - 1) * x).long()  # (batch_size, input_size)
        else:
            invalid = torch.any(
                (x_long < 0) | (x_long >= self.input_level), dim=1
            )

        if self.perfect_hash:
            x_indices = torch.sum(x_long * self.idx_stride, dim=1)  # (batch_size,)
            x_indices = torch.where(invalid, self.offsets[-1], x_indices)
        else:
            x_long = torch.where(
                invalid.unsqueeze(1), self.input_level, x_long
            )
            x_onthot = torch.zeros(
                (x_long.shape[0], self.input_size, self.input_level), dtype=torch.bool, device=x_long.device
            )
            x_onthot.scatter_(2, x_long.unsqueeze(-1), 1)  # (batch_size, input_size, input_level)
            x_hash = x_onthot * self.hashs  # (batch_size, input_size, input_level)
            x_hash = torch.sum(x_hash, dim=(1, 2))  # (batch_size,)

            x_indices = x_hash % (2**self.hash_logsize)  # (batch_size,)

        if self.sub_features > 1:
            x_features = []
            for i in range(self.sub_features):
                x_features.append(self.features(x_indices + self.offsets[i]))
                x_indices = torch.floor_divide(x_indices, self.sub_divisor)
            x_features = torch.cat(x_features, dim=1)  # (batch_size, level_dim * sub_features)
            x_features = self.feature_mapping(x_features)  # (batch_size, dim_feature)
        else:
            x_features = self.features(x_indices)  # (batch_size, dim_feature)

        if self.quant_int8:
            x_features = fake_quant(torch.clamp(x_features, min=-1, max=127 / 128), scale=128)
        return x_features
