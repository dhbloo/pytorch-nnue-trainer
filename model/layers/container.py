"""Containers for layers with non-standard call signatures."""

import torch.nn as nn


class SequentialWithExtraArguments(nn.Sequential):
    """Sequential container that forwards extra positional and keyword arguments to each module."""
    def forward(self, x, *args, **kwargs):
        for module in self:
            x = module(x, *args, **kwargs)
        return x
