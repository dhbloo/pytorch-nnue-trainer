"""Export-only SizeBatchNorm adapter; training checkpoints remain unchanged."""

import torch
from torch import nn

from .normalization import SizeBatchNorm


class _SizeBNAffine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scales, biases, sizes):
        pairs = sizes.tolist()
        index = pairs.index([int(x.shape[-2]), int(x.shape[-1])])
        return x * scales[index].reshape(1, -1, 1, 1) + biases[index].reshape(1, -1, 1, 1)

    @staticmethod
    def symbolic(g, x, scales, biases, sizes):
        # This marker is lowered to standard ONNX operators before saving.
        return g.op("ntr::SizeBNAffine", x, scales, biases, sizes).setType(x.type())


class _ExportSizeBatchNorm(nn.Module):
    def __init__(self, source, board_sizes):
        super().__init__()
        scales, biases = [], []
        for side in board_sizes:
            bn = source.materialize(side, side)  # Reject missing or untrained buckets.
            scale = bn.weight.detach() * torch.rsqrt(bn.running_var + bn.eps)
            scales.append(scale)
            biases.append(bn.bias.detach() - bn.running_mean * scale)
        self.register_buffer("scales", torch.stack(scales))
        self.register_buffer("biases", torch.stack(biases))
        self.register_buffer("sizes", torch.tensor([[s, s] for s in board_sizes], dtype=torch.int64))

    def forward(self, x, mask=None):
        return _SizeBNAffine.apply(x, self.scales, self.biases, self.sizes)


def prepare_dynamic_size_batch_norms(model, board_sizes):
    """Replace eval SizeBN with immutable per-size affine tables for ONNX export."""
    targets = [(name, m) for name, m in model.named_modules() if isinstance(m, SizeBatchNorm)]
    if model.training or any(m.training for _, m in targets):
        raise ValueError("Dynamic sizebn export requires an evaluation model")
    replacements = [(name, _ExportSizeBatchNorm(m, board_sizes)) for name, m in targets]
    for name, replacement in replacements:
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, replacement)
    return len(replacements)
