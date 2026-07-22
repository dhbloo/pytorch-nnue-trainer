"""Directional convolution layers for board-shaped inputs."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectionalConvLayer(nn.Module):
    """Apply one learned three-tap kernel along each board direction."""

    def __init__(
        self,
        dim_in,
        dim_out,
        use_nonzero_padding=False,
        use_channel_last=True,
        fix_direction_order=False,
    ):
        super().__init__()
        weight_shape = (dim_out, dim_in, 3) if use_channel_last else (3, dim_out, dim_in)
        self.weight = nn.Parameter(torch.empty(weight_shape))
        self.bias = nn.Parameter(torch.zeros((dim_out,)))
        self.use_nonzero_padding = use_nonzero_padding
        self.use_channel_last = use_channel_last
        self.fix_direction_order = fix_direction_order

    def initialize(self):
        if self.use_channel_last:
            nn.init.kaiming_normal_(self.weight.data)
        else:
            nn.init.kaiming_normal_(self.weight.data.permute(1, 0, 2))
        self.bias.data.zero_()

    def _conv1d_direction(self, x, dir):
        if self.use_channel_last:
            dim_out, dim_in, kernel_size = self.weight.shape
            weight_1d = self.weight
        else:
            kernel_size, dim_out, dim_in = self.weight.shape
            weight_1d = self.weight.permute(1, 2, 0)
        # plain int: under torch.jit.trace shape elements are traced tensors,
        # and conv2d/F.pad reject a 0-dim tensor as padding
        kernel_size = int(kernel_size)

        # Horizontal and vertical directions map directly to rectangular
        # convolutions.  Avoid embedding their three taps into a dense 3x3
        # kernel whose other six weights are zero; cuDNN otherwise performs
        # all nine multiplies in both the forward and backward passes.
        if not self.use_nonzero_padding and dir == 0:
            return F.conv2d(x, weight_1d.unsqueeze(2), self.bias, padding=(0, kernel_size // 2))
        if not self.use_nonzero_padding and dir == 1:
            return F.conv2d(x, weight_1d.unsqueeze(3), self.bias, padding=(kernel_size // 2, 0))

        # A dense 3x3 convolution wastes two thirds of its work for a diagonal
        # three-tap kernel.  Materialize the three shifted inputs next to each
        # other in the channel dimension so the actual computation is a 1x1
        # convolution.  Inductor fuses the padding/slicing/concatenation into a
        # single channels-last copy and the convolution uses a tensor-core GEMM.
        if not self.use_nonzero_padding and dir >= 2:
            height, width = x.shape[-2:]
            padded = F.pad(x, (1, 1, 1, 1))
            if dir == (3 if self.fix_direction_order else 2):
                shifted = (
                    padded[:, :, :height, :width],
                    x,
                    padded[:, :, 2 : height + 2, 2 : width + 2],
                )
            else:
                shifted = (
                    padded[:, :, 2 : height + 2, :width],
                    x,
                    padded[:, :, :height, 2 : width + 2],
                )
            input_1x1 = torch.cat(shifted, dim=1)
            weight_1x1 = torch.cat(tuple(weight_1d.unbind(dim=2)), dim=1)
            return F.conv2d(input_1x1, weight_1x1[:, :, None, None], self.bias)

        zero = torch.zeros((dim_out, dim_in), dtype=self.weight.dtype, device=self.weight.device)

        if self.use_channel_last:
            w_0_, w_1_, w_2_ = self.weight[..., 0], self.weight[..., 1], self.weight[..., 2]
        else:
            w_0_, w_1_, w_2_ = self.weight[0], self.weight[1], self.weight[2]

        if self.fix_direction_order:
            weight_func_map = [
                lambda w: (zero, zero, zero, w_0_, w_1_, w_2_, zero, zero, zero),
                lambda w: (zero, w_0_, zero, zero, w_1_, zero, zero, w_2_, zero),
                lambda w: (zero, zero, w_2_, zero, w_1_, zero, w_0_, zero, zero),
                lambda w: (w_0_, zero, zero, zero, w_1_, zero, zero, zero, w_2_),
            ]
        else:
            weight_func_map = [
                lambda w: (zero, zero, zero, w_0_, w_1_, w_2_, zero, zero, zero),
                lambda w: (zero, w_0_, zero, zero, w_1_, zero, zero, w_2_, zero),
                lambda w: (w_0_, zero, zero, zero, w_1_, zero, zero, zero, w_2_),
                lambda w: (zero, zero, w_2_, zero, w_1_, zero, w_0_, zero, zero),
            ]

        weight = torch.stack(weight_func_map[dir](self.weight), dim=2)
        weight = weight.reshape(dim_out, dim_in, kernel_size, kernel_size)
        if self.use_nonzero_padding:
            x = F.pad(
                x,
                pad=(kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2),
                mode="constant",
                value=1,
            )
            return torch.conv2d(x, weight, self.bias, padding=0)
        return torch.conv2d(x, weight, self.bias, padding=kernel_size // 2)

    def forward(self, x):
        assert len(x) == 4, f"must be 4 directions, got {len(x)}"
        return tuple((self._conv1d_direction(xi, i) if xi is not None else None) for i, xi in enumerate(x))
