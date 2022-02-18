import torch
import torch.nn as nn


def build_input_plane(input_type):
    if input_type == 'basic':
        return BasicInputPlane(with_stm=True)
    elif input_type == 'basic-nostm':
        return BasicInputPlane(with_stm=False)
    elif input_type == 'raw':
        return lambda x: x  # identity transform
    else:
        assert 0, f"Unsupported input: {input_type}"


class BasicInputPlane(nn.Module):
    def __init__(self, with_stm=True):
        super().__init__()
        self.with_stm = with_stm

    def forward(self, data):
        board_input = data['board_input'].float()
        stm_input = data['stm_input'].float()

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
