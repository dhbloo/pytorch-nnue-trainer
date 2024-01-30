import torch


def fake_quant(x: torch.Tensor, scale=128, zero_point=0, num_bits=8, signed=True, floor=False):
    """Fake quantization while keep float gradient."""
    if num_bits is not None:
        if signed:
            qmin = -(2**(num_bits - 1))
            qmax = 2**(num_bits - 1) - 1
        else:
            qmin = 0
            qmax = 2**num_bits - 1
        x = torch.clamp(x, qmin / scale, qmax / scale)
    x_quant = (x.detach() * scale + zero_point)
    x_quant = x_quant.floor() if floor else x_quant.round()
    x_dequant = (x_quant - zero_point) / scale
    x = x - x.detach() + x_dequant  # stop gradient
    return x