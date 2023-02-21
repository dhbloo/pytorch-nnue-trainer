import torch


def fake_quant(x: torch.Tensor, scale=128, zero_point=0, num_bits=8, signed=True):
    """Fake quantization while keep float gradient."""
    x_quant = (x.detach() * scale + zero_point).round().int()
    if num_bits is not None:
        if signed:
            qmin = -(2**(num_bits - 1))
            qmax = 2**(num_bits - 1) - 1
        else:
            qmin = 0
            qmax = 2**num_bits - 1
        x_quant = torch.clamp(x_quant, qmin, qmax)
    x_dequant = (x_quant - zero_point).float() / scale
    x = x - x.detach() + x_dequant  # stop gradient
    return x