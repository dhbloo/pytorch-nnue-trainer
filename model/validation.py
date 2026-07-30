"""Low-cost validation helpers for model input contracts."""

import torch


def validate_batch_shared_value(name, values, expected):
    """Validate batch metadata without reading a CUDA scalar on the host."""
    if values.numel() == 0:
        raise ValueError(f"{name} cannot be empty")
    matches = torch.all(values == expected)
    message = f"{name} does not match the model value {expected}"
    if matches.device.type == "cpu":
        if not matches:
            raise ValueError(message)
    else:
        torch._assert_async(matches, message)
