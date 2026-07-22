"""Small CUDA runtime configuration helpers."""

import torch


def configure_cuda_memory_limit(
    device: torch.device,
    max_memory_fraction: float | None,
) -> int | None:
    """Apply a per-process allocator ceiling and return its byte limit."""
    if max_memory_fraction is None:
        return None
    if device.type != "cuda":
        raise ValueError("max_memory_fraction is only valid for CUDA devices")
    if not 0 < max_memory_fraction <= 1:
        raise ValueError("max_memory_fraction must be in the interval (0, 1]")

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.set_per_process_memory_fraction(max_memory_fraction, device_index)
    total_memory = torch.cuda.get_device_properties(device_index).total_memory
    return int(total_memory * max_memory_fraction)
