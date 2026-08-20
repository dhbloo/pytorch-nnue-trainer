"""Exact consistency checks for replicated optimizer parameters."""

import torch
import torch.distributed as dist


_CHECK_CHUNK_BYTES = 16 * 1024 * 1024


def optimizer_parameters(optimizers):
    """Return uniquely owned optimizer parameters in a stable order."""
    parameters = []
    seen = set()
    for optimizer_name in sorted(optimizers):
        optimizer = optimizers[optimizer_name]
        for group_index, group in enumerate(optimizer.param_groups):
            for parameter_index, parameter in enumerate(group["params"]):
                identity = id(parameter)
                if identity in seen:
                    continue
                seen.add(identity)
                parameters.append(
                    (
                        f"{optimizer_name}.group{group_index}.param{parameter_index}",
                        parameter,
                    )
                )
    return parameters


def _parameter_manifest(parameters):
    return tuple(
        (
            label,
            tuple(parameter.shape),
            str(parameter.dtype),
            tuple(parameter.stride()),
            str(parameter.layout),
            parameter.device.type,
        )
        for label, parameter in parameters
    )


def _parameter_chunks(parameter, chunk_bytes):
    chunk_elements = max(1, chunk_bytes // parameter.element_size())
    flat = parameter.detach().view(-1) if parameter.is_contiguous() else None
    for start in range(0, parameter.numel(), chunk_elements):
        stop = min(start + chunk_elements, parameter.numel())
        if flat is not None:
            yield flat[start:stop]
        else:
            indices = torch.arange(start, stop, device=parameter.device)
            yield torch.take(parameter.detach(), indices)


@torch.no_grad()
def assert_replicated_parameters_equal(
    parameters,
    *,
    process_group=None,
    source_rank=0,
    chunk_bytes=_CHECK_CHUNK_BYTES,
):
    """Require exact parameter equality within a replicated process group.

    A small manifest collective first guarantees that every rank will issue
    the same tensor collectives. Parameter values are then compared against a
    rank-zero reference in bounded chunks, without modifying live weights.
    """
    parameters = list(parameters)
    world_size = dist.get_world_size(process_group)
    manifest = _parameter_manifest(parameters)
    manifests = [None] * world_size
    dist.all_gather_object(manifests, manifest, group=process_group)
    if any(candidate != manifest for candidate in manifests):
        raise RuntimeError(
            "replicated optimizer parameter layouts differ across ranks"
        )
    if not parameters:
        return
    if any(parameter.layout != torch.strided for _, parameter in parameters):
        raise RuntimeError(
            "replica consistency checks require strided optimizer parameters"
        )
    device_types = {parameter.device.type for _, parameter in parameters}
    if len(device_types) != 1:
        raise RuntimeError(
            "replica consistency checks require one parameter device type per rank"
        )

    mismatch = torch.zeros(
        (),
        dtype=torch.long,
        device=parameters[0][1].device,
    )
    rank = dist.get_rank()
    for _label, parameter in parameters:
        for local_chunk in _parameter_chunks(parameter, chunk_bytes):
            local_bytes = local_chunk.view(torch.uint8)
            reference = torch.empty_like(local_bytes)
            if rank == source_rank:
                reference.copy_(local_bytes)
            dist.broadcast(
                reference,
                src=source_rank,
                group=process_group,
            )
            mismatch.bitwise_or_(
                torch.any(local_bytes != reference).to(dtype=torch.long)
            )

    dist.all_reduce(mismatch, op=dist.ReduceOp.MAX, group=process_group)
    if mismatch.item():
        raise RuntimeError(
            "replicated optimizer parameters differ across ranks"
        )
