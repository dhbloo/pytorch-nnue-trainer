import random
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from itertools import chain

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import IterableDataset, Sampler, default_convert
from torch.utils.data.dataloader import DataLoader

from utils.fused_adamw import MultiTensorAdamW, multi_tensor_adamw_available
from utils.misc_utils import Registry


def _seed_dataset_worker(worker_id, *, root_seed, rank):
    from dataset.core import rng_u64

    seed = rng_u64(root_seed, "worker_rng", (rank, worker_id))
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    torch.manual_seed(seed)


OPTIMIZERS = Registry("optimizer")
"""Registry of optimizer factories.

Each entry is a callable ``(parameters, model_or_models, lr, weight_decay, **kwargs)
-> Optimizer``, where *parameters* is the (already filtered) parameter list and
*model_or_models* is the original model or list of models for factories that
need module structure (e.g. muon's per-parameter routing).
"""


def clip_grad_norm(parameters, max_norm: float, norm_type: float = 2.0):
    """Streamlined gradient clipping for the single-(device, dtype) case.

    Numerically identical to ``torch.nn.utils.clip_grad_norm_`` with
    ``error_if_nonfinite=False`` (same per-tensor foreach norms, same total
    norm, same clamped coefficient scaled in place), but skips the per-step
    python ceremony of the stock implementation (device/dtype regrouping,
    generator plumbing, wrapper dispatch), which costs several hundred
    microseconds per step on high-latency hosts.  Parameters spanning
    multiple devices or dtypes fall back to the stock implementation.
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        # Mirrors torch.nn.utils._get_total_norm on the empty-grads path.
        return torch.tensor(0.0)
    first_device, first_dtype = grads[0].device, grads[0].dtype
    for grad in grads[1:]:
        if grad.device != first_device or grad.dtype != first_dtype:
            return torch.nn.utils.clip_grad_norm_(
                parameters, max_norm, norm_type=norm_type
            )
    norms = torch._foreach_norm(grads, norm_type)
    total_norm = torch.linalg.vector_norm(torch.stack(norms), norm_type)
    clip_coef = float(max_norm) / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    torch._foreach_mul_(grads, clip_coef_clamped)
    return total_norm


# Keywords for which MultiTensorAdamW is the better implementation. Anything
# else falls back to torch.optim.AdamW: the implementation selectors (`fused`,
# `foreach`, `capturable`, ...), which is also the documented opt-out, and the
# `amsgrad` / `maximize` variants, which MultiTensorAdamW computes correctly but
# one parameter at a time.
MULTI_TENSOR_ADAMW_ARGS = frozenset({"lr", "betas", "eps", "weight_decay"})


@OPTIMIZERS.register("adamw")
def _make_adamw(parameters, model, lr, weight_decay, **kwargs):
    args = {"lr": lr, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": weight_decay}
    args.update(kwargs)
    # torch's fused AdamW costs ~0.48 ms per step on every network trained here,
    # independently of parameter count, because its kernel recomputes
    # pow(beta, step) in double precision per thread and sm_89 runs FP64 at 1/64
    # rate. MultiTensorAdamW is one Triton launch with the same per-parameter
    # state layout; pass `fused: false` (or `fused: true`) to select torch's.
    if set(args) <= MULTI_TENSOR_ADAMW_ARGS and multi_tensor_adamw_available():
        return MultiTensorAdamW(parameters, **args)
    if "fused" not in args and "foreach" not in args and torch.cuda.is_available():
        args["fused"] = True
    try:
        return optim.AdamW(parameters, **args)
    except (RuntimeError, ValueError):
        # older torch versions reject fused=True for the CPU parameters the
        # optimizer is constructed with (accelerate moves them to CUDA later)
        if not args.get("fused") or "fused" in kwargs:
            raise
        args.pop("fused")
        return optim.AdamW(parameters, **args)


@OPTIMIZERS.register("adamw-ams")
def _make_adamw_ams(parameters, model, lr, weight_decay, **kwargs):
    args = {"lr": lr, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": weight_decay, "amsgrad": True}
    args.update(kwargs)
    return optim.AdamW(parameters, **args)


@OPTIMIZERS.register("sgd")
def _make_sgd(parameters, model, lr, weight_decay, **kwargs):
    args = {"lr": lr, "momentum": 0, "dampening": 0, "weight_decay": weight_decay}
    args.update(kwargs)
    return optim.SGD(parameters, **args)


@OPTIMIZERS.register("sgd-momentum")
def _make_sgd_momentum(parameters, model, lr, weight_decay, **kwargs):
    args = {"lr": lr, "momentum": 0.9, "dampening": 0.1, "nesterov": False, "weight_decay": weight_decay}
    args.update(kwargs)
    return optim.SGD(parameters, **args)


@OPTIMIZERS.register("sgd-nesterov")
def _make_sgd_nesterov(parameters, model, lr, weight_decay, **kwargs):
    # Nesterov momentum requires zero dampening
    args = {"lr": lr, "momentum": 0.9, "dampening": 0, "nesterov": True, "weight_decay": weight_decay}
    args.update(kwargs)
    return optim.SGD(parameters, **args)


@OPTIMIZERS.register("muon-adamw")
def _make_muon_adamw(parameters, model, lr, weight_decay, **kwargs):
    from utils.muon import Muon, get_params_for_muon
    from utils.chained_optimizer import ChainedOptimizer, OptimizerSpec

    models = model if isinstance(model, (list, tuple)) else [model]
    params_id_to_name = {}
    muon_params_id_set = set()
    for m in models:
        params_id_to_name.update({id(p): name for name, p in m.named_parameters()})
        muon_params_id_set.update(id(p) for p in get_params_for_muon(m))
    # Default Muon's weight_decay to 1e-2 only when unset; a configured value
    # (including 0.0) must be respected, not floored.
    muon_args = {"weight_decay": 1e-2 if weight_decay is None else weight_decay}
    muon_args.update(kwargs.pop("muon_args", {}))
    adamw_args = {"betas": (0.9, 0.999), "eps": 1e-8}
    adamw_args.update(kwargs.pop("adamw_args", {}))
    # Muon routes biases and normalization parameters to AdamW.  ResNets contain
    # many such small tensors, which is exactly the case where torch's fused
    # kernel is pure fixed cost: its per-thread double-precision bias correction
    # makes every launch cost the same regardless of how little work it carries.
    if set(adamw_args) <= MULTI_TENSOR_ADAMW_ARGS and multi_tensor_adamw_available():
        adamw_class = MultiTensorAdamW
    else:
        adamw_class = optim.AdamW
        if "fused" not in adamw_args and "foreach" not in adamw_args and torch.cuda.is_available():
            adamw_args["fused"] = True
    spec_muon = OptimizerSpec(Muon, muon_args, lambda param: id(param) in muon_params_id_set)
    spec_adamw = OptimizerSpec(adamw_class, adamw_args, None)
    specs = [spec_muon, spec_adamw]
    callback = None
    if kwargs.pop("verbose", False):
        callback = lambda p, spec_idx: print(
            f"Adding param {params_id_to_name[id(p)]} ({p.shape}) to "
            f"optimizer{spec_idx} {str(specs[spec_idx].class_type)}"
        )
    kwargs.update({"lr": lr, "weight_decay": weight_decay, "optimizer_selection_callback": callback})
    return ChainedOptimizer(parameters, specs, **kwargs)


def build_optimizer(
    optim_type: str,
    model: torch.nn.Module | list[torch.nn.Module],
    lr: float,
    weight_decay: float = 0.0,
    only_track_requires_grad=True,
    **kwargs,
):
    if optim_type not in OPTIMIZERS:
        raise ValueError(f"Unsupported optimizer: {optim_type}")

    if isinstance(model, (list, tuple)):
        parameters = chain(*(m.parameters() for m in model))
    else:
        parameters = model.parameters()
    if only_track_requires_grad:
        # only track parameters with requires_grad=True
        parameters = [p for p in parameters if p.requires_grad]
    else:
        # materialize so factories can retry construction (e.g. fused fallback)
        parameters = list(parameters)

    return OPTIMIZERS[optim_type](parameters, model, lr, weight_decay, **kwargs)


def build_lr_scheduler(optimizer, lr_schedule_type, iterations, last_it=-1, **kwargs):
    if lr_schedule_type == "constant":
        scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=iterations, last_epoch=last_it)
    elif lr_schedule_type == "step":
        step_size = kwargs.get("step_size", 50000)
        step_gamma = kwargs.get("step_gamma", 0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=step_gamma, last_epoch=last_it)
    elif lr_schedule_type == "cosine":
        eta_min = kwargs.get("eta_min", 1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations, eta_min=eta_min, last_epoch=last_it)
    else:
        raise ValueError(f"Unsupported lr scheduler: {lr_schedule_type}")

    return scheduler


class DeviceLoaderWrapper:
    """Iterate *dataloader* and move each batch to *device*.

    Minimal replacement for accelerate's dataloader preparation for iterable
    datasets that already partition work across ranks and workers themselves:
    only device placement is needed, without a second rank-sharding layer.
    """

    def __init__(self, dataloader, device, non_blocking=True):
        self.dataloader = dataloader
        self.device = device
        self.non_blocking = non_blocking

    @property
    def dataset(self):
        return self.dataloader.dataset

    def __iter__(self):
        from accelerate.utils import send_to_device
        from dataclasses import replace
        from dataset.stream import BatchEnvelope

        for batch in self.dataloader:
            if isinstance(batch, BatchEnvelope):
                yield replace(
                    batch,
                    data=send_to_device(
                        default_convert(batch.data),
                        self.device,
                        non_blocking=self.non_blocking,
                    ),
                    is_real=send_to_device(
                        torch.as_tensor(batch.is_real),
                        self.device,
                        non_blocking=self.non_blocking,
                    ),
                )
            else:
                yield send_to_device(batch, self.device, non_blocking=self.non_blocking)


class ObservedDeviceLoaderWrapper(DeviceLoaderWrapper):
    """Device placement with opt-in source-wait and copy-launch metrics."""

    def __init__(self, dataloader, device, pipeline_stats, non_blocking=True):
        super().__init__(dataloader, device, non_blocking=non_blocking)
        self.pipeline_stats = pipeline_stats

    def __iter__(self):
        import time
        from accelerate.utils import send_to_device
        from dataclasses import replace
        from dataset.stream import BatchEnvelope
        from dataset.telemetry import tensor_bytes

        iterator = iter(self.dataloader)
        while True:
            wait_start = time.perf_counter_ns()
            try:
                batch = next(iterator)
            except StopIteration:
                return
            self.pipeline_stats.record_source_wait(
                time.perf_counter_ns() - wait_start
            )
            transfer_start = time.perf_counter_ns()
            if isinstance(batch, BatchEnvelope):
                converted = default_convert(batch.data)
                byte_count = tensor_bytes(converted) + tensor_bytes(batch.is_real)
                batch = replace(
                    batch,
                    data=send_to_device(
                        converted,
                        self.device,
                        non_blocking=self.non_blocking,
                    ),
                    is_real=send_to_device(
                        torch.as_tensor(batch.is_real),
                        self.device,
                        non_blocking=self.non_blocking,
                    ),
                )
            else:
                byte_count = tensor_bytes(batch)
                batch = send_to_device(
                    batch,
                    self.device,
                    non_blocking=self.non_blocking,
                )
            self.pipeline_stats.record_h2d(
                time.perf_counter_ns() - transfer_start,
                byte_count,
            )
            yield batch


def _move_to_device_and_collect(
    value,
    device,
    non_blocking,
    cuda_tensors,
):
    """Move one nested value while collecting its CUDA tensor leaves."""
    if isinstance(value, torch.Tensor):
        moved = value.to(device, non_blocking=non_blocking)
        if moved.device.type == "cuda":
            cuda_tensors.append(moved)
        return moved
    if hasattr(value, "to"):
        try:
            moved = value.to(device, non_blocking=non_blocking)
        except TypeError:
            moved = value.to(device)
        if isinstance(moved, torch.Tensor) and moved.device.type == "cuda":
            cuda_tensors.append(moved)
        return moved
    if isinstance(value, (list, tuple)):
        moved = (
            _move_to_device_and_collect(
                item,
                device,
                non_blocking,
                cuda_tensors,
            )
            for item in value
        )
        if hasattr(value, "_fields"):
            return type(value)(*moved)
        return type(value)(moved)
    if isinstance(value, Mapping):
        return type(value)(
            {
                key: _move_to_device_and_collect(
                    item,
                    device,
                    non_blocking,
                    cuda_tensors,
                )
                for key, item in value.items()
            }
        )
    return value


class CudaPrefetchLoaderWrapper(DeviceLoaderWrapper):
    """Bounded H2D lookahead on a dedicated CUDA stream.

    ``prefetch_batches=1`` is double buffering: the currently consumed batch
    plus one queued batch. Host batches remain referenced until their copy
    event completes, independently of how far the CPU runs ahead of the GPU.
    """

    MAX_PREFETCH_BATCHES = 4
    MAX_PENDING_TIMINGS = 1024

    def __init__(
        self,
        dataloader,
        device,
        *,
        prefetch_batches: int = 1,
        pipeline_stats=None,
        non_blocking=True,
    ):
        super().__init__(dataloader, device, non_blocking=non_blocking)
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("CUDA prefetch requires a CUDA device")
        if (
            type(prefetch_batches) is not int
            or not 1 <= prefetch_batches <= self.MAX_PREFETCH_BATCHES
        ):
            raise ValueError(
                "cuda_prefetch_batches must be an integer in [1, 4]"
            )
        self.prefetch_batches = prefetch_batches
        self.pipeline_stats = pipeline_stats
        self._batch_tensor_keys = None
        self.prefetch_audit = {
            "configured_batches": prefetch_batches,
            "max_queued_batches": 0,
            "max_retired_host_batches": 0,
            "submitted_batches": 0,
            "completed_batches": 0,
        }

    def _move_batch(self, batch, device, non_blocking):
        from dataclasses import replace
        from dataset.stream import BatchEnvelope

        cuda_tensors = []
        if isinstance(batch, BatchEnvelope):
            data = batch.data
            if type(data) is dict and all(
                isinstance(value, torch.Tensor) for value in data.values()
            ):
                keys = tuple(data)
                if self._batch_tensor_keys is None:
                    self._batch_tensor_keys = keys
                if keys == self._batch_tensor_keys:
                    converted = data
                    device_data = {}
                    for key in keys:
                        moved = data[key].to(
                            device,
                            non_blocking=non_blocking,
                        )
                        device_data[key] = moved
                        cuda_tensors.append(moved)
                else:
                    converted = default_convert(data)
                    device_data = _move_to_device_and_collect(
                        converted,
                        device,
                        non_blocking,
                        cuda_tensors,
                    )
            else:
                converted = default_convert(data)
                device_data = _move_to_device_and_collect(
                    converted,
                    device,
                    non_blocking,
                    cuda_tensors,
                )
            converted_mask = torch.as_tensor(batch.is_real)
            device_mask = converted_mask.to(
                device,
                non_blocking=non_blocking,
            )
            cuda_tensors.append(device_mask)
            return (
                replace(
                    batch,
                    data=device_data,
                    is_real=device_mask,
                ),
                (converted, converted_mask),
                tuple(cuda_tensors),
            )
        device_batch = _move_to_device_and_collect(
            batch,
            device,
            non_blocking,
            cuda_tensors,
        )
        return (
            device_batch,
            batch,
            tuple(cuda_tensors),
        )

    @staticmethod
    def _record_tensor_streams(cuda_tensors, stream) -> None:
        for tensor in cuda_tensors:
            tensor.record_stream(stream)

    @staticmethod
    def _is_terminal_batch(batch) -> bool:
        """Return whether a transactional envelope ends its source epoch."""
        token = getattr(batch, "token", None)
        planned_batch = getattr(token, "batch", None)
        if planned_batch is not None:
            return bool(getattr(planned_batch, "is_last", False))
        return bool(getattr(token, "is_last", False))

    @staticmethod
    def _synchronize_and_close(prefetch_stream, source) -> None:
        """Close the source even when an asynchronous CUDA failure surfaces."""
        sync_error = None
        try:
            prefetch_stream.synchronize()
        except BaseException as error:
            sync_error = error
        close = getattr(source, "close", None)
        try:
            if close is not None:
                close()
        except BaseException as close_error:
            if sync_error is None:
                raise
            sync_error.add_note(
                "Source close also failed after CUDA synchronization: "
                f"{type(close_error).__name__}: {close_error}"
            )
        if sync_error is not None:
            raise sync_error

    def __iter__(self):
        import time

        source = iter(self.dataloader)
        prefetch_stream = torch.cuda.Stream(device=self.device)
        pending = deque()
        retired_hosts = deque()
        timing_records = deque()
        exhausted = False
        stats = self.pipeline_stats
        if stats is not None:
            from dataset.stream import BatchEnvelope
            from dataset.telemetry import tensor_bytes

            def batch_bytes(batch):
                if isinstance(batch, BatchEnvelope):
                    return tensor_bytes(batch.data) + tensor_bytes(batch.is_real)
                return tensor_bytes(batch)

        def flush_completed_timings():
            if stats is None:
                return
            while timing_records and timing_records[0][3].query():
                copy_start, copy_end, wait_start, wait_end = timing_records.popleft()
                stats.record_cuda_prefetch(
                    copy_start.elapsed_time(copy_end),
                    wait_start.elapsed_time(wait_end),
                )

        def release_completed_hosts():
            while retired_hosts and retired_hosts[0][0].query():
                retired_hosts.popleft()
            if len(retired_hosts) >= self.prefetch_batches + 1:
                retired_hosts[0][0].synchronize()
                retired_hosts.popleft()

        def enqueue_one():
            nonlocal exhausted
            if exhausted:
                return False
            wait_start_ns = time.perf_counter_ns() if stats is not None else None
            try:
                host_batch = next(source)
            except StopIteration:
                exhausted = True
                return False
            if stats is not None:
                stats.record_source_wait(
                    time.perf_counter_ns() - wait_start_ns
                )
                byte_count = batch_bytes(host_batch)
                launch_start_ns = time.perf_counter_ns()
            with torch.cuda.stream(prefetch_stream):
                copy_start = (
                    torch.cuda.Event(enable_timing=True)
                    if stats is not None
                    else None
                )
                if copy_start is not None:
                    copy_start.record(prefetch_stream)
                device_batch, transfer_owner, cuda_tensors = self._move_batch(
                    host_batch,
                    self.device,
                    self.non_blocking,
                )
                copy_end = torch.cuda.Event(enable_timing=stats is not None)
                copy_end.record(prefetch_stream)
            if stats is not None:
                stats.record_h2d(
                    time.perf_counter_ns() - launch_start_ns,
                    byte_count,
                )
            pending.append(
                (
                    device_batch,
                    copy_start,
                    copy_end,
                    transfer_owner,
                    cuda_tensors,
                )
            )
            # Do not probe a transactional source beyond its terminal envelope.
            # Leaving its generator suspended lets close() roll back queued but
            # uncommitted lookahead, including at the exact end of an epoch.
            if self._is_terminal_batch(host_batch):
                exhausted = True
            self.prefetch_audit["submitted_batches"] += 1
            self.prefetch_audit["max_queued_batches"] = max(
                self.prefetch_audit["max_queued_batches"],
                len(pending),
            )
            return True

        try:
            while len(pending) < self.prefetch_batches and enqueue_one():
                pass
            while pending:
                flush_completed_timings()
                release_completed_hosts()
                (
                    device_batch,
                    copy_start,
                    copy_end,
                    transfer_owner,
                    cuda_tensors,
                ) = pending.popleft()
                current_stream = torch.cuda.current_stream(self.device)
                if stats is not None:
                    wait_start = torch.cuda.Event(enable_timing=True)
                    wait_end = torch.cuda.Event(enable_timing=True)
                    wait_start.record(current_stream)
                current_stream.wait_event(copy_end)
                if stats is not None:
                    wait_end.record(current_stream)
                    timing_records.append(
                        (copy_start, copy_end, wait_start, wait_end)
                    )
                    while len(timing_records) > self.MAX_PENDING_TIMINGS:
                        timing_records.popleft()
                self._record_tensor_streams(cuda_tensors, current_stream)
                retired_hosts.append((copy_end, transfer_owner))
                self.prefetch_audit["max_retired_host_batches"] = max(
                    self.prefetch_audit["max_retired_host_batches"],
                    len(retired_hosts),
                )
                self.prefetch_audit["completed_batches"] += 1
                while len(pending) < self.prefetch_batches and enqueue_one():
                    pass
                yield device_batch
            flush_completed_timings()
        finally:
            self._synchronize_and_close(prefetch_stream, source)


class StaticSlotLoaderWrapper(DeviceLoaderWrapper):
    """H2D every batch into one fixed set of device tensors on a side stream.

    When a ``torch.compile`` region runs under Inductor CUDA graphs (the
    trainer default), every input whose data pointer changes between steps is
    first copied into graph-owned storage: one tiny DtoD memcpy plus
    host-side ceremony per tensor per step, which on high-latency hosts
    serializes into several hundred microseconds of compute-stream idle time
    per step. Delivering the batch in *persistent* device slots marked with
    ``torch._dynamo.mark_static_address`` skips that stabilization entirely;
    the only per-tensor work left is the H2D copy itself on a dedicated copy
    stream.

    Ordering is guaranteed by two events per step: the compute stream waits
    for "copy ready", and the next copy waits for "previous consumption
    recorded" (enqueued after the consumer's forward/backward launches), so
    a slot is never overwritten while its contents can still be read. A fresh
    iterator (epoch restart) starts by ordering its first refill behind the
    compute stream's current tail, which covers the previous epoch's final
    step still executing against the slots.

    Requires a constant batch schema (keys, shapes, dtypes) for the whole
    run, which the planned-batch datasets already guarantee.
    """

    def __init__(self, dataloader, device, *, pipeline_stats=None, non_blocking=True):
        super().__init__(dataloader, device, non_blocking)
        self._copy_stream = torch.cuda.Stream(device=self.device)
        self._pipeline_stats = pipeline_stats
        self._slots = None
        self._schema = None

    def _resolve_schema(self, data, mask):
        if not (type(data) is dict and all(
            isinstance(value, torch.Tensor) for value in data.values()
        )):
            raise RuntimeError(
                "StaticSlotLoaderWrapper requires dict[str, Tensor] batches"
            )
        entries = list(data.items())
        if mask is not None:
            entries.append(("__is_real", torch.as_tensor(mask)))
        return tuple(
            (name, tuple(t.shape), t.dtype) for name, t in entries
        ), entries

    def _build_slots(self, entries):
        slots = {}
        for name, tensor in entries:
            slot = torch.empty(
                tuple(tensor.shape), dtype=tensor.dtype, device=self.device
            )
            torch._dynamo.mark_static_address(slot)
            slots[name] = slot
        return slots

    def __iter__(self):
        import time
        from dataclasses import replace

        from dataset.stream import BatchEnvelope
        from dataset.telemetry import tensor_bytes

        source = iter(self.dataloader)
        compute_stream = torch.cuda.current_stream(self.device)
        # Ordered behind whatever is enqueued on the compute stream right now:
        # epoch restarts reach here while the previous epoch's final step may
        # still be executing against the slots.
        read_done = torch.cuda.Event()
        read_done.record(compute_stream)
        try:
            while True:
                wait_start = time.perf_counter_ns()
                try:
                    host_batch = next(source)
                except StopIteration:
                    return
                if self._pipeline_stats is not None:
                    self._pipeline_stats.record_source_wait(
                        time.perf_counter_ns() - wait_start
                    )
                if isinstance(host_batch, BatchEnvelope):
                    data, mask = host_batch.data, host_batch.is_real
                else:
                    data, mask = host_batch, None
                schema, entries = self._resolve_schema(data, mask)
                if self._slots is None:
                    self._schema = schema
                    self._slots = self._build_slots(entries)
                elif schema != self._schema:
                    raise RuntimeError(
                        "batch schema changed mid-run; static input slots "
                        "require constant keys/shapes/dtypes"
                    )
                copy_start = time.perf_counter_ns()
                with torch.cuda.stream(self._copy_stream):
                    self._copy_stream.wait_event(read_done)
                    for name, tensor in entries:
                        self._slots[name].copy_(tensor, non_blocking=self.non_blocking)
                    ready = torch.cuda.Event()
                    ready.record(self._copy_stream)
                if self._pipeline_stats is not None:
                    self._pipeline_stats.record_h2d(
                        time.perf_counter_ns() - copy_start,
                        tensor_bytes(dict(entries)),
                    )
                compute_stream.wait_event(ready)
                slot_data = {
                    name: self._slots[name]
                    for name, _shape, _dtype in schema
                    if name != "__is_real"
                }
                slot_mask = self._slots.get("__is_real")
                if isinstance(host_batch, BatchEnvelope):
                    yield replace(host_batch, data=slot_data, is_real=slot_mask)
                else:
                    yield slot_data
                # The consumer has enqueued this step's work by the time it
                # asks for the next batch; the event lands behind forward and
                # backward on the compute stream, so the next slot refill is
                # ordered after the last possible reader.
                read_done = torch.cuda.Event()
                read_done.record(compute_stream)
        finally:
            self._copy_stream.synchronize()
            close = getattr(source, "close", None)
            if close is not None:
                close()


@dataclass(frozen=True)
class SamplerBatchToken:
    before_epoch: int
    before_offset: int
    after_epoch: int
    after_offset: int


@dataclass(frozen=True)
class PreparedSamplerCommit:
    before_epoch: int
    before_offset: int
    after_epoch: int
    after_offset: int
    token_count: int


class ResumableSampler(Sampler):
    """Deterministic map-style sampler with an O(1) checkpoint cursor.

    The current epoch permutation is derived solely from ``seed + epoch``.
    ``offset`` counts source samples already handed to the loader, so restoring
    state never decodes dataset entries or replays transforms.
    """

    def __init__(
        self,
        data_source,
        *,
        shuffle: bool,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.data_source = data_source
        self.shuffle = shuffle
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        if self.world_size <= 0 or not 0 <= self.rank < self.world_size:
            raise ValueError(
                f"Invalid distributed sampler identity rank={self.rank}, "
                f"world_size={self.world_size}."
            )
        self.epoch = 0
        self.offset = 0
        self._yield_epoch = 0
        self._yielded_offset = 0
        self._staged_epoch = 0
        self._staged_offset = 0

    @property
    def local_length(self):
        return len(self.data_source) // self.world_size

    def __len__(self):
        return max(0, self.local_length - self._yielded_offset)

    def __iter__(self):
        target = self.data_source
        while target is not None:
            try:
                setattr(target, "_active_epoch", self._yield_epoch)
            except (AttributeError, TypeError):
                pass
            target = getattr(target, "dataset", None)
        length = len(self.data_source)
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self._yield_epoch)
            indices = torch.randperm(length, generator=generator).tolist()
        else:
            indices = range(length)
        while self._yielded_offset < self.local_length:
            global_position = self._yielded_offset * self.world_size + self.rank
            index = indices[global_position]
            self._yielded_offset += 1
            yield index

    def stage_batch(self, count: int) -> SamplerBatchToken:
        """Stage one delivered batch without publishing its checkpoint cursor.

        DataLoader wrappers may read ahead by one batch.  Checkpoints therefore
        persist only a later committed cursor; staged or prefetched samples are
        replayed after a failed optimizer transaction.
        """
        count = int(count)
        if count <= 0:
            raise RuntimeError(f"Cannot stage a non-positive batch size {count}.")
        before_epoch = self._staged_epoch
        before_offset = self._staged_offset
        if self._yield_epoch == self._staged_epoch + 1:
            self._staged_epoch = self._yield_epoch
            self._staged_offset = 0
        elif self._yield_epoch != self._staged_epoch:
            raise RuntimeError(
                "Cannot stage a non-contiguous sampler epoch transition from "
                f"{self._staged_epoch} to {self._yield_epoch}."
            )
        available = self.local_length - self._staged_offset
        delivered = min(count, available)
        if (
            delivered <= 0
            or self._staged_offset + delivered > self._yielded_offset
        ):
            raise RuntimeError(
                f"Cannot stage {count} samples at epoch {self._staged_epoch}, "
                f"offset {self._staged_offset}; sampler has yielded "
                f"{self._yielded_offset}."
            )
        self._staged_offset += delivered
        return SamplerBatchToken(
            before_epoch,
            before_offset,
            self._staged_epoch,
            self._staged_offset,
        )

    def advance_yield_epoch(self):
        if self._yielded_offset < self.local_length:
            raise RuntimeError(
                "Cannot advance a resumable sampler before its epoch is exhausted."
            )
        self._yield_epoch += 1
        self._yielded_offset = 0

    def prepare_commit(self, tokens) -> PreparedSamplerCommit:
        """Validate a complete optimizer-step cursor transaction."""
        token_tuple = tuple(tokens)
        if not token_tuple:
            raise ValueError("sampler commit transaction must contain at least one token")
        epoch = self.epoch
        offset = self.offset
        for token in token_tuple:
            if (token.before_epoch, token.before_offset) != (epoch, offset):
                raise RuntimeError(
                    "non-contiguous sampler token starts at "
                    f"{(token.before_epoch, token.before_offset)}, expected "
                    f"{(epoch, offset)}"
                )
            epoch = token.after_epoch
            offset = token.after_offset
            if epoch < self.epoch or epoch > self._yield_epoch:
                raise RuntimeError("sampler token epoch is outside the yielded range")
            if not 0 <= offset <= self.local_length:
                raise RuntimeError("sampler token offset is outside the local partition")
        return PreparedSamplerCommit(
            self.epoch,
            self.offset,
            epoch,
            offset,
            len(token_tuple),
        )

    def commit_prepared(self, candidate: PreparedSamplerCommit):
        if (candidate.before_epoch, candidate.before_offset) != (
            self.epoch,
            self.offset,
        ):
            raise RuntimeError("prepared sampler commit no longer matches current state")
        self.epoch = candidate.after_epoch
        self.offset = candidate.after_offset

    def rollback_uncommitted(self):
        """Discard speculative reads so a reused trainer restarts exactly."""
        self._yield_epoch = self.epoch
        self._yielded_offset = self.offset
        self._staged_epoch = self.epoch
        self._staged_offset = self.offset

    def state_dict(self):
        return {
            "version": 2,
            "length": len(self.data_source),
            "shuffle": self.shuffle,
            "seed": self.seed,
            "world_size": self.world_size,
            "rank": self.rank,
            "epoch": self.epoch,
            "offset": self.offset,
        }

    def load_state_dict(self, state):
        if state.get("version") == 1:
            if self.world_size != 1 or self.rank != 0:
                raise RuntimeError(
                    "Cannot restore a legacy single-process sampler into a "
                    "distributed topology."
                )
            expected_legacy = {
                "version": 1,
                "length": len(self.data_source),
                "shuffle": self.shuffle,
                "seed": self.seed,
            }
            actual_legacy = {
                key: state.get(key) for key in expected_legacy
            }
            if actual_legacy != expected_legacy:
                raise RuntimeError(
                    "Cannot restore legacy sampler state: expected "
                    f"{expected_legacy}, got {actual_legacy}."
                )
            offset = int(state["offset"])
            if not 0 <= offset <= self.local_length:
                raise RuntimeError(
                    f"Cannot restore legacy sampler offset {offset} for "
                    f"dataset length {self.local_length}."
                )
            self.epoch = int(state["epoch"])
            self.offset = offset
            self._yield_epoch = self.epoch
            self._yielded_offset = offset
            self._staged_epoch = self.epoch
            self._staged_offset = offset
            return
        expected = {
            "version": 2,
            "length": len(self.data_source),
            "shuffle": self.shuffle,
            "seed": self.seed,
            "world_size": self.world_size,
            "rank": self.rank,
        }
        actual = {key: state.get(key) for key in expected}
        if actual != expected:
            raise RuntimeError(
                f"Cannot restore sampler state: expected {expected}, got {actual}."
            )
        offset = int(state["offset"])
        if not 0 <= offset <= self.local_length:
            raise RuntimeError(
                f"Cannot restore sampler offset {offset} for"
                f" local dataset length {self.local_length}."
            )
        self.epoch = int(state["epoch"])
        self.offset = offset
        self._yield_epoch = self.epoch
        self._yielded_offset = offset
        self._staged_epoch = self.epoch
        self._staged_offset = offset


def build_data_loader(
    dataset,
    batch_size=1,
    shuffle=False,
    shuffle_buffer_size=None,
    num_workers=0,
    drop_last=True,
    batch_by_boardsize=False,
    **kwargs,
):
    capabilities = getattr(dataset, "capabilities", None)
    runtime_context = getattr(dataset, "runtime_context", None)
    built_in_stream = (
        isinstance(dataset, IterableDataset)
        and capabilities is not None
        and capabilities.resumable
        and capabilities.yields_batches
    )
    if built_in_stream:
        forbidden = sorted(
            key
            for key in ("sampler", "batch_sampler", "generator", "worker_init_fn")
            if key in kwargs
        )
        if forbidden:
            raise ValueError(
                "resumable built-in stream rejects loader option(s): "
                + ", ".join(forbidden)
            )
        if num_workers != 0:
            raise ValueError(
                f"resumable built-in stream requires num_workers=0, got {num_workers}"
            )
        if kwargs.get("persistent_workers", False):
            raise ValueError("persistent_workers is invalid for a built-in stream")
        if kwargs.get("in_order") is False:
            raise ValueError("in_order=False is nondeterministic and cannot resume exactly")
        planner = getattr(dataset, "_partitioned_stream", None)
        configured_window = int(
            planner.shuffle_window_size
            if planner is not None
            else getattr(
                dataset,
                "shuffle_window_size",
                getattr(dataset, "extra_kwargs", {}).get(
                    "shuffle_window_size", 32768
                ),
            )
        )
        if shuffle_buffer_size is not None:
            requested_window = int(shuffle_buffer_size)
            if requested_window <= 0:
                raise ValueError("shuffle_buffer_size must be positive")
            if (
                getattr(dataset, "_explicit_shuffle_window_size", False)
                and configured_window != requested_window
            ):
                raise ValueError(
                    "shuffle_buffer_size conflicts with dataset shuffle_window_size: "
                    f"{requested_window} != {configured_window}"
                )
            dataset.shuffle_window_size = requested_window
            if planner is not None:
                planner.shuffle_window_size = requested_window
        else:
            requested_window = configured_window
        mode = getattr(runtime_context, "mode", None)
        if mode == "train" and not drop_last:
            raise ValueError(
                "built-in training streams require drop_last=True because "
                "incomplete global shape buckets are discarded collectively"
            )
        source = getattr(planner, "source", None)
        shape_codes = getattr(source, "shape_codes", {})
        shapes = {tuple(shape) for shape in shape_codes}
        if not batch_by_boardsize and len(shapes) > 1:
            raise ValueError(
                "batch_by_boardsize=False requires a single source output shape; "
                f"found {sorted(shapes)}"
            )
        loader_pin_memory = kwargs.pop("pin_memory", None)
        dataset_pin_memory = getattr(dataset, "pin_memory", None)
        if loader_pin_memory is not None:
            loader_pin_memory = bool(loader_pin_memory)
            if (
                getattr(dataset, "_explicit_pin_memory", False)
                and dataset_pin_memory is not None
                and bool(dataset_pin_memory) != loader_pin_memory
            ):
                raise ValueError(
                    "dataloader pin_memory conflicts with dataset pin_memory: "
                    f"{loader_pin_memory} != {bool(dataset_pin_memory)}"
                )
            if hasattr(dataset, "pin_memory"):
                dataset.pin_memory = loader_pin_memory
            dataset_pin_memory = loader_pin_memory
    if shuffle and isinstance(dataset, IterableDataset):
        if not bool(getattr(dataset, "is_internal_shuffleable", False)):
            raise ValueError(
                "shuffle=True for an iterable dataset requires intrinsic "
                "deterministic shuffling"
            )
        shuffle = False

    if bool(getattr(dataset, "yields_batches", getattr(dataset, "YIELDS_BATCHES", False))):
        # Dataset yields whole collated batches and prefetches with internal
        # threads: run it in-process (worker->main IPC costs more per batch
        # than the vectorized assembly itself) and disable automatic batching.
        # Pinning costs more per batch than the pageable H2D copy it saves at
        # these batch sizes, so default it off here (still overridable).
        return DataLoader(
            dataset,
            batch_size=None,
            num_workers=0,
            **kwargs,
        )

    if runtime_context is not None:
        from dataset.core import rng_u64

        if "generator" not in kwargs:
            generator_seed = rng_u64(
                runtime_context.seed,
                "loader_rng",
                (runtime_context.rank,),
            )
            kwargs["generator"] = torch.Generator().manual_seed(
                generator_seed & ((1 << 63) - 1)
            )
        if num_workers > 0 and "worker_init_fn" not in kwargs:
            kwargs["worker_init_fn"] = partial(
                _seed_dataset_worker,
                root_seed=runtime_context.seed,
                rank=runtime_context.rank,
            )

    # Default to pin_memory=True for better performance
    if "pin_memory" not in kwargs:
        kwargs["pin_memory"] = True

    # Default to persistent workers to avoid worker restart cost between epochs
    if "persistent_workers" not in kwargs:
        kwargs["persistent_workers"] = num_workers > 0

    if batch_by_boardsize:
        assert isinstance(dataset, IterableDataset), "batch_by_boardsize must be used with IterableDataset"
        dataset = BatchByBoardSizeDataset(dataset, batch_size)

    dataloader = DataLoader(
        dataset,
        batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        **kwargs,
    )
    return dataloader


def resolve_weight_clipping(named_parameters, clip_parameters):
    """Resolve parameter names in *clip_parameters* to parameter tensors.

    Done once at setup so the per-iteration apply does not rebuild the name
    lookup.  Returns a list of ``(min_weight, max_weight, params,
    virtual_params)`` tuples; ``virtual_params`` is ``None`` for plain clamping.
    """
    named_parameters = dict(named_parameters)
    resolved = []
    for group in clip_parameters:
        params = [named_parameters[name] for name in group["params"]]
        virtual_params = None
        if "virtual_params" in group:
            virtual_params = [named_parameters[name] for name in group["virtual_params"]]
            if len(virtual_params) != len(params):
                raise ValueError(
                    f"weight clipping group has {len(params)} params"
                    f" but {len(virtual_params)} virtual_params"
                )
        resolved.append((group["min_weight"], group["max_weight"], params, virtual_params))
    return resolved


def apply_weight_clipping(resolved_groups):
    """Clamp parameters per the groups produced by :func:`resolve_weight_clipping`."""
    for min_weight, max_weight, params, virtual_params in resolved_groups:
        if virtual_params is None:
            for p in params:
                p.data.clamp_(min_weight, max_weight)
        else:
            for p, virtual_param in zip(params, virtual_params):
                p_data = p.data
                virtual = virtual_param.repeat(
                    *[p_data.shape[i] // virtual_param.shape[i] for i in range(virtual_param.ndim)]
                )
                min_weight_t = p_data.new_full(p_data.shape, min_weight) - virtual
                p_data = torch.max(p_data, min_weight_t)
                max_weight_t = p_data.new_full(p_data.shape, max_weight) - virtual
                p_data = torch.min(p_data, max_weight_t)
                p.data.copy_(p_data)


def cross_entropy_with_softlabel(
    input, target, reduction="mean", weight=None, focal_gamma=0.0, use_kl_divergence=False, eps=1e-8
):
    """
    :param input: (batch, *) logits before sigmoid/softmax activation.
    :param target: (batch, *) same shape as input, must be a valid distribution
        (sum(target[i, ...]) == 1) for multi-class classification or in [0, 1] for binary classification.
    :param weight: (batch, *) same shape as input. If specified, a weight is applied for each category.
        If used in binary classification, this is treated as positive weight.
    :param focal_gamma: focal loss gamma parameter. Default to 0 as disabled.
    :param use_kl_divergence: subtract soft-label bias from the loss
    :param eps: small value to prevent log(0) when target is 0. Default to 1e-8.
    """
    assert (
        focal_gamma == 0.0 or use_kl_divergence is False
    ), "Focal loss and KL divergence cannot be used together."

    if input.ndim > 1:
        # Cross-entropy Loss
        input = input.view(input.shape[0], -1)
        target = target.view(target.shape[0], -1)
        if weight is not None:
            target = target * weight.view(weight.shape[0], -1)

        logprobs = F.log_softmax(input, dim=1)
        if focal_gamma > 0.0:
            focal_weight = (1 - torch.exp(logprobs)) ** focal_gamma
            logprobs = logprobs * focal_weight
        batchloss = -torch.sum(target * logprobs, dim=1)

        if use_kl_divergence:
            logprobs_target = torch.log(torch.clamp(target, min=eps))
            batchloss += torch.sum(target * logprobs_target, dim=1)
    else:
        # Binary Cross-entropy Loss
        batchloss = F.binary_cross_entropy_with_logits(input, target, reduction="none", pos_weight=weight)

        if focal_gamma > 0.0:
            probs = torch.sigmoid(input)
            pt = target * probs + (1 - target) * (1 - probs)
            focal_weight = (1 - pt) ** focal_gamma
            batchloss = batchloss * focal_weight

        if use_kl_divergence:
            logprobs_target = torch.log(torch.clamp(target, min=eps))
            loginvprobs_target = torch.log(torch.clamp(1 - target, min=eps))
            batchloss += target * logprobs_target + (1 - target) * loginvprobs_target

    if reduction == "none":
        return batchloss
    elif reduction == "mean":
        return torch.mean(batchloss)
    elif reduction == "sum":
        return torch.sum(batchloss)
    else:
        raise ValueError(f"Unsupported reduction mode {reduction}.")


class BatchByBoardSizeDataset(IterableDataset):
    def __init__(self, dataset, batch_size):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        boardsize_to_databuf = {}
        try:
            dataset_iter = iter(self.dataset)
            while True:
                try:
                    data = next(dataset_iter)
                    board_size = tuple(data["board_size"])
                    if board_size not in boardsize_to_databuf:
                        boardsize_to_databuf[board_size] = []
                    databuf = boardsize_to_databuf[board_size]
                    databuf.append(data)

                    assert len(databuf) <= self.batch_size
                    if len(databuf) == self.batch_size:
                        while len(databuf) > 0:
                            yield databuf.pop()
                except StopIteration:
                    break  # discard last incomplete batch for all board size
        except GeneratorExit:
            pass


def state_dict_drop_size_unmatched(model: torch.nn.Module, loaded_state_dict: dict) -> dict:
    """
    Drop key and values from loaded_state_dict that have shape
    unmatched with the current model's parameters.
    This will not drop other unmatched keys.
    """
    current_model_dict = model.state_dict()
    new_state_dict = {}
    for k, v in loaded_state_dict.items():
        if k not in current_model_dict or current_model_dict[k].size() == v.size():
            new_state_dict[k] = v
    return new_state_dict
