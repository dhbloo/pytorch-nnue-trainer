"""Asynchronous checkpoint serialization.

Periodic saves otherwise stall the training loop twice: once copying the live
device tensors during pickling, and again writing the pickled bytes to a
latency-bound filesystem. This module moves everything except a device-resident
snapshot off the training loop:

1. ``submit()`` clones every tensor in the payloads on a dedicated copy stream
   (ordered after the compute stream's current tail, so the snapshot holds the
   exact save-point values) and makes the compute stream wait on that snapshot
   event. The host-side cost is one sub-millisecond DtoD burst; training then
   resumes immediately with a consistent snapshot.
2. A single writer thread waits on the event, serializes the staged payloads
   (DtoH during pickling overlaps training), commits each file through its
   caller-provided finalizer (atomic tmp+rename), and finally runs an optional
   post-write hook such as checkpoint pruning.
3. ``flush()`` joins the writer and re-raises any failure. Callers flush before
   the next save, before declaring a run finished, and at resource shutdown,
   so a failed asynchronous save fails the run exactly like a failed
   synchronous one.

Snapshot content is frozen at ``submit()`` time; the only behavioural delta is
that the bytes reach the filesystem shortly after the save iteration instead
of before it. A crash inside that window resumes from the previous completed
checkpoint, which the resume machinery already handles.
"""

import os
import threading

import torch
from torch import Tensor


def _stage_tree(obj, memo, stream):
    """Clone every tensor in a nested dict/list/tuple structure.

    Clones are issued on ``stream``; identical tensors (shared storage across
    payloads) are staged once.
    """
    if isinstance(obj, Tensor):
        key = id(obj)
        staged = memo.get(key)
        if staged is None:
            if obj.device.type == "cpu":
                staged = obj.clone()
            else:
                with torch.cuda.stream(stream):
                    staged = torch.empty_like(obj, device=obj.device)
                    staged.copy_(obj, non_blocking=True)
            memo[key] = staged
        return staged
    if isinstance(obj, dict):
        return {key: _stage_tree(value, memo, stream) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        staged_items = [_stage_tree(item, memo, stream) for item in obj]
        return type(obj)(staged_items) if isinstance(obj, tuple) else staged_items
    return obj


class AsyncCheckpointWriter:
    """Serializes staged checkpoint payloads on one background thread.

    Args:
        device: Training device. On non-CUDA devices (or when disabled) every
            call degrades to the plain synchronous path.
    """

    def __init__(self, device):
        self._device = device
        self._copy_stream = (
            torch.cuda.Stream(device) if device.type == "cuda" else None
        )
        self._thread = None
        self._error = None
        self._disabled = False

    def flush(self):
        """Wait for any in-flight save and re-raise its failure, if any."""
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join()
        if self._error is not None:
            error, self._error = self._error, None
            raise RuntimeError(
                f"asynchronous checkpoint write failed: {type(error).__name__}: {error}"
            ) from error

    def submit(self, payloads, post_hook=None):
        """Snapshot ``payloads`` and write them in the background.

        Args:
            payloads: Iterable of ``(obj, finalize)`` pairs. ``obj`` is any
                nested structure of tensors; ``finalize(staged_obj)`` must
                atomically commit exactly one checkpoint file.
            post_hook: Optional callable run on the writer thread after all
                finalizers succeed (e.g. checkpoint pruning).
        """
        self.flush()
        if (
            self._disabled
            or self._copy_stream is None
            or os.environ.get("NNUE_SYNC_CHECKPOINT")
        ):
            for obj, finalize in payloads:
                finalize(obj)
            if post_hook is not None:
                post_hook()
            return

        compute_stream = torch.cuda.current_stream(self._device)
        self._copy_stream.wait_stream(compute_stream)
        memo = {}
        try:
            staged = [
                (_stage_tree(obj, memo, self._copy_stream), finalize)
                for obj, finalize in payloads
            ]
        except torch.cuda.OutOfMemoryError:
            # The snapshot transiently duplicates the checkpoint on device;
            # if that does not fit, stay correct and pay the synchronous cost.
            # Drain the queued partial clones first so no copy-stream work
            # touches the staged blocks once their references are dropped.
            self._copy_stream.synchronize()
            memo.clear()
            self._disabled = True
            for obj, finalize in payloads:
                finalize(obj)
            if post_hook is not None:
                post_hook()
            return
        snapshot_done = torch.cuda.Event()
        snapshot_done.record(self._copy_stream)
        # Later compute-stream work may mutate the live tensors only after the
        # snapshot clones have completed.
        compute_stream.wait_event(snapshot_done)

        thread = threading.Thread(
            target=self._write,
            name="checkpoint-writer",
            args=(staged, post_hook, snapshot_done),
        )
        self._thread = thread
        thread.start()

    def _write(self, staged, post_hook, snapshot_done):
        try:
            with torch.cuda.stream(self._copy_stream):
                snapshot_done.synchronize()
                for staged_obj, finalize in staged:
                    finalize(staged_obj)
            if post_hook is not None:
                post_hook()
        except BaseException as exc:
            self._error = exc
