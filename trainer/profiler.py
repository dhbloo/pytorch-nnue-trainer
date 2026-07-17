"""Pluggable training profiler: per-region timing and on-demand torch.profiler windows.

Disabled (the default) the trainer holds the zero-overhead :data:`NULL_PROFILER`,
whose ``region()`` returns a shared ``nullcontext`` — the hot loop pays only a few
attribute lookups per iteration.

Enabled via ``profiler_args``:

    timing: true            # per-region iteration timing, logged at every log_interval
    trace_at: [100000]      # start a torch.profiler trace window at these iterations
    trace_iters: 30         # active iterations per trace window

Timing regions are recorded with CUDA event pairs (asynchronous, no host sync);
elapsed times are read only at the log-interval boundary where training already
synchronizes to gather metrics.  The pure host-side wait for the next batch
("data") is measured with a wall clock instead.  Trace windows dump chrome
traces to ``rundir/profile_trace`` (main process only), viewable in TensorBoard
or ``chrome://tracing``.
"""

import os
import time
from contextlib import nullcontext

import torch


_NULL_CONTEXT = nullcontext()

# regions measured with a wall clock instead of CUDA events (pure host-side waits)
_CPU_REGIONS = frozenset({"data"})


class NullProfiler:
    """Zero-overhead stand-in used when profiling is disabled."""

    __slots__ = ()

    def region(self, name):
        return _NULL_CONTEXT

    def mark_iteration(self):
        pass

    def step(self, iteration):
        pass

    def flush_timings(self):
        return {}

    def close(self):
        pass


NULL_PROFILER = NullProfiler()


class _CudaRegion:
    """Records an asynchronous CUDA event pair around a code region."""

    __slots__ = ("store", "start")

    def __init__(self, store):
        self.store = store

    def __enter__(self):
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        self.start = start

    def __exit__(self, exc_type, exc_value, traceback):
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self.store.append((self.start, end))


class _CpuRegion:
    """Records wall-clock milliseconds around a code region."""

    __slots__ = ("store", "start")

    def __init__(self, store):
        self.store = store

    def __enter__(self):
        self.start = time.perf_counter()

    def __exit__(self, exc_type, exc_value, traceback):
        self.store.append((time.perf_counter() - self.start) * 1e3)


class TrainerProfiler:
    """Training profiler with per-region timing and scheduled trace windows."""

    _TRACE_WARMUP = 2

    def __init__(self, *, timing, trace_at, trace_iters, trace_dir, is_main_process, use_cuda):
        self._timing = timing
        self._use_cuda = use_cuda
        self._regions = {}
        self._iters = 0
        self._trace_targets = sorted(trace_at)
        self._trace_iters = trace_iters
        self._trace_dir = trace_dir
        self._is_main_process = is_main_process
        self._trace = None
        self._trace_steps_left = 0

    def region(self, name):
        """Context manager timing one occurrence of region *name*."""
        if not self._timing:
            return _NULL_CONTEXT
        store = self._regions.get(name)
        if store is None:
            store = self._regions[name] = []
        if self._use_cuda and name not in _CPU_REGIONS:
            return _CudaRegion(store)
        return _CpuRegion(store)

    def mark_iteration(self):
        """Count one finished training iteration for the timing averages.

        Called before the log-interval flush so the current iteration's
        regions and the divisor stay consistent.
        """
        if self._timing:
            self._iters += 1

    def step(self, iteration):
        """Drive trace windows; called at the very end of each iteration so a
        window's step boundaries match training iterations (including any
        logging/saving/validation that followed the step)."""
        if self._trace is not None:
            self._trace.step()
            self._trace_steps_left -= 1
            if self._trace_steps_left > 0:
                return
            self._trace.__exit__(None, None, None)
            self._trace = None
            print(f"Profile trace saved at {self._trace_dir}.", flush=True)
            # fall through: a target at exactly this iteration starts next
        # drop targets already passed (resumed beyond them, or inside a window)
        while self._trace_targets and self._trace_targets[0] < iteration:
            self._trace_targets.pop(0)
        if self._trace_targets and self._trace_targets[0] == iteration:
            self._trace_targets.pop(0)
            if self._is_main_process:
                self._start_trace()

    def _start_trace(self):
        activities = [torch.profiler.ProfilerActivity.CPU]
        if self._use_cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self._trace = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=0, warmup=self._TRACE_WARMUP, active=self._trace_iters, repeat=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(self._trace_dir, use_gzip=True),
            record_shapes=True,
            with_stack=True,
            with_flops=True,
            with_modules=True,
        )
        self._trace.__enter__()
        self._trace_steps_left = self._TRACE_WARMUP + self._trace_iters

    def flush_timings(self):
        """Average per-iteration region times since the last flush, in milliseconds.

        Must be called on every process (it clears the recording buffers).  The
        single CUDA synchronize lands at the log-interval boundary where the
        training loop has already synchronized to gather metrics.
        """
        if not self._timing or self._iters == 0:
            return {}
        if self._use_cuda:
            torch.cuda.synchronize()
        stats = {}
        for name, store in self._regions.items():
            if not store:
                continue
            if isinstance(store[0], tuple):
                total = sum(start.elapsed_time(end) for start, end in store)
            else:
                total = sum(store)
            stats[f"prof/{name}_ms"] = total / self._iters
            store.clear()
        self._iters = 0
        return stats

    def close(self):
        """Finalize an in-flight trace window (called at the end of training).

        torch.profiler only writes the trace on the schedule's final
        RECORD_AND_SAVE transition, so a window cut mid-recording is driven
        through its remaining (empty) steps to save what was captured; a
        window still warming up has captured nothing and is dropped.
        """
        if self._trace is None:
            return
        if self._trace_steps_left > self._trace_iters:
            print("Profiler trace window ended during warmup; no trace saved.", flush=True)
        else:
            for _ in range(self._trace_steps_left):
                self._trace.step()
            print(f"Profile trace saved at {self._trace_dir}.", flush=True)
        self._trace.__exit__(None, None, None)
        self._trace = None


def build_profiler(profiler_args, *, rundir, is_main_process, use_cpu):
    """Build a :class:`TrainerProfiler` from *profiler_args*, or :data:`NULL_PROFILER` when disabled."""
    if not profiler_args:
        return NULL_PROFILER
    unknown = set(profiler_args) - {"timing", "trace_at", "trace_iters"}
    if unknown:
        raise ValueError(f"Unknown profiler_args keys: {', '.join(sorted(unknown))}")
    timing = bool(profiler_args.get("timing", False))
    trace_at = list(profiler_args.get("trace_at") or [])
    trace_iters = int(profiler_args.get("trace_iters", 30))
    if not timing and not trace_at:
        return NULL_PROFILER
    return TrainerProfiler(
        timing=timing,
        trace_at=trace_at,
        trace_iters=trace_iters,
        trace_dir=os.path.join(rundir, "profile_trace"),
        is_main_process=is_main_process,
        use_cuda=torch.cuda.is_available() and not use_cpu,
    )
