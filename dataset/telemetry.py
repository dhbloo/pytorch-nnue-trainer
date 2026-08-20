"""Minimal data-pipeline measurements for adaptive runtime control."""

from __future__ import annotations

from dataclasses import dataclass
import os
import threading
import time

from utils.system_resources import probe_process_resident_bytes

from .decoder import ProcessedNpzDecoder
from .source_dataset import SourceBatchDataset


@dataclass(frozen=True, slots=True)
class PipelineObservabilityConfig:
    enabled: bool = False

    @classmethod
    def parse(cls, value) -> "PipelineObservabilityConfig":
        if value is None or value is False:
            return cls(False)
        if value is True:
            return cls(True)
        if not isinstance(value, dict):
            raise TypeError("observability must be a boolean or mapping")
        unknown = set(value).difference({"enabled"})
        if unknown:
            raise ValueError(
                "observability has unknown option(s): " + ", ".join(sorted(unknown))
            )
        enabled = value.get("enabled", True)
        if type(enabled) is not bool:
            raise TypeError("observability.enabled must be a boolean")
        return cls(enabled)


@dataclass(frozen=True, slots=True)
class PipelineMetricWindow:
    """One rank-local measurement window consumed by rank aggregation."""

    producer_capacity_rows_s: float
    data_wait_fraction: float
    prefetch_wait_fraction: float
    source_tail_wait_fraction: float
    h2d_wait_fraction: float
    cache_reloads: int
    rss_gib: float
    host_memory_used_fraction: float | None = None
    host_memory_high_water_fraction: float | None = None
    host_memory_backpressure_events: int | None = None


@dataclass(frozen=True, slots=True)
class AggregatedPipelineMetrics:
    """Controller inputs plus pipeline and process metric surfaces."""

    observation: dict[str, float]
    public: dict[str, float]
    process: dict[str, float]


class PipelineStats:
    """Thread-safe window totals required by the adaptive controller."""

    __slots__ = (
        "_lock",
        "_last_snapshot_ns",
        "_decoded_rows",
        "_decode_ns",
        "_prefetch_wait_ns",
        "_source_wait_ns",
        "_h2d_launch_wait_ns",
        "_cuda_exposed_wait_ns",
        "_cache_reloads",
    )

    def __init__(self):
        self._lock = threading.Lock()
        self._last_snapshot_ns = time.perf_counter_ns()
        self._reset()

    def _reset(self) -> None:
        self._decoded_rows = 0
        self._decode_ns = 0
        self._prefetch_wait_ns = 0
        self._source_wait_ns = 0
        self._h2d_launch_wait_ns = 0
        self._cuda_exposed_wait_ns = 0
        self._cache_reloads = 0

    def record_decode(self, elapsed_ns: int, rows: int) -> None:
        with self._lock:
            self._decode_ns += max(0, int(elapsed_ns))
            self._decoded_rows += int(rows)

    def record_prefetch_wait(self, elapsed_ns: int) -> None:
        with self._lock:
            self._prefetch_wait_ns += max(0, int(elapsed_ns))

    def record_source_wait(self, elapsed_ns: int) -> None:
        with self._lock:
            self._source_wait_ns += max(0, int(elapsed_ns))

    def record_h2d(self, elapsed_ns: int) -> None:
        with self._lock:
            self._h2d_launch_wait_ns += max(0, int(elapsed_ns))

    def record_cuda_exposed_wait(self, elapsed_ms: float) -> None:
        with self._lock:
            self._cuda_exposed_wait_ns += max(
                0,
                round(float(elapsed_ms) * 1e6),
            )

    def record_cache_reload(self) -> None:
        with self._lock:
            self._cache_reloads += 1

    def snapshot(self, *, active_workers: int) -> PipelineMetricWindow:
        now = time.perf_counter_ns()
        with self._lock:
            elapsed_ns = max(1, now - self._last_snapshot_ns)
            input_wait_ns = max(self._source_wait_ns, self._prefetch_wait_ns)
            h2d_wait_ns = self._h2d_launch_wait_ns + self._cuda_exposed_wait_ns
            window = PipelineMetricWindow(
                producer_capacity_rows_s=(
                    self._decoded_rows
                    * max(1, int(active_workers))
                    * 1e9
                    / max(1, self._decode_ns)
                ),
                data_wait_fraction=min(
                    1.0,
                    (input_wait_ns + h2d_wait_ns) / elapsed_ns,
                ),
                prefetch_wait_fraction=min(
                    1.0,
                    self._prefetch_wait_ns / elapsed_ns,
                ),
                source_tail_wait_fraction=min(
                    1.0,
                    max(0, self._source_wait_ns - self._prefetch_wait_ns)
                    / elapsed_ns,
                ),
                h2d_wait_fraction=min(1.0, h2d_wait_ns / elapsed_ns),
                cache_reloads=self._cache_reloads,
                rss_gib=(probe_process_resident_bytes().value or 0) / 1024**3,
            )
            self._last_snapshot_ns = now
            self._reset()
            return window


class ObservedProcessedNpzDecoder(ProcessedNpzDecoder):
    """Processed decoder that counts successful private-cache reloads."""

    def __init__(self, *args, pipeline_stats: PipelineStats, **kwargs):
        self.pipeline_stats = pipeline_stats
        self._observed_loaded_paths: set[str] = set()
        super().__init__(*args, **kwargs)

    def _acquire_cached_arrays(self, path):
        arrays, lease, loaded = super()._acquire_cached_arrays(path)
        canonical = os.path.abspath(os.fspath(path))
        if loaded and not self.is_mapped_source(canonical):
            with self._array_cache_condition:
                reloaded = canonical in self._observed_loaded_paths
                self._observed_loaded_paths.add(canonical)
            if reloaded:
                self.pipeline_stats.record_cache_reload()
        return arrays, lease, loaded


class _AdjustableConcurrencyGate:
    """Limit active decode calls while retaining one reusable executor."""

    def __init__(self, limit: int):
        self._condition = threading.Condition()
        self._limit = int(limit)
        self._active = 0

    def set_limit(self, limit: int) -> None:
        with self._condition:
            self._limit = int(limit)
            self._condition.notify_all()

    def run(self, callback, *args):
        with self._condition:
            while self._active >= self._limit:
                self._condition.wait()
            self._active += 1
        try:
            return callback(*args)
        finally:
            with self._condition:
                self._active -= 1
                self._condition.notify_all()


class ObservedSourceBatchDataset(SourceBatchDataset):
    """Measured adapter for the adaptive processed-NPZ pipeline."""

    def __init__(
        self,
        *args,
        pipeline_stats,
        adaptive_runtime=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pipeline_stats = pipeline_stats
        self.adaptive_runtime = adaptive_runtime
        self._maximum_prefetch_workers = int(
            self._effective_prefetch_workers
            if adaptive_runtime is None
            else adaptive_runtime.maximum_prefetch_workers
        )
        self._concurrency_gate = (
            _AdjustableConcurrencyGate(self.active_prefetch_workers)
            if adaptive_runtime is not None and self._effective_prefetch_workers > 0
            else None
        )

    @property
    def active_prefetch_workers(self) -> int:
        if self.adaptive_runtime is None:
            return self._effective_prefetch_workers
        return int(self.adaptive_runtime.settings.decode_workers)

    @property
    def maximum_prefetch_workers(self) -> int:
        return self._maximum_prefetch_workers

    @property
    def active_prefetch_chunk_batches(self) -> int:
        if self.adaptive_runtime is None:
            return super().active_prefetch_chunk_batches
        return int(self.adaptive_runtime.settings.decode_chunk_batches)

    @staticmethod
    def _decoded_row_count(decoded) -> int:
        return (
            len(decoded[1])
            if isinstance(decoded, tuple) and len(decoded) == 3
            else 0
        )

    def _decode_batch(self, batch):
        start = time.perf_counter_ns()
        decoded = super()._decode_batch(batch)
        self.pipeline_stats.record_decode(
            time.perf_counter_ns() - start,
            self._decoded_row_count(decoded),
        )
        return decoded

    def _decode_batches(self, batches):
        start = time.perf_counter_ns()
        decoded = super()._decode_batches(batches)
        self.pipeline_stats.record_decode(
            time.perf_counter_ns() - start,
            sum(self._decoded_row_count(item) for item in decoded),
        )
        return decoded

    def _run_decode_batches(self, batches):
        if self._concurrency_gate is None:
            return self._decode_batches(batches)
        return self._concurrency_gate.run(self._decode_batches, batches)

    def _await_prefetch(self, future):
        start = time.perf_counter_ns()
        try:
            return future.result()
        finally:
            self.pipeline_stats.record_prefetch_wait(
                time.perf_counter_ns() - start
            )

    def pipeline_metrics_snapshot(self) -> PipelineMetricWindow:
        return self.pipeline_stats.snapshot(
            active_workers=self.active_prefetch_workers,
        )

    def _apply_adaptive_settings(self) -> None:
        if self.adaptive_runtime is None:
            return
        if self._concurrency_gate is not None:
            self._concurrency_gate.set_limit(self.active_prefetch_workers)
        decoder = getattr(self.source, "decoder", None)
        if decoder is None or not hasattr(decoder, "configure_cache"):
            return
        cache_state = decoder.cache_state()
        decoder.configure_cache(
            entries=int(cache_state["capacity_entries"]),
            byte_capacity=int(
                self.adaptive_runtime.settings.decoded_cache_bytes
            ),
        )

    def pipeline_tuning_update(
        self,
        metrics,
        iteration: int,
        *,
        epoch_changed: bool = False,
    ):
        if self.adaptive_runtime is None:
            return None
        state = self.adaptive_runtime.update(
            metrics,
            iteration,
            epoch_changed=epoch_changed,
        )
        if state is not None:
            self._apply_adaptive_settings()
        return state

    def pipeline_tuning_state_dict(self):
        if self.adaptive_runtime is None:
            return None
        return self.adaptive_runtime.state_dict()

    def load_pipeline_tuning_state_dict(self, state) -> None:
        if self.adaptive_runtime is None:
            if state is not None:
                raise ValueError("adaptive data pipeline is disabled")
            return
        self.adaptive_runtime.load_state_dict(state)
        self._apply_adaptive_settings()

    def restore_pipeline_tuning_state_dict(self, state) -> bool:
        """Restore a same-runtime performance state, or ignore a stale one."""
        if self.adaptive_runtime is None:
            return False
        restored = self.adaptive_runtime.restore_state_dict(state)
        if restored:
            self._apply_adaptive_settings()
        return restored


def aggregate_pipeline_snapshots(
    snapshots: list[PipelineMetricWindow],
    *,
    consumer_batches_s: float,
    rows_per_batch: int,
) -> AggregatedPipelineMetrics:
    if not snapshots:
        return AggregatedPipelineMetrics({}, {}, {})
    consumer_rows_s = consumer_batches_s * rows_per_batch
    data_wait_fraction = max(snapshot.data_wait_fraction for snapshot in snapshots)
    prefetch_wait_fraction = max(
        snapshot.prefetch_wait_fraction for snapshot in snapshots
    )
    source_tail_wait_fraction = max(
        snapshot.source_tail_wait_fraction for snapshot in snapshots
    )
    cache_reloads = max(snapshot.cache_reloads for snapshot in snapshots)
    producer_capacity = min(
        snapshot.producer_capacity_rows_s for snapshot in snapshots
    )
    observation = {
        "throughput/consumer_rows_s": float(consumer_rows_s),
        "distributed/producer_capacity_min_rows_s": producer_capacity,
        "data_wait_fraction": data_wait_fraction,
        "distributed/prefetch_wait_fraction_max": prefetch_wait_fraction,
        "distributed/source_tail_wait_fraction_max": source_tail_wait_fraction,
        "distributed/cache_reloads_max": float(cache_reloads),
    }
    public = {
        "data_wait_fraction": data_wait_fraction,
        "producer_headroom": producer_capacity / max(1e-12, consumer_rows_s),
        "distributed/prefetch_wait_fraction_max": prefetch_wait_fraction,
        "distributed/source_tail_wait_fraction_max": source_tail_wait_fraction,
        "distributed/cache_reloads_max": float(cache_reloads),
        "h2d_wait_fraction": max(
            snapshot.h2d_wait_fraction for snapshot in snapshots
        ),
    }
    process = {
        "process_rss_gib_mean": sum(snapshot.rss_gib for snapshot in snapshots)
        / len(snapshots),
    }
    has_host_memory = all(
        snapshot.host_memory_used_fraction is not None
        and snapshot.host_memory_high_water_fraction is not None
        and snapshot.host_memory_backpressure_events is not None
        for snapshot in snapshots
    )
    if has_host_memory:
        public.update(
            {
                "host_memory/used_fraction": max(
                    float(snapshot.host_memory_used_fraction)
                    for snapshot in snapshots
                ),
                "host_memory/high_water_fraction": max(
                    float(snapshot.host_memory_high_water_fraction)
                    for snapshot in snapshots
                ),
                "host_memory/backpressure_events": float(
                    max(
                        int(snapshot.host_memory_backpressure_events)
                        for snapshot in snapshots
                    )
                ),
            }
        )
    return AggregatedPipelineMetrics(observation, public, process)
