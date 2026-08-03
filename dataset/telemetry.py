"""Opt-in data-pipeline metrics and conservative runtime tuning.

The normal dataset path never imports or calls these counters from its hot
methods.  Instrumented dataset and decoder subclasses are selected once at
construction time, which keeps disabled observability out of the steady-state
batch path entirely.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import threading
import time
from typing import Mapping

import numpy as np
import torch

try:
    import resource
except ImportError:  # Windows
    resource = None

from .decoder import ProcessedNpzDecoder
from .source_dataset import SourceBatchDataset


PIPELINE_TUNING_SCHEMA = "pipeline-tuning-v1"
PIPELINE_PERFORMANCE_ABI = "processed-dense-prefetch-v1"

_LATENCY_BUCKETS_NS = (
    10_000,
    25_000,
    50_000,
    100_000,
    250_000,
    500_000,
    1_000_000,
    2_500_000,
    5_000_000,
    10_000_000,
    25_000_000,
    50_000_000,
    100_000_000,
    250_000_000,
    500_000_000,
    1_000_000_000,
)


def _positive_int(name: str, value, *, allow_zero: bool = False) -> int:
    if type(value) is not int or value < (0 if allow_zero else 1):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


def _finite_positive_float(name: str, value) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


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
class PipelineAutotuneConfig:
    enabled: bool = False
    reuse: str = "exact"
    warmup_iterations: int = 1000
    verify_iterations: int = 500
    decision_interval: int = 500
    freeze_after: int = 10_000
    target_producer_headroom: float = 1.5
    max_data_wait_fraction: float = 0.02
    max_prefetch_threads: int = 4
    max_prefetch_batches: int = 64
    host_cache_budget_bytes: int = 2 * 1024**3
    respect_explicit: bool = True
    cache_dir: str | None = None

    @classmethod
    def parse(cls, value) -> "PipelineAutotuneConfig":
        if value is None or value is False:
            return cls()
        if value is True:
            value = {}
        if not isinstance(value, dict):
            raise TypeError("autotune must be a boolean or mapping")
        allowed = set(cls.__dataclass_fields__)
        unknown = set(value).difference(allowed)
        if unknown:
            raise ValueError(
                "autotune has unknown option(s): " + ", ".join(sorted(unknown))
            )
        options = dict(value)
        options.setdefault("enabled", True)
        config = cls(**options)
        if type(config.enabled) is not bool:
            raise TypeError("autotune.enabled must be a boolean")
        if config.reuse not in {"off", "exact", "compatible"}:
            raise ValueError("autotune.reuse must be 'off', 'exact', or 'compatible'")
        _positive_int(
            "autotune.warmup_iterations",
            config.warmup_iterations,
            allow_zero=True,
        )
        _positive_int(
            "autotune.verify_iterations",
            config.verify_iterations,
            allow_zero=True,
        )
        _positive_int("autotune.decision_interval", config.decision_interval)
        _positive_int("autotune.freeze_after", config.freeze_after)
        _finite_positive_float(
            "autotune.target_producer_headroom", config.target_producer_headroom
        )
        wait_fraction = float(config.max_data_wait_fraction)
        if not np.isfinite(wait_fraction) or not 0 <= wait_fraction <= 1:
            raise ValueError("autotune.max_data_wait_fraction must be in [0, 1]")
        _positive_int("autotune.max_prefetch_threads", config.max_prefetch_threads)
        _positive_int("autotune.max_prefetch_batches", config.max_prefetch_batches)
        _positive_int(
            "autotune.host_cache_budget_bytes",
            config.host_cache_budget_bytes,
        )
        if type(config.respect_explicit) is not bool:
            raise TypeError("autotune.respect_explicit must be a boolean")
        if config.cache_dir is not None and not isinstance(config.cache_dir, str):
            raise TypeError("autotune.cache_dir must be a string or null")
        return config


class _LatencyHistogram:
    __slots__ = ("counts", "total", "maximum")

    def __init__(self):
        self.counts = [0] * (len(_LATENCY_BUCKETS_NS) + 1)
        self.total = 0
        self.maximum = 0

    def add(self, elapsed_ns: int) -> None:
        elapsed_ns = max(0, int(elapsed_ns))
        index = int(np.searchsorted(_LATENCY_BUCKETS_NS, elapsed_ns, side="left"))
        self.counts[index] += 1
        self.total += elapsed_ns
        self.maximum = max(self.maximum, elapsed_ns)

    def clear(self) -> None:
        self.counts[:] = [0] * len(self.counts)
        self.total = 0
        self.maximum = 0

    def count(self) -> int:
        return sum(self.counts)

    def percentile_ns(self, percentile: float) -> int:
        count = self.count()
        if count == 0:
            return 0
        target = max(1, int(np.ceil(count * percentile)))
        cumulative = 0
        for index, bucket_count in enumerate(self.counts):
            cumulative += bucket_count
            if cumulative >= target:
                return (
                    self.maximum
                    if index == len(_LATENCY_BUCKETS_NS)
                    else _LATENCY_BUCKETS_NS[index]
                )
        return self.maximum


class PipelineStats:
    """Thread-safe aggregate counters used only by the instrumented path."""

    __slots__ = (
        "_lock",
        "_last_snapshot_ns",
        "_decode_latency",
        "_prefetch_wait_latency",
        "_source_wait_latency",
        "_h2d_latency",
        "_cuda_h2d_latency",
        "_cuda_wait_latency",
        "_decoded_rows",
        "_decoded_batches",
        "_submitted_batches",
        "_completed_batches",
        "_queue_depth_sum",
        "_queue_depth_samples",
        "_queue_depth_max",
        "_queue_empty_events",
        "_cache_hits",
        "_cache_misses",
        "_cache_waits",
        "_cache_reloads",
        "_cache_evictions",
        "_cache_load_ns",
        "_h2d_bytes",
        "_h2d_batches",
        "_cuda_prefetch_batches",
        "_manifest_ms",
        "_manifest_files",
        "_manifest_rows",
    )

    def __init__(self):
        self._lock = threading.Lock()
        self._last_snapshot_ns = time.perf_counter_ns()
        self._decode_latency = _LatencyHistogram()
        self._prefetch_wait_latency = _LatencyHistogram()
        self._source_wait_latency = _LatencyHistogram()
        self._h2d_latency = _LatencyHistogram()
        self._cuda_h2d_latency = _LatencyHistogram()
        self._cuda_wait_latency = _LatencyHistogram()
        self._decoded_rows = 0
        self._decoded_batches = 0
        self._submitted_batches = 0
        self._completed_batches = 0
        self._queue_depth_sum = 0
        self._queue_depth_samples = 0
        self._queue_depth_max = 0
        self._queue_empty_events = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_waits = 0
        self._cache_reloads = 0
        self._cache_evictions = 0
        self._cache_load_ns = 0
        self._h2d_bytes = 0
        self._h2d_batches = 0
        self._cuda_prefetch_batches = 0
        self._manifest_ms = 0.0
        self._manifest_files = 0
        self._manifest_rows = 0

    def record_manifest(self, elapsed_ns: int, files: int, rows: int) -> None:
        with self._lock:
            self._manifest_ms = elapsed_ns / 1e6
            self._manifest_files = int(files)
            self._manifest_rows = int(rows)

    def record_decode(self, elapsed_ns: int, rows: int, batches: int) -> None:
        with self._lock:
            self._decode_latency.add(elapsed_ns)
            self._decoded_rows += int(rows)
            self._decoded_batches += int(batches)

    def record_prefetch_wait(self, elapsed_ns: int) -> None:
        with self._lock:
            self._prefetch_wait_latency.add(elapsed_ns)
            if elapsed_ns >= 100_000:
                self._queue_empty_events += 1

    def record_queue(self, depth: int, *, submitted: int = 0, completed: int = 0) -> None:
        with self._lock:
            self._queue_depth_sum += int(depth)
            self._queue_depth_samples += 1
            self._queue_depth_max = max(self._queue_depth_max, int(depth))
            self._submitted_batches += int(submitted)
            self._completed_batches += int(completed)

    def record_source_wait(self, elapsed_ns: int) -> None:
        with self._lock:
            self._source_wait_latency.add(elapsed_ns)

    def record_h2d(self, elapsed_ns: int, byte_count: int) -> None:
        with self._lock:
            self._h2d_latency.add(elapsed_ns)
            self._h2d_bytes += int(byte_count)
            self._h2d_batches += 1

    def record_cuda_prefetch(
        self,
        copy_elapsed_ms: float,
        exposed_wait_ms: float,
    ) -> None:
        with self._lock:
            self._cuda_h2d_latency.add(round(float(copy_elapsed_ms) * 1e6))
            self._cuda_wait_latency.add(round(float(exposed_wait_ms) * 1e6))
            self._cuda_prefetch_batches += 1

    def record_cache_access(
        self,
        kind: str,
        elapsed_ns: int,
        *,
        reload: bool = False,
        evictions: int = 0,
    ) -> None:
        with self._lock:
            if kind == "hit":
                self._cache_hits += 1
            elif kind == "wait":
                self._cache_waits += 1
            elif kind == "miss":
                self._cache_misses += 1
                self._cache_load_ns += int(elapsed_ns)
                if reload:
                    self._cache_reloads += 1
            else:
                raise ValueError(f"unknown cache access kind {kind!r}")
            self._cache_evictions += int(evictions)

    @staticmethod
    def _current_rss_bytes() -> int:
        if sys.platform.startswith("linux"):
            try:
                with open("/proc/self/statm", encoding="ascii") as stream:
                    resident_pages = int(stream.read().split()[1])
                return resident_pages * os.sysconf("SC_PAGE_SIZE")
            except (OSError, ValueError, IndexError):
                pass
        if resource is None:
            return 0
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak * (1 if sys.platform == "darwin" else 1024))

    def snapshot(
        self,
        *,
        active_workers: int,
        prefetch_batches: int,
        cache_state: Mapping[str, int] | None = None,
    ) -> dict[str, float]:
        now = time.perf_counter_ns()
        with self._lock:
            elapsed_ns = max(1, now - self._last_snapshot_ns)
            decode_count = self._decode_latency.count()
            prefetch_wait_count = self._prefetch_wait_latency.count()
            source_wait_count = self._source_wait_latency.count()
            h2d_count = self._h2d_latency.count()
            cuda_h2d_count = self._cuda_h2d_latency.count()
            cuda_wait_count = self._cuda_wait_latency.count()
            service_rows_s = (
                self._decoded_rows
                * max(1, int(active_workers))
                * 1e9
                / max(1, self._decode_latency.total)
            )
            values = {
                "throughput/decoded_rows_s": self._decoded_rows * 1e9 / elapsed_ns,
                "throughput/producer_capacity_rows_s": service_rows_s,
                "latency/decode_mean_ms": (
                    self._decode_latency.total / max(1, decode_count) / 1e6
                ),
                "latency/decode_p95_ms": self._decode_latency.percentile_ns(0.95) / 1e6,
                "latency/prefetch_wait_mean_ms": (
                    self._prefetch_wait_latency.total
                    / max(1, prefetch_wait_count)
                    / 1e6
                ),
                "latency/prefetch_wait_p95_ms": (
                    self._prefetch_wait_latency.percentile_ns(0.95) / 1e6
                ),
                "latency/source_wait_mean_ms": (
                    self._source_wait_latency.total / max(1, source_wait_count) / 1e6
                ),
                "latency/source_wait_p95_ms": (
                    self._source_wait_latency.percentile_ns(0.95) / 1e6
                ),
                "transfer/h2d_launch_mean_ms": (
                    self._h2d_latency.total / max(1, h2d_count) / 1e6
                ),
                "transfer/h2d_launch_p95_ms": (
                    self._h2d_latency.percentile_ns(0.95) / 1e6
                ),
                "transfer/h2d_submit_gib_s": (
                    self._h2d_bytes * 1e9 / max(1, self._h2d_latency.total) / 1024**3
                ),
                "transfer/h2d_device_mean_ms": (
                    self._cuda_h2d_latency.total
                    / max(1, cuda_h2d_count)
                    / 1e6
                ),
                "transfer/h2d_device_p95_ms": (
                    self._cuda_h2d_latency.percentile_ns(0.95) / 1e6
                ),
                "latency/cuda_prefetch_exposed_wait_mean_ms": (
                    self._cuda_wait_latency.total
                    / max(1, cuda_wait_count)
                    / 1e6
                ),
                "latency/cuda_prefetch_exposed_wait_p95_ms": (
                    self._cuda_wait_latency.percentile_ns(0.95) / 1e6
                ),
                "transfer/h2d_bytes": float(self._h2d_bytes),
                "prefetch/queue_depth_mean": (
                    self._queue_depth_sum / max(1, self._queue_depth_samples)
                ),
                "prefetch/queue_depth_max": float(self._queue_depth_max),
                "prefetch/starvation_ratio": (
                    self._queue_empty_events / max(1, prefetch_wait_count)
                ),
                "prefetch/submitted_batches": float(self._submitted_batches),
                "prefetch/completed_batches": float(self._completed_batches),
                "prefetch/active_workers": float(active_workers),
                "prefetch/configured_batches": float(prefetch_batches),
                "cache/hits": float(self._cache_hits),
                "cache/misses": float(self._cache_misses),
                "cache/waits": float(self._cache_waits),
                "cache/reloads": float(self._cache_reloads),
                "cache/evictions": float(self._cache_evictions),
                "cache/hit_ratio": (
                    self._cache_hits
                    / max(1, self._cache_hits + self._cache_misses + self._cache_waits)
                ),
                "cache/load_mean_ms": self._cache_load_ns / max(1, self._cache_misses) / 1e6,
                "memory/rss_gib": self._current_rss_bytes() / 1024**3,
                "manifest/elapsed_ms": self._manifest_ms,
                "manifest/files": float(self._manifest_files),
                "manifest/rows": float(self._manifest_rows),
                "interval/seconds": elapsed_ns / 1e9,
                "interval/decoded_batches": float(self._decoded_batches),
                "interval/h2d_batches": float(self._h2d_batches),
                "interval/cuda_prefetch_batches": float(
                    self._cuda_prefetch_batches
                ),
            }
            if cache_state is not None:
                values.update(
                    {
                        "cache/resident_entries": float(cache_state["entries"]),
                        "cache/resident_bytes": float(cache_state["bytes"]),
                        "cache/capacity_entries": float(cache_state["capacity_entries"]),
                        "cache/capacity_bytes": float(cache_state["capacity_bytes"]),
                    }
                )
            self._last_snapshot_ns = now
            self._decode_latency.clear()
            self._prefetch_wait_latency.clear()
            self._source_wait_latency.clear()
            self._h2d_latency.clear()
            self._cuda_h2d_latency.clear()
            self._cuda_wait_latency.clear()
            self._decoded_rows = 0
            self._decoded_batches = 0
            self._submitted_batches = 0
            self._completed_batches = 0
            self._queue_depth_sum = 0
            self._queue_depth_samples = 0
            self._queue_depth_max = 0
            self._queue_empty_events = 0
            self._cache_hits = 0
            self._cache_misses = 0
            self._cache_waits = 0
            self._cache_reloads = 0
            self._cache_evictions = 0
            self._cache_load_ns = 0
            self._h2d_bytes = 0
            self._h2d_batches = 0
            self._cuda_prefetch_batches = 0
            return values


class ObservedProcessedNpzDecoder(ProcessedNpzDecoder):
    """Processed decoder with file-cache measurements on the opt-in path."""

    def __init__(self, *args, pipeline_stats: PipelineStats, **kwargs):
        self.pipeline_stats = pipeline_stats
        self._observed_loaded_paths: set[str] = set()
        super().__init__(*args, **kwargs)

    def _load(self, path: str) -> dict[str, np.ndarray]:
        canonical = os.path.abspath(path)
        start = time.perf_counter_ns()
        with self._array_cache_condition:
            waited = canonical in self._array_cache_loading
            while canonical in self._array_cache_loading:
                self._array_cache_condition.wait()
            cached = self._array_cache_values.get(canonical)
            if cached is not None:
                self._array_cache_values.move_to_end(canonical)
                self.pipeline_stats.record_cache_access(
                    "wait" if waited else "hit",
                    time.perf_counter_ns() - start,
                )
                return cached
            before_keys = frozenset(self._array_cache_values)
            self._array_cache_loading.add(canonical)
        reloaded = canonical in self._observed_loaded_paths
        try:
            arrays = self._load_uncached(canonical)
        except BaseException:
            with self._array_cache_condition:
                self._array_cache_loading.remove(canonical)
                self._array_cache_condition.notify_all()
            raise
        with self._array_cache_condition:
            self._array_cache_values[canonical] = arrays
            self._array_cache_values.move_to_end(canonical)
            self._evict_array_cache()
            self._array_cache_loading.remove(canonical)
            self._array_cache_condition.notify_all()
            after_keys = frozenset(self._array_cache_values)
        self._observed_loaded_paths.add(canonical)
        self.pipeline_stats.record_cache_access(
            "miss",
            time.perf_counter_ns() - start,
            reload=reloaded,
            evictions=len(before_keys.difference(after_keys)),
        )
        return arrays


def _default_profile_cache_dir() -> Path:
    configured = os.environ.get("XDG_CACHE_HOME")
    if configured:
        root = Path(configured)
    elif os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        root = Path(os.environ["LOCALAPPDATA"])
    else:
        root = Path.home() / ".cache"
    return root / "pytorch-nnue-trainer" / "pipeline-tuning" / "v1"


class PipelineProfileStore:
    """Small atomic cache of resolved machine-specific tuning profiles."""

    def __init__(self, cache_dir: str | None, *, max_entries: int = 64):
        self.cache_dir = Path(cache_dir) if cache_dir else _default_profile_cache_dir()
        self.max_entries = max_entries

    @staticmethod
    def _valid(record) -> bool:
        return (
            isinstance(record, dict)
            and record.get("schema") == PIPELINE_TUNING_SCHEMA
            and isinstance(record.get("exact_key"), str)
            and isinstance(record.get("compatible_key"), str)
            and isinstance(record.get("resolved"), dict)
        )

    def _profile_paths(self):
        return [
            path
            for path in self.cache_dir.glob("*.json")
            if len(path.stem) == 64
            and all(character in "0123456789abcdef" for character in path.stem)
        ]

    def load(self, exact_key: str, compatible_key: str, reuse: str):
        if reuse == "off" or not self.cache_dir.is_dir():
            return None, None
        exact_path = self.cache_dir / f"{exact_key}.json"
        candidates = [exact_path]
        if reuse == "compatible" and not exact_path.exists():
            candidates.extend(
                sorted(
                    self._profile_paths(),
                    key=lambda path: path.stat().st_mtime_ns,
                    reverse=True,
                )
            )
        for path in candidates:
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not self._valid(record):
                continue
            if record["exact_key"] == exact_key:
                return record, "exact"
            if reuse == "compatible" and record["compatible_key"] == compatible_key:
                return record, "compatible"
        return None, None

    def save(self, record: dict) -> None:
        if not self._valid(record):
            raise ValueError("cannot save an invalid pipeline tuning profile")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        destination = self.cache_dir / f"{record['exact_key']}.json"
        temporary = self.cache_dir / (
            f".{record['exact_key']}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        temporary.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
        profiles = sorted(
            self._profile_paths(),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        for expired in profiles[self.max_entries :]:
            try:
                expired.unlink()
            except FileNotFoundError:
                pass


def _canonical_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_tuning_keys(
    *,
    source_manifest: dict,
    batch_size: int,
    world_size: int,
    pin_memory: bool,
    pipeline_signatures,
    cache_budget_bytes: int,
    tuning_contract: Mapping[str, object],
) -> tuple[str, str]:
    gpu = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        gpu = {
            "name": properties.name,
            "major": properties.major,
            "minor": properties.minor,
            "total_memory": properties.total_memory,
        }
    hardware = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "gpu": gpu,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "numpy": np.__version__,
    }
    common = {
        "schema": PIPELINE_TUNING_SCHEMA,
        "performance_abi": PIPELINE_PERFORMANCE_ABI,
        "hardware": hardware,
        "batch_size": int(batch_size),
        "world_size": int(world_size),
        "pin_memory": bool(pin_memory),
        "pipelines": pipeline_signatures,
        "cache_budget_bytes": int(cache_budget_bytes),
        "tuning_contract": dict(tuning_contract),
    }
    exact = {**common, "source": source_manifest}
    files = source_manifest.get("files", [])
    compatible_source = {
        "schema": source_manifest.get("schema"),
        "decoder": source_manifest.get("decoder"),
        "file_count": len(files),
        "row_count": source_manifest.get("logical_row_count"),
        "file_sizes": sorted(int(item.get("size", 0)) for item in files),
        "row_counts": sorted(int(item.get("logical_row_count", 0)) for item in files),
        "board_sizes": sorted({tuple(item.get("board_size", ())) for item in files}),
    }
    compatible = {**common, "source": compatible_source}
    return _canonical_hash(exact), _canonical_hash(compatible)


class PipelineAutotuner:
    """Bounded controller for timing-only processed-NPZ parameters."""

    def __init__(
        self,
        config: PipelineAutotuneConfig,
        *,
        initial_workers: int,
        initial_prefetch_batches: int,
        initial_cache_entries: int,
        initial_cache_bytes: int,
        exact_key: str,
        compatible_key: str,
        locked_options=(),
    ):
        self.config = config
        self.exact_key = exact_key
        self.compatible_key = compatible_key
        self.locked_options = frozenset(locked_options)
        self.store = PipelineProfileStore(config.cache_dir)
        self.settings = {
            "prefetch_threads": int(initial_workers),
            "prefetch_batches": int(initial_prefetch_batches),
            "cache_entries": int(initial_cache_entries),
            "cache_bytes": int(initial_cache_bytes),
        }
        self.requested_settings = dict(self.settings)
        self.reused = None
        self.frozen = False
        self.started_iteration: int | None = None
        self.last_decision_iteration: int | None = None
        self.decisions: list[dict] = []
        self.run_record_path: Path | None = None
        profile, reuse_kind = self.store.load(exact_key, compatible_key, config.reuse)
        if profile is not None:
            for name, value in profile["resolved"].items():
                if name in self.settings and name not in self.locked_options:
                    self.settings[name] = int(value)
            self.reused = reuse_kind
        self._clamp_settings()

    def _clamp_settings(self) -> None:
        self.settings["prefetch_threads"] = min(
            max(1, self.settings["prefetch_threads"]),
            self.config.max_prefetch_threads,
        )
        self.settings["prefetch_batches"] = min(
            max(1, self.settings["prefetch_batches"]),
            self.config.max_prefetch_batches,
        )
        self.settings["cache_entries"] = max(1, self.settings["cache_entries"])
        self.settings["cache_bytes"] = min(
            max(1, self.settings["cache_bytes"]),
            self.config.host_cache_budget_bytes,
        )

    def attach_run_dir(self, rundir: str | None) -> None:
        self.run_record_path = None if rundir is None else Path(rundir) / "pipeline_tuning.json"
        self._write_run_record()

    def _record(self) -> dict:
        return {
            "schema": PIPELINE_TUNING_SCHEMA,
            "exact_key": self.exact_key,
            "compatible_key": self.compatible_key,
            "resolved": dict(self.settings),
            "requested": dict(self.requested_settings),
            "config": {
                name: getattr(self.config, name)
                for name in self.config.__dataclass_fields__
                if name != "cache_dir"
            },
            "locked_options": sorted(self.locked_options),
            "reused": self.reused,
            "frozen": self.frozen,
            "started_iteration": self.started_iteration,
            "last_decision_iteration": self.last_decision_iteration,
            "decisions": list(self.decisions),
            "updated_unix_ns": time.time_ns(),
        }

    def _write_run_record(self) -> None:
        if self.run_record_path is None:
            return
        self.run_record_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.run_record_path.with_name(
            f".{self.run_record_path.name}.{os.getpid()}.tmp"
        )
        temporary.write_text(
            json.dumps(self._record(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.run_record_path)

    def _freeze(self, iteration: int, reason: str, metrics: Mapping[str, float]) -> None:
        self.frozen = True
        self.last_decision_iteration = int(iteration)
        self.decisions.append(
            {
                "iteration": int(iteration),
                "action": "freeze",
                "reason": reason,
                "producer_headroom": float(metrics.get("producer_headroom", 0.0)),
                "data_wait_fraction": float(metrics.get("data_wait_fraction", 0.0)),
            }
        )
        record = self._record()
        try:
            self.store.save(record)
        except OSError as exc:
            self.decisions.append(
                {
                    "iteration": int(iteration),
                    "action": "profile_cache_write_failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
        self._write_run_record()

    def update(self, metrics: Mapping[str, float], iteration: int) -> dict | None:
        iteration = int(iteration)
        if self.started_iteration is None:
            self.started_iteration = iteration
        elapsed_iterations = max(0, iteration - self.started_iteration)
        minimum_observation = (
            self.config.verify_iterations
            if self.reused is not None
            else self.config.warmup_iterations
        )
        if self.frozen or elapsed_iterations < minimum_observation:
            return None
        if (
            self.last_decision_iteration is not None
            and iteration - self.last_decision_iteration
            < self.config.decision_interval
        ):
            return None
        self.last_decision_iteration = iteration
        headroom = float(metrics.get("producer_headroom", 0.0))
        wait_fraction = float(metrics.get("data_wait_fraction", 0.0))
        healthy = (
            headroom >= self.config.target_producer_headroom
            and wait_fraction <= self.config.max_data_wait_fraction
        )
        if (
            self.reused is not None
            and elapsed_iterations >= self.config.verify_iterations
        ):
            if healthy:
                self._freeze(iteration, f"validated {self.reused} cached profile", metrics)
                return self.state_dict()
            self.decisions.append(
                {
                    "iteration": int(iteration),
                    "action": "invalidate_reuse",
                    "reason": "cached profile missed the runtime health target",
                }
            )
            self.reused = None

        if elapsed_iterations >= self.config.freeze_after:
            self._freeze(iteration, "configured tuning window ended", metrics)
            return self.state_dict()

        before = dict(self.settings)
        reason = None
        if not healthy:
            if (
                "prefetch_threads" not in self.locked_options
                and self.settings["prefetch_threads"] < self.config.max_prefetch_threads
            ):
                self.settings["prefetch_threads"] += 1
                reason = "producer headroom or exposed-wait target was missed"
            elif (
                "prefetch_batches" not in self.locked_options
                and self.settings["prefetch_batches"] < self.config.max_prefetch_batches
            ):
                self.settings["prefetch_batches"] = min(
                    self.config.max_prefetch_batches,
                    max(
                        self.settings["prefetch_batches"] + 1,
                        self.settings["prefetch_batches"] * 2,
                    ),
                )
                reason = "prefetch remained starved at the worker limit"
            elif (
                metrics.get("cache/reloads", 0.0) > 0
                and "cache_bytes" not in self.locked_options
                and self.settings["cache_bytes"] < self.config.host_cache_budget_bytes
            ):
                self.settings["cache_entries"] += 1
                self.settings["cache_bytes"] = min(
                    self.config.host_cache_budget_bytes,
                    max(
                        self.settings["cache_bytes"] + 128 * 1024**2,
                        int(self.settings["cache_bytes"] * 1.25),
                    ),
                )
                reason = "decoded files were reloaded within the observation window"
        else:
            queue_depth = float(metrics.get("prefetch/queue_depth_mean", 0.0))
            minimum_depth = max(2, self.settings["prefetch_threads"] * 2)
            if (
                "prefetch_batches" not in self.locked_options
                and self.settings["prefetch_batches"] > minimum_depth
                and queue_depth >= self.settings["prefetch_batches"] * 0.7
                and headroom >= self.config.target_producer_headroom * 2
            ):
                self.settings["prefetch_batches"] = max(
                    minimum_depth, self.settings["prefetch_batches"] // 2
                )
                reason = "the queue stayed full with excess producer headroom"
            elif (
                "prefetch_threads" not in self.locked_options
                and self.settings["prefetch_threads"] > 1
                and headroom >= self.config.target_producer_headroom * 2
            ):
                self.settings["prefetch_threads"] -= 1
                reason = "fewer workers still satisfy the producer target"

        self._clamp_settings()
        if self.settings != before:
            self.decisions.append(
                {
                    "iteration": int(iteration),
                    "action": "adjust",
                    "reason": reason,
                    "before": before,
                    "after": dict(self.settings),
                    "producer_headroom": headroom,
                    "data_wait_fraction": wait_fraction,
                }
            )
            self._write_run_record()
        elif healthy:
            self._freeze(
                iteration,
                "health target satisfied with no further useful adjustment",
                metrics,
            )
        return self.state_dict()

    def state_dict(self) -> dict:
        return {
            "schema": PIPELINE_TUNING_SCHEMA,
            "exact_key": self.exact_key,
            "compatible_key": self.compatible_key,
            "settings": dict(self.settings),
            "reused": self.reused,
            "frozen": self.frozen,
            "started_iteration": self.started_iteration,
            "last_decision_iteration": self.last_decision_iteration,
            "decisions": list(self.decisions),
        }

    def load_state_dict(self, state: dict) -> None:
        if not isinstance(state, dict) or state.get("schema") != PIPELINE_TUNING_SCHEMA:
            raise ValueError("pipeline tuning state has an incompatible schema")
        if (
            state.get("exact_key") != self.exact_key
            or state.get("compatible_key") != self.compatible_key
        ):
            raise ValueError("pipeline tuning state belongs to a different runtime")
        settings = state.get("settings")
        if not isinstance(settings, dict) or set(settings) != set(self.settings):
            raise ValueError("pipeline tuning state has invalid settings")
        self.settings = {name: int(value) for name, value in settings.items()}
        self._clamp_settings()
        self.reused = state.get("reused")
        self.frozen = bool(state.get("frozen", False))
        started_iteration = state.get("started_iteration")
        self.started_iteration = (
            None if started_iteration is None else int(started_iteration)
        )
        last_decision_iteration = state.get("last_decision_iteration")
        self.last_decision_iteration = (
            None
            if last_decision_iteration is None
            else int(last_decision_iteration)
        )
        decisions = state.get("decisions", [])
        if not isinstance(decisions, list):
            raise ValueError("pipeline tuning decisions must be a list")
        self.decisions = list(decisions)
        self._write_run_record()


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
    """Opt-in measured adapter with timing-only runtime controls."""

    def __init__(
        self,
        *args,
        pipeline_stats,
        autotuner=None,
        maximum_prefetch_workers: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prefetch_audit = {
            "configured_workers": self.prefetch_workers,
            "effective_workers": self._effective_prefetch_workers,
            "configured_batches": self.prefetch_batches,
            "finalize_in_worker": self._finalize_in_prefetch,
            "disabled_reason": (
                "stateful batch pipelines require ordered planning"
                if self.prefetch_workers > 0
                and self.planner.pipeline_composer is not None
                else None
            ),
            "max_queued_batches": 0,
            "submitted_batches": 0,
            "completed_batches": 0,
        }
        self.pipeline_stats = pipeline_stats
        self.autotuner = autotuner
        self._maximum_prefetch_workers = max(
            self._effective_prefetch_workers,
            maximum_prefetch_workers or self._effective_prefetch_workers,
        )
        self._concurrency_gate = (
            _AdjustableConcurrencyGate(self.active_prefetch_workers)
            if autotuner is not None and self._effective_prefetch_workers > 0
            else None
        )

    @property
    def active_prefetch_workers(self) -> int:
        if self.autotuner is None:
            return self._effective_prefetch_workers
        return int(self.autotuner.settings["prefetch_threads"])

    @property
    def active_prefetch_batches(self) -> int:
        if self.autotuner is None:
            return self.prefetch_batches
        return int(self.autotuner.settings["prefetch_batches"])

    @staticmethod
    def _decoded_row_count(decoded) -> int:
        return len(decoded[1]) if isinstance(decoded, tuple) and len(decoded) == 3 else 0

    def _decode_batch(self, batch):
        start = time.perf_counter_ns()
        decoded = super()._decode_batch(batch)
        self.pipeline_stats.record_decode(
            time.perf_counter_ns() - start,
            self._decoded_row_count(decoded),
            1,
        )
        return decoded

    def _decode_batches(self, batches):
        start = time.perf_counter_ns()
        decoded = super()._decode_batches(batches)
        self.pipeline_stats.record_decode(
            time.perf_counter_ns() - start,
            sum(self._decoded_row_count(item) for item in decoded),
            len(decoded),
        )
        return decoded

    def _run_decode_batches(self, batches):
        if self._concurrency_gate is None:
            return self._decode_batches(batches)
        return self._concurrency_gate.run(self._decode_batches, batches)

    def _iter_prefetched(self):
        planned = iter(self._planned_transactions())
        pending = deque()
        pending_batches = 0
        exhausted = False

        def submit_chunk(executor):
            nonlocal exhausted, pending_batches
            limit = self.active_prefetch_batches
            workers = self.active_prefetch_workers
            if exhausted or pending_batches >= limit:
                return
            chunk_size = max(1, limit // max(2, workers))
            items = []
            capacity = limit - pending_batches
            for _ in range(min(chunk_size, capacity)):
                try:
                    items.append(next(planned))
                except StopIteration:
                    exhausted = True
                    break
            if not items:
                return
            pending.append(
                (
                    tuple(items),
                    executor.submit(
                        self._run_decode_batches,
                        tuple(batch for batch, _ in items),
                    ),
                )
            )
            pending_batches += len(items)
            self.prefetch_audit["submitted_batches"] += len(items)
            self.prefetch_audit["max_queued_batches"] = max(
                self.prefetch_audit["max_queued_batches"],
                pending_batches,
            )
            self.pipeline_stats.record_queue(pending_batches, submitted=len(items))

        with ThreadPoolExecutor(max_workers=self._maximum_prefetch_workers) as executor:
            while pending_batches < self.active_prefetch_batches and not exhausted:
                submit_chunk(executor)
            while pending:
                items, future = pending.popleft()
                wait_start = time.perf_counter_ns()
                decoded_batches = future.result()
                self.pipeline_stats.record_prefetch_wait(
                    time.perf_counter_ns() - wait_start
                )
                if len(decoded_batches) != len(items):
                    raise RuntimeError("prefetch worker changed the batch count")
                for (batch, token), decoded in zip(items, decoded_batches):
                    pending_batches -= 1
                    self.prefetch_audit["completed_batches"] += 1
                    self.pipeline_stats.record_queue(pending_batches, completed=1)
                    yield self._publish(batch, token, decoded)
                if self._concurrency_gate is not None:
                    self._concurrency_gate.set_limit(self.active_prefetch_workers)
                while pending_batches < self.active_prefetch_batches and not exhausted:
                    submit_chunk(executor)

    def pipeline_metrics_snapshot(self) -> dict[str, float]:
        decoder = getattr(self.source, "decoder", None)
        cache_state = (
            decoder.cache_state()
            if decoder is not None and hasattr(decoder, "cache_state")
            else None
        )
        return self.pipeline_stats.snapshot(
            active_workers=self.active_prefetch_workers,
            prefetch_batches=self.active_prefetch_batches,
            cache_state=cache_state,
        )

    def attach_pipeline_run_dir(self, rundir: str | None) -> None:
        if self.autotuner is not None:
            self.autotuner.attach_run_dir(rundir)

    def pipeline_tuning_update(self, metrics, iteration: int):
        if self.autotuner is None:
            return None
        state = self.autotuner.update(metrics, iteration)
        if state is not None:
            self.load_pipeline_tuning_state_dict(state)
        return state

    def pipeline_tuning_state_dict(self):
        return None if self.autotuner is None else self.autotuner.state_dict()

    def load_pipeline_tuning_state_dict(self, state) -> None:
        if self.autotuner is None:
            if state is not None:
                raise ValueError("pipeline autotuning is disabled")
            return
        self.autotuner.load_state_dict(state)
        if self._concurrency_gate is not None:
            self._concurrency_gate.set_limit(self.active_prefetch_workers)
        decoder = getattr(self.source, "decoder", None)
        if decoder is not None and hasattr(decoder, "configure_cache"):
            decoder.configure_cache(
                entries=self.autotuner.settings["cache_entries"],
                byte_capacity=self.autotuner.settings["cache_bytes"],
            )

    def restore_pipeline_tuning_state_dict(self, state) -> bool:
        """Restore a same-runtime performance state, or ignore a stale one."""
        if self.autotuner is None:
            return False
        if (
            not isinstance(state, dict)
            or state.get("exact_key") != self.autotuner.exact_key
            or state.get("compatible_key") != self.autotuner.compatible_key
        ):
            return False
        self.load_pipeline_tuning_state_dict(state)
        return True


def aggregate_pipeline_snapshots(
    snapshots: list[Mapping[str, float]],
    *,
    consumer_batches_s: float,
    rows_per_batch: int,
) -> dict[str, float]:
    if not snapshots:
        return {}
    keys = set(snapshots[0])
    if any(set(snapshot) != keys for snapshot in snapshots[1:]):
        raise ValueError("pipeline metric schemas differ across ranks")
    aggregate = {
        key: sum(float(snapshot[key]) for snapshot in snapshots) / len(snapshots)
        for key in keys
    }
    capacities = [
        float(snapshot["throughput/producer_capacity_rows_s"])
        for snapshot in snapshots
    ]
    waits = []
    for snapshot in snapshots:
        transfer_wait = (
            float(snapshot["latency/cuda_prefetch_exposed_wait_p95_ms"])
            if snapshot.get("interval/cuda_prefetch_batches", 0.0) > 0
            else float(snapshot["transfer/h2d_launch_p95_ms"])
        )
        waits.append(
            float(snapshot["latency/source_wait_p95_ms"]) + transfer_wait
        )
    consumer_rows_s = consumer_batches_s * rows_per_batch
    aggregate["throughput/consumer_rows_s"] = float(consumer_rows_s)
    aggregate["producer_headroom"] = min(capacities) / max(1e-12, consumer_rows_s)
    step_ms = 1000.0 / max(1e-12, consumer_batches_s)
    aggregate["data_wait_fraction"] = max(waits) / step_ms
    aggregate["distributed/rank_count"] = float(len(snapshots))
    aggregate["distributed/producer_capacity_min_rows_s"] = min(capacities)
    aggregate["distributed/exposed_data_wait_p95_max_ms"] = max(waits)
    return aggregate


def tensor_bytes(value) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, Mapping):
        return sum(tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(item) for item in value)
    return 0
