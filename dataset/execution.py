"""Choose a physical record backend independently of its resource policy."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from .source_dataset import SourceBatchDataset
from .stream import reject_duplicate_physical_files


Backend = Literal["packed", "generic", "indexed", "sequential", "map"]
Policy = Literal["continuous", "manual", "legacy_fixed"]


@dataclass(frozen=True, slots=True)
class ExecutionCapabilities:
    packed_records: bool
    thread_safe_materialization: bool
    chunk_materialization: bool
    budgeted_decode: bool
    mapped_cache: bool
    parallel_stateless_pipeline: bool
    memory_accounting: Literal["reserved", "observed", "untracked"]


@dataclass(frozen=True, slots=True)
class ExecutionDecision:
    backend: Backend
    policy: Policy
    capabilities: ExecutionCapabilities
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ResumeExecutionHint:
    backend: Backend | None = None
    legacy: bool = False


@dataclass(frozen=True, slots=True)
class FileMemoryBound:
    resident_bytes: int
    transient_bytes: int
    catalog_bytes: int = 0

    def __post_init__(self):
        if any(
            type(value) is not int or value < 0
            for value in (
                self.resident_bytes,
                self.transient_bytes,
                self.catalog_bytes,
            )
        ):
            raise ValueError("file memory bounds must be non-negative integers")
        if self.transient_bytes < self.resident_bytes:
            raise ValueError("transient file bound cannot be below retained bytes")


class PipelineLifecycleMixin:
    """Shared node-cache and adaptive telemetry lifecycle for NPZ streams."""

    def prepare_node_decoded_cache(self):
        """Build this run's node-local immutable cache on the node leader."""

        if not self._node_decoded_cache_is_supported():
            return
        paths = reject_duplicate_physical_files(self.file_list)
        cache = self.adaptive_pipeline.node_decoded_cache
        cache.prepare(
            paths,
            workers=self.adaptive_pipeline.resources.per_rank_cpu_limit,
        )

    def activate_node_decoded_cache(self):
        """Install the collectively prepared source-to-mmap catalog."""

        if not self._node_decoded_cache_is_supported():
            return
        paths = reject_duplicate_physical_files(self.file_list)
        self._node_decoded_cache_catalog = (
            self.adaptive_pipeline.node_decoded_cache.catalog(paths)
        )

    def node_decoded_cache_ready(self):
        if not self._node_decoded_cache_is_supported():
            return False
        return self.adaptive_pipeline.node_decoded_cache.is_ready()

    def set_node_decoded_cache_enabled(self, enabled):
        """Apply one globally coordinated cache availability decision."""

        if type(enabled) is not bool:
            raise TypeError("node decoded-cache enablement must be a boolean")
        if enabled:
            self.activate_node_decoded_cache()
            return
        self._node_decoded_cache_catalog = None
        cache = (
            None
            if self.adaptive_pipeline is None
            else self.adaptive_pipeline.node_decoded_cache
        )
        if cache is not None:
            cache.cleanup()

    def cleanup_node_decoded_cache(self):
        cache = (
            None
            if self.adaptive_pipeline is None
            else self.adaptive_pipeline.node_decoded_cache
        )
        if cache is not None:
            cache.cleanup()

    def pipeline_metrics_snapshot(self):
        if hasattr(self._planned_decoder, "pipeline_metrics_snapshot"):
            metrics = self._planned_decoder.pipeline_metrics_snapshot()
        elif hasattr(self, "passive_pipeline_stats"):
            metrics = self.passive_pipeline_stats.snapshot(
                active_workers=self._planned_decoder.active_prefetch_workers,
            )
        else:
            return None
        runtime = self._adaptive_pipeline_runtime
        if runtime is not None and metrics is not None and hasattr(runtime, "memory_snapshot"):
            memory = runtime.memory_snapshot()
            total_bytes = memory["total_bytes"]
            metrics = replace(
                metrics,
                host_memory_used_fraction=memory["used_bytes"] / total_bytes,
                host_memory_high_water_fraction=(
                    memory["high_water_bytes"] / total_bytes
                ),
                host_memory_backpressure_events=memory["backpressure_events"],
            )
        return metrics

    def pipeline_tuning_update(
        self,
        metrics,
        iteration,
        *,
        epoch_changed=False,
    ):
        if not hasattr(self._planned_decoder, "pipeline_tuning_update"):
            return None
        return self._planned_decoder.pipeline_tuning_update(
            metrics,
            iteration,
            epoch_changed=epoch_changed,
        )

    def pipeline_tuning_state_dict(self):
        if not hasattr(self._planned_decoder, "pipeline_tuning_state_dict"):
            return None
        return self._planned_decoder.pipeline_tuning_state_dict()

    def load_pipeline_tuning_state_dict(self, state):
        if not hasattr(self._planned_decoder, "load_pipeline_tuning_state_dict"):
            if state is not None:
                raise ValueError("adaptive data pipeline is disabled")
            return
        self._planned_decoder.load_pipeline_tuning_state_dict(state)

    def restore_pipeline_tuning_state_dict(self, state):
        restore = getattr(
            self._planned_decoder,
            "restore_pipeline_tuning_state_dict",
            None,
        )
        return False if restore is None else restore(state)


class IndexedLifecycleMixin(PipelineLifecycleMixin):
    """Attach serial indexed NPZ materialization to the shared runtime."""

    def _node_decoded_cache_is_supported(self):
        return False

    def _finish_indexed_execution(self, decoder, catalogs, planner_config):
        from .pipeline_runtime import AdaptivePipelineRuntime
        from .telemetry import PipelineStats

        spec = self.adaptive_pipeline
        runtime = None
        stats = None
        if spec is not None:
            context = self.runtime_context
            composer = self._partitioned_stream.pipeline_composer
            runtime_manifests = (
                [
                    {
                        **manifest,
                        "output_row_bytes": (
                            manifest["output_row_bytes"]
                            + composer.added_output_row_bytes(manifest["board_size"])
                        ),
                    }
                    for manifest in catalogs
                ]
                if composer is not None else catalogs
            )
            runtime = AdaptivePipelineRuntime(
                spec,
                runtime_manifests,
                local_batch_size=context.local_batch_size,
                global_batch_size=context.global_batch_size,
                shuffle=planner_config.shuffle,
                shuffle_window_size=planner_config.shuffle_buffer_size,
                memory_budget=decoder.host_memory_budget,
                serial_execution=True,
                serial_shape_count=len(self._record_source.shape_codes),
                pre_reserved_semantic_bytes=decoder.catalog_reserved_bytes,
            )
            runtime.reserve_semantic_floor()
            stats = PipelineStats()
        try:
            self.execution_decision = resolve_execution(
                self._record_source,
                requested_policy=spec,
                pipeline_composer=self._partitioned_stream.pipeline_composer,
            )
            self._planned_decoder = build_execution(
                self._partitioned_stream,
                self._record_source,
                decision=self.execution_decision,
                runtime=runtime,
                pipeline_stats=stats,
                prefetch_workers=0,
                prefetch_batches=(
                    1 if runtime is None else runtime.settings.ready_queue_batches
                ),
                host_memory_budget=(None if runtime is None else runtime.memory_budget),
                output_batch_bytes=(0 if runtime is None else runtime.output_batch_bytes),
                planner_token_bytes=(0 if runtime is None else runtime.planner_token_bytes),
            )
        except BaseException:
            if runtime is not None:
                runtime.close()
            raise
        self._adaptive_pipeline_runtime = runtime
        if stats is not None:
            self.pipeline_stats = stats


class SequentialLifecycleMixin(PipelineLifecycleMixin):
    """Run ordered binary readers with a shared observable execution contract."""

    def _node_decoded_cache_is_supported(self):
        return False

    def _finish_sequential_execution(self):
        from .telemetry import PipelineStats

        spec = self.adaptive_pipeline
        runtime = None if spec is None else ObservedExecutionRuntime(spec)
        stats = None if runtime is None else PipelineStats()
        self._record_source.observed_sequential = runtime is not None
        try:
            self.execution_decision = resolve_execution(
                self._record_source,
                requested_policy=spec,
                pipeline_composer=self._partitioned_stream.pipeline_composer,
            )
            self._planned_decoder = build_execution(
                self._partitioned_stream,
                self._record_source,
                decision=self.execution_decision,
                runtime=runtime,
                pipeline_stats=stats,
                prefetch_workers=0,
                prefetch_batches=1,
            )
        except BaseException:
            if runtime is not None:
                runtime.close()
            raise
        self._adaptive_pipeline_runtime = runtime
        if stats is not None:
            self.pipeline_stats = stats


@dataclass(frozen=True, slots=True)
class _ObservedSerialSettings:
    decode_workers: int = 0
    decode_chunk_batches: int = 1
    ready_queue_batches: int = 1
    decoded_cache_bytes: int = 0
    pin_memory: bool = False


class ObservedExecutionRuntime:
    """Share policy and metrics without claiming a host-memory hard cap."""

    memory_accounting = "observed"
    maximum_prefetch_workers = 0
    settings = _ObservedSerialSettings()

    def __init__(self, spec, *, legacy_executor=None):
        config = spec.config
        if config.host_memory_budget != "auto":
            raise ValueError(
                "observational execution cannot enforce an explicit "
                "host_memory_budget"
            )
        if config.adaptation != "continuous" or config.data_cpu_budget != "auto":
            raise ValueError(
                "observational execution has no tunable worker or "
                "cache layout; use the automatic continuous policy"
            )
        if legacy_executor is not None:
            self.settings = _ObservedSerialSettings(
                decode_workers=legacy_executor.active_prefetch_workers,
                decode_chunk_batches=legacy_executor.active_prefetch_chunk_batches,
                ready_queue_batches=legacy_executor.active_prefetch_batches,
                pin_memory=legacy_executor.output_is_pinned,
            )
            self.maximum_prefetch_workers = legacy_executor.active_prefetch_workers

    def update(self, metrics, iteration, *, epoch_changed=False):
        return None

    def state_dict(self):
        return None

    def load_state_dict(self, state):
        if state is not None:
            raise ValueError("observational serial execution has no tuning state")

    def restore_state_dict(self, state):
        return state is None

    def close(self):
        pass


def attach_observed_map_execution(dataset, spec, runtime):
    """Attach policy to an eager map dataset without changing its sampler."""
    if not isinstance(runtime, ObservedExecutionRuntime):
        raise TypeError("map observation requires a preflighted runtime")
    dataset.observed_map = True
    try:
        decision = resolve_execution(dataset, requested_policy=spec)
    except BaseException:
        runtime.close()
        raise
    dataset.adaptive_pipeline = spec
    dataset.execution_decision = decision
    dataset._adaptive_pipeline_runtime = runtime
    return dataset


def attach_observed_legacy_execution(dataset, spec):
    """Observe an existing built-in executor without changing its layout."""
    from .telemetry import PassivePipelineStats

    if getattr(dataset, "_partitioned_stream", None) is None:
        raise RuntimeError("legacy executor must be built before observation")
    executor = dataset._planned_decoder
    runtime = ObservedExecutionRuntime(spec, legacy_executor=executor)
    try:
        decision = resolve_execution(
            dataset._record_source,
            requested_policy=spec,
            pipeline_composer=dataset._partitioned_stream.pipeline_composer,
            observational_legacy=True,
        )
    except BaseException:
        runtime.close()
        raise
    dataset._adaptive_pipeline_runtime = runtime
    dataset.execution_decision = decision
    dataset.passive_pipeline_stats = PassivePipelineStats()
    return dataset


def resolve_execution(
    source,
    *,
    requested_policy=None,
    legacy_options=None,
    pipeline_composer=None,
    resume_hint: ResumeExecutionHint | None = None,
    observational_generic: bool = False,
    observational_legacy: bool = False,
) -> ExecutionDecision:
    """Resolve capabilities without decoding data or changing stream semantics.

    A tuple of child sources denotes a composite whose concrete source has not
    yet been constructed. Its backend is fixed here, before planner creation.
    """
    from .npz_source import DenseNpzSource, IndexedNpzSource
    from .packed_composite import PackedCompositeRecordSource
    from .sequential_source import InterleavedSequentialSource
    from torch.utils.data.dataset import Dataset, IterableDataset

    composite = isinstance(source, (tuple, list))
    sources = tuple(source) if composite else (source,)
    packed = (
        PackedCompositeRecordSource.supports(sources)
        if composite
        else type(source) is DenseNpzSource
        or isinstance(source, PackedCompositeRecordSource)
    )
    if requested_policy is None:
        policy: Policy = "legacy_fixed"
    else:
        policy = requested_policy.config.adaptation
        if policy not in {"continuous", "manual"}:
            raise ValueError(f"unsupported execution policy {policy!r}")
    parallel_pipeline = (
        pipeline_composer is None or pipeline_composer.is_parallel_stateless
    )
    preferred_backend = (
        None if requested_policy is None else requested_policy.preferred_backend
    )
    if packed and (not composite or (
        requested_policy is not None and preferred_backend != "generic"
    )):
        backend: Backend = "packed"
    elif not composite and isinstance(source, IndexedNpzSource):
        backend = "indexed"
    elif not composite and isinstance(source, InterleavedSequentialSource):
        backend = "sequential"
    elif not composite and isinstance(source, Dataset) and not isinstance(
        source, IterableDataset
    ):
        backend = "map"
    else:
        backend = "generic"
    indexed_budgeted = (
        backend == "indexed"
        and getattr(getattr(source, "decoder", None), "host_memory_budget", None)
        is not None
    )
    sequential_observed = (
        backend == "sequential" and bool(getattr(source, "observed_sequential", False))
    )
    map_observed = backend == "map" and bool(getattr(source, "observed_map", False))
    if (
        requested_policy is not None
        and not parallel_pipeline
        and not (sequential_observed or map_observed or observational_generic
                 or observational_legacy)
    ):
        raise ValueError("adaptive execution requires parallel-stateless pipelines")
    child_budgets = tuple(
        getattr(getattr(item, "decoder", None), "host_memory_budget", None)
        or getattr(getattr(item, "decoder", None), "_host_memory_budget", None)
        for item in sources
    )
    generic_budgeted = (
        composite
        and bool(child_budgets)
        and child_budgets[0] is not None
        and all(budget is child_budgets[0] for budget in child_budgets)
    )
    if observational_generic and (not composite or packed):
        raise ValueError("observational generic execution requires a generic composite")
    if requested_policy is not None and not (
        packed or indexed_budgeted or generic_budgeted or sequential_observed
        or map_observed or observational_generic or observational_legacy
    ):
        raise ValueError("adaptive execution requires a supported memory accounting mode")
    reasons = []
    if resume_hint is not None and resume_hint.backend is not None:
        if resume_hint.backend not in {"packed", "generic"}:
            raise ValueError("unsupported saved composite execution backend")
        if resume_hint.backend == "packed" and not packed:
            raise ValueError("saved packed execution is incompatible with current sources")
        if requested_policy is not None and resume_hint.backend == "generic" and not (
            generic_budgeted or observational_generic
        ):
            raise ValueError("saved generic execution requires supported child sources")
        backend = resume_hint.backend
        reasons.append("saved execution backend")
    else:
        reasons.append("dense NPZ capability" if backend == "packed" else "source policy")
    if composite and packed and requested_policy is None and resume_hint is None:
        reasons.append("preserve fixed-policy sample order")
    if requested_policy is None and legacy_options:
        reasons.append("legacy fixed resource options")
    capabilities = ExecutionCapabilities(
        packed_records=packed,
        thread_safe_materialization=(not composite or backend == "packed") and all(
            bool(getattr(item, "thread_safe_materialization", False))
            for item in sources
        ),
        chunk_materialization=(not composite or backend == "packed") and all(
            hasattr(item, "materialize_batches")
            or hasattr(item, "materialize_batches_with_keys")
            for item in sources
        ),
        budgeted_decode=packed or indexed_budgeted or generic_budgeted,
        mapped_cache=all(
            type(item) is DenseNpzSource
            and getattr(getattr(item, "decoder", None), "format_id", "")
            != "batched-raw-katago-npz"
            for item in sources
        ),
        parallel_stateless_pipeline=parallel_pipeline,
        memory_accounting=(
            "observed" if sequential_observed or map_observed or observational_generic
            or observational_legacy else
            "reserved" if requested_policy is not None else "untracked"
        ),
    )
    return ExecutionDecision(backend, policy, capabilities, tuple(reasons))


def build_execution(
    planner,
    source,
    *,
    decision: ExecutionDecision,
    runtime=None,
    pipeline_stats=None,
    **kwargs,
) -> SourceBatchDataset:
    """Create the existing ordered executor; selection adds no per-batch layer."""
    if decision.policy != "legacy_fixed" and runtime is None:
        raise ValueError("adaptive execution requires a runtime")
    if runtime is not None or pipeline_stats is not None:
        from .telemetry import ObservedSourceBatchDataset

        return ObservedSourceBatchDataset(
            planner,
            source,
            pipeline_stats=pipeline_stats,
            adaptive_runtime=runtime,
            **kwargs,
        )
    return SourceBatchDataset(planner, source, **kwargs)
