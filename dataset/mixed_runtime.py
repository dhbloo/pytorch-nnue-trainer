"""Share one adaptive runtime across compatible composite NPZ sources."""

import os

import numpy as np
import torch

from .katago import BatchedProcessedKatagoNumpyDataset, IterativeProcessedKatagoNumpyDataset
from .execution import PipelineLifecycleMixin
from .pipeline_runtime import AdaptivePipelineRuntime
from .telemetry import PipelineStats
from .stream import reject_duplicate_physical_files


class _CompositeDecoderCache:
    def __init__(self, decoders, manifests):
        self.decoders = tuple(decoders)
        self.floors = tuple(
            max(m["decoded_file_bytes"] for m in group) for group in manifests
        )

    def cache_state(self):
        return {
            "capacity_entries": sum(
                d.cache_state()["capacity_entries"] for d in self.decoders
            )
        }

    def configure_cache(
        self, *, entries, byte_capacity, host_memory_budget=None,
        file_size_catalog=(),
    ):
        if byte_capacity < sum(self.floors):
            raise ValueError("mixed decoded cache cannot fit one file from each child")
        remaining = byte_capacity - sum(self.floors)
        for decoder, floor in zip(self.decoders, self.floors):
            state = decoder.cache_state()
            decoder.configure_cache(
                entries=state["capacity_entries"],
                byte_capacity=floor + remaining // len(self.decoders),
            )


class AdaptiveMultiMixin(PipelineLifecycleMixin):
    # The cache is prepared over the union once. Per-child ordinal catalogs
    # must never be independently installed into the shared directory.

    def _init_adaptive_multi(self, spec):
        self.adaptive_pipeline = spec
        self._adaptive_pipeline_runtime = None
        self._observed_composite = False
        self._node_decoded_cache_catalog = None
        self._shared_host_memory_budget = None
        if spec is None:
            return
        from .raw_npz import BatchedKatagoNumpyDataset
        from .katago import IterativeKatagoNumpyDataset
        from .sparse_numpy import IterativeSparseNumpyDataset
        from .simple_binary import SimpleBinaryDataset
        from .packed_binary import PackedBinaryDataset
        from .host_memory import HostMemoryBudget

        supported = {
            BatchedProcessedKatagoNumpyDataset,
            IterativeProcessedKatagoNumpyDataset,
            BatchedKatagoNumpyDataset,
            IterativeKatagoNumpyDataset,
            IterativeSparseNumpyDataset,
        }
        binary_types = {SimpleBinaryDataset, PackedBinaryDataset}
        if not all(type(child) in supported | binary_types for child in self.datasets):
            raise ValueError("adaptive iterative_multi requires supported record children")
        if any(type(child) in binary_types for child in self.datasets):
            from .execution import ObservedExecutionRuntime

            self._adaptive_pipeline_runtime = ObservedExecutionRuntime(spec)
            self._observed_composite = True
            self.file_list = reject_duplicate_physical_files(
                [path for child in self.datasets for path in child.file_list]
            )
            return
        if self.batch_pipelines:
            from .core import PipelineStateComposer

            if not PipelineStateComposer(self.batch_pipelines).is_parallel_stateless:
                raise ValueError(
                    "adaptive iterative_multi requires parallel-stateless batch_pipelines"
                )
        self._shared_host_memory_budget = HostMemoryBudget(
            spec.resources.per_rank_host_budget_bytes
        )
        self.file_list = reject_duplicate_physical_files(
            [path for child in self.datasets for path in child.file_list]
        )

    def _node_decoded_cache_is_supported(self):
        return (
            self.adaptive_pipeline is not None
            and self.adaptive_pipeline.node_decoded_cache is not None
            and all(type(child) is BatchedProcessedKatagoNumpyDataset for child in self.datasets)
        )

    def _configure_adaptive_children(self):
        if self.adaptive_pipeline is None or self._observed_composite:
            return
        catalog = self._node_decoded_cache_catalog
        for child in self.datasets:
            child._shared_host_memory_budget = self._shared_host_memory_budget
            if hasattr(child, "observability"):
                child.observability = True
            if catalog is not None:
                child._node_decoded_cache_catalog = {
                    os.path.abspath(path): catalog[os.path.abspath(path)]
                    for path in child.file_list
                }

    def _install_adaptive_multi(self):
        if self.adaptive_pipeline is None:
            return
        if self._observed_composite:
            from .execution import build_execution

            self.pipeline_stats = PipelineStats()
            self._planned_decoder = build_execution(
                self._partitioned_stream,
                self._record_source,
                decision=self.execution_decision,
                runtime=self._adaptive_pipeline_runtime,
                pipeline_stats=self.pipeline_stats,
                prefetch_workers=0,
                prefetch_batches=1,
            )
            return
        from .packed_composite import PackedCompositeRecordSource

        packed = isinstance(self._record_source, PackedCompositeRecordSource)
        composer = self._partitioned_stream.pipeline_composer
        child_manifests = [
            getattr(child, "_record_manifests", None)
            or getattr(child, "_indexed_manifests", None)
            for child in self.datasets
        ]
        if any(not group for group in child_manifests):
            raise ValueError("adaptive composite child has no budgeted manifest")
        manifests = [
            {
                **manifest,
                "output_row_bytes": (
                    manifest["output_row_bytes"]
                    + composer.added_output_row_bytes(manifest["board_size"])
                ),
            }
            if composer is not None
            else manifest
            for group in child_manifests
            for manifest in group
        ]
        cache_floors = [
            max(manifest["decoded_file_bytes"] for manifest in group)
            for group in child_manifests
        ]
        indexed_catalog_bytes = sum(
            getattr(child._record_source.decoder, "catalog_reserved_bytes", 0)
            for child in self.datasets
        )
        context = self.runtime_context
        runtime = AdaptivePipelineRuntime(
            self.adaptive_pipeline,
            manifests,
            local_batch_size=context.local_batch_size,
            global_batch_size=context.global_batch_size,
            shuffle=self.shuffle,
            shuffle_window_size=self.shuffle_window_size,
            memory_budget=self._shared_host_memory_budget,
            shared_decoded_cache=self._node_decoded_cache_catalog is not None,
            packed_mixed_shapes=len(self._record_source.shape_codes),
            packed_source_chunk_size=(
                getattr(self._record_source, "packed_source_chunk_size", None)
                if packed else None
            ),
            minimum_cache_bytes=sum(cache_floors),
            serial_execution=not packed,
            pre_reserved_semantic_bytes=indexed_catalog_bytes,
            fixed_execution_overhead_bytes=(
                2 * 1024**3
                if packed and self._record_source._quota_child_prefetcher is not None
                else 0
            ),
            generic_source_count=len(self.datasets),
        )
        self._adaptive_pipeline_runtime = runtime
        self.pipeline_stats = PipelineStats()
        settings = runtime.settings
        dense_children = [
            (child, group, floor)
            for child, group, floor in zip(self.datasets, child_manifests, cache_floors)
            if hasattr(child, "_record_decoder")
        ]
        for child, group, floor in dense_children:
            child._record_decoder.pipeline_stats = self.pipeline_stats
            child._record_decoder.configure_cache(
                entries=max(1, len(group)),
                byte_capacity=(
                    settings.decoded_cache_bytes if packed else floor
                ),
                host_memory_budget=runtime.memory_budget,
                file_size_catalog=group,
            )
        if packed:
            self._record_source.decoder = _CompositeDecoderCache(
                (child._record_decoder for child in self.datasets),
                child_manifests,
            )
            self._record_source.decoder.configure_cache(
                entries=1, byte_capacity=settings.decoded_cache_bytes
            )
        runtime.reserve_semantic_floor()

        def finalize(data):
            if not settings.pin_memory:
                return data
            return {
                key: (
                    torch.from_numpy(np.ascontiguousarray(value)).pin_memory()
                    if isinstance(value, np.ndarray) and value.dtype.kind in "biuf"
                    else value
                )
                for key, value in data.items()
            }

        from .execution import build_execution

        self._planned_decoder = build_execution(
            self._partitioned_stream,
            self._record_source,
            decision=self.execution_decision,
            runtime=runtime,
            pipeline_stats=self.pipeline_stats,
            finalize_batch=finalize,
            prefetch_workers=(settings.decode_workers if packed else 0),
            prefetch_batches=settings.ready_queue_batches,
            prefetch_chunk_batches=settings.decode_chunk_batches,
            finalize_in_prefetch=packed,
            host_memory_budget=runtime.memory_budget,
            output_batch_bytes=runtime.output_batch_bytes,
            planner_token_bytes=runtime.planner_token_bytes,
            epoch_lookahead_bytes=runtime.epoch_lookahead_bytes,
            output_is_pinned=settings.pin_memory,
        )

    def close(self):
        try:
            error = None
            adapter = getattr(self, "_planned_decoder", None)
            if adapter is not None:
                try:
                    adapter.close()
                except Exception as exc:
                    error = exc
            planner = getattr(self, "_partitioned_stream", None)
            if planner is not None:
                try:
                    planner.close()
                except Exception as exc:
                    error = error or exc
            worker = getattr(
                getattr(self, "_record_source", None), "_quota_child_prefetcher", None
            )
            if worker is not None:
                try:
                    worker.close()
                except Exception as exc:
                    error = error or exc
            for child in self.datasets:
                close = getattr(child, "close", None)
                if close is not None:
                    try:
                        close()
                    except Exception as exc:
                        error = error or exc
            if error is not None:
                raise error
        finally:
            if self._adaptive_pipeline_runtime is not None:
                worker = getattr(
                    getattr(self, "_record_source", None), "_quota_child_prefetcher", None
                )
                if worker is None or not worker.worker_alive:
                    self._adaptive_pipeline_runtime.close()
                    self._adaptive_pipeline_runtime = None
