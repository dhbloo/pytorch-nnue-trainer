"""Reuse the adaptive NPZ runtime for a compatible composite source."""

import os

import numpy as np
import torch

from .katago import BatchedProcessedKatagoNumpyDataset
from .pipeline_runtime import AdaptivePipelineRuntime
from .telemetry import ObservedSourceBatchDataset, PipelineStats
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

    def configure_cache(self, *, entries, byte_capacity):
        if byte_capacity < sum(self.floors):
            raise ValueError("mixed decoded cache cannot fit one file from each child")
        remaining = byte_capacity - sum(self.floors)
        for decoder, floor in zip(self.decoders, self.floors):
            state = decoder.cache_state()
            decoder.configure_cache(
                entries=state["capacity_entries"],
                byte_capacity=floor + remaining // len(self.decoders),
            )


class AdaptiveMultiMixin:
    # The cache is prepared over the union once. Per-child ordinal catalogs
    # must never be independently installed into the shared directory.
    prepare_node_decoded_cache = BatchedProcessedKatagoNumpyDataset.prepare_node_decoded_cache
    activate_node_decoded_cache = BatchedProcessedKatagoNumpyDataset.activate_node_decoded_cache
    node_decoded_cache_ready = BatchedProcessedKatagoNumpyDataset.node_decoded_cache_ready
    set_node_decoded_cache_enabled = BatchedProcessedKatagoNumpyDataset.set_node_decoded_cache_enabled
    cleanup_node_decoded_cache = BatchedProcessedKatagoNumpyDataset.cleanup_node_decoded_cache
    pipeline_metrics_snapshot = BatchedProcessedKatagoNumpyDataset.pipeline_metrics_snapshot
    pipeline_tuning_update = BatchedProcessedKatagoNumpyDataset.pipeline_tuning_update
    pipeline_tuning_state_dict = BatchedProcessedKatagoNumpyDataset.pipeline_tuning_state_dict
    load_pipeline_tuning_state_dict = BatchedProcessedKatagoNumpyDataset.load_pipeline_tuning_state_dict
    restore_pipeline_tuning_state_dict = BatchedProcessedKatagoNumpyDataset.restore_pipeline_tuning_state_dict

    def _init_adaptive_multi(self, spec):
        self.adaptive_pipeline = spec
        self._adaptive_pipeline_runtime = None
        self._node_decoded_cache_catalog = None
        if spec is None:
            return
        if not all(type(child) is BatchedProcessedKatagoNumpyDataset for child in self.datasets):
            raise ValueError("adaptive iterative_multi requires batched processed NPZ children")
        unsupported = (
            "filter_stm", "filter_condition", "board_input_channels",
            "stm_input_channel", "value_target_channels",
        )
        if any(
            any(child.extra_kwargs.get(key) is not None for key in unsupported)
            for child in self.datasets
        ):
            raise ValueError("adaptive iterative_multi requires dense unfiltered NPZ children")
        if any(len(child.boardsizes) != 1 for child in self.datasets):
            raise ValueError("adaptive iterative_multi requires one board size per child")
        if self.batch_pipelines:
            raise ValueError("adaptive iterative_multi does not support batch_pipelines")
        self.file_list = reject_duplicate_physical_files(
            [path for child in self.datasets for path in child.file_list]
        )

    def _node_decoded_cache_is_supported(self):
        return (
            self.adaptive_pipeline is not None
            and self.adaptive_pipeline.node_decoded_cache is not None
        )

    def _configure_adaptive_children(self):
        if self.adaptive_pipeline is None:
            return
        catalog = self._node_decoded_cache_catalog
        for child in self.datasets:
            child.observability = True
            if catalog is not None:
                child._node_decoded_cache_catalog = {
                    os.path.abspath(path): catalog[os.path.abspath(path)]
                    for path in child.file_list
                }

    def _install_adaptive_multi(self):
        if self.adaptive_pipeline is None:
            return
        from .packed_composite import PackedCompositeRecordSource

        if not isinstance(self._record_source, PackedCompositeRecordSource):
            raise ValueError("adaptive iterative_multi requires packed dense NPZ sources")
        manifests = [
            manifest for child in self.datasets for manifest in child._record_manifests
        ]
        context = self.runtime_context
        runtime = AdaptivePipelineRuntime(
            self.adaptive_pipeline,
            manifests,
            local_batch_size=context.local_batch_size,
            global_batch_size=context.global_batch_size,
            shuffle=self.shuffle,
            shuffle_window_size=self.shuffle_window_size,
            shared_decoded_cache=self._node_decoded_cache_catalog is not None,
            packed_mixed_shapes=len(self._record_source.shape_codes),
            minimum_cache_bytes=sum(
                max(m["decoded_file_bytes"] for m in child._record_manifests)
                for child in self.datasets
            ),
        )
        self._adaptive_pipeline_runtime = runtime
        self.pipeline_stats = PipelineStats()
        settings = runtime.settings
        for child in self.datasets:
            child._record_decoder.pipeline_stats = self.pipeline_stats
            child._record_decoder.configure_cache(
                entries=max(1, len(child._record_manifests)),
                byte_capacity=settings.decoded_cache_bytes,
                host_memory_budget=runtime.memory_budget,
                file_size_catalog=child._record_manifests,
            )
        self._record_source.decoder = _CompositeDecoderCache(
            (child._record_decoder for child in self.datasets),
            [child._record_manifests for child in self.datasets],
        )
        self._record_source.decoder.configure_cache(
            entries=1, byte_capacity=settings.decoded_cache_bytes
        )
        runtime.reserve_semantic_floor()

        def finalize(data):
            if not settings.pin_memory:
                return data
            return {
                key: torch.from_numpy(np.ascontiguousarray(value)).pin_memory()
                for key, value in data.items()
            }

        self._planned_decoder = ObservedSourceBatchDataset(
            self._partitioned_stream,
            self._record_source,
            finalize_batch=finalize,
            prefetch_workers=settings.decode_workers,
            prefetch_batches=settings.ready_queue_batches,
            prefetch_chunk_batches=settings.decode_chunk_batches,
            finalize_in_prefetch=True,
            host_memory_budget=runtime.memory_budget,
            output_batch_bytes=runtime.output_batch_bytes,
            planner_token_bytes=runtime.planner_token_bytes,
            output_is_pinned=settings.pin_memory,
            pipeline_stats=self.pipeline_stats,
            adaptive_runtime=runtime,
        )

    def close(self):
        try:
            error = None
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
                self._adaptive_pipeline_runtime.close()
                self._adaptive_pipeline_runtime = None
