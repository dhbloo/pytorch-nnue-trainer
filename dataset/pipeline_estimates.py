"""Portable memory footprint estimates for dense NPZ execution."""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import Mapping, Sequence, TYPE_CHECKING

from .packed import PACKED_RESERVOIR_UNDO_BYTES_PER_REPLACEMENT
from .pipeline_controller import PipelineControllerConstraints
from .planner import SOURCE_CHUNK_SIZE, PACKED_MIXED_SOURCE_CHUNK_SIZE

if TYPE_CHECKING:
    from .pipeline_runtime import AdaptivePipelineRuntimeSpec

_PACKED_RECORD_ID_BYTES = 8
_BATCH_MASK_BYTES = 1
_READY_BATCH_METADATA_BYTES_PER_ROW = (
    2 * struct.calcsize("P") + _BATCH_MASK_BYTES
)
_PINNED_FINALIZATION_COPY_COUNT = 2
_COMPOSITE_ROUTING_BYTES_PER_ROW = 64


@dataclass(frozen=True, slots=True)
class ExecutionFootprint:
    constraints: PipelineControllerConstraints
    packed_uniform: bool
    total_logical_rows: int

    @property
    def output_batch_bytes(self) -> int:
        return self.constraints.output_batch_bytes

    @property
    def planner_token_bytes(self) -> int:
        return self.constraints.planner_token_bytes_per_queued_batch

    @property
    def semantic_floor_bytes(self) -> int:
        return self.constraints.fixed_semantic_floor_bytes

    @property
    def minimum_cache_bytes(self) -> int:
        return self.constraints.largest_decoded_file_bytes

    @property
    def largest_inflight_bytes(self) -> int:
        return self.constraints.inflight_file_bytes

    @property
    def total_cache_bytes(self) -> int:
        return self.constraints.total_decoded_bytes

    @property
    def initial_cache_bytes(self) -> int:
        return self.constraints.initial_decoded_cache_bytes


def _manifest_size(catalog: Mapping, name: str) -> int:
    value = catalog.get(name)
    if type(value) is not int or value < 0:
        raise ValueError(f"processed-NPZ manifest {name} must be non-negative")
    return value


def estimate_npz_execution(
    spec: AdaptivePipelineRuntimeSpec,
    manifests: Sequence[Mapping],
    *,
    local_batch_size: int,
    global_batch_size: int,
    shuffle: bool,
    shuffle_window_size: int,
    shared_decoded_cache: bool,
    packed_mixed_shapes: int,
    minimum_cache_bytes: int,
) -> ExecutionFootprint:
    """Estimate the existing controller inputs without changing its constants."""
    decoded_sizes = tuple(
        _manifest_size(catalog, "decoded_file_bytes")
        for catalog in manifests
    )
    inflight_sizes = tuple(
        _manifest_size(catalog, "inflight_file_bytes")
        if "inflight_file_bytes" in catalog
        else decoded_size
        for catalog, decoded_size in zip(manifests, decoded_sizes)
    )
    output_row_sizes = tuple(
        _manifest_size(catalog, "output_row_bytes")
        for catalog in manifests
    )
    positive_decoded_sizes = tuple(size for size in decoded_sizes if size > 0)
    positive_inflight_sizes = tuple(
        inflight_size
        for decoded_size, inflight_size in zip(
            decoded_sizes,
            inflight_sizes,
        )
        if decoded_size > 0
    )
    positive_output_sizes = tuple(size for size in output_row_sizes if size > 0)
    if not positive_decoded_sizes or not positive_output_sizes:
        raise ValueError(
            "adaptive pipeline requires at least one non-empty accepted NPZ file"
        )

    planned_pin_memory = (
        spec.config.advanced.pin_memory
        if spec.config.advanced is not None
        else spec.pin_memory_supported
    )
    output_copy_count = (
        _PINNED_FINALIZATION_COPY_COUNT if planned_pin_memory else 1
    )
    if packed_mixed_shapes:
        # Child decode chunks coexist with their scattered composite output.
        # Routing indices and lazy keys also outlive child materialization.
        output_copy_count += 1
    composite_metadata_bytes = (
        _COMPOSITE_ROUTING_BYTES_PER_ROW if packed_mixed_shapes else 0
    )
    output_batch_bytes = (
        (
            max(positive_output_sizes) * output_copy_count
            + _READY_BATCH_METADATA_BYTES_PER_ROW
            + composite_metadata_bytes
        )
        * local_batch_size
    )

    reservoir_capacity = shuffle_window_size if shuffle else 1
    reservoir_bytes = reservoir_capacity * _PACKED_RECORD_ID_BYTES
    planned_batch_bytes = global_batch_size * (
        _PACKED_RECORD_ID_BYTES + _BATCH_MASK_BYTES
    )
    board_sizes = []
    for catalog in manifests:
        board_size = catalog.get("board_size")
        if (
            not isinstance(board_size, (list, tuple))
            or len(board_size) != 2
        ):
            board_sizes = []
            break
        board_sizes.append(tuple(int(value) for value in board_size))
    packed_uniform = bool(packed_mixed_shapes) or (
        bool(board_sizes) and len(set(board_sizes)) == 1
    )
    source_chunk_size = (
        PACKED_MIXED_SOURCE_CHUNK_SIZE
        if packed_mixed_shapes > 1 else SOURCE_CHUNK_SIZE
    )
    packed_journal_bytes = (
        (global_batch_size * max(1, packed_mixed_shapes) + source_chunk_size)
        * PACKED_RESERVOIR_UNDO_BYTES_PER_REPLACEMENT
    )
    planner_token_bytes = (
        planned_batch_bytes + packed_journal_bytes
        if packed_uniform
        else reservoir_bytes + planned_batch_bytes
    )
    # Mixed shape queues retain at most one global batch per shape.
    # Snapshot arrays are immutable and shared until consumed.
    shape_queue_bytes = packed_mixed_shapes * global_batch_size * _PACKED_RECORD_ID_BYTES
    planner_token_bytes += shape_queue_bytes
    queued_batch_bytes = output_batch_bytes + planner_token_bytes
    # Packed transactions retain deltas per queued batch. At the terminal
    # drain, rollback temporarily retains the live reservoir allocation,
    # one pre-drain image, one zero-copy ready array, and the absorbed
    # no-batch probe's journal. Generic planners keep the previous
    # live-plus-committed snapshot accounting.
    fixed_semantic_floor_bytes = (
        3 * reservoir_bytes + planned_batch_bytes + packed_journal_bytes
        if packed_uniform
        else 2 * reservoir_bytes + planned_batch_bytes
    )

    fixed_semantic_floor_bytes += 2 * shape_queue_bytes

    resources = spec.resources
    total_decoded_bytes = sum(positive_decoded_sizes)
    total_logical_rows = sum(
        int(catalog.get("logical_row_count", 0))
        for catalog in manifests
        if int(catalog.get("decoded_file_bytes", 0)) > 0
    )
    initial_queue_batches = max(1, resources.per_rank_cpu_limit * 8)
    lookahead_rows = global_batch_size * initial_queue_batches
    if shuffle:
        lookahead_rows += shuffle_window_size
    initial_cache_bytes = max(max(positive_decoded_sizes), minimum_cache_bytes)
    if total_logical_rows > 0:
        working_fraction = min(
            1.0,
            1.25 * lookahead_rows / total_logical_rows,
        )
        initial_cache_bytes = min(
            total_decoded_bytes,
            max(
                initial_cache_bytes,
                math.ceil(total_decoded_bytes * working_fraction),
            ),
        )
    if shared_decoded_cache:
        initial_cache_bytes = total_decoded_bytes
    constraints = PipelineControllerConstraints(
        local_rank_count=resources.local_rank_count,
        per_rank_host_budget_bytes=resources.per_rank_host_budget_bytes,
        per_rank_cpu_limit=resources.per_rank_cpu_limit,
        largest_decoded_file_bytes=max(max(positive_decoded_sizes), minimum_cache_bytes),
        output_batch_bytes=output_batch_bytes,
        fixed_semantic_floor_bytes=fixed_semantic_floor_bytes,
        planner_token_bytes_per_queued_batch=planner_token_bytes,
        pin_memory_supported=spec.pin_memory_supported,
        largest_inflight_file_bytes=max(positive_inflight_sizes),
        consumer_retained_bytes=(
            spec.consumer_retained_batches * queued_batch_bytes
        ),
        h2d_retained_bytes=(
            0
            if spec.h2d_lookahead_batches == 0
            else (
                spec.h2d_lookahead_batches * queued_batch_bytes
                + (spec.h2d_lookahead_batches + 1)
                * output_batch_bytes
            )
        ),
        total_decoded_bytes=total_decoded_bytes,
        initial_decoded_cache_bytes=initial_cache_bytes,
        shared_decoded_cache=shared_decoded_cache,
    )
    return ExecutionFootprint(
        constraints=constraints,
        packed_uniform=packed_uniform,
        total_logical_rows=total_logical_rows,
    )
