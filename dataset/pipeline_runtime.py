"""Runtime bridge for the adaptive processed-NPZ data pipeline.

System probing and distributed topology resolution happen before this module is
constructed. This layer turns the resolved rank-local limits plus dataset
metadata into one memory budget and one observation-driven controller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import struct
from typing import Mapping, Sequence

from .host_memory import HostMemoryBudget, HostMemoryCategory
from .node_decoded_cache import NodeDecodedCache
from .packed import PACKED_RESERVOIR_UNDO_BYTES_PER_REPLACEMENT
from .pipeline_config import AdaptiveDataPipelineConfig
from .pipeline_controller import (
    AdaptivePipelineController,
    DecodeLayout,
    PipelineCapacityError,
    PipelineControllerConstraints,
    PipelineObservation,
    PipelineSettings,
)
from .pipeline_topology import DistributedPipelineResources
from .planner import SOURCE_CHUNK_SIZE


ADAPTIVE_PIPELINE_RUNTIME_SCHEMA = "adaptive-pipeline-runtime-v5"

_PACKED_RECORD_ID_BYTES = 8
_BATCH_MASK_BYTES = 1
_READY_BATCH_METADATA_BYTES_PER_ROW = (
    2 * struct.calcsize("P") + _BATCH_MASK_BYTES
)
_PINNED_FINALIZATION_COPY_COUNT = 2


@dataclass(frozen=True, slots=True)
class AdaptivePipelineRuntimeSpec:
    """Portable user policy paired with collectively resolved rank limits."""

    config: AdaptiveDataPipelineConfig
    resources: DistributedPipelineResources
    pin_memory_supported: bool
    consumer_retained_batches: int = 1
    h2d_lookahead_batches: int = 0
    node_decoded_cache: NodeDecodedCache | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.config, AdaptiveDataPipelineConfig):
            raise TypeError("config must be AdaptiveDataPipelineConfig")
        if not isinstance(self.resources, DistributedPipelineResources):
            raise TypeError("resources must be DistributedPipelineResources")
        if type(self.pin_memory_supported) is not bool:
            raise TypeError("pin_memory_supported must be a boolean")
        if (
            type(self.consumer_retained_batches) is not int
            or self.consumer_retained_batches <= 0
        ):
            raise ValueError(
                "consumer_retained_batches must be a positive integer"
            )
        if (
            type(self.h2d_lookahead_batches) is not int
            or self.h2d_lookahead_batches < 0
        ):
            raise ValueError(
                "h2d_lookahead_batches must be a non-negative integer"
            )
        if self.node_decoded_cache is not None and not isinstance(
            self.node_decoded_cache,
            NodeDecodedCache,
        ):
            raise TypeError("node_decoded_cache must be NodeDecodedCache or null")


def _positive_int(name: str, value) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _manifest_size(catalog: Mapping, name: str) -> int:
    value = catalog.get(name)
    if type(value) is not int or value < 0:
        raise ValueError(f"processed-NPZ manifest {name} must be non-negative")
    return value


def _config_state(config: AdaptiveDataPipelineConfig) -> dict:
    advanced = config.advanced
    return {
        "host_memory_budget": config.host_memory_budget,
        "data_wait_budget": config.data_wait_budget,
        "data_cpu_budget": config.data_cpu_budget,
        "adaptation": config.adaptation,
        "advanced": (
            None
            if advanced is None
            else {
                "decoded_cache_bytes": advanced.decoded_cache_bytes,
                "decode_workers": advanced.decode_workers,
                "decode_chunk_batches": advanced.decode_chunk_batches,
                "ready_queue_batches": advanced.ready_queue_batches,
                "pin_memory": advanced.pin_memory,
            }
        ),
    }


class AdaptivePipelineRuntime:
    """Own rank-local accounting, tuning state, and portable size estimates."""

    def __init__(
        self,
        spec: AdaptivePipelineRuntimeSpec,
        manifests: Sequence[Mapping],
        *,
        local_batch_size: int,
        global_batch_size: int,
        shuffle: bool,
        shuffle_window_size: int,
        memory_budget: HostMemoryBudget | None = None,
        shared_decoded_cache: bool = False,
    ) -> None:
        if not isinstance(spec, AdaptivePipelineRuntimeSpec):
            raise TypeError("spec must be AdaptivePipelineRuntimeSpec")
        if not manifests:
            raise ValueError("adaptive pipeline requires processed-NPZ manifests")
        local_batch_size = _positive_int("local_batch_size", local_batch_size)
        global_batch_size = _positive_int("global_batch_size", global_batch_size)
        shuffle_window_size = _positive_int(
            "shuffle_window_size", shuffle_window_size
        )
        if type(shuffle) is not bool:
            raise TypeError("shuffle must be a boolean")
        if type(shared_decoded_cache) is not bool:
            raise TypeError("shared_decoded_cache must be a boolean")
        if memory_budget is not None:
            if not isinstance(memory_budget, HostMemoryBudget):
                raise TypeError("memory_budget must be HostMemoryBudget or null")
            if memory_budget.total_bytes != spec.resources.per_rank_host_budget_bytes:
                raise ValueError(
                    "memory_budget total must match the resolved per-rank host budget"
                )

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
        output_batch_bytes = (
            (
                max(positive_output_sizes) * output_copy_count
                + _READY_BATCH_METADATA_BYTES_PER_ROW
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
        packed_uniform = bool(board_sizes) and len(set(board_sizes)) == 1
        packed_journal_bytes = (
            (global_batch_size + SOURCE_CHUNK_SIZE)
            * PACKED_RESERVOIR_UNDO_BYTES_PER_REPLACEMENT
        )
        planner_token_bytes = (
            planned_batch_bytes + packed_journal_bytes
            if packed_uniform
            else reservoir_bytes + planned_batch_bytes
        )
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
        initial_cache_bytes = max(positive_decoded_sizes)
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
            largest_decoded_file_bytes=max(positive_decoded_sizes),
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
        self.spec = spec
        self.constraints = constraints
        self.memory_budget = memory_budget or HostMemoryBudget(
            resources.per_rank_host_budget_bytes
        )
        self.controller = AdaptivePipelineController(spec.config, constraints)
        self._semantic_reservation = None
        self._events: list[dict] = []
        self._total_logical_rows = total_logical_rows
        self._runtime_key = self._build_runtime_key()
        self._legacy_runtime_key = self._build_legacy_runtime_key()

    @property
    def settings(self) -> PipelineSettings:
        return self.controller.settings

    @property
    def maximum_prefetch_workers(self) -> int:
        return self.constraints.per_rank_cpu_limit

    @property
    def output_batch_bytes(self) -> int:
        return self.constraints.output_batch_bytes

    @property
    def planner_token_bytes(self) -> int:
        return self.constraints.planner_token_bytes_per_queued_batch

    def reserve_semantic_floor(self) -> None:
        if self._semantic_reservation is not None:
            return
        self._semantic_reservation = self.memory_budget.reserve(
            HostMemoryCategory.SEMANTIC_FIXED,
            self.constraints.fixed_semantic_floor_bytes,
            label="planner fixed state",
        )

    def close(self) -> None:
        reservation = self._semantic_reservation
        self._semantic_reservation = None
        if reservation is not None:
            reservation.release()

    def update(
        self,
        metrics: Mapping[str, float],
        iteration: int,
        *,
        epoch_changed: bool = False,
    ):
        """Consume one distributed metric window and return state on change."""

        if not isinstance(metrics, Mapping):
            raise TypeError("pipeline metrics must be a mapping")
        if type(iteration) is not int or iteration < 0:
            raise ValueError("pipeline iteration must be a non-negative integer")
        observation = self._observation(metrics)
        decision = self.controller.observe(
            observation,
            epoch_changed=epoch_changed,
        )
        if decision is None:
            return None
        event = decision.as_dict()
        event["iteration"] = iteration
        self._events[:] = [event]
        return self.state_dict()

    def state_dict(self) -> dict:
        return {
            "schema": ADAPTIVE_PIPELINE_RUNTIME_SCHEMA,
            "runtime_key": self._runtime_key,
            "settings": self.settings.as_dict(),
            "transition": self.controller.transition_state_dict(),
            "frozen": self.controller.frozen,
            "decisions": list(self._events),
        }

    def load_state_dict(self, state) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("adaptive pipeline runtime state must be a mapping")
        if state.get("schema") != ADAPTIVE_PIPELINE_RUNTIME_SCHEMA:
            raise ValueError("adaptive pipeline runtime state schema changed")
        if state.get("runtime_key") not in {
            self._runtime_key,
            self._legacy_runtime_key,
        }:
            raise ValueError("adaptive pipeline runtime state is incompatible")
        settings_state = state.get("settings")
        if not isinstance(settings_state, Mapping):
            raise ValueError("adaptive pipeline runtime settings are malformed")
        try:
            settings = PipelineSettings(**dict(settings_state))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "adaptive pipeline runtime settings are malformed"
            ) from exc
        frozen = state.get("frozen")
        if type(frozen) is not bool:
            raise ValueError("adaptive pipeline runtime frozen flag is malformed")
        decisions = state.get("decisions", [])
        if not isinstance(decisions, list) or not all(
            isinstance(item, dict) for item in decisions
        ):
            raise ValueError("adaptive pipeline runtime decisions are malformed")
        self.controller.restore_performance_settings(settings, frozen=frozen)
        try:
            self.controller.restore_transition_state(state.get("transition"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "adaptive pipeline runtime transition is malformed"
            ) from exc
        self._events = list(decisions[-1:])

    def restore_state_dict(self, state) -> bool:
        """Restore a compatible checkpoint from an accepted layout boundary."""

        if (
            not isinstance(state, Mapping)
            or state.get("schema") != ADAPTIVE_PIPELINE_RUNTIME_SCHEMA
            or state.get("runtime_key")
            not in {self._runtime_key, self._legacy_runtime_key}
        ):
            return False
        restored_state = state
        transition = state.get("transition")
        if isinstance(transition, Mapping) and transition.get("phase") == "trial":
            accepted = DecodeLayout.parse(transition.get("accepted_layout"))
            settings_state = state.get("settings")
            if not isinstance(settings_state, Mapping):
                raise ValueError("adaptive pipeline runtime settings are malformed")
            try:
                settings = PipelineSettings(**dict(settings_state))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "adaptive pipeline runtime settings are malformed"
                ) from exc
            settings = accepted.apply(settings)
            restored_state = dict(state)
            restored_state["settings"] = settings.as_dict()
            restored_state["transition"] = {
                "phase": "calibrating",
                "accepted_layout": accepted.as_dict(),
            }
        try:
            self.load_state_dict(restored_state)
        except PipelineCapacityError:
            return False
        return True

    def memory_snapshot(self) -> dict:
        return self.memory_budget.snapshot().as_dict()

    def _build_runtime_key(self) -> str:
        constraints = self.constraints.as_dict()
        if self.spec.config.host_memory_budget == "auto":
            constraints.pop("per_rank_host_budget_bytes")
        if self.spec.config.data_cpu_budget == "auto":
            constraints.pop("per_rank_cpu_limit")
            constraints.pop("initial_decoded_cache_bytes")
        payload = {
            "config": _config_state(self.spec.config),
            "constraints": constraints,
            "total_logical_rows": self._total_logical_rows,
        }
        return self._hash_runtime_key(payload)

    def _build_legacy_runtime_key(self) -> str:
        payload = {
            "config": _config_state(self.spec.config),
            "constraints": self.constraints.as_dict(),
        }
        return self._hash_runtime_key(payload)

    @staticmethod
    def _hash_runtime_key(payload) -> str:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _observation(self, metrics: Mapping[str, float]) -> PipelineObservation:
        def finite(name: str) -> float:
            value = metrics.get(name)
            if isinstance(value, bool) or type(value) not in {int, float}:
                raise ValueError(f"pipeline metric {name!r} is missing or invalid")
            normalized = float(value)
            if not math.isfinite(normalized):
                raise ValueError(f"pipeline metric {name!r} must be finite")
            return normalized

        consumer_rows_s = finite("throughput/consumer_rows_s")
        if consumer_rows_s <= 0:
            raise ValueError("pipeline consumer throughput must be positive")
        producer_rows_s = finite("distributed/producer_capacity_min_rows_s")
        data_wait = min(1.0, max(0.0, finite("data_wait_fraction")))
        compute_demand_rows_s = consumer_rows_s / max(0.05, 1.0 - data_wait)
        head_wait = min(
            1.0,
            max(
                0.0,
                finite("distributed/prefetch_wait_fraction_max"),
            ),
        )
        reloads = finite("distributed/cache_reloads_max")
        source_tail_wait = min(
            1.0,
            max(
                0.0,
                finite("distributed/source_tail_wait_fraction_max"),
            ),
        )
        return PipelineObservation(
            producer_rows_per_second=max(0.0, producer_rows_s),
            consumer_demand_rows_per_second=compute_demand_rows_s,
            data_wait_fraction=data_wait,
            cache_reloads=math.ceil(max(0.0, reloads)),
            head_of_line_wait_fraction=head_wait,
            tail_wait_fraction=source_tail_wait,
            consumer_rows_per_second=consumer_rows_s,
        )


__all__ = [
    "ADAPTIVE_PIPELINE_RUNTIME_SCHEMA",
    "AdaptivePipelineRuntime",
    "AdaptivePipelineRuntimeSpec",
]
