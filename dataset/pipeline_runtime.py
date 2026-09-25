"""Runtime bridge for the adaptive processed-NPZ data pipeline.

System probing and distributed topology resolution happen before this module is
constructed. This layer turns the resolved rank-local limits plus dataset
metadata into one memory budget and one observation-driven controller.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from typing import Mapping, Sequence

from .host_memory import HostMemoryBudget, HostMemoryCategory
from .pipeline_estimates import estimate_npz_execution
from .node_decoded_cache import NodeDecodedCache
from .pipeline_config import AdaptiveDataPipelineConfig
from .pipeline_controller import (
    AdaptivePipelineController,
    DecodeLayout,
    PipelineCapacityError,
    PipelineObservation,
    PipelineSettings,
)
from .pipeline_topology import DistributedPipelineResources


ADAPTIVE_PIPELINE_RUNTIME_SCHEMA = "adaptive-pipeline-runtime-v5"

@dataclass(frozen=True, slots=True)
class AdaptivePipelineRuntimeSpec:
    """Portable user policy paired with collectively resolved rank limits."""

    config: AdaptiveDataPipelineConfig
    resources: DistributedPipelineResources
    pin_memory_supported: bool
    origin: str = "explicit"
    compatibility_layout: bool = False
    preferred_backend: str | None = None
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
        if self.origin not in {"explicit", "implicit"}:
            raise ValueError("origin must be explicit or implicit")
        if type(self.compatibility_layout) is not bool:
            raise TypeError("compatibility_layout must be a boolean")
        if self.preferred_backend not in {None, "packed", "generic"}:
            raise ValueError("preferred_backend must be packed, generic, or null")
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
        packed_mixed_shapes: int = 0,
        minimum_cache_bytes: int = 0,
        serial_execution: bool = False,
        pre_reserved_semantic_bytes: int = 0,
        generic_source_count: int = 1,
        serial_shape_count: int = 1,
    ) -> None:
        if type(packed_mixed_shapes) is not int or packed_mixed_shapes < 0:
            raise ValueError("packed_mixed_shapes must be a non-negative integer")
        if type(minimum_cache_bytes) is not int or minimum_cache_bytes < 0:
            raise ValueError("minimum_cache_bytes must be a non-negative integer")
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
        if type(serial_execution) is not bool:
            raise TypeError("serial_execution must be a boolean")
        if type(pre_reserved_semantic_bytes) is not int or pre_reserved_semantic_bytes < 0:
            raise ValueError("pre_reserved_semantic_bytes must be non-negative")
        generic_source_count = _positive_int("generic_source_count", generic_source_count)
        serial_shape_count = _positive_int("serial_shape_count", serial_shape_count)
        if memory_budget is not None:
            if not isinstance(memory_budget, HostMemoryBudget):
                raise TypeError("memory_budget must be HostMemoryBudget or null")
            if memory_budget.total_bytes != spec.resources.per_rank_host_budget_bytes:
                raise ValueError(
                    "memory_budget total must match the resolved per-rank host budget"
                )

        footprint = estimate_npz_execution(
            spec,
            manifests,
            local_batch_size=local_batch_size,
            global_batch_size=global_batch_size,
            shuffle=shuffle,
            shuffle_window_size=shuffle_window_size,
            shared_decoded_cache=shared_decoded_cache,
            packed_mixed_shapes=packed_mixed_shapes,
            minimum_cache_bytes=minimum_cache_bytes,
        )
        constraints = footprint.constraints
        if serial_execution:
            # Generic NPZ planners retain Python envelopes and pointer-based
            # cursor snapshots. Packed uint64 accounting alone is too small.
            reservoir_rows = shuffle_window_size if shuffle else 1
            shape_rows = max(serial_shape_count, packed_mixed_shapes) * global_batch_size
            pending_rows = generic_source_count * 1024
            record_bytes = 2048
            snapshot_pointer_bytes = 16 * (
                reservoir_rows + shape_rows + pending_rows
            )
            generic_fixed_bytes = record_bytes * (
                reservoir_rows + shape_rows + 2 * global_batch_size
                + pending_rows + len(manifests)
            )
            generic_token_bytes = (
                record_bytes * (global_batch_size + pending_rows)
                + snapshot_pointer_bytes
            )
            constraints = replace(
                constraints,
                per_rank_cpu_limit=1,
                total_decoded_bytes=constraints.largest_decoded_file_bytes,
                initial_decoded_cache_bytes=constraints.largest_decoded_file_bytes,
                fixed_semantic_floor_bytes=(
                    constraints.fixed_semantic_floor_bytes + generic_fixed_bytes
                ),
                planner_token_bytes_per_queued_batch=(
                    constraints.planner_token_bytes_per_queued_batch
                    + generic_token_bytes
                ),
                consumer_retained_bytes=(
                    constraints.consumer_retained_bytes
                    + spec.consumer_retained_batches * generic_token_bytes
                ),
                h2d_retained_bytes=(
                    constraints.h2d_retained_bytes
                    + spec.h2d_lookahead_batches * generic_token_bytes
                ),
            )
        self._semantic_reservation_bytes = constraints.fixed_semantic_floor_bytes
        if pre_reserved_semantic_bytes:
            constraints = replace(
                constraints,
                fixed_semantic_floor_bytes=(
                    constraints.fixed_semantic_floor_bytes
                    + pre_reserved_semantic_bytes
                ),
            )
        packed_uniform = footprint.packed_uniform and not serial_execution
        fixed_semantic_floor_bytes = footprint.semantic_floor_bytes
        total_logical_rows = footprint.total_logical_rows
        resources = spec.resources
        self.spec = spec
        self.serial_execution = serial_execution
        self.constraints = constraints
        self.memory_budget = memory_budget or HostMemoryBudget(
            resources.per_rank_host_budget_bytes
        )
        self.controller = AdaptivePipelineController(spec.config, constraints)
        if serial_execution:
            self.controller.restore_performance_settings(
                self.controller.settings, frozen=True
            )
        # An overlapping epoch owns another reservoir and cursor. Include its
        # full floor in every future layout/capacity decision, not just the
        # currently charged bytes when decode workers happen to be idle.
        self.epoch_lookahead_bytes = 0
        if (
            packed_uniform
            and self.controller.capacity_bytes() + fixed_semantic_floor_bytes
            <= resources.per_rank_host_budget_bytes
        ):
            self.epoch_lookahead_bytes = fixed_semantic_floor_bytes
            self.constraints = replace(
                constraints,
                fixed_semantic_floor_bytes=2 * fixed_semantic_floor_bytes,
            )
            self.controller.constraints = self.constraints
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
        return 0 if self.serial_execution else self.constraints.per_rank_cpu_limit

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
            self._semantic_reservation_bytes,
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
        if self.serial_execution:
            payload["serial_execution"] = True
        return self._hash_runtime_key(payload)

    def _build_legacy_runtime_key(self) -> str:
        payload = {
            "config": _config_state(self.spec.config),
            "constraints": self.constraints.as_dict(),
        }
        if self.serial_execution:
            payload["serial_execution"] = True
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
