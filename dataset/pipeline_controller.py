"""Small adaptive policy for processed-NPZ pipeline resources.

The controller owns no operating-system, CUDA, dataset, or trainer behavior.
It adjusts only a grow-only private decoded cache and one atomic decode layout
containing worker, chunk, and ready-queue sizes.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import math
from typing import Literal

from .pipeline_config import AdaptiveDataPipelineConfig


_DECISION_WINDOWS = 2
_TRIAL_SETTLE_WINDOWS = 1
_TRIAL_VERIFY_WINDOWS = 2
_MIN_PRODUCER_HEADROOM = 1.10
_LAYOUT_PROBE_HEADROOM = 1.50
_MIN_GRANULARITY_GAIN = 1.03
_MAX_THROUGHPUT_REGRESSION = 0.90
_MAX_WAIT_REGRESSION = 2.0

ControllerPhase = Literal["calibrating", "trial", "steady", "manual"]
Bottleneck = Literal[
    "calibrating",
    "healthy",
    "cache_reloads",
    "starved",
    "memory_limited",
    "cpu_limited",
    "manual",
]
_Signal = Literal["healthy", "cache", "head", "producer", "tail"]
_TrialKind = Literal["right_size", "granularity", "recovery", "resumed"]


class PipelineCapacityError(ValueError):
    """Raised when no valid controller setting can fit the rank-local cap."""


def _positive_int(name: str, value) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_int(name: str, value) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _finite_number(name: str, value, *, positive: bool = False) -> float:
    if isinstance(value, bool) or type(value) not in {int, float}:
        qualifier = "positive " if positive else "non-negative "
        raise TypeError(f"{name} must be a {qualifier}finite number")
    normalized = float(value)
    invalid = normalized <= 0 if positive else normalized < 0
    if not math.isfinite(normalized) or invalid:
        qualifier = "positive " if positive else "non-negative "
        raise ValueError(f"{name} must be a {qualifier}finite number")
    return normalized


def _fraction(name: str, value) -> float:
    normalized = _finite_number(name, value)
    if normalized > 1:
        raise ValueError(f"{name} must be at most 1")
    return normalized


@dataclass(frozen=True, slots=True)
class PipelineControllerConstraints:
    """Resolved limits and size estimates supplied by runtime integration."""

    local_rank_count: int
    per_rank_host_budget_bytes: int
    per_rank_cpu_limit: int
    largest_decoded_file_bytes: int
    output_batch_bytes: int
    fixed_semantic_floor_bytes: int
    planner_token_bytes_per_queued_batch: int
    pin_memory_supported: bool = True
    largest_inflight_file_bytes: int | None = None
    consumer_retained_bytes: int = 0
    h2d_retained_bytes: int = 0
    total_decoded_bytes: int | None = None
    initial_decoded_cache_bytes: int | None = None
    shared_decoded_cache: bool = False

    def __post_init__(self) -> None:
        _positive_int("local_rank_count", self.local_rank_count)
        _positive_int("per_rank_host_budget_bytes", self.per_rank_host_budget_bytes)
        _positive_int("per_rank_cpu_limit", self.per_rank_cpu_limit)
        _positive_int("largest_decoded_file_bytes", self.largest_decoded_file_bytes)
        _positive_int("output_batch_bytes", self.output_batch_bytes)
        _non_negative_int("fixed_semantic_floor_bytes", self.fixed_semantic_floor_bytes)
        _non_negative_int(
            "planner_token_bytes_per_queued_batch",
            self.planner_token_bytes_per_queued_batch,
        )
        if type(self.pin_memory_supported) is not bool:
            raise TypeError("pin_memory_supported must be a boolean")
        if self.largest_inflight_file_bytes is not None:
            _positive_int(
                "largest_inflight_file_bytes",
                self.largest_inflight_file_bytes,
            )
        _non_negative_int("consumer_retained_bytes", self.consumer_retained_bytes)
        _non_negative_int("h2d_retained_bytes", self.h2d_retained_bytes)
        if self.total_decoded_bytes is not None:
            _positive_int("total_decoded_bytes", self.total_decoded_bytes)
            if self.total_decoded_bytes < self.largest_decoded_file_bytes:
                raise ValueError(
                    "total_decoded_bytes cannot be smaller than one decoded file"
                )
        if self.initial_decoded_cache_bytes is not None:
            _positive_int(
                "initial_decoded_cache_bytes",
                self.initial_decoded_cache_bytes,
            )
            if self.initial_decoded_cache_bytes < self.largest_decoded_file_bytes:
                raise ValueError(
                    "initial_decoded_cache_bytes cannot be smaller than one file"
                )
            if (
                self.total_decoded_bytes is not None
                and self.initial_decoded_cache_bytes > self.total_decoded_bytes
            ):
                raise ValueError(
                    "initial_decoded_cache_bytes exceeds total decoded bytes"
                )
        if type(self.shared_decoded_cache) is not bool:
            raise TypeError("shared_decoded_cache must be a boolean")

    @property
    def inflight_file_bytes(self) -> int:
        return (
            self.largest_decoded_file_bytes
            if self.largest_inflight_file_bytes is None
            else self.largest_inflight_file_bytes
        )

    @property
    def queued_batch_bytes(self) -> int:
        return self.output_batch_bytes + self.planner_token_bytes_per_queued_batch

    @property
    def minimum_capacity_bytes(self) -> int:
        return (
            self.fixed_semantic_floor_bytes
            + self.consumer_retained_bytes
            + self.h2d_retained_bytes
            + self.queued_batch_bytes
            + (
                0
                if self.shared_decoded_cache
                else self.inflight_file_bytes + self.largest_decoded_file_bytes
            )
        )

    @property
    def initial_cache_bytes(self) -> int:
        return (
            self.largest_decoded_file_bytes
            if self.initial_decoded_cache_bytes is None
            else self.initial_decoded_cache_bytes
        )

    @property
    def maximum_cache_bytes(self) -> int:
        return (
            self.per_rank_host_budget_bytes
            if self.total_decoded_bytes is None
            else self.total_decoded_bytes
        )

    def as_dict(self) -> dict[str, int | bool]:
        return {
            "local_rank_count": self.local_rank_count,
            "per_rank_host_budget_bytes": self.per_rank_host_budget_bytes,
            "per_rank_cpu_limit": self.per_rank_cpu_limit,
            "largest_decoded_file_bytes": self.largest_decoded_file_bytes,
            "output_batch_bytes": self.output_batch_bytes,
            "fixed_semantic_floor_bytes": self.fixed_semantic_floor_bytes,
            "planner_token_bytes_per_queued_batch": (
                self.planner_token_bytes_per_queued_batch
            ),
            "pin_memory_supported": self.pin_memory_supported,
            "largest_inflight_file_bytes": self.inflight_file_bytes,
            "consumer_retained_bytes": self.consumer_retained_bytes,
            "h2d_retained_bytes": self.h2d_retained_bytes,
            "total_decoded_bytes": self.maximum_cache_bytes,
            "initial_decoded_cache_bytes": self.initial_cache_bytes,
            "shared_decoded_cache": self.shared_decoded_cache,
        }


@dataclass(frozen=True, slots=True)
class DecodeLayout:
    """The decoder settings that are always trialled as one unit."""

    workers: int
    chunk_batches: int
    queue_batches: int

    def __post_init__(self) -> None:
        _positive_int("workers", self.workers)
        _positive_int("chunk_batches", self.chunk_batches)
        _positive_int("queue_batches", self.queue_batches)
        if self.queue_batches < self.chunk_batches:
            raise ValueError("queue_batches must be at least chunk_batches")

    @classmethod
    def from_settings(cls, settings: "PipelineSettings") -> "DecodeLayout":
        return cls(
            workers=settings.decode_workers,
            chunk_batches=settings.decode_chunk_batches,
            queue_batches=settings.ready_queue_batches,
        )

    def apply(self, settings: "PipelineSettings") -> "PipelineSettings":
        return replace(
            settings,
            decode_workers=self.workers,
            decode_chunk_batches=self.chunk_batches,
            ready_queue_batches=self.queue_batches,
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "decode_workers": self.workers,
            "decode_chunk_batches": self.chunk_batches,
            "ready_queue_batches": self.queue_batches,
        }

    @classmethod
    def parse(cls, state: Mapping) -> "DecodeLayout":
        if not isinstance(state, Mapping) or set(state) != {
            "decode_workers",
            "decode_chunk_batches",
            "ready_queue_batches",
        }:
            raise ValueError("controller layout boundary is malformed")
        try:
            return cls(
                workers=state["decode_workers"],
                chunk_batches=state["decode_chunk_batches"],
                queue_batches=state["ready_queue_batches"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("controller layout boundary is malformed") from exc


@dataclass(frozen=True, slots=True)
class PipelineSettings:
    """Rank-local performance settings selected by the controller."""

    decoded_cache_bytes: int
    decode_workers: int
    decode_chunk_batches: int
    ready_queue_batches: int
    pin_memory: bool

    def __post_init__(self) -> None:
        _positive_int("decoded_cache_bytes", self.decoded_cache_bytes)
        _positive_int("decode_workers", self.decode_workers)
        _positive_int("decode_chunk_batches", self.decode_chunk_batches)
        _positive_int("ready_queue_batches", self.ready_queue_batches)
        if type(self.pin_memory) is not bool:
            raise TypeError("pin_memory must be a boolean")
        if self.ready_queue_batches < self.decode_chunk_batches:
            raise ValueError(
                "ready_queue_batches must be at least decode_chunk_batches"
            )

    def as_dict(self) -> dict[str, int | bool]:
        return {
            "decoded_cache_bytes": self.decoded_cache_bytes,
            "decode_workers": self.decode_workers,
            "decode_chunk_batches": self.decode_chunk_batches,
            "ready_queue_batches": self.ready_queue_batches,
            "pin_memory": self.pin_memory,
        }


@dataclass(frozen=True, slots=True)
class PipelineObservation:
    """One already-aggregated controller observation window."""

    producer_rows_per_second: float
    consumer_demand_rows_per_second: float
    data_wait_fraction: float
    cache_reloads: int = 0
    head_of_line_wait_fraction: float = 0.0
    tail_wait_fraction: float = 0.0
    consumer_rows_per_second: float | None = None

    def __post_init__(self) -> None:
        _finite_number("producer_rows_per_second", self.producer_rows_per_second)
        _finite_number(
            "consumer_demand_rows_per_second",
            self.consumer_demand_rows_per_second,
            positive=True,
        )
        _fraction("data_wait_fraction", self.data_wait_fraction)
        _non_negative_int("cache_reloads", self.cache_reloads)
        _fraction("head_of_line_wait_fraction", self.head_of_line_wait_fraction)
        _fraction("tail_wait_fraction", self.tail_wait_fraction)
        if self.consumer_rows_per_second is not None:
            _finite_number(
                "consumer_rows_per_second",
                self.consumer_rows_per_second,
                positive=True,
            )

    @property
    def observed_throughput_rows_per_second(self) -> float:
        return (
            self.producer_rows_per_second
            if self.consumer_rows_per_second is None
            else self.consumer_rows_per_second
        )


@dataclass(frozen=True, slots=True)
class ControllerDecision:
    """One controller event that changes settings or reaches a limit."""

    window: int
    action: str
    reason: str
    bottleneck: Bottleneck
    before: PipelineSettings | None
    after: PipelineSettings
    capacity_bytes: int

    def as_dict(self) -> dict:
        return {
            "window": self.window,
            "action": self.action,
            "reason": self.reason,
            "bottleneck": self.bottleneck,
            "before": None if self.before is None else self.before.as_dict(),
            "after": self.after.as_dict(),
            "capacity_bytes": self.capacity_bytes,
        }


@dataclass(slots=True)
class _LayoutTrial:
    kind: _TrialKind
    accepted: DecodeLayout
    baseline_throughput: float | None
    baseline_wait: float | None
    observations: list[PipelineObservation]


class AdaptivePipelineController:
    """Bounded observation-driven processed-NPZ resource policy."""

    def __init__(
        self,
        config: AdaptiveDataPipelineConfig,
        constraints: PipelineControllerConstraints,
    ) -> None:
        if not isinstance(config, AdaptiveDataPipelineConfig):
            raise TypeError("config must be an AdaptiveDataPipelineConfig")
        if not isinstance(constraints, PipelineControllerConstraints):
            raise TypeError("constraints must be PipelineControllerConstraints")
        self.config = config
        self.constraints = constraints
        self._validate_resolved_budgets()
        self._validate_minimum_capacity()

        self._window = 0
        self._signal: _Signal | None = None
        self._signal_windows: deque[PipelineObservation] = deque(
            maxlen=_DECISION_WINDOWS
        )
        self._blocked_signal: _Signal | None = None
        self._rejected_layouts: set[DecodeLayout] = set()
        self._trial: _LayoutTrial | None = None
        if config.adaptation == "manual":
            self._settings = self._manual_settings()
            self._phase: ControllerPhase = "manual"
            self._frozen = True
            self._bottleneck: Bottleneck = "manual"
            reason = "manual settings were validated and split across local ranks"
        else:
            self._settings = self._initial_auto_settings()
            self._phase = "calibrating"
            self._frozen = False
            self._bottleneck = "calibrating"
            reason = "initialized a balanced layout within resolved budgets"
        self._record_decision("initialize", reason, None, self._settings)

    @property
    def settings(self) -> PipelineSettings:
        return self._settings

    @property
    def bottleneck(self) -> Bottleneck:
        return self._bottleneck

    @property
    def phase(self) -> ControllerPhase:
        return self._phase

    @property
    def frozen(self) -> bool:
        return self._frozen

    def transition_state_dict(self) -> dict:
        """Return the phase and accepted layout, never trial measurements."""

        accepted = (
            DecodeLayout.from_settings(self._settings)
            if self._trial is None
            else self._trial.accepted
        )
        return {
            "phase": self._phase,
            "accepted_layout": accepted.as_dict(),
        }

    def restore_transition_state(self, state: Mapping | None) -> None:
        """Restore a minimal layout boundary and restart any trial evidence."""

        self._clear_signal()
        self._trial = None
        self._blocked_signal = None
        if state is None:
            self._phase = "manual" if self._frozen else "calibrating"
            return
        if not isinstance(state, Mapping) or set(state) != {
            "phase",
            "accepted_layout",
        }:
            raise ValueError("controller transition state has invalid fields")
        phase = state["phase"]
        if phase not in {"calibrating", "trial", "steady", "manual"}:
            raise ValueError("controller phase is malformed")
        accepted = DecodeLayout.parse(state["accepted_layout"])
        self._ensure_capacity(accepted.apply(self._settings))
        if self._frozen != (phase == "manual"):
            raise ValueError("controller phase is incompatible with frozen state")
        self._phase = phase
        if phase == "trial":
            self._trial = _LayoutTrial(
                kind="resumed",
                accepted=accepted,
                baseline_throughput=None,
                baseline_wait=None,
                observations=[],
            )

    def capacity_bytes(self, settings: PipelineSettings | None = None) -> int:
        settings = self._settings if settings is None else settings
        return (
            self.constraints.fixed_semantic_floor_bytes
            + self.constraints.consumer_retained_bytes
            + self.constraints.h2d_retained_bytes
            + settings.ready_queue_batches * self.constraints.queued_batch_bytes
            + (
                0
                if self.constraints.shared_decoded_cache
                else (
                    settings.decode_workers * self.constraints.inflight_file_bytes
                    + settings.decoded_cache_bytes
                )
            )
        )

    def observe(
        self,
        observation: PipelineObservation,
        *,
        epoch_changed: bool = False,
    ) -> ControllerDecision | None:
        if not isinstance(observation, PipelineObservation):
            raise TypeError("observation must be a PipelineObservation")
        self._window += 1
        if self._frozen:
            return None
        if epoch_changed:
            if self._trial is None:
                self._clear_signal()
            return None
        if self._trial is not None:
            return self._observe_trial(observation)

        signal = self._classify(observation)
        self._bottleneck = self._bottleneck_for_signal(signal)
        if self._phase == "steady" and signal == "healthy":
            self._blocked_signal = None
            self._clear_signal()
            return None
        if signal != self._signal:
            self._signal = signal
            self._signal_windows.clear()
        self._signal_windows.append(observation)
        if len(self._signal_windows) < _DECISION_WINDOWS:
            return None

        baseline = self._combine_observations(tuple(self._signal_windows))
        self._clear_signal()
        if signal == "healthy":
            if self._phase == "steady":
                return None
            if self._layout_probe_has_headroom(baseline):
                candidate = self._right_size_candidate()
                if (
                    candidate is not None
                    and candidate not in self._rejected_layouts
                ):
                    return self._try_layout(
                        candidate,
                        baseline,
                        kind="right_size",
                        reason="measured producer headroom permits a smaller decoder layout",
                    )
            self._phase = "steady"
            self._bottleneck = "healthy"
            return None

        self._phase = "calibrating"
        if signal == "cache" and not self.constraints.shared_decoded_cache:
            decision = self._grow_cache()
            if decision is not None:
                return decision
        layout_signal = (
            self._wait_signal(baseline) if signal == "cache" else signal
        )
        candidates = self._layout_candidates(layout_signal, baseline)
        for candidate, kind, reason in candidates:
            if candidate not in self._rejected_layouts:
                return self._try_layout(
                    candidate,
                    baseline,
                    kind=kind,
                    reason=reason,
                )
        self._phase = "steady"
        if candidates:
            return None
        if layout_signal == self._blocked_signal:
            return None
        self._blocked_signal = layout_signal
        return self._blocked(layout_signal)

    def restore_performance_settings(
        self,
        settings: PipelineSettings,
        *,
        frozen: bool | None = None,
    ) -> None:
        """Install a coordinated plan and restart local timing evidence."""

        if not isinstance(settings, PipelineSettings):
            raise TypeError("settings must be PipelineSettings")
        if frozen is not None and type(frozen) is not bool:
            raise TypeError("frozen must be a boolean or null")
        self._ensure_capacity(settings)
        self._settings = settings
        if frozen is not None:
            self._frozen = frozen
        self._phase = "manual" if self._frozen else "calibrating"
        self._trial = None
        self._blocked_signal = None
        self._clear_signal()
        self._bottleneck = "manual" if self._frozen else "calibrating"

    def _validate_resolved_budgets(self) -> None:
        ranks = self.constraints.local_rank_count
        if (
            self.config.host_memory_budget != "auto"
            and self.constraints.per_rank_host_budget_bytes * ranks
            > self.config.host_memory_budget
        ):
            raise ValueError(
                "resolved per-rank host budgets exceed host_memory_budget"
            )
        if (
            self.config.data_cpu_budget != "auto"
            and self.constraints.per_rank_cpu_limit * ranks
            > self.config.data_cpu_budget
        ):
            raise ValueError("resolved per-rank CPU limits exceed data_cpu_budget")

    def _validate_minimum_capacity(self) -> None:
        required = self.constraints.minimum_capacity_bytes
        budget = self.constraints.per_rank_host_budget_bytes
        if required > budget:
            raise PipelineCapacityError(
                "minimum processed-NPZ pipeline does not fit the per-rank host "
                f"budget: required={required}, budget={budget}; "
                f"fixed={self.constraints.fixed_semantic_floor_bytes}, "
                f"consumer_retained={self.constraints.consumer_retained_bytes}, "
                f"h2d_retained={self.constraints.h2d_retained_bytes}, "
                f"one_inflight_file={self.constraints.inflight_file_bytes}, "
                f"one_ready_batch_and_token={self.constraints.queued_batch_bytes}, "
                f"one_cached_file={self.constraints.largest_decoded_file_bytes}"
            )

    def _manual_settings(self) -> PipelineSettings:
        plan = self.config.advanced
        if plan is None:
            raise ValueError("manual adaptation requires advanced settings")
        ranks = self.constraints.local_rank_count
        if plan.decode_workers % ranks:
            raise ValueError(
                "advanced.decode_workers must be divisible by local_rank_count"
            )
        cache_bytes = plan.decoded_cache_bytes // ranks
        workers = plan.decode_workers // ranks
        if cache_bytes < self.constraints.largest_decoded_file_bytes:
            raise PipelineCapacityError(
                "rank-local manual decoded cache cannot retain one decoded file"
            )
        if workers < 1:
            raise ValueError(
                "advanced.decode_workers must provide at least one worker per rank"
            )
        if workers > self.constraints.per_rank_cpu_limit:
            raise ValueError(
                "rank-local manual decode workers exceed per_rank_cpu_limit"
            )
        if plan.pin_memory and not self.constraints.pin_memory_supported:
            raise ValueError(
                "advanced.pin_memory requires a runtime with pinned-memory support"
            )
        settings = PipelineSettings(
            decoded_cache_bytes=cache_bytes,
            decode_workers=workers,
            decode_chunk_batches=plan.decode_chunk_batches,
            ready_queue_batches=plan.ready_queue_batches,
            pin_memory=plan.pin_memory,
        )
        self._ensure_capacity(settings)
        return settings

    def _initial_auto_settings(self) -> PipelineSettings:
        """Start with a balanced layout and reduce it only when needed to fit."""

        workers = max(1, (self.constraints.per_rank_cpu_limit + 1) // 2)
        while workers >= 1:
            multipliers = (8, 4, 2, 1) if workers == 1 else (8, 4, 2)
            for multiplier in multipliers:
                queue = max(1, workers * multiplier)
                fixed = PipelineSettings(
                    decoded_cache_bytes=self.constraints.largest_decoded_file_bytes,
                    decode_workers=workers,
                    decode_chunk_batches=self._balanced_chunk(workers, queue),
                    ready_queue_batches=queue,
                    pin_memory=self.constraints.pin_memory_supported,
                )
                if self.constraints.shared_decoded_cache:
                    if self._fits(fixed):
                        return replace(
                            fixed,
                            decoded_cache_bytes=self.constraints.initial_cache_bytes,
                        )
                    continue
                cache_room = (
                    self.constraints.per_rank_host_budget_bytes
                    - self.capacity_bytes(fixed)
                    + fixed.decoded_cache_bytes
                )
                if cache_room < self.constraints.largest_decoded_file_bytes:
                    continue
                candidate = replace(
                    fixed,
                    decoded_cache_bytes=min(
                        self.constraints.initial_cache_bytes,
                        cache_room,
                    ),
                )
                if self._fits(candidate):
                    return candidate
            if workers == 1:
                break
            workers = max(1, workers // 2)
        raise PipelineCapacityError(
            "minimum processed-NPZ pipeline does not fit the per-rank host budget"
        )

    def _ensure_capacity(self, settings: PipelineSettings) -> None:
        if settings.decode_workers > self.constraints.per_rank_cpu_limit:
            raise PipelineCapacityError("decode_workers exceed per_rank_cpu_limit")
        required = self.capacity_bytes(settings)
        budget = self.constraints.per_rank_host_budget_bytes
        if required > budget:
            raise PipelineCapacityError(
                "pipeline settings exceed the per-rank host budget: "
                f"required={required}, budget={budget}, "
                f"settings={settings.as_dict()}"
            )

    def _fits(self, settings: PipelineSettings) -> bool:
        return (
            settings.decode_workers <= self.constraints.per_rank_cpu_limit
            and self.capacity_bytes(settings)
            <= self.constraints.per_rank_host_budget_bytes
        )

    def _classify(self, observation: PipelineObservation) -> _Signal:
        budget = self.config.data_wait_budget
        if (
            observation.head_of_line_wait_fraction <= budget
            and observation.tail_wait_fraction <= budget
        ):
            return "healthy"
        if observation.cache_reloads > 0:
            return "cache"
        return self._wait_signal(observation)

    def _wait_signal(self, observation: PipelineObservation) -> _Signal:
        """Classify exposed wait after cache growth is unavailable."""

        budget = self.config.data_wait_budget
        if observation.head_of_line_wait_fraction > budget:
            return "head"
        if self._producer_is_short(observation):
            return "producer"
        return "tail"

    @staticmethod
    def _bottleneck_for_signal(signal: _Signal) -> Bottleneck:
        return {
            "healthy": "healthy",
            "cache": "cache_reloads",
            "head": "starved",
            "producer": "starved",
            "tail": "starved",
        }[signal]

    def _producer_is_short(self, observation: PipelineObservation) -> bool:
        return (
            observation.producer_rows_per_second
            < observation.consumer_demand_rows_per_second
            * _MIN_PRODUCER_HEADROOM
        )

    def _layout_probe_has_headroom(self, observation: PipelineObservation) -> bool:
        return (
            observation.head_of_line_wait_fraction <= self.config.data_wait_budget
            and observation.producer_rows_per_second
            >= observation.consumer_demand_rows_per_second
            * _LAYOUT_PROBE_HEADROOM
        )

    def _grow_cache(self) -> ControllerDecision | None:
        current = self._settings.decoded_cache_bytes
        room = (
            self.constraints.per_rank_host_budget_bytes
            - self.capacity_bytes()
            + current
        )
        target = min(
            self.constraints.maximum_cache_bytes,
            room,
            max(current * 2, current + self.constraints.largest_decoded_file_bytes),
        )
        if target <= current:
            return None
        candidate = replace(self._settings, decoded_cache_bytes=target)
        if not self._fits(candidate):
            return None
        before = self._settings
        self._settings = candidate
        self._phase = "calibrating"
        self._blocked_signal = None
        self._bottleneck = "cache_reloads"
        return self._record_decision(
            "grow_cache",
            "decoded-file reloads exposed data wait",
            before,
            candidate,
        )

    def _layout_candidates(
        self,
        signal: _Signal,
        observation: PipelineObservation,
    ) -> list[tuple[DecodeLayout, _TrialKind, str]]:
        candidates: list[tuple[DecodeLayout, _TrialKind, str]] = []
        if signal == "head":
            chunk_batches = max(1, self._settings.decode_chunk_batches // 2)
            if self._producer_is_short(observation):
                candidate = self._worker_recovery_candidate(
                    chunk_batches=chunk_batches,
                )
                if candidate is not None:
                    candidates.append(
                        (
                            candidate,
                            "recovery",
                            "persistent head-of-line wait and producer starvation "
                            "require a smaller chunk and more workers",
                        )
                    )
            if chunk_batches < self._settings.decode_chunk_batches:
                candidates.append(
                    (
                        DecodeLayout(
                            workers=self._settings.decode_workers,
                            chunk_batches=chunk_batches,
                            queue_batches=self._settings.ready_queue_batches,
                        ),
                        "recovery",
                        "persistent head-of-line wait requires a smaller decode chunk",
                    )
                )
        if signal in {"producer", "cache"} and self._producer_is_short(observation):
            candidate = self._worker_recovery_candidate()
            if candidate is not None:
                candidates.append(
                    (
                        candidate,
                        "recovery",
                        "persistent producer starvation requires a stronger decoder layout",
                    )
                )
        if signal in {"tail", "cache"}:
            if self._layout_probe_has_headroom(observation):
                candidate = self._right_size_candidate()
                if candidate is not None:
                    candidates.append(
                        (
                            candidate,
                            "right_size",
                            "source-tail wait still has enough producer headroom for fewer workers",
                        )
                    )
            candidate = self._granularity_candidate()
            if candidate is not None:
                candidates.append(
                    (
                        candidate,
                        "granularity",
                        "persistent source-tail wait warrants a coarser decode layout",
                    )
                )
        return candidates

    def _right_size_candidate(self) -> DecodeLayout | None:
        if self._settings.decode_workers <= 1:
            return None
        workers = max(1, (self._settings.decode_workers + 1) // 2)
        return DecodeLayout(
            workers=workers,
            chunk_batches=self._balanced_chunk(
                workers,
                self._settings.ready_queue_batches,
            ),
            queue_batches=self._settings.ready_queue_batches,
        )

    def _worker_recovery_candidate(
        self,
        *,
        chunk_batches: int | None = None,
    ) -> DecodeLayout | None:
        current = self._settings.decode_workers
        if current >= self.constraints.per_rank_cpu_limit:
            self._bottleneck = "cpu_limited"
            return None
        chunk_batches = (
            self._settings.decode_chunk_batches
            if chunk_batches is None
            else chunk_batches
        )
        desired = min(
            self.constraints.per_rank_cpu_limit,
            max(current + 1, current * 2),
        )
        for workers in range(desired, current, -1):
            queue = max(self._settings.ready_queue_batches, workers * 2)
            layout = DecodeLayout(
                workers=workers,
                chunk_batches=min(
                    chunk_batches,
                    self._balanced_chunk(workers, queue),
                ),
                queue_batches=queue,
            )
            if self._fits(layout.apply(self._settings)):
                return layout
        return None

    def _granularity_candidate(self) -> DecodeLayout | None:
        queue = self._settings.ready_queue_batches * 2
        layout = DecodeLayout(
            workers=self._settings.decode_workers,
            chunk_batches=self._balanced_chunk(self._settings.decode_workers, queue),
            queue_batches=queue,
        )
        return layout if self._fits(layout.apply(self._settings)) else None

    def _try_layout(
        self,
        candidate: DecodeLayout,
        baseline: PipelineObservation,
        *,
        kind: _TrialKind,
        reason: str,
    ) -> ControllerDecision:
        before = self._settings
        after = candidate.apply(before)
        self._ensure_capacity(after)
        if after == before:
            raise RuntimeError("layout trial did not change the active settings")
        self._settings = after
        self._phase = "trial"
        self._blocked_signal = None
        self._trial = _LayoutTrial(
            kind=kind,
            accepted=DecodeLayout.from_settings(before),
            baseline_throughput=baseline.observed_throughput_rows_per_second,
            baseline_wait=baseline.data_wait_fraction,
            observations=[],
        )
        return self._record_decision("try_layout", reason, before, after)

    def _observe_trial(
        self,
        observation: PipelineObservation,
    ) -> ControllerDecision | None:
        trial = self._trial
        if trial is None:
            raise RuntimeError("trial phase lacks a layout boundary")
        trial.observations.append(observation)
        required = _TRIAL_SETTLE_WINDOWS + _TRIAL_VERIFY_WINDOWS
        if len(trial.observations) < required:
            return None
        verification = trial.observations[-_TRIAL_VERIFY_WINDOWS:]
        combined = self._combine_observations(verification)
        reasons = self._trial_rejection_reasons(trial, verification, combined)
        self._clear_signal()
        self._trial = None
        if reasons:
            before = self._settings
            if trial.kind != "resumed":
                self._rejected_layouts.add(DecodeLayout.from_settings(before))
            self._settings = trial.accepted.apply(before)
            self._ensure_capacity(self._settings)
            self._phase = "steady"
            rejected_signal = self._classify(combined)
            self._blocked_signal = None
            self._bottleneck = self._bottleneck_for_signal(rejected_signal)
            return self._record_decision(
                "reject_layout",
                "; ".join(reasons),
                before,
                self._settings,
            )

        self._phase = "steady" if trial.kind == "recovery" else "calibrating"
        self._blocked_signal = None
        self._bottleneck = self._bottleneck_for_signal(self._classify(combined))
        return self._record_decision(
            "accept_layout",
            "candidate layout preserved throughput and producer headroom",
            trial.accepted.apply(self._settings),
            self._settings,
        )

    def _trial_rejection_reasons(
        self,
        trial: _LayoutTrial,
        verification: Sequence[PipelineObservation],
        combined: PipelineObservation,
    ) -> list[str]:
        if trial.baseline_throughput is None or trial.baseline_wait is None:
            return ["trial restarted without persisted performance evidence"]
        reasons = []
        if (
            combined.observed_throughput_rows_per_second
            < trial.baseline_throughput * _MAX_THROUGHPUT_REGRESSION
        ):
            reasons.append("consumer throughput regressed by more than 10%")
        if self._producer_is_short(combined):
            reasons.append("producer capacity fell below its safety margin")
        if all(
            item.head_of_line_wait_fraction > self.config.data_wait_budget
            for item in verification
        ):
            reasons.append("head-of-line wait remained above budget")
        wait_limit = max(
            self.config.data_wait_budget,
            trial.baseline_wait * _MAX_WAIT_REGRESSION,
        )
        if all(item.data_wait_fraction > wait_limit for item in verification):
            reasons.append("data wait more than doubled")
        if (
            trial.kind == "granularity"
            and combined.observed_throughput_rows_per_second
            < trial.baseline_throughput * _MIN_GRANULARITY_GAIN
        ):
            reasons.append("coarser layout improved throughput by less than 3%")
        return reasons

    def _blocked(self, signal: _Signal) -> ControllerDecision:
        if self._bottleneck == "cpu_limited":
            reason = "producer starvation persists at the per-rank CPU limit"
        else:
            self._bottleneck = "memory_limited"
            reason = (
                "cache or layout cannot grow within the per-rank host budget"
                if signal in {"cache", "tail"}
                else "no useful layout change remains within resolved limits"
            )
        return self._record_decision(
            "blocked",
            reason,
            self._settings,
            self._settings,
        )

    @staticmethod
    def _balanced_chunk(workers: int, queue: int) -> int:
        return max(1, queue // max(1, workers * 2))

    @staticmethod
    def _combine_observations(
        observations: Sequence[PipelineObservation],
    ) -> PipelineObservation:
        if not observations:
            raise ValueError("cannot combine an empty observation window")
        count = len(observations)
        return PipelineObservation(
            producer_rows_per_second=sum(
                item.producer_rows_per_second for item in observations
            )
            / count,
            consumer_demand_rows_per_second=sum(
                item.consumer_demand_rows_per_second for item in observations
            )
            / count,
            data_wait_fraction=max(item.data_wait_fraction for item in observations),
            cache_reloads=sum(item.cache_reloads for item in observations),
            head_of_line_wait_fraction=max(
                item.head_of_line_wait_fraction for item in observations
            ),
            tail_wait_fraction=max(
                item.tail_wait_fraction for item in observations
            ),
            consumer_rows_per_second=(
                None
                if any(item.consumer_rows_per_second is None for item in observations)
                else sum(
                    item.consumer_rows_per_second for item in observations
                )
                / count
            ),
        )

    def _clear_signal(self) -> None:
        self._signal = None
        self._signal_windows.clear()

    def _record_decision(
        self,
        action: str,
        reason: str,
        before: PipelineSettings | None,
        after: PipelineSettings,
    ) -> ControllerDecision:
        return ControllerDecision(
            window=self._window,
            action=action,
            reason=reason,
            bottleneck=self._bottleneck,
            before=before,
            after=after,
            capacity_bytes=self.capacity_bytes(after),
        )


__all__ = [
    "AdaptivePipelineController",
    "Bottleneck",
    "ControllerDecision",
    "ControllerPhase",
    "DecodeLayout",
    "PipelineCapacityError",
    "PipelineControllerConstraints",
    "PipelineObservation",
    "PipelineSettings",
]
