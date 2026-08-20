"""Pure resource-policy bridge for adaptive data pipelines.

This module resolves already-probed system facts and normalized user caps into
rank-local limits.  It never probes the operating system and does not apply the
result to a running pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from utils.system_resources import Confidence, ResourceValue, SystemResources

from .pipeline_config import AdaptiveDataPipelineConfig


_CONFIDENT_MEMORY_LEVELS = {Confidence.MEDIUM, Confidence.HIGH}


class PipelineResourceError(ValueError):
    """Raised when resource facts and user caps cannot form a usable allocation."""


@dataclass(frozen=True, slots=True)
class PipelineResourceAllocation:
    """Resolved node-wide and evenly split rank-local resource limits."""

    local_rank_count: int
    node_host_budget_bytes: int
    per_rank_host_budget_bytes: int
    node_cpu_limit: int
    per_rank_cpu_limit: int
    host_budget_provenance: str
    cpu_limit_provenance: str
    clamp_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_positive_int("local_rank_count", self.local_rank_count)
        _require_positive_int("node_host_budget_bytes", self.node_host_budget_bytes)
        _require_positive_int(
            "per_rank_host_budget_bytes",
            self.per_rank_host_budget_bytes,
        )
        _require_positive_int("node_cpu_limit", self.node_cpu_limit)
        _require_positive_int("per_rank_cpu_limit", self.per_rank_cpu_limit)
        if not isinstance(self.host_budget_provenance, str) or not self.host_budget_provenance:
            raise ValueError("host_budget_provenance must be a non-empty string")
        if not isinstance(self.cpu_limit_provenance, str) or not self.cpu_limit_provenance:
            raise ValueError("cpu_limit_provenance must be a non-empty string")
        if not isinstance(self.clamp_reasons, tuple) or not all(
            isinstance(reason, str) and reason for reason in self.clamp_reasons
        ):
            raise TypeError("clamp_reasons must be a tuple of non-empty strings")

    def as_dict(self) -> dict[str, int | str | list[str]]:
        """Return a JSON-safe representation of this allocation."""

        return {
            "local_rank_count": self.local_rank_count,
            "node_host_budget_bytes": self.node_host_budget_bytes,
            "per_rank_host_budget_bytes": self.per_rank_host_budget_bytes,
            "node_cpu_limit": self.node_cpu_limit,
            "per_rank_cpu_limit": self.per_rank_cpu_limit,
            "host_budget_provenance": self.host_budget_provenance,
            "cpu_limit_provenance": self.cpu_limit_provenance,
            "clamp_reasons": list(self.clamp_reasons),
        }


def _require_positive_int(name: str, value) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _memory_fact(metric: ResourceValue[int]) -> int | None:
    value = metric.value
    if type(value) is not int or value < 0:
        return None
    return value


def _cpu_fact(metric: ResourceValue[int] | ResourceValue[float]) -> float | None:
    value = metric.value
    if isinstance(value, bool) or type(value) not in {int, float}:
        return None
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0:
        return None
    return normalized


def _fact_provenance(metric: ResourceValue) -> str:
    return f"{metric.source} ({metric.confidence.value} confidence)"


def _resolve_host_budget(
    config: AdaptiveDataPipelineConfig,
    resources: SystemResources,
) -> tuple[int, str, str | None]:
    available_metric = resources.memory.effective_available_bytes
    total_metric = resources.memory.effective_total_bytes
    available = _memory_fact(available_metric)
    total = _memory_fact(total_metric)

    if config.host_memory_budget == "auto":
        if available is None or available < 2:
            raise PipelineResourceError(
                "automatic host memory selection requires effective available "
                "memory; set host_memory_budget to an explicit positive byte "
                "size when safe available-memory probing is unavailable"
            )
        candidates: list[tuple[int, str]] = [
            (
                available // 2,
                "50% of effective available memory from "
                + _fact_provenance(available_metric),
            )
        ]
        if total is not None:
            candidates.append(
                (
                    total // 4,
                    "25% of effective total memory from "
                    + _fact_provenance(total_metric),
                )
            )
        selected = min(value for value, _ in candidates)
        if selected <= 0:
            raise PipelineResourceError(
                "automatic host memory facts do not provide a positive budget; "
                "set host_memory_budget explicitly"
            )
        provenance = "auto: " + "; ".join(description for _, description in candidates)
        return selected, provenance, None

    requested = _require_positive_int("host_memory_budget", config.host_memory_budget)
    selected = requested
    provenance = "explicit host_memory_budget"
    clamp_reason = None
    if (
        available is not None
        and available_metric.confidence in _CONFIDENT_MEMORY_LEVELS
    ):
        safety_cap = available * 4 // 5
        provenance += "; 80% effective-available safety ceiling from " + _fact_provenance(
            available_metric
        )
        if safety_cap < selected:
            selected = safety_cap
            clamp_reason = (
                f"host_memory_budget clamped from {requested} to {selected} bytes "
                "by the 80% effective-available safety ceiling"
            )
    if selected <= 0:
        raise PipelineResourceError(
            "the effective-available safety ceiling leaves no positive host "
            "memory budget; free host memory or retry with a safe explicit "
            "host_memory_budget after available memory increases"
        )
    return selected, provenance, clamp_reason


def _observed_cpu_limit(resources: SystemResources) -> tuple[int | None, str | None]:
    effective_metric = resources.cpu.effective_count
    effective = _cpu_fact(effective_metric)
    if effective is not None:
        return math.floor(effective), _fact_provenance(effective_metric)

    logical_metric = resources.cpu.logical_count
    logical = _cpu_fact(logical_metric)
    if logical is not None:
        return math.floor(logical), _fact_provenance(logical_metric)
    return None, None


def _resolve_cpu_limit(
    config: AdaptiveDataPipelineConfig,
    resources: SystemResources,
) -> tuple[int, str, str | None]:
    observed, observed_provenance = _observed_cpu_limit(resources)

    if config.data_cpu_budget == "auto":
        if observed is None:
            raise PipelineResourceError(
                "automatic CPU selection requires an effective or logical CPU count; "
                "set data_cpu_budget to an explicit positive integer when probing is "
                "unavailable"
            )
        return observed, f"auto: floor of {observed_provenance}", None

    requested = _require_positive_int("data_cpu_budget", config.data_cpu_budget)
    if observed is None:
        return requested, "explicit data_cpu_budget; no observed CPU ceiling", None

    selected = min(requested, observed)
    provenance = f"explicit data_cpu_budget; observed ceiling from {observed_provenance}"
    clamp_reason = None
    if selected < requested:
        clamp_reason = (
            f"data_cpu_budget clamped from {requested} to {selected} CPUs "
            "by the observed CPU ceiling"
        )
    return selected, provenance, clamp_reason


def resolve_pipeline_resources(
    config: AdaptiveDataPipelineConfig,
    system_resources: SystemResources,
    local_rank_count: int,
) -> PipelineResourceAllocation:
    """Resolve node caps and deterministic, evenly split per-rank limits."""

    if not isinstance(config, AdaptiveDataPipelineConfig):
        raise TypeError("config must be an AdaptiveDataPipelineConfig")
    if not isinstance(system_resources, SystemResources):
        raise TypeError("system_resources must be a SystemResources snapshot")
    ranks = _require_positive_int("local_rank_count", local_rank_count)

    node_host_budget, host_provenance, host_clamp = _resolve_host_budget(
        config,
        system_resources,
    )
    node_cpu_limit, cpu_provenance, cpu_clamp = _resolve_cpu_limit(
        config,
        system_resources,
    )

    per_rank_host_budget = node_host_budget // ranks
    if per_rank_host_budget <= 0:
        raise PipelineResourceError(
            f"resolved node host budget ({node_host_budget} bytes) provides no "
            f"positive byte budget across {ranks} local ranks; reduce "
            "local_rank_count or raise host_memory_budget"
        )
    per_rank_cpu_limit = node_cpu_limit // ranks
    if per_rank_cpu_limit < 1:
        raise PipelineResourceError(
            f"resolved node CPU limit ({node_cpu_limit}) provides fewer than 1 CPU "
            f"per local rank across {ranks} ranks; reduce local_rank_count or raise "
            "data_cpu_budget"
        )

    clamp_reasons = tuple(
        reason for reason in (host_clamp, cpu_clamp) if reason is not None
    )
    return PipelineResourceAllocation(
        local_rank_count=ranks,
        node_host_budget_bytes=node_host_budget,
        per_rank_host_budget_bytes=per_rank_host_budget,
        node_cpu_limit=node_cpu_limit,
        per_rank_cpu_limit=per_rank_cpu_limit,
        host_budget_provenance=host_provenance,
        cpu_limit_provenance=cpu_provenance,
        clamp_reasons=clamp_reasons,
    )


__all__ = [
    "PipelineResourceAllocation",
    "PipelineResourceError",
    "resolve_pipeline_resources",
]
