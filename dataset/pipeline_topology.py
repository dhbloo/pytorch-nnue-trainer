"""Pure coordination of resource policy across distributed rank reports."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from utils.system_resources import SystemResources

from .pipeline_config import AdaptiveDataPipelineConfig
from .pipeline_resources import (
    PipelineResourceAllocation,
    PipelineResourceError,
    resolve_pipeline_resources,
)


class PipelineTopologyError(ValueError):
    """Raised when rank reports cannot produce one distributed allocation."""


@dataclass(frozen=True, slots=True)
class NodeRankLayout:
    """Ephemeral rank layout for one node without exposing its identity."""

    ranks: tuple[int, ...]
    current_rank: int

    def __post_init__(self) -> None:
        if not self.ranks or tuple(sorted(set(self.ranks))) != self.ranks:
            raise ValueError("node ranks must be a non-empty sorted unique tuple")
        if self.current_rank not in self.ranks:
            raise ValueError("current rank is not part of the node rank layout")

    @property
    def leader_rank(self) -> int:
        return self.ranks[0]

    @property
    def is_leader(self) -> bool:
        return self.current_rank == self.leader_rank


@dataclass(frozen=True, slots=True)
class RankResourceReport:
    """Resource snapshot reported by one rank on an opaque node identity."""

    rank: int
    node_key: str = field(repr=False)
    resources: SystemResources

    def __post_init__(self) -> None:
        if type(self.rank) is not int:
            raise TypeError("rank must be a non-negative integer")
        if self.rank < 0:
            raise ValueError("rank must be a non-negative integer")
        if not isinstance(self.node_key, str):
            raise TypeError("node_key must be a non-empty string")
        if not self.node_key:
            raise ValueError("node_key must be a non-empty string")
        if not isinstance(self.resources, SystemResources):
            raise TypeError("resources must be a SystemResources snapshot")


@dataclass(frozen=True, slots=True)
class DistributedPipelineResources:
    """One conservative rank-local allocation shared by every DDP rank."""

    local_rank_count: int
    per_rank_host_budget_bytes: int
    per_rank_cpu_limit: int
    host_budget_provenance: str
    cpu_limit_provenance: str
    clamp_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_positive_int("local_rank_count", self.local_rank_count)
        _require_positive_int(
            "per_rank_host_budget_bytes",
            self.per_rank_host_budget_bytes,
        )
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
        """Return a JSON-safe view that excludes opaque node identities."""

        return {
            "local_rank_count": self.local_rank_count,
            "per_rank_host_budget_bytes": self.per_rank_host_budget_bytes,
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


def _validate_reports(
    reports: Iterable[RankResourceReport],
) -> tuple[RankResourceReport, ...]:
    try:
        normalized = tuple(reports)
    except TypeError as exc:
        raise TypeError("reports must be an iterable of RankResourceReport values") from exc
    if not normalized:
        raise PipelineTopologyError("at least one rank resource report is required")
    if not all(isinstance(report, RankResourceReport) for report in normalized):
        raise TypeError("reports must contain only RankResourceReport values")

    ranks = [report.rank for report in normalized]
    if len(set(ranks)) != len(ranks):
        raise PipelineTopologyError("rank resource reports must have unique ranks")
    if sorted(ranks) != list(range(len(normalized))):
        raise PipelineTopologyError(
            "rank resource reports must have contiguous ranks starting at 0"
        )
    return tuple(sorted(normalized, key=lambda report: report.rank))


def _minimum_with_provenance(
    allocations: list[tuple[int, PipelineResourceAllocation]],
    value_field: str,
    provenance_field: str,
) -> tuple[int, str]:
    minimum = min(getattr(allocation, value_field) for _, allocation in allocations)
    rank, allocation = next(
        (rank, allocation)
        for rank, allocation in allocations
        if getattr(allocation, value_field) == minimum
    )
    provenance = getattr(allocation, provenance_field)
    return minimum, f"conservative global minimum selected from rank {rank}: {provenance}"


def _collect_clamp_reasons(
    allocations: list[tuple[int, PipelineResourceAllocation]],
) -> tuple[str, ...]:
    reasons: list[str] = []
    seen: set[str] = set()
    for rank, allocation in allocations:
        for reason in allocation.clamp_reasons:
            described = f"rank {rank}: {reason}"
            if described not in seen:
                reasons.append(described)
                seen.add(described)
    return tuple(reasons)


def coordinate_pipeline_resources(
    config: AdaptiveDataPipelineConfig,
    reports: Iterable[RankResourceReport],
) -> DistributedPipelineResources:
    """Resolve all reports and return identical conservative per-rank limits."""

    if not isinstance(config, AdaptiveDataPipelineConfig):
        raise TypeError("config must be an AdaptiveDataPipelineConfig")
    normalized = _validate_reports(reports)

    ranks_per_node: dict[str, int] = {}
    for report in normalized:
        ranks_per_node[report.node_key] = ranks_per_node.get(report.node_key, 0) + 1

    allocations: list[tuple[int, PipelineResourceAllocation]] = []
    for report in normalized:
        try:
            allocation = resolve_pipeline_resources(
                config,
                report.resources,
                ranks_per_node[report.node_key],
            )
        except PipelineResourceError as exc:
            raise PipelineTopologyError(
                f"resource resolution failed for rank {report.rank}: {exc}"
            ) from exc
        allocations.append((report.rank, allocation))

    per_rank_host_budget, host_provenance = _minimum_with_provenance(
        allocations,
        "per_rank_host_budget_bytes",
        "host_budget_provenance",
    )
    per_rank_cpu_limit, cpu_provenance = _minimum_with_provenance(
        allocations,
        "per_rank_cpu_limit",
        "cpu_limit_provenance",
    )
    return DistributedPipelineResources(
        local_rank_count=max(ranks_per_node.values()),
        per_rank_host_budget_bytes=per_rank_host_budget,
        per_rank_cpu_limit=per_rank_cpu_limit,
        host_budget_provenance=host_provenance,
        cpu_limit_provenance=cpu_provenance,
        clamp_reasons=_collect_clamp_reasons(allocations),
    )


def resolve_node_rank_layout(
    reports: Iterable[RankResourceReport],
    current_rank: int,
) -> NodeRankLayout:
    """Resolve a rank's node-local peers while keeping the node key ephemeral."""

    normalized = _validate_reports(reports)
    if type(current_rank) is not int or not 0 <= current_rank < len(normalized):
        raise ValueError("current_rank must identify one reported rank")
    node_key = normalized[current_rank].node_key
    ranks = tuple(
        report.rank for report in normalized if report.node_key == node_key
    )
    return NodeRankLayout(ranks=ranks, current_rank=current_rank)


__all__ = [
    "DistributedPipelineResources",
    "NodeRankLayout",
    "PipelineTopologyError",
    "RankResourceReport",
    "coordinate_pipeline_resources",
    "resolve_node_rank_layout",
]
