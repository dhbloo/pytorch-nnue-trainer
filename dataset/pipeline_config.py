"""Configuration schema for adaptive processed-NPZ data pipelines.

The schema normalizes YAML-friendly values without selecting or applying a
runtime controller.  Host memory and CPU budgets are totals for one node, not
per-rank allowances, so adding local training ranks does not multiply them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import math
import re
from typing import Literal


AUTO = "auto"
AdaptationMode = Literal["continuous", "manual"]

_BYTE_SIZE_PATTERN = re.compile(
    r"^\s*(?P<amount>(?:\d+(?:\.\d*)?|\.\d+))\s*"
    r"(?P<unit>b|[kmgtp]i?b)?\s*$",
    re.IGNORECASE,
)
_BYTE_MULTIPLIERS = {
    "b": 1,
    "kb": 1000,
    "mb": 1000**2,
    "gb": 1000**3,
    "tb": 1000**4,
    "pb": 1000**5,
    "kib": 1024,
    "mib": 1024**2,
    "gib": 1024**3,
    "tib": 1024**4,
    "pib": 1024**5,
}


def _parse_byte_size(name: str, value, *, allow_auto: bool) -> int | str:
    if allow_auto and isinstance(value, str) and value.strip().lower() == AUTO:
        return AUTO
    if type(value) is int:
        if value <= 0:
            raise ValueError(f"{name} must be positive")
        return value
    if not isinstance(value, str):
        qualifier = "'auto' or " if allow_auto else ""
        raise TypeError(f"{name} must be {qualifier}a positive byte size")

    match = _BYTE_SIZE_PATTERN.fullmatch(value)
    if match is None:
        qualifier = "'auto' or " if allow_auto else ""
        raise ValueError(
            f"{name} must be {qualifier}a byte size such as '512 MiB' or '2.5 GiB'"
        )
    amount = Decimal(match.group("amount"))
    unit = (match.group("unit") or "b").lower()
    byte_count = amount * _BYTE_MULTIPLIERS[unit]
    if byte_count <= 0 or byte_count != byte_count.to_integral_value():
        raise ValueError(f"{name} must resolve to a positive whole number of bytes")
    return int(byte_count)


def _parse_wait_fraction(name: str, value) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a fraction or percentage")
    if type(value) in {int, float}:
        fraction = float(value)
    elif isinstance(value, str):
        text = value.strip()
        percentage = text.endswith("%")
        if percentage:
            text = text[:-1].strip()
        try:
            fraction = float(Decimal(text))
        except (InvalidOperation, ValueError) as exc:
            raise ValueError(f"{name} must be a fraction or percentage") from exc
        if percentage:
            fraction /= 100.0
    else:
        raise TypeError(f"{name} must be a fraction or percentage")
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError(f"{name} must be greater than 0% and at most 100%")
    return fraction


def _parse_positive_int(name: str, value) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _parse_cpu_budget(name: str, value) -> int | str:
    if isinstance(value, str) and value.strip().lower() == AUTO:
        return AUTO
    return _parse_positive_int(name, value)


@dataclass(frozen=True, slots=True)
class ManualDataPipelinePlan:
    """Advanced fixed plan used only with ``adaptation: manual``.

    ``decoded_cache_bytes`` and ``decode_workers`` are node totals.  Batch
    counts describe each rank-local decode pipeline and are kept independent
    from the number of local ranks.
    """

    decoded_cache_bytes: int
    decode_workers: int
    decode_chunk_batches: int
    ready_queue_batches: int
    pin_memory: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "decoded_cache_bytes",
            _parse_byte_size(
                "advanced.decoded_cache_bytes",
                self.decoded_cache_bytes,
                allow_auto=False,
            ),
        )
        object.__setattr__(
            self,
            "decode_workers",
            _parse_positive_int("advanced.decode_workers", self.decode_workers),
        )
        object.__setattr__(
            self,
            "decode_chunk_batches",
            _parse_positive_int(
                "advanced.decode_chunk_batches", self.decode_chunk_batches
            ),
        )
        object.__setattr__(
            self,
            "ready_queue_batches",
            _parse_positive_int(
                "advanced.ready_queue_batches", self.ready_queue_batches
            ),
        )
        if type(self.pin_memory) is not bool:
            raise TypeError("advanced.pin_memory must be a boolean")
        if self.ready_queue_batches < self.decode_chunk_batches:
            raise ValueError(
                "advanced.ready_queue_batches must be at least "
                "advanced.decode_chunk_batches"
            )

    @classmethod
    def parse(cls, value) -> "ManualDataPipelinePlan":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("advanced must be a mapping")
        fields = set(cls.__dataclass_fields__)
        unknown = set(value).difference(fields)
        missing = fields.difference(value)
        if unknown:
            raise ValueError(
                "advanced has unknown option(s): " + ", ".join(sorted(unknown))
            )
        if missing:
            raise ValueError(
                "advanced is missing required option(s): "
                + ", ".join(sorted(missing))
            )
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class AdaptiveDataPipelineConfig:
    """Normalized public configuration for the adaptive controller."""

    host_memory_budget: int | Literal["auto"] = AUTO
    data_wait_budget: float = 0.01
    data_cpu_budget: int | Literal["auto"] = AUTO
    adaptation: AdaptationMode = "continuous"
    advanced: ManualDataPipelinePlan | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "host_memory_budget",
            _parse_byte_size(
                "host_memory_budget",
                self.host_memory_budget,
                allow_auto=True,
            ),
        )
        object.__setattr__(
            self,
            "data_wait_budget",
            _parse_wait_fraction("data_wait_budget", self.data_wait_budget),
        )
        object.__setattr__(
            self,
            "data_cpu_budget",
            _parse_cpu_budget("data_cpu_budget", self.data_cpu_budget),
        )
        if not isinstance(self.adaptation, str):
            raise TypeError("adaptation must be a string")
        if self.adaptation not in {"continuous", "manual"}:
            raise ValueError("adaptation must be 'continuous' or 'manual'")
        if self.advanced is not None and not isinstance(
            self.advanced, ManualDataPipelinePlan
        ):
            object.__setattr__(
                self,
                "advanced",
                ManualDataPipelinePlan.parse(self.advanced),
            )
        if self.adaptation == "manual":
            if self.advanced is None:
                raise ValueError("adaptation 'manual' requires advanced")
        elif self.advanced is not None:
            raise ValueError("advanced is only valid with adaptation 'manual'")
        if self.advanced is not None:
            if (
                self.host_memory_budget != AUTO
                and self.advanced.decoded_cache_bytes > self.host_memory_budget
            ):
                raise ValueError(
                    "advanced.decoded_cache_bytes exceeds host_memory_budget"
                )
            if (
                self.data_cpu_budget != AUTO
                and self.advanced.decode_workers > self.data_cpu_budget
            ):
                raise ValueError("advanced.decode_workers exceeds data_cpu_budget")

    @classmethod
    def parse(cls, value) -> "AdaptiveDataPipelineConfig":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("adaptive data-pipeline configuration must be a mapping")
        fields = set(cls.__dataclass_fields__)
        unknown = set(value).difference(fields)
        if unknown:
            raise ValueError(
                "adaptive data-pipeline configuration has unknown option(s): "
                + ", ".join(sorted(unknown))
            )

        return cls(**dict(value))


def parse_adaptive_data_pipeline_config(value) -> AdaptiveDataPipelineConfig:
    """Parse a YAML-decoded mapping into the public adaptive schema."""

    return AdaptiveDataPipelineConfig.parse(value)


__all__ = [
    "AdaptiveDataPipelineConfig",
    "AdaptationMode",
    "ManualDataPipelinePlan",
    "parse_adaptive_data_pipeline_config",
]
