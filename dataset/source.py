from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Literal, Protocol, Sequence, TypeVar


SOURCE_MANIFEST_SCHEMA = "record-source-manifest-v2"
SOURCE_CURSOR_SCHEMA = "record-source-cursor-v2"


@dataclass(frozen=True, slots=True)
class SourceCapabilities:
    """Physical and semantic capabilities exposed to the dataset planner."""

    access_mode: Literal["random", "sequential"]
    known_length: bool
    exact_distributed_partition: bool
    resumable: bool
    deterministic: bool

    def __post_init__(self) -> None:
        if self.access_mode not in {"random", "sequential"}:
            raise ValueError(f"unsupported source access mode {self.access_mode!r}")
        for name in (
            "known_length",
            "exact_distributed_partition",
            "resumable",
            "deterministic",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"source capability {name} must be a bool")

    @property
    def random_access(self) -> bool:
        return self.access_mode == "random"

    @property
    def sequential_access(self) -> bool:
        return self.access_mode == "sequential"


@dataclass(frozen=True, slots=True)
class RecordEnvelope:
    """Bounded planner-facing record metadata with an opaque native payload."""

    source_id: int
    record_key: Hashable
    shape_code: int
    payload: object
    resident_bytes: int = 0


SourceCursorT = TypeVar("SourceCursorT")


class RecordSource(Protocol[SourceCursorT]):
    """Format-native record stream consumed by the unified planner."""

    capabilities: SourceCapabilities

    def manifest_state(self) -> dict: ...

    def start_epoch(self, epoch: int, rank: int) -> SourceCursorT: ...

    def next_envelope(
        self, cursor: SourceCursorT
    ) -> tuple[RecordEnvelope | None, SourceCursorT]: ...

    def materialize_batch(
        self, envelopes: Sequence[RecordEnvelope]
    ) -> dict: ...

    def save_cursor(self, cursor: SourceCursorT) -> dict: ...

    def restore_cursor(self, state: dict) -> SourceCursorT: ...
