from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

from .core import canonical_pipeline_state_bytes
from .source import RecordEnvelope, SourceCapabilities
from .stream import collate_sample_dicts


COMPOSITE_SOURCE_SCHEMA = "composite-record-source-v3"
COMPOSITE_ENVELOPE_OVERHEAD_BYTES = 128


@dataclass(frozen=True, slots=True)
class CompositePayload:
    child_index: int
    envelope: RecordEnvelope


@dataclass(slots=True)
class CompositeCursor:
    epoch: int
    source_cycle: int
    rank: int
    child_cursors: list
    exhausted: list[bool]
    pending: list[RecordEnvelope]
    pending_position: int
    cycle_index: int
    terminal: bool


def _weighted_cycle(weights: tuple[int, ...]) -> tuple[int, ...]:
    """Return one evenly interleaved cycle with exact integer child counts."""
    if not weights or any(type(weight) is not int or weight <= 0 for weight in weights):
        raise ValueError("composite weights must be positive integers")
    total = sum(weights)
    scores = [0] * len(weights)
    schedule = []
    for _ in range(total):
        for child, weight in enumerate(weights):
            scores[child] += weight
        selected = max(range(len(weights)), key=lambda child: (scores[child], -child))
        scores[selected] -= total
        schedule.append(selected)
    return tuple(schedule)


class CompositeRecordSource:
    """Deterministically mix native child sources without changing payloads."""

    def __init__(self, child_sources, child_ids, weights, *, sync_length: bool):
        self.child_sources = tuple(child_sources)
        self.child_ids = tuple(str(child_id) for child_id in child_ids)
        self.weights = tuple(int(weight) for weight in weights)
        self.sync_length = bool(sync_length)
        if (
            not self.child_sources
            or len(self.child_sources) != len(self.child_ids)
            or len(self.child_sources) != len(self.weights)
        ):
            raise ValueError("composite child, ID, and weight counts must match")
        if len(set(self.child_ids)) != len(self.child_ids):
            raise ValueError("composite child IDs must be unique")
        if any(not source.capabilities.deterministic for source in self.child_sources):
            raise ValueError("composite sources must be deterministic")
        self.schedule = _weighted_cycle(self.weights)

        shapes = set()
        child_shape_maps = []
        for child, source in enumerate(self.child_sources):
            shape_codes = getattr(source, "shape_codes", None)
            if not isinstance(shape_codes, dict) or not shape_codes:
                raise TypeError(
                    f"composite child {self.child_ids[child]!r} does not expose "
                    "shape_codes"
                )
            normalized = {tuple(shape): int(code) for shape, code in shape_codes.items()}
            if len(set(normalized.values())) != len(normalized):
                raise ValueError(
                    f"composite child {self.child_ids[child]!r} has duplicate shape codes"
                )
            child_shape_maps.append({code: shape for shape, code in normalized.items()})
            shapes.update(normalized)
        self.shape_codes = {
            shape: code for code, shape in enumerate(sorted(shapes))
        }
        self._child_shape_maps = tuple(child_shape_maps)
        self._record_key_prefixes = tuple(
            b"NNUE-composite-record-key-v2\0"
            + struct.pack("<I", len(child_id.encode("utf-8")))
            + child_id.encode("utf-8")
            for child_id in self.child_ids
        )
        self._world_size = 1
        self._rank = 0
        self._rank_sharded = False

        capabilities = [source.capabilities for source in self.child_sources]
        self.capabilities = SourceCapabilities(
            access_mode=(
                "random"
                if all(capability.random_access for capability in capabilities)
                else "sequential"
            ),
            known_length=all(capability.known_length for capability in capabilities),
            exact_distributed_partition=all(
                capability.exact_distributed_partition
                for capability in capabilities
            ),
            resumable=all(capability.resumable for capability in capabilities),
            deterministic=True,
        )

    def configure_distributed(
        self,
        world_size: int,
        rank: int,
        *,
        rank_sharded: bool,
    ) -> None:
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("invalid composite distributed identity")
        for child_id, source in zip(self.child_ids, self.child_sources):
            if hasattr(source, "configure_distributed"):
                source.configure_distributed(
                    world_size,
                    rank,
                    rank_sharded=rank_sharded,
                )
            elif rank_sharded:
                raise TypeError(
                    f"composite child {child_id!r} cannot be rank-sharded"
                )
        self._world_size = int(world_size)
        self._rank = int(rank)
        self._rank_sharded = bool(rank_sharded)

    def manifest_state(self) -> dict:
        return {
            "schema": COMPOSITE_SOURCE_SCHEMA,
            "child_ids": list(self.child_ids),
            "weights": list(self.weights),
            "schedule": list(self.schedule),
            "sync_length": self.sync_length,
            "shape_codes": [
                [list(shape), code]
                for shape, code in sorted(self.shape_codes.items())
            ],
            "children": [
                source.manifest_state() for source in self.child_sources
            ],
        }

    def start_epoch(self, epoch: int, rank: int) -> CompositeCursor:
        return self.start_cycle(epoch, 0, rank)

    def start_cycle(self, epoch: int, cycle: int, rank: int) -> CompositeCursor:
        if type(epoch) is not int or epoch < 0 or cycle < 0:
            raise ValueError("composite epoch/cycle must be non-negative")
        if rank != self._rank or not 0 <= rank < self._world_size:
            raise ValueError("composite rank differs from its configuration")
        cursors = []
        try:
            for source in self.child_sources:
                cursors.append(
                    source.start_cycle(epoch, cycle, rank)
                    if hasattr(source, "start_cycle")
                    else source.start_epoch(epoch, rank)
                )
        except BaseException:
            for source, cursor in zip(self.child_sources, cursors):
                if hasattr(source, "close_cursor"):
                    source.close_cursor(cursor)
            raise
        cursor_rank = rank if self._rank_sharded else 0
        return CompositeCursor(
            epoch,
            cycle,
            cursor_rank,
            cursors,
            [False] * len(cursors),
            [],
            0,
            0,
            False,
        )

    def _wrap(self, child: int, envelope: RecordEnvelope) -> RecordEnvelope:
        try:
            shape = self._child_shape_maps[child][envelope.shape_code]
        except KeyError as exc:
            raise RuntimeError(
                f"composite child {self.child_ids[child]!r} emitted an unknown shape code"
            ) from exc
        record_key = (
            "composite",
            self.child_ids[child],
            envelope.record_key,
        )
        return RecordEnvelope(
            source_id=child,
            record_key=record_key,
            shape_code=self.shape_codes[shape],
            payload=CompositePayload(child, envelope),
            resident_bytes=(
                envelope.resident_bytes + COMPOSITE_ENVELOPE_OVERHEAD_BYTES
            ),
        )

    def _next_child(self, cursor: CompositeCursor, child: int):
        envelope, child_cursor = self.child_sources[child].next_envelope(
            cursor.child_cursors[child]
        )
        cursor.child_cursors[child] = child_cursor
        if envelope is None:
            cursor.exhausted[child] = True
            return None
        return self._wrap(child, envelope)

    def _next_child_many(
        self,
        cursor: CompositeCursor,
        child: int,
        limit: int,
    ) -> list[RecordEnvelope]:
        if cursor.exhausted[child]:
            return []
        source = self.child_sources[child]
        child_cursor = cursor.child_cursors[child]
        envelopes = []
        while len(envelopes) < limit:
            remaining = limit - len(envelopes)
            if hasattr(source, "next_envelopes"):
                chunk, child_cursor = source.next_envelopes(
                    child_cursor,
                    remaining,
                )
                envelopes.extend(chunk)
                if not chunk:
                    cursor.exhausted[child] = True
                    break
            else:
                envelope, child_cursor = source.next_envelope(child_cursor)
                if envelope is None:
                    cursor.exhausted[child] = True
                    break
                envelopes.append(envelope)
        cursor.child_cursors[child] = child_cursor
        return envelopes

    def _fill_cycles(self, cursor: CompositeCursor, cycle_count: int) -> bool:
        child_envelopes = [
            self._next_child_many(cursor, child, weight * cycle_count)
            for child, weight in enumerate(self.weights)
        ]
        if self.sync_length:
            completed_cycles = min(
                len(envelopes) // weight
                for envelopes, weight in zip(child_envelopes, self.weights)
            )
            if completed_cycles == 0:
                cursor.terminal = True
                cursor.pending = []
                cursor.pending_position = 0
                return False
        else:
            if all(not envelopes for envelopes in child_envelopes):
                cursor.terminal = True
                return False
            completed_cycles = cycle_count

        positions = [0] * len(self.child_sources)
        pending = []
        nonempty_cycles = 0
        for _ in range(completed_cycles):
            cycle_begin = len(pending)
            for child in self.schedule:
                position = positions[child]
                if position >= len(child_envelopes[child]):
                    continue
                pending.append(self._wrap(child, child_envelopes[child][position]))
                positions[child] += 1
            nonempty_cycles += len(pending) != cycle_begin
        if not pending:
            cursor.terminal = True
            return False
        cursor.pending = pending
        cursor.pending_position = 0
        cursor.cycle_index += nonempty_cycles
        return True

    def _fill_cycle(self, cursor: CompositeCursor) -> bool:
        pending = []
        if self.sync_length:
            for child in self.schedule:
                envelope = self._next_child(cursor, child)
                if envelope is None:
                    cursor.terminal = True
                    cursor.pending = []
                    cursor.pending_position = 0
                    return False
                pending.append(envelope)
        else:
            if all(cursor.exhausted):
                cursor.terminal = True
                return False
            for child in self.schedule:
                if cursor.exhausted[child]:
                    continue
                envelope = self._next_child(cursor, child)
                if envelope is not None:
                    pending.append(envelope)
            if not pending:
                cursor.terminal = True
                return False
        cursor.pending = pending
        cursor.pending_position = 0
        cursor.cycle_index += 1
        return True

    def next_envelope(
        self, cursor: CompositeCursor
    ) -> tuple[RecordEnvelope | None, CompositeCursor]:
        if cursor.terminal:
            return None, cursor
        if cursor.pending_position >= len(cursor.pending):
            cursor.pending = []
            cursor.pending_position = 0
            if not self._fill_cycle(cursor):
                return None, cursor
        envelope = cursor.pending[cursor.pending_position]
        cursor.pending_position += 1
        if cursor.pending_position == len(cursor.pending):
            cursor.pending = []
            cursor.pending_position = 0
        return envelope, cursor

    def next_envelopes(
        self,
        cursor: CompositeCursor,
        limit: int,
    ) -> tuple[tuple[RecordEnvelope, ...], CompositeCursor]:
        if type(limit) is not int or limit <= 0:
            raise ValueError("composite chunk limit must be a positive integer")
        output = []
        while len(output) < limit and not cursor.terminal:
            if cursor.pending_position >= len(cursor.pending):
                cursor.pending = []
                cursor.pending_position = 0
                remaining = limit - len(output)
                cycle_count = max(
                    1,
                    (remaining + len(self.schedule) - 1) // len(self.schedule),
                )
                if not self._fill_cycles(cursor, cycle_count):
                    break
            available = min(
                limit - len(output),
                len(cursor.pending) - cursor.pending_position,
            )
            output.extend(
                cursor.pending[
                    cursor.pending_position : cursor.pending_position + available
                ]
            )
            cursor.pending_position += available
            if cursor.pending_position == len(cursor.pending):
                cursor.pending = []
                cursor.pending_position = 0
        return tuple(output), cursor

    def materialize_batch(self, envelopes) -> dict:
        grouped: list[list[tuple[int, RecordEnvelope]]] = [
            [] for _ in self.child_sources
        ]
        for output_index, envelope in enumerate(envelopes):
            payload = envelope.payload
            if (
                not isinstance(payload, CompositePayload)
                or payload.child_index != envelope.source_id
                or not 0 <= payload.child_index < len(self.child_sources)
                or envelope.record_key
                != (
                    "composite",
                    self.child_ids[payload.child_index],
                    payload.envelope.record_key,
                )
            ):
                raise RuntimeError("composite envelope identity is inconsistent")
            grouped[payload.child_index].append((output_index, payload.envelope))

        samples = [None] * len(envelopes)
        for child, routed in enumerate(grouped):
            if not routed:
                continue
            child_batch = self.child_sources[child].materialize_batch(
                tuple(envelope for _, envelope in routed)
            )
            for child_row, (output_index, _) in enumerate(routed):
                samples[output_index] = {
                    key: value[child_row]
                    for key, value in child_batch.items()
                }
        if any(sample is None for sample in samples):
            raise RuntimeError("composite materialization left an unrouted row")
        return collate_sample_dicts(samples, validate_core_fields=True)

    def _save_child_envelope(self, child: int, envelope: RecordEnvelope) -> dict:
        source = self.child_sources[child]
        payload = (
            source.save_payload(envelope.payload)
            if hasattr(source, "save_payload")
            else envelope.payload
        )
        return {
            "source_id": envelope.source_id,
            "record_key": envelope.record_key,
            "shape_code": envelope.shape_code,
            "payload": payload,
            "resident_bytes": envelope.resident_bytes,
        }

    @staticmethod
    def _tuple_tree(value):
        if isinstance(value, list):
            return tuple(CompositeRecordSource._tuple_tree(item) for item in value)
        if isinstance(value, tuple):
            return tuple(CompositeRecordSource._tuple_tree(item) for item in value)
        return value

    def _restore_child_envelope(self, child: int, state: dict) -> RecordEnvelope:
        source = self.child_sources[child]
        payload = (
            source.restore_payload(state["payload"])
            if hasattr(source, "restore_payload")
            else state["payload"]
        )
        return RecordEnvelope(
            source_id=int(state["source_id"]),
            record_key=self._tuple_tree(state["record_key"]),
            shape_code=int(state["shape_code"]),
            payload=payload,
            resident_bytes=int(state["resident_bytes"]),
        )

    def save_payload(self, payload: CompositePayload) -> dict:
        if not isinstance(payload, CompositePayload):
            raise ValueError("invalid composite payload")
        return {
            "child_index": payload.child_index,
            "envelope": self._save_child_envelope(
                payload.child_index,
                payload.envelope,
            ),
        }

    def restore_payload(self, state: dict) -> CompositePayload:
        try:
            child = int(state["child_index"])
            envelope_state = state["envelope"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed composite payload") from exc
        if not 0 <= child < len(self.child_sources):
            raise ValueError("composite payload child is out of range")
        return CompositePayload(
            child,
            self._restore_child_envelope(child, envelope_state),
        )

    def _save_envelope(self, envelope: RecordEnvelope) -> dict:
        return {
            "source_id": envelope.source_id,
            "record_key": envelope.record_key,
            "shape_code": envelope.shape_code,
            "payload": self.save_payload(envelope.payload),
            "resident_bytes": envelope.resident_bytes,
        }

    def _restore_envelope(self, state: dict) -> RecordEnvelope:
        return RecordEnvelope(
            source_id=int(state["source_id"]),
            record_key=self._tuple_tree(state["record_key"]),
            shape_code=int(state["shape_code"]),
            payload=self.restore_payload(state["payload"]),
            resident_bytes=int(state["resident_bytes"]),
        )

    def save_cursor(self, cursor: CompositeCursor) -> dict:
        if not self.capabilities.resumable:
            raise RuntimeError("one or more composite children cannot resume exactly")
        return {
            "schema": COMPOSITE_SOURCE_SCHEMA,
            "epoch": cursor.epoch,
            "source_cycle": cursor.source_cycle,
            "rank": cursor.rank,
            "child_cursors": [
                source.save_cursor(child_cursor)
                for source, child_cursor in zip(
                    self.child_sources,
                    cursor.child_cursors,
                )
            ],
            "exhausted": list(cursor.exhausted),
            "pending": [
                self._save_envelope(envelope)
                for envelope in cursor.pending[cursor.pending_position :]
            ],
            "cycle_index": cursor.cycle_index,
            "terminal": cursor.terminal,
        }

    def restore_cursor(self, state: dict) -> CompositeCursor:
        if not self.capabilities.resumable:
            raise RuntimeError("one or more composite children cannot resume exactly")
        try:
            if state["schema"] != COMPOSITE_SOURCE_SCHEMA:
                raise ValueError("composite cursor schema changed")
            epoch = int(state["epoch"])
            source_cycle = int(state["source_cycle"])
            rank = int(state["rank"])
            child_states = tuple(state["child_cursors"])
            exhausted_values = tuple(state["exhausted"])
            pending_states = tuple(state["pending"])
            cycle_index = int(state["cycle_index"])
            terminal = state["terminal"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed composite cursor") from exc
        if any(type(value) is not bool for value in exhausted_values):
            raise ValueError("composite exhausted flags must be booleans")
        exhausted = list(exhausted_values)
        if (
            len(child_states) != len(self.child_sources)
            or len(exhausted) != len(self.child_sources)
            or type(terminal) is not bool
            or epoch < 0
            or source_cycle < 0
            or rank != (self._rank if self._rank_sharded else 0)
            or cycle_index < 0
        ):
            raise ValueError("invalid composite cursor dimensions")
        child_cursors = []
        try:
            for source, child_state in zip(self.child_sources, child_states):
                child_cursor = source.restore_cursor(child_state)
                if (
                    getattr(child_cursor, "epoch", epoch) != epoch
                    or getattr(child_cursor, "cycle", source_cycle)
                    != source_cycle
                    or getattr(child_cursor, "rank", rank) != rank
                ):
                    raise ValueError(
                        "composite child cursor identity differs from its parent"
                    )
                child_cursors.append(child_cursor)
            pending = [self._restore_envelope(item) for item in pending_states]
        except BaseException:
            for source, child_cursor in zip(self.child_sources, child_cursors):
                if hasattr(source, "close_cursor"):
                    source.close_cursor(child_cursor)
            raise
        if terminal and pending:
            self.close_cursor(
                CompositeCursor(
                    epoch,
                    source_cycle,
                    rank,
                    child_cursors,
                    exhausted,
                    pending,
                    0,
                    cycle_index,
                    terminal,
                )
            )
            raise ValueError("terminal composite cursor contains pending records")
        return CompositeCursor(
            epoch,
            source_cycle,
            rank,
            child_cursors,
            exhausted,
            pending,
            0,
            cycle_index,
            terminal,
        )

    def update_record_key_digest(self, digest, record_key: tuple) -> None:
        try:
            namespace, child_id, child_key = record_key
            child = self.child_ids.index(child_id)
        except (TypeError, ValueError) as exc:
            raise ValueError("malformed composite record key") from exc
        if namespace != "composite":
            raise ValueError("malformed composite record key")
        digest.update(self._record_key_prefixes[child])
        source = self.child_sources[child]
        if hasattr(source, "update_record_key_digest"):
            source.update_record_key_digest(digest, child_key)
        else:
            encoded = canonical_pipeline_state_bytes(child_key)
            digest.update(hashlib.sha256(encoded).digest())

    def close_cursor(self, cursor: CompositeCursor) -> None:
        for source, child_cursor in zip(
            self.child_sources,
            cursor.child_cursors,
        ):
            if hasattr(source, "close_cursor"):
                source.close_cursor(child_cursor)
