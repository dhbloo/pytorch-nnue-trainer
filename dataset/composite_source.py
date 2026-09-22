from __future__ import annotations

import hashlib
import struct
from dataclasses import asdict, dataclass

import numpy as np

from .core import FIELD_SPECS, canonical_pipeline_state_bytes, validate_field_dict
from .mixing import MixingConfig
from .source import RecordEnvelope, SourceCapabilities
from .stream import collate_sample_dicts


COMPOSITE_SOURCE_SCHEMA = "composite-record-source-v4"
COMPOSITE_ENVELOPE_OVERHEAD_BYTES = 128
MIXING_BLOCK_ROWS = 1024


def _merge_child_batches(groups, size):
    """Scatter decoded child arrays once, retaining the shared field contract."""
    if not groups:
        raise ValueError("cannot materialize an empty composite batch")
    first = groups[0][1]
    for positions, data in groups:
        if data.keys() != first.keys():
            raise ValueError("decoded children have inconsistent field schemas")
        if len(data["board_size"]) != len(positions):
            raise ValueError("decoded child changed the batch row count")

    # Keep the generic contract for tensor and ragged/list-valued sources.
    if any(
        not isinstance(value, np.ndarray) or value.dtype.kind == "O" or value.ndim < 2
        for _, data in groups for value in data.values()
    ):
        samples = [None] * size
        for positions, data in groups:
            for row, position in enumerate(positions):
                samples[int(position)] = {key: value[row] for key, value in data.items()}
        return collate_sample_dicts(samples, validate_core_fields=True)

    for _, data in groups:
        validate_field_dict(data, batched=True)
    for key, value in first.items():
        for _, data in groups[1:]:
            if data[key].shape[1:] != value.shape[1:]:
                raise ValueError(f"field {key!r} has incompatible child shapes")
            if FIELD_SPECS[key].scope == "batch_shared" and not np.array_equal(
                data[key][0], value[0]
            ):
                raise ValueError(f"batch-shared field {key!r} differs between children")
    if len(groups) == 1:
        return first
    output = {
        key: np.empty(
            (size, *value.shape[1:]),
            dtype=np.result_type(*(data[key].dtype for _, data in groups)),
        )
        for key, value in first.items()
    }
    for positions, data in groups:
        for key, value in data.items():
            output[key][positions] = value
    return output


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
    remaining: list[int]


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

    def __init__(self, child_sources, child_ids, weights, *, mixing: MixingConfig, seed: int):
        self.child_sources = tuple(child_sources)
        self.child_ids = tuple(str(child_id) for child_id in child_ids)
        self.weights = tuple(int(weight) for weight in weights)
        self.mixing = mixing
        self.seed = int(seed)
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
        counts = tuple(getattr(source, "logical_row_count", None) for source in self.child_sources)
        exact_counts = all(
            type(count) is int and count >= 0 and getattr(source, "sample_rate", 1.0) == 1.0
            for source, count in zip(self.child_sources, counts)
        )
        self.quotas = None
        if mixing.mode in {"natural", "tempered"}:
            if not exact_counts:
                raise ValueError(
                    "natural/tempered mixing requires known filtered row counts and "
                    "sample_rate=1 on every child"
                )
            self.quotas = mixing.epoch_quotas(counts)
            self.schedule = ()
            self.logical_row_count = sum(self.quotas)
        else:
            self.schedule = _weighted_cycle(self.weights)
            self.logical_row_count = (
                min(count // weight for count, weight in zip(counts, self.weights))
                * sum(self.weights) if exact_counts else None
            )
            if self.logical_row_count == 0:
                raise ValueError("balanced/weighted mixing cannot supply one complete source cycle")

        shapes = set()
        child_shape_maps = []
        for source in self.child_sources:
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
        if self.quotas is not None and rank_sharded:
            raise ValueError("size-based mixing requires globally planned source counts")
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
            "mixing": asdict(self.mixing),
            "quotas": self.quotas,
            "seed": self.seed,
            "mixing_rng": ("numpy-pcg64-choice-v1", np.__version__, MIXING_BLOCK_ROWS),
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
            list(self.quotas) if self.quotas is not None else [],
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
        completed_cycles = min(
            len(envelopes) // weight
            for envelopes, weight in zip(child_envelopes, self.weights)
        )
        if completed_cycles == 0:
            cursor.terminal = True
            cursor.pending = []
            cursor.pending_position = 0
            return False

        positions = [0] * len(self.child_sources)
        pending = []
        for _ in range(completed_cycles):
            for child in self.schedule:
                position = positions[child]
                pending.append(self._wrap(child, child_envelopes[child][position]))
                positions[child] += 1
        cursor.pending = pending
        cursor.pending_position = 0
        cursor.cycle_index += completed_cycles
        return True

    def _draw_quota_sources(self, cursor):
        total = sum(cursor.remaining)
        if not total:
            return np.empty(0, dtype=np.intp)
        seed = hashlib.sha256(canonical_pipeline_state_bytes(
            (self.seed, cursor.epoch, cursor.source_cycle, cursor.cycle_index)
        )).digest()
        rng = np.random.Generator(np.random.PCG64(int.from_bytes(seed[:16], "little")))
        # Sample virtual ranks, not a full row permutation. Retained state is bounded
        # by this fixed block and the number of sources, independent of data size.
        ranks = rng.choice(total, size=min(MIXING_BLOCK_ROWS, total), replace=False)
        children = np.searchsorted(np.cumsum(cursor.remaining, dtype=np.int64), ranks, side="right")
        counts = np.bincount(children, minlength=len(cursor.remaining))
        cursor.remaining = [n - int(count) for n, count in zip(cursor.remaining, counts)]
        cursor.cycle_index += 1
        return children

    def _fill_quota_block(self, cursor):
        children = self._draw_quota_sources(cursor)
        cursor.pending_position = 0
        if not len(children):
            cursor.terminal = True
            cursor.pending = []
            return False
        pending = [None] * len(children)
        for child in np.unique(children):
            child = int(child)
            positions = np.flatnonzero(children == child)
            values = self._next_child_many(cursor, child, len(positions))
            if len(values) != len(positions):
                raise RuntimeError("source exhausted before its declared mixing quota")
            for position, value in zip(positions, values):
                pending[int(position)] = self._wrap(child, value)
        cursor.pending = pending
        return True

    def _fill_cycle(self, cursor: CompositeCursor) -> bool:
        if self.quotas is not None:
            return self._fill_quota_block(cursor)
        return self._fill_cycles(cursor, 1)

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
                if self.quotas is not None:
                    filled = self._fill_quota_block(cursor)
                else:
                    remaining = limit - len(output)
                    cycle_count = max(
                        1, (remaining + len(self.schedule) - 1) // len(self.schedule)
                    )
                    filled = self._fill_cycles(cursor, cycle_count)
                if not filled:
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

        groups = []
        for child, routed in enumerate(grouped):
            if not routed:
                continue
            child_batch = self.child_sources[child].materialize_batch(
                tuple(envelope for _, envelope in routed)
            )
            groups.append((np.asarray([index for index, _ in routed], dtype=np.intp), child_batch))
        return _merge_child_batches(groups, len(envelopes))

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
            "remaining": list(cursor.remaining),
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
            remaining = list(state["remaining"])
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
        quotas = self.quotas or ()
        if len(remaining) != len(quotas) or any(
            type(n) is not int or not 0 <= n <= quota
            for n, quota in zip(remaining, quotas)
        ) or (terminal and any(remaining)):
            raise ValueError("invalid remaining composite quotas")
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
                    remaining,
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
            remaining,
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
