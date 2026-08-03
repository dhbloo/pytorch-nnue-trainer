from __future__ import annotations

import hashlib
import io
import os
import struct
from dataclasses import dataclass

from .core import (
    canonical_pipeline_state_bytes,
    deterministic_permutation,
    rng_u64,
    sample_is_admitted,
)
from .source import RecordEnvelope, SourceCapabilities
from .stream import collate_sample_dicts, reject_duplicate_physical_files


SEQUENTIAL_SOURCE_SCHEMA = "sequential-record-source-v3"
SEQUENTIAL_ENVELOPE_OVERHEAD_BYTES = 256


@dataclass(frozen=True, slots=True)
class SequentialFileDescriptor:
    path: str
    file_ordinal: int
    size: int
    mtime_ns: int
    file_key: bytes
    compressed: bool


@dataclass(frozen=True, slots=True)
class SequentialPayload:
    raw_entry: bytes
    subrecord: int | None = None
    augmentation_epoch: int = 0


@dataclass(slots=True)
class _PendingEntry:
    raw_entry: bytes
    entry_index: int
    shape_code: int
    next_subrecord: int
    subrecord_count: int


@dataclass(slots=True)
class _ActiveReader:
    descriptor_index: int
    stream: object
    entry_index: int = 0
    pending: _PendingEntry | None = None


@dataclass(slots=True)
class SequentialCursor:
    epoch: int
    cycle: int
    rank: int
    file_order: tuple[int, ...]
    next_file_position: int
    active: list[_ActiveReader]
    turn: int
    quantum_used: int


class _CaptureReader:
    __slots__ = ("source", "parts")

    def __init__(self, source):
        self.source = source
        self.parts: list[bytes] = []

    def readinto(self, buffer):
        count = self.source.readinto(buffer)
        if count:
            self.parts.append(bytes(memoryview(buffer).cast("B")[:count]))
        return count

    def read(self, size=-1):
        value = self.source.read(size)
        if value:
            self.parts.append(bytes(value))
        return value

    def tell(self):
        return self.source.tell()

    @property
    def payload(self) -> bytes:
        return b"".join(self.parts)


class InterleavedSequentialSource:
    """Deterministically interleave bounded sequential readers without indexing."""

    def __init__(
        self,
        paths,
        *,
        format_id: str,
        schema_version: int,
        seed: int,
        shuffle: bool,
        sample_rate: float,
        active_streams: int,
        read_quantum: int,
        output_shapes,
        open_file,
        read_entry,
        shape_of,
        materialize_entry,
        read_raw_entry=None,
        decode_raw_entry=None,
        subrecord_count=None,
        semantic_state: dict | None = None,
    ) -> None:
        if not 0.0 <= sample_rate <= 1.0:
            raise ValueError(f"sample_rate must be in [0, 1], got {sample_rate}")
        if type(active_streams) is not int or active_streams <= 0:
            raise ValueError("sequential active_streams must be a positive integer")
        if type(read_quantum) is not int or read_quantum <= 0:
            raise ValueError("sequential read_quantum must be a positive integer")
        paths = reject_duplicate_physical_files(paths)
        if not paths:
            raise ValueError("sequential source requires at least one file")
        descriptors = []
        identity_semantics = {
            "schema": SEQUENTIAL_SOURCE_SCHEMA,
            "format_id": format_id,
            "schema_version": int(schema_version),
            "semantic_state": dict(semantic_state or {}),
        }
        semantic_digest = hashlib.sha256(
            canonical_pipeline_state_bytes(identity_semantics)
        ).digest()
        for ordinal, path in enumerate(paths):
            stat = os.stat(path)
            identity = canonical_pipeline_state_bytes(
                {
                    "path": path,
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                    "file_ordinal": ordinal,
                    "source_semantics": semantic_digest,
                }
            )
            descriptors.append(
                SequentialFileDescriptor(
                    path=path,
                    file_ordinal=ordinal,
                    size=int(stat.st_size),
                    mtime_ns=int(stat.st_mtime_ns),
                    file_key=hashlib.sha256(identity).digest(),
                    compressed=path.lower().endswith(".lz4"),
                )
            )
        shapes = sorted({tuple(shape) for shape in output_shapes})
        if not shapes:
            raise ValueError("sequential source must declare at least one output shape")
        self.descriptors = tuple(descriptors)
        self.format_id = str(format_id)
        self.schema_version = int(schema_version)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.sample_rate = float(sample_rate)
        self.active_streams = active_streams
        self.read_quantum = read_quantum
        self.shape_codes = {shape: index for index, shape in enumerate(shapes)}
        self.open_file = open_file
        self.read_entry = read_entry
        self.read_raw_entry = read_raw_entry
        self.decode_raw_entry = decode_raw_entry
        self.subrecord_count = subrecord_count
        self.shape_of = shape_of
        self.materialize_entry = materialize_entry
        self.semantic_state = dict(semantic_state or {})
        encoded_format = self.format_id.encode("utf-8")
        self._record_key_prefix = (
            b"NNUE-sequential-record-key-v2\0"
            + struct.pack("<I", len(encoded_format))
            + encoded_format
        )
        self._active_epoch = 0
        self._world_size = 1
        self._rank = 0
        self._rank_sharded = False
        self._file_identity = hashlib.sha256(
            b"".join(descriptor.file_key for descriptor in self.descriptors)
        ).digest()
        self.capabilities = SourceCapabilities(
            access_mode="sequential",
            known_length=False,
            exact_distributed_partition=False,
            resumable=not any(
                descriptor.compressed for descriptor in self.descriptors
            ),
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
            raise ValueError("invalid sequential distributed identity")
        if rank_sharded and len(self.descriptors) < world_size:
            raise ValueError(
                "rank-sharded sequential training requires at least one file "
                "per rank"
            )
        self._world_size = int(world_size)
        self._rank = int(rank)
        self._rank_sharded = bool(rank_sharded)

    def manifest_state(self) -> dict:
        return {
            "schema": SEQUENTIAL_SOURCE_SCHEMA,
            "format_id": self.format_id,
            "schema_version": self.schema_version,
            "shuffle": self.shuffle,
            "sample_rate": self.sample_rate,
            "active_streams": self.active_streams,
            "read_quantum": self.read_quantum,
            "distributed_file_policy": "greedy-size-balanced-v1",
            "logical_records": (
                "entry" if self.subrecord_count is None else "entry-subrecord"
            ),
            "shape_codes": [
                [list(shape), code]
                for shape, code in sorted(self.shape_codes.items())
            ],
            "semantic_state": self.semantic_state,
            "identity_policy": "path-size-mtime-ordinal-semantics-v1",
            "files": [
                {
                    "path": descriptor.path,
                    "file_ordinal": descriptor.file_ordinal,
                    "size": descriptor.size,
                    "mtime_ns": descriptor.mtime_ns,
                    "file_key": descriptor.file_key.hex(),
                    "compressed": descriptor.compressed,
                }
                for descriptor in self.descriptors
            ],
        }

    def _open_reader(self, descriptor_index: int, entry_index: int = 0):
        descriptor = self.descriptors[descriptor_index]
        return _ActiveReader(
            descriptor_index,
            self.open_file(descriptor.path),
            entry_index,
        )

    def _file_order(self, epoch: int, cycle: int, rank: int) -> tuple[int, ...]:
        all_files = tuple(range(len(self.descriptors)))
        if not self._rank_sharded:
            if not self.shuffle:
                return all_files
            return tuple(
                deterministic_permutation(
                    len(all_files),
                    self.seed,
                    "sequential_file_order",
                    (
                        (epoch, self._file_identity)
                        if cycle == 0
                        else (epoch, cycle, self._file_identity)
                    ),
                )
            )
        assignments = [[] for _ in range(self._world_size)]
        loads = [0] * self._world_size
        by_size = sorted(
            all_files,
            key=lambda index: (-self.descriptors[index].size, index),
        )
        for position, descriptor_index in enumerate(by_size):
            assigned_rank = (
                position
                if position < self._world_size
                else min(
                    range(self._world_size),
                    key=lambda candidate: (loads[candidate], candidate),
                )
            )
            assignments[assigned_rank].append(descriptor_index)
            loads[assigned_rank] += self.descriptors[descriptor_index].size
        shard = tuple(sorted(assignments[rank]))
        if not self.shuffle:
            return shard
        permutation = deterministic_permutation(
            len(shard),
            self.seed,
            "sequential_shard_file_order",
            (epoch, cycle, rank, self._file_identity),
        )
        return tuple(shard[index] for index in permutation)

    def start_epoch(self, epoch: int, rank: int) -> SequentialCursor:
        return self.start_cycle(epoch, 0, rank)

    def start_cycle(self, epoch: int, cycle: int, rank: int) -> SequentialCursor:
        if epoch < 0 or cycle < 0:
            raise ValueError("sequential source epoch/cycle must be non-negative")
        if rank != self._rank or not 0 <= rank < self._world_size:
            raise ValueError("sequential source rank differs from its configuration")
        order = self._file_order(epoch, cycle, rank)
        count = min(self.active_streams, len(order))
        active = [self._open_reader(order[index]) for index in range(count)]
        self._active_epoch = epoch
        return SequentialCursor(epoch, cycle, rank, order, count, active, 0, 0)

    def _augmentation_epoch(self, cursor: SequentialCursor) -> int:
        if cursor.cycle == 0:
            return cursor.epoch
        return int(
            rng_u64(
                self.seed,
                "sequential_cycle_epoch",
                (cursor.epoch, cursor.cycle),
            )
            & ((1 << 63) - 1)
        )

    @staticmethod
    def _has_data(stream) -> bool:
        if hasattr(stream, "peek"):
            return stream.peek(1) != b""
        position = stream.tell()
        value = stream.read(1)
        stream.seek(position)
        return value != b""

    def _replace_or_remove(self, cursor: SequentialCursor) -> None:
        reader = cursor.active[cursor.turn]
        if reader.pending is not None:
            raise RuntimeError("cannot replace a reader with pending subrecords")
        reader.stream.close()
        if cursor.next_file_position < len(cursor.file_order):
            descriptor_index = cursor.file_order[cursor.next_file_position]
            cursor.next_file_position += 1
            cursor.active[cursor.turn] = self._open_reader(descriptor_index)
        else:
            cursor.active.pop(cursor.turn)
            if cursor.active:
                cursor.turn %= len(cursor.active)
        cursor.quantum_used = 0

    def _advance_turn(self, cursor: SequentialCursor) -> None:
        if cursor.active:
            cursor.turn = (cursor.turn + 1) % len(cursor.active)
        cursor.quantum_used = 0

    def next_envelope(
        self, cursor: SequentialCursor
    ) -> tuple[RecordEnvelope | None, SequentialCursor]:
        while cursor.active:
            reader = cursor.active[cursor.turn]
            descriptor = self.descriptors[reader.descriptor_index]
            if reader.pending is not None:
                envelope = self._next_pending_envelope(
                    cursor,
                    reader,
                    descriptor,
                )
                if envelope is not None:
                    return envelope, cursor
                continue
            if not self._has_data(reader.stream):
                self._replace_or_remove(cursor)
                continue
            entry_index = reader.entry_index
            try:
                if self.read_raw_entry is not None:
                    raw_entry = self.read_raw_entry(reader.stream)
                    entry = raw_entry
                else:
                    capture = _CaptureReader(reader.stream)
                    entry = self.read_entry(capture)
                    raw_entry = capture.payload
            except EOFError as exc:
                raise RuntimeError(
                    f"truncated {self.format_id} entry {entry_index} in "
                    f"{descriptor.path}: {exc}"
                ) from exc
            if not raw_entry:
                raise RuntimeError(
                    f"{self.format_id} reader consumed no bytes for entry "
                    f"{entry_index} in {descriptor.path}"
                )
            reader.entry_index += 1
            cursor.quantum_used += 1
            if cursor.quantum_used >= self.read_quantum:
                self._advance_turn(cursor)

            shape = self.shape_of(entry)
            if shape is None:
                continue
            shape = tuple(shape)
            shape_code = self.shape_codes.get(shape)
            if shape_code is None:
                raise RuntimeError(
                    f"{self.format_id} produced undeclared output shape {shape}"
                )
            count = (
                1
                if self.subrecord_count is None
                else int(self.subrecord_count(entry))
            )
            if count < 0:
                raise RuntimeError(
                    f"{self.format_id} produced a negative subrecord count"
                )
            if count == 0:
                continue
            reader.pending = _PendingEntry(
                raw_entry,
                entry_index,
                shape_code,
                0,
                count,
            )
            envelope = self._next_pending_envelope(cursor, reader, descriptor)
            if envelope is not None:
                return envelope, cursor
        return None, cursor

    def _next_pending_envelope(
        self,
        cursor: SequentialCursor,
        reader: _ActiveReader,
        descriptor: SequentialFileDescriptor,
    ) -> RecordEnvelope | None:
        pending = reader.pending
        while pending is not None:
            subrecord = pending.next_subrecord
            pending.next_subrecord += 1
            if pending.next_subrecord == pending.subrecord_count:
                reader.pending = None
            base_address = (
                (pending.entry_index,)
                if self.subrecord_count is None
                else (pending.entry_index, subrecord)
            )
            address = (
                base_address
                if cursor.cycle == 0
                else (cursor.cycle, *base_address)
            )
            record_key = (
                "sequential",
                descriptor.file_key,
                self.format_id,
                address,
            )
            if sample_is_admitted(
                self.sample_rate,
                self.seed,
                self._augmentation_epoch(cursor),
                record_key,
            ):
                return RecordEnvelope(
                    source_id=descriptor.file_ordinal,
                    record_key=record_key,
                    shape_code=pending.shape_code,
                    payload=SequentialPayload(
                        pending.raw_entry,
                        None if self.subrecord_count is None else subrecord,
                        self._augmentation_epoch(cursor),
                    ),
                    resident_bytes=(
                        len(pending.raw_entry)
                        + SEQUENTIAL_ENVELOPE_OVERHEAD_BYTES
                    ),
                )
            pending = reader.pending
        return None

    def materialize_batch(self, envelopes) -> dict:
        samples = []
        for envelope in envelopes:
            try:
                descriptor = self.descriptors[envelope.source_id]
            except IndexError as exc:
                raise RuntimeError("sequential envelope source is out of range") from exc
            key_address = envelope.record_key[3]
            base_length = 1 if self.subrecord_count is None else 2
            cycle = 0 if len(key_address) == base_length else key_address[0]
            logical_address = (
                key_address
                if cycle == 0
                else key_address[1:]
            )
            entry_index = logical_address[0]
            subrecord = envelope.payload.subrecord
            if (subrecord is None) != (self.subrecord_count is None):
                raise RuntimeError("sequential payload subrecord mode is inconsistent")
            base_address = (
                (entry_index,)
                if subrecord is None
                else (entry_index, subrecord)
            )
            address = base_address if cycle == 0 else (cycle, *base_address)
            expected_key = (
                "sequential",
                descriptor.file_key,
                self.format_id,
                address,
            )
            if envelope.record_key != expected_key:
                raise RuntimeError("sequential envelope identity is inconsistent")
            raw_entry = envelope.payload.raw_entry
            if self.decode_raw_entry is not None:
                entry = self.decode_raw_entry(raw_entry)
            else:
                stream = io.BytesIO(raw_entry)
                entry = self.read_entry(stream)
                if stream.tell() != len(raw_entry):
                    raise RuntimeError("sequential payload contains trailing entry bytes")
            sample = (
                self.materialize_entry(
                    entry,
                    envelope.record_key,
                    envelope.payload.augmentation_epoch,
                )
                if self.subrecord_count is None
                else self.materialize_entry(
                    entry,
                    envelope.record_key,
                    envelope.payload.augmentation_epoch,
                    subrecord,
                )
            )
            if sample is None:
                raise RuntimeError("admitted sequential entry did not materialize")
            samples.append(sample)
        return collate_sample_dicts(samples, validate_core_fields=True)

    def update_record_key_digest(self, digest, record_key: tuple) -> None:
        try:
            namespace, file_key, format_id, address = record_key
            base_length = 1 if self.subrecord_count is None else 2
            if len(address) == base_length:
                cycle, logical_address = 0, address
            else:
                cycle, logical_address = address[0], address[1:]
            entry_index = logical_address[0]
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError("malformed sequential record key") from exc
        if (
            namespace != "sequential"
            or format_id != self.format_id
            or not isinstance(file_key, bytes)
            or len(file_key) != 32
            or type(entry_index) is not int
            or entry_index < 0
            or type(cycle) is not int
            or cycle < 0
            or len(address) not in {base_length, base_length + 1}
            or any(type(value) is not int or value < 0 for value in address)
        ):
            raise ValueError("malformed sequential record key")
        digest.update(self._record_key_prefix)
        digest.update(file_key)
        digest.update(struct.pack("<I", len(address)))
        for value in address:
            digest.update(struct.pack("<Q", value))

    @staticmethod
    def save_payload(payload: SequentialPayload) -> dict:
        return {
            "raw_entry": payload.raw_entry,
            "subrecord": payload.subrecord,
            "augmentation_epoch": payload.augmentation_epoch,
        }

    def restore_payload(self, state: dict) -> SequentialPayload:
        try:
            raw_entry = state["raw_entry"]
        except (KeyError, TypeError) as exc:
            raise ValueError("malformed sequential payload state") from exc
        if not isinstance(raw_entry, bytes) or not raw_entry:
            raise ValueError("sequential payload state must contain nonempty bytes")
        subrecord = state.get("subrecord")
        if subrecord is not None and (type(subrecord) is not int or subrecord < 0):
            raise ValueError("sequential payload has an invalid subrecord")
        if (subrecord is None) != (self.subrecord_count is None):
            raise ValueError("sequential payload subrecord mode changed")
        try:
            augmentation_epoch = int(state["augmentation_epoch"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("sequential payload augmentation epoch is invalid") from exc
        if not 0 <= augmentation_epoch < 1 << 63:
            raise ValueError("sequential payload augmentation epoch is out of range")
        return SequentialPayload(raw_entry, subrecord, augmentation_epoch)

    def save_cursor(self, cursor: SequentialCursor) -> dict:
        if not self.capabilities.resumable:
            raise RuntimeError("exact resume is unsupported for continuous LZ4 streams")
        return {
            "schema": SEQUENTIAL_SOURCE_SCHEMA,
            "epoch": cursor.epoch,
            "cycle": cursor.cycle,
            "rank": cursor.rank,
            "file_order": list(cursor.file_order),
            "next_file_position": cursor.next_file_position,
            "turn": cursor.turn,
            "quantum_used": cursor.quantum_used,
            "active": [
                {
                    "descriptor_index": reader.descriptor_index,
                    "entry_index": reader.entry_index,
                    "byte_offset": int(reader.stream.tell()),
                    "pending": (
                        None
                        if reader.pending is None
                        else {
                            "raw_entry": reader.pending.raw_entry,
                            "entry_index": reader.pending.entry_index,
                            "shape_code": reader.pending.shape_code,
                            "next_subrecord": reader.pending.next_subrecord,
                            "subrecord_count": reader.pending.subrecord_count,
                        }
                    ),
                }
                for reader in cursor.active
            ],
        }

    def restore_cursor(self, state: dict) -> SequentialCursor:
        if not self.capabilities.resumable:
            raise RuntimeError("exact resume is unsupported for continuous LZ4 streams")
        try:
            if state["schema"] != SEQUENTIAL_SOURCE_SCHEMA:
                raise ValueError("sequential cursor schema changed")
            epoch = int(state["epoch"])
            cycle = int(state["cycle"])
            rank = int(state["rank"])
            order = tuple(int(value) for value in state["file_order"])
            next_file_position = int(state["next_file_position"])
            turn = int(state["turn"])
            quantum_used = int(state["quantum_used"])
            active_state = tuple(state["active"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed sequential cursor") from exc
        initial = self.start_cycle(epoch, cycle, rank)
        expected_order = initial.file_order
        self.close_cursor(initial)
        if order != expected_order:
            raise ValueError("sequential file order changed")
        if not 0 <= next_file_position <= len(order):
            raise ValueError("sequential next file position is out of range")
        if not active_state:
            if turn != 0 or quantum_used != 0 or next_file_position != len(order):
                raise ValueError("terminal sequential cursor is inconsistent")
            return SequentialCursor(
                epoch,
                cycle,
                rank,
                order,
                next_file_position,
                [],
                0,
                0,
            )
        if not 0 <= turn < len(active_state):
            raise ValueError("sequential reader turn is out of range")
        if not 0 <= quantum_used < self.read_quantum:
            raise ValueError("sequential quantum cursor is out of range")
        if len(active_state) > self.active_streams:
            raise ValueError("sequential cursor has too many active readers")
        scheduled = set(order[:next_file_position])
        descriptor_indices = [int(item["descriptor_index"]) for item in active_state]
        if (
            len(descriptor_indices) != len(set(descriptor_indices))
            or any(index not in scheduled for index in descriptor_indices)
        ):
            raise ValueError("sequential active reader assignment is inconsistent")
        active = []
        try:
            for item in active_state:
                descriptor_index = int(item["descriptor_index"])
                entry_index = int(item["entry_index"])
                byte_offset = int(item["byte_offset"])
                pending_state = item.get("pending")
                if not 0 <= descriptor_index < len(self.descriptors):
                    raise ValueError("sequential descriptor index is out of range")
                if entry_index < 0 or byte_offset < 0:
                    raise ValueError("sequential reader cursor is negative")
                reader = self._open_reader(descriptor_index, entry_index)
                active.append(reader)
                if pending_state is not None:
                    try:
                        raw_entry = pending_state["raw_entry"]
                        pending_entry_index = int(pending_state["entry_index"])
                        shape_code = int(pending_state["shape_code"])
                        next_subrecord = int(pending_state["next_subrecord"])
                        subrecord_count = int(pending_state["subrecord_count"])
                    except (KeyError, TypeError, ValueError) as exc:
                        raise ValueError(
                            "malformed pending sequential entry"
                        ) from exc
                    if (
                        self.subrecord_count is None
                        or not isinstance(raw_entry, bytes)
                        or not raw_entry
                        or pending_entry_index != entry_index - 1
                        or not 0 <= shape_code < len(self.shape_codes)
                        or not 0 < next_subrecord < subrecord_count
                    ):
                        raise ValueError("invalid pending sequential entry")
                    if self.decode_raw_entry is not None:
                        pending_entry = self.decode_raw_entry(raw_entry)
                    else:
                        pending_stream = io.BytesIO(raw_entry)
                        pending_entry = self.read_entry(pending_stream)
                        if pending_stream.tell() != len(raw_entry):
                            raise ValueError(
                                "pending sequential entry has trailing bytes"
                            )
                    expected_shape = self.shape_of(
                        raw_entry if self.read_raw_entry is not None else pending_entry
                    )
                    expected_shape_code = self.shape_codes.get(
                        None if expected_shape is None else tuple(expected_shape)
                    )
                    expected_subrecords = int(
                        self.subrecord_count(
                            raw_entry
                            if self.read_raw_entry is not None
                            else pending_entry
                        )
                    )
                    if (
                        expected_shape_code != shape_code
                        or expected_subrecords != subrecord_count
                    ):
                        raise ValueError(
                            "pending sequential entry semantics changed"
                        )
                    reader.pending = _PendingEntry(
                        raw_entry,
                        pending_entry_index,
                        shape_code,
                        next_subrecord,
                        subrecord_count,
                    )
                reader.stream.seek(byte_offset)
                if int(reader.stream.tell()) != byte_offset:
                    raise ValueError("sequential reader failed to seek to its cursor")
        except BaseException:
            for reader in active:
                reader.stream.close()
            raise
        self._active_epoch = epoch
        return SequentialCursor(
            epoch,
            cycle,
            rank,
            order,
            next_file_position,
            active,
            turn,
            quantum_used,
        )

    @staticmethod
    def close_cursor(cursor: SequentialCursor) -> None:
        for reader in cursor.active:
            reader.stream.close()
        cursor.active.clear()
