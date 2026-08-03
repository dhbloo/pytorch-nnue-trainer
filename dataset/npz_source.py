from __future__ import annotations

import hashlib
import struct
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .core import (
    canonical_pipeline_state_bytes,
    deterministic_permutation,
    sample_is_admitted,
)
from .packed import PackedEnvelopeBatch, PackedRecordBlock
from .source import RecordEnvelope, SourceCapabilities
from .stream import collate_sample_dicts


DENSE_NPZ_SOURCE_SCHEMA = "dense-npz-source-v2"


def _validate_packed_batch_owner(source, batch: PackedEnvelopeBatch) -> None:
    expected = getattr(source, "_packed_batch_identity", None)
    if expected is None:
        expected = hashlib.sha256(
            canonical_pipeline_state_bytes(source.manifest_state())
        ).hexdigest()
        source._packed_batch_identity = expected
    if batch.identity != expected:
        raise RuntimeError("packed NPZ batch belongs to a different source manifest")


@dataclass(frozen=True, slots=True)
class DenseNpzFileDescriptor:
    path: str
    file_ordinal: int
    file_sha256: bytes
    logical_row_count: int
    global_row_begin: int
    board_size: tuple[int, int]
    shape_code: int


@dataclass(frozen=True, slots=True)
class _DenseEnvelope:
    """Compact active-window identity for a dense NPZ row."""

    descriptor: DenseNpzFileDescriptor
    payload: int
    cycle: int

    @property
    def source_id(self) -> int:
        return self.descriptor.file_ordinal

    @property
    def shape_code(self) -> int:
        return self.descriptor.shape_code

    @property
    def record_key(self):
        row = self.payload - self.descriptor.global_row_begin
        address = (row,) if self.cycle == 0 else (self.cycle, row)
        return ("root", self.descriptor.file_sha256, "npz-row", address)

    @property
    def resident_bytes(self) -> int:
        return 0


@dataclass(frozen=True, slots=True)
class DenseNpzCursor:
    epoch: int
    cycle: int
    rank: int
    file_order: tuple[int, ...]
    file_position: int
    row_position: int


@dataclass(frozen=True, slots=True)
class _DenseDecodeRef:
    path: str
    address: tuple[int]
    board_size: tuple[int, int]
    sample_key: tuple
    record_digest: bytes = b""


@dataclass(frozen=True, slots=True)
class _PackedDenseDecodeBatch:
    paths: tuple[str, ...]
    content_digests: tuple[bytes, ...]
    file_indices: np.ndarray
    rows: np.ndarray
    board_size: tuple[int, int]
    sample_keys: Sequence

    def __len__(self) -> int:
        return len(self.rows)


@dataclass(frozen=True, slots=True, eq=False)
class _PackedDenseSampleKeys(Sequence):
    """Lazily expose canonical row keys without retaining per-row tuples."""

    content_digests: tuple[bytes, ...]
    file_indices: np.ndarray
    rows: np.ndarray

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[position] for position in range(*index.indices(len(self))))
        position = int(index)
        if position < 0:
            position += len(self)
        if not 0 <= position < len(self):
            raise IndexError("packed dense sample-key index out of range")
        file_index = int(self.file_indices[position])
        return (
            "root",
            self.content_digests[file_index],
            "npz-row",
            (int(self.rows[position]),),
        )

    def __iter__(self):
        for file_index, row in zip(self.file_indices, self.rows):
            yield (
                "root",
                self.content_digests[int(file_index)],
                "npz-row",
                (int(row),),
            )

    def __eq__(self, other):
        if not isinstance(other, Sequence):
            return NotImplemented
        return len(self) == len(other) and all(
            left == right for left, right in zip(self, other)
        )


class DenseNpzSource:
    """Random-access NPZ rows backed by O(file-count) retained metadata."""

    thread_safe_materialization = True
    zero_resident_envelopes = True
    capabilities = SourceCapabilities(
        access_mode="random",
        known_length=True,
        exact_distributed_partition=True,
        resumable=True,
        deterministic=True,
    )

    def __init__(
        self,
        catalogs,
        decoder,
        *,
        seed: int,
        shuffle: bool,
        sample_rate: float,
    ):
        catalogs = tuple(catalogs)
        if not catalogs:
            raise ValueError("dense NPZ source requires at least one file catalog")
        if not 0.0 <= sample_rate <= 1.0:
            raise ValueError(f"sample_rate must be in [0, 1], got {sample_rate}")
        shapes = sorted(
            {
                tuple(int(value) for value in catalog["board_size"])
                for catalog in catalogs
            }
        )
        shape_codes = {shape: index for index, shape in enumerate(shapes)}
        self.shape_codes = shape_codes
        descriptors = []
        row_begin = 0
        file_ordinals = set()
        for catalog in catalogs:
            path = str(catalog["path"])
            file_ordinal = int(catalog["file_ordinal"])
            file_sha256 = bytes(catalog["file_sha256"])
            logical_row_count = int(catalog["logical_row_count"])
            board_size = tuple(int(value) for value in catalog["board_size"])
            if file_ordinal < 0 or file_ordinal in file_ordinals:
                raise ValueError(
                    "dense NPZ file ordinals must be unique and non-negative"
                )
            if len(file_sha256) != 32:
                raise ValueError("dense NPZ file digest must contain 32 bytes")
            if logical_row_count < 0:
                raise ValueError("dense NPZ logical row count must be non-negative")
            if board_size not in shape_codes:
                raise ValueError("dense NPZ board size is inconsistent")
            file_ordinals.add(file_ordinal)
            descriptors.append(
                DenseNpzFileDescriptor(
                    path=path,
                    file_ordinal=file_ordinal,
                    file_sha256=file_sha256,
                    logical_row_count=logical_row_count,
                    global_row_begin=row_begin,
                    board_size=board_size,
                    shape_code=shape_codes[board_size],
                )
            )
            row_begin += logical_row_count
        if row_begin >= 1 << 64:
            raise ValueError("dense NPZ source exceeds the uint64 record ID space")
        self.descriptors = tuple(descriptors)
        self._descriptors_by_ordinal = {
            descriptor.file_ordinal: descriptor for descriptor in self.descriptors
        }
        self.decoder = decoder
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.sample_rate = float(sample_rate)
        self.logical_row_count = row_begin
        self._row_begins = tuple(
            descriptor.global_row_begin for descriptor in self.descriptors
        )
        self._row_begins_array = np.asarray(self._row_begins, dtype=np.uint64)
        self._row_begins_array.flags.writeable = False
        self._row_counts_array = np.asarray(
            [descriptor.logical_row_count for descriptor in self.descriptors],
            dtype=np.uint64,
        )
        self._row_counts_array.flags.writeable = False
        self._descriptor_paths = tuple(
            descriptor.path for descriptor in self.descriptors
        )
        self._descriptor_digests = tuple(
            descriptor.file_sha256 for descriptor in self.descriptors
        )
        self._file_identity = hashlib.sha256(
            b"".join(descriptor.file_sha256 for descriptor in self.descriptors)
        ).digest()
        self._active_epoch = 0
        self._world_size = 1
        self._rank = 0
        self._rank_sharded = False

    def configure_distributed(
        self,
        world_size: int,
        rank: int,
        *,
        rank_sharded: bool,
    ) -> None:
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("invalid dense NPZ distributed identity")
        self._world_size = int(world_size)
        self._rank = int(rank)
        self._rank_sharded = bool(rank_sharded)

    def manifest_state(self) -> dict:
        return {
            "schema": DENSE_NPZ_SOURCE_SCHEMA,
            "decoder": self.decoder.signature_state(),
            "shuffle": self.shuffle,
            "sample_rate": self.sample_rate,
            "logical_row_count": self.logical_row_count,
            "files": [
                {
                    "path": descriptor.path,
                    "file_ordinal": descriptor.file_ordinal,
                    "file_sha256": descriptor.file_sha256.hex(),
                    "logical_row_count": descriptor.logical_row_count,
                    "global_row_begin": descriptor.global_row_begin,
                    "board_size": list(descriptor.board_size),
                    "shape_code": descriptor.shape_code,
                }
                for descriptor in self.descriptors
            ],
        }

    def start_epoch(self, epoch: int, rank: int) -> DenseNpzCursor:
        return self.start_cycle(epoch, 0, rank)

    def start_cycle(self, epoch: int, cycle: int, rank: int) -> DenseNpzCursor:
        if epoch < 0 or cycle < 0:
            raise ValueError("dense NPZ epoch/cycle must be non-negative")
        if rank != self._rank or not 0 <= rank < self._world_size:
            raise ValueError("dense NPZ rank differs from its configuration")
        order = tuple(range(len(self.descriptors)))
        if self.shuffle:
            order = tuple(
                deterministic_permutation(
                    len(order),
                    self.seed,
                    "file_order",
                    (
                        (epoch, self._file_identity)
                        if cycle == 0
                        else (epoch, cycle, self._file_identity)
                    ),
                )
            )
        self._active_epoch = epoch
        if hasattr(self.decoder, "set_epoch"):
            self.decoder.set_epoch(epoch)
        cursor_rank = rank if self._rank_sharded else 0
        return DenseNpzCursor(epoch, cycle, cursor_rank, order, 0, 0)

    @staticmethod
    def _record_key(
        descriptor: DenseNpzFileDescriptor,
        row: int,
        cycle: int,
    ):
        address = (row,) if cycle == 0 else (cycle, row)
        return ("root", descriptor.file_sha256, "npz-row", address)

    @staticmethod
    def _record_cycle(record_key) -> int:
        address = record_key[3]
        return 0 if len(address) == 1 else address[0]

    def next_envelope(
        self, cursor: DenseNpzCursor
    ) -> tuple[RecordEnvelope | None, DenseNpzCursor]:
        envelopes, cursor = self.next_envelopes(cursor, 1)
        return (envelopes[0] if envelopes else None), cursor

    def next_envelopes(
        self,
        cursor: DenseNpzCursor,
        limit: int,
    ) -> tuple[tuple[RecordEnvelope, ...], DenseNpzCursor]:
        if type(limit) is not int or limit <= 0:
            raise ValueError("dense NPZ chunk limit must be positive")
        file_position = cursor.file_position
        row_position = cursor.row_position
        envelopes = []
        while (
            len(envelopes) < limit
            and file_position < len(cursor.file_order)
        ):
            descriptor = self.descriptors[cursor.file_order[file_position]]
            if self._rank_sharded and row_position < descriptor.logical_row_count:
                global_row = descriptor.global_row_begin + row_position
                row_position += (cursor.rank - global_row) % self._world_size
            if row_position >= descriptor.logical_row_count:
                file_position += 1
                row_position = 0
                continue
            row = row_position
            row_position += self._world_size if self._rank_sharded else 1
            if self.sample_rate < 1.0 and not sample_is_admitted(
                self.sample_rate,
                self.seed,
                cursor.epoch,
                self._record_key(descriptor, row, cursor.cycle),
            ):
                continue
            record_id = descriptor.global_row_begin + row
            envelopes.append(
                _DenseEnvelope(descriptor, record_id, cursor.cycle)
            )
        while file_position < len(cursor.file_order) and row_position >= self.descriptors[
            cursor.file_order[file_position]
        ].logical_row_count:
            file_position += 1
            row_position = 0
        if file_position == len(cursor.file_order):
            row_position = 0
        return tuple(envelopes), DenseNpzCursor(
            cursor.epoch,
            cursor.cycle,
            cursor.rank,
            cursor.file_order,
            file_position,
            row_position,
        )

    def next_packed_records(
        self,
        cursor: DenseNpzCursor,
        limit: int,
    ) -> tuple[PackedRecordBlock, DenseNpzCursor]:
        if type(limit) is not int or limit <= 0:
            raise ValueError("dense NPZ chunk limit must be positive")
        file_position = cursor.file_position
        row_position = cursor.row_position
        parts = []
        count = 0
        while count < limit and file_position < len(cursor.file_order):
            descriptor = self.descriptors[cursor.file_order[file_position]]
            if row_position >= descriptor.logical_row_count:
                file_position += 1
                row_position = 0
                continue
            take = min(limit - count, descriptor.logical_row_count - row_position)
            if self.sample_rate == 1.0:
                begin = descriptor.global_row_begin + row_position
                parts.append(np.arange(begin, begin + take, dtype=np.uint64))
                row_position += take
                count += take
                continue
            admitted = []
            stop = row_position + take
            while row_position < stop:
                row = row_position
                row_position += 1
                if sample_is_admitted(
                    self.sample_rate,
                    self.seed,
                    cursor.epoch,
                    self._record_key(descriptor, row, cursor.cycle),
                ):
                    admitted.append(descriptor.global_row_begin + row)
            if admitted:
                values = np.asarray(admitted, dtype=np.uint64)
                parts.append(values)
                count += len(values)
        while file_position < len(cursor.file_order) and row_position >= self.descriptors[
            cursor.file_order[file_position]
        ].logical_row_count:
            file_position += 1
            row_position = 0
        if file_position == len(cursor.file_order):
            row_position = 0
        record_ids = (
            np.empty(0, dtype=np.uint64)
            if not parts
            else parts[0]
            if len(parts) == 1
            else np.concatenate(parts)
        )
        return PackedRecordBlock(
            record_ids,
            default_cycle=cursor.cycle,
        ), DenseNpzCursor(
            cursor.epoch,
            cursor.cycle,
            cursor.rank,
            cursor.file_order,
            file_position,
            row_position,
        )

    def envelopes_from_record_ids(self, record_ids) -> tuple[_DenseEnvelope, ...]:
        envelopes = []
        for record_id in record_ids:
            descriptor, _ = self._resolve(int(record_id))
            envelopes.append(_DenseEnvelope(descriptor, int(record_id), 0))
        return tuple(envelopes)

    def _resolve(self, record_id: int) -> tuple[DenseNpzFileDescriptor, int]:
        record_id = int(record_id)
        descriptor_index = bisect_right(self._row_begins, record_id) - 1
        if descriptor_index < 0:
            raise ValueError(f"dense NPZ record ID {record_id} is out of range")
        descriptor = self.descriptors[descriptor_index]
        row = record_id - descriptor.global_row_begin
        if not 0 <= row < descriptor.logical_row_count:
            raise ValueError(f"dense NPZ record ID {record_id} is out of range")
        return descriptor, row

    def _decode_refs(self, envelopes):
        refs = []
        if isinstance(envelopes, PackedEnvelopeBatch):
            _validate_packed_batch_owner(self, envelopes)
            for record_id in envelopes.record_ids:
                descriptor, row = self._resolve(int(record_id))
                record_key = self._record_key(descriptor, row, 0)
                refs.append(
                    _DenseDecodeRef(
                        descriptor.path,
                        (row,),
                        descriptor.board_size,
                        record_key,
                        struct.pack("<Q", row),
                    )
                )
            return tuple(refs)
        for envelope in envelopes:
            if isinstance(envelope, _DenseEnvelope):
                descriptor = envelope.descriptor
                row = envelope.payload - descriptor.global_row_begin
                if (
                    self._descriptors_by_ordinal.get(descriptor.file_ordinal)
                    != descriptor
                    or not 0 <= row < descriptor.logical_row_count
                ):
                    raise RuntimeError("dense NPZ envelope identity is inconsistent")
                record_key = envelope.record_key
                refs.append(
                    _DenseDecodeRef(
                        descriptor.path,
                        (row,),
                        descriptor.board_size,
                        record_key,
                        struct.pack("<Q", row),
                    )
                )
                continue
            descriptor, row = self._resolve(envelope.payload)
            if (
                envelope.source_id != descriptor.file_ordinal
                or envelope.shape_code != descriptor.shape_code
                or envelope.record_key
                != self._record_key(
                    descriptor,
                    row,
                    self._record_cycle(envelope.record_key),
                )
            ):
                raise RuntimeError("dense NPZ envelope identity is inconsistent")
            refs.append(
                _DenseDecodeRef(
                    descriptor.path,
                    (row,),
                    descriptor.board_size,
                    envelope.record_key,
                    struct.pack("<Q", row),
                )
            )
        return tuple(refs)

    def _packed_decode_batch(self, envelopes):
        _validate_packed_batch_owner(self, envelopes)
        record_ids = envelopes.record_ids
        file_indices = np.searchsorted(
            self._row_begins_array, record_ids, side="right"
        ).astype(np.intp, copy=False) - 1
        if np.any(file_indices < 0):
            raise RuntimeError("packed dense NPZ batch contains an invalid record ID")
        rows_u64 = record_ids - self._row_begins_array[file_indices]
        counts = self._row_counts_array[file_indices]
        if np.any(rows_u64 >= counts):
            raise RuntimeError("packed dense NPZ batch contains an invalid record ID")
        rows = rows_u64.astype(np.intp, copy=False)
        file_indices = np.ascontiguousarray(file_indices)
        rows = np.ascontiguousarray(rows)
        file_indices.setflags(write=False)
        rows.setflags(write=False)
        sample_keys = _PackedDenseSampleKeys(
            self._descriptor_digests,
            file_indices,
            rows,
        )
        return _PackedDenseDecodeBatch(
            paths=self._descriptor_paths,
            content_digests=self._descriptor_digests,
            file_indices=file_indices,
            rows=rows,
            board_size=self.descriptors[int(file_indices[0])].board_size,
            sample_keys=sample_keys,
        )

    def _materialize_refs(self, refs) -> dict:
        decoded = (
            self.decoder.decode_batch(refs)
            if hasattr(self.decoder, "decode_batch")
            else None
        )
        if decoded is not None:
            return decoded
        return collate_sample_dicts(
            [self.decoder.decode_one(ref) for ref in refs],
            validate_core_fields=True,
        )

    def materialize_batch(self, envelopes) -> dict:
        return self._materialize_refs(self._decode_refs(envelopes))

    def materialize_batch_with_keys(self, envelopes):
        if isinstance(envelopes, PackedEnvelopeBatch) and hasattr(
            self.decoder, "decode_packed_batch"
        ):
            packed = self._packed_decode_batch(envelopes)
            decoded = self.decoder.decode_packed_batch(packed)
            if decoded is not None:
                return decoded, packed.sample_keys
        refs = self._decode_refs(envelopes)
        return self._materialize_refs(refs), tuple(ref.sample_key for ref in refs)

    def materialize_batches(self, envelope_batches):
        ref_batches = tuple(
            self._decode_refs(envelopes) for envelopes in envelope_batches
        )
        decoded_batches = (
            self.decoder.decode_batches(ref_batches)
            if hasattr(self.decoder, "decode_batches")
            else None
        )
        if decoded_batches is not None:
            if len(decoded_batches) != len(ref_batches):
                raise RuntimeError("dense NPZ decoder changed the batch count")
            return tuple(decoded_batches)
        return tuple(self._materialize_refs(refs) for refs in ref_batches)

    def materialize_batches_with_keys(self, envelope_batches):
        if envelope_batches and all(
            isinstance(envelopes, PackedEnvelopeBatch)
            for envelopes in envelope_batches
        ) and hasattr(self.decoder, "decode_packed_batch"):
            requests = tuple(
                self._packed_decode_batch(envelopes)
                for envelopes in envelope_batches
            )
            combined_file_indices = np.concatenate(
                [request.file_indices for request in requests]
            )
            combined_rows = np.concatenate([request.rows for request in requests])
            combined_file_indices.setflags(write=False)
            combined_rows.setflags(write=False)
            combined = _PackedDenseDecodeBatch(
                paths=requests[0].paths,
                content_digests=requests[0].content_digests,
                file_indices=combined_file_indices,
                rows=combined_rows,
                board_size=requests[0].board_size,
                sample_keys=_PackedDenseSampleKeys(
                    requests[0].content_digests,
                    combined_file_indices,
                    combined_rows,
                ),
            )
            decoded = self.decoder.decode_packed_batch(combined)
            if decoded is not None:
                batches = []
                offset = 0
                for request in requests:
                    stop = offset + len(request)
                    batches.append(
                        (
                            {
                                key: value[offset:stop]
                                for key, value in decoded.items()
                            },
                            request.sample_keys,
                        )
                    )
                    offset = stop
                return tuple(batches)
        ref_batches = tuple(
            self._decode_refs(envelopes) for envelopes in envelope_batches
        )
        decoded_batches = (
            self.decoder.decode_batches(ref_batches)
            if hasattr(self.decoder, "decode_batches")
            else None
        )
        if decoded_batches is None:
            decoded_batches = tuple(
                self._materialize_refs(refs) for refs in ref_batches
            )
        elif len(decoded_batches) != len(ref_batches):
            raise RuntimeError("dense NPZ decoder changed the batch count")
        return tuple(
            (data, tuple(ref.sample_key for ref in refs))
            for data, refs in zip(decoded_batches, ref_batches)
        )

    def sample_keys_for_batch(self, envelopes) -> tuple:
        if not isinstance(envelopes, PackedEnvelopeBatch):
            return tuple(envelope.record_key for envelope in envelopes)
        _validate_packed_batch_owner(self, envelopes)
        return tuple(
            self._record_key(*self._resolve(int(record_id)), 0)
            for record_id in envelopes.record_ids
        )

    @staticmethod
    def update_record_key_digest(digest, record_key: tuple) -> None:
        try:
            namespace, file_digest, kind, address = record_key
            if len(address) == 1:
                cycle, row = 0, address[0]
            else:
                cycle, row = address
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError("malformed dense NPZ record key") from exc
        if (
            namespace != "root"
            or kind != "npz-row"
            or not isinstance(file_digest, bytes)
            or len(file_digest) != 32
            or len(address) not in {1, 2}
            or type(cycle) is not int
            or cycle < 0
            or type(row) is not int
            or row < 0
        ):
            raise ValueError("malformed dense NPZ record key")
        digest.update(b"NNUE-dense-npz-record-key-v2\0")
        digest.update(file_digest)
        if len(address) == 2:
            digest.update(struct.pack("<Q", cycle))
        digest.update(struct.pack("<Q", row))

    @staticmethod
    def update_batch_record_digest(digest, envelopes, is_real) -> None:
        if isinstance(envelopes, PackedEnvelopeBatch):
            digest.update(b"NNUE-dense-npz-batch-ids-v2\0")
            digest.update(
                envelopes.record_ids.astype("<u8", copy=False).tobytes()
            )
            digest.update(bytes(is_real))
            return

        def record_cycle(envelope):
            return (
                envelope.cycle
                if isinstance(envelope, _DenseEnvelope)
                else DenseNpzSource._record_cycle(envelope.record_key)
            )

        cycles = tuple(record_cycle(envelope) for envelope in envelopes)
        if all(cycle == 0 for cycle in cycles):
            record_ids = np.fromiter(
                (int(envelope.payload) for envelope in envelopes),
                dtype="<u8",
                count=len(envelopes),
            )
            digest.update(b"NNUE-dense-npz-batch-ids-v2\0")
        else:
            record_ids = np.fromiter(
                (
                    value
                    for envelope, cycle in zip(envelopes, cycles)
                    for value in (
                        cycle,
                        int(envelope.payload),
                    )
                ),
                dtype="<u8",
                count=len(envelopes) * 2,
            )
            digest.update(b"NNUE-dense-npz-batch-cycle-ids-v3\0")
        digest.update(record_ids.tobytes())
        digest.update(bytes(is_real))

    def save_cursor(self, cursor: DenseNpzCursor) -> dict:
        return {
            "schema": DENSE_NPZ_SOURCE_SCHEMA,
            "epoch": cursor.epoch,
            "cycle": cursor.cycle,
            "rank": cursor.rank,
            "file_order": list(cursor.file_order),
            "file_position": cursor.file_position,
            "row_position": cursor.row_position,
        }

    def restore_cursor(self, state: dict) -> DenseNpzCursor:
        try:
            if state["schema"] != DENSE_NPZ_SOURCE_SCHEMA:
                raise ValueError("dense NPZ cursor schema changed")
            epoch = int(state["epoch"])
            cycle = int(state.get("cycle", 0))
            rank = int(state.get("rank", self._rank))
            order = tuple(int(value) for value in state["file_order"])
            file_position = int(state["file_position"])
            row_position = int(state["row_position"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed dense NPZ cursor") from exc
        expected_rank = self._rank if self._rank_sharded else 0
        if rank != expected_rank:
            raise ValueError("dense NPZ cursor rank changed")
        expected = self.start_cycle(epoch, cycle, self._rank).file_order
        if order != expected:
            raise ValueError("dense NPZ file order changed")
        if not 0 <= file_position <= len(order):
            raise ValueError("dense NPZ file position is out of range")
        if file_position == len(order):
            if row_position != 0:
                raise ValueError("terminal dense NPZ row position must be zero")
        elif not 0 <= row_position <= self.descriptors[
            order[file_position]
        ].logical_row_count:
            raise ValueError("dense NPZ row position is out of range")
        return DenseNpzCursor(
            epoch,
            cycle,
            rank,
            order,
            file_position,
            row_position,
        )


INDEXED_NPZ_SOURCE_SCHEMA = "indexed-npz-source-v2"


@dataclass(frozen=True, slots=True)
class IndexedNpzFileDescriptor:
    path: str
    file_ordinal: int
    file_sha256: bytes
    logical_row_count: int
    global_row_begin: int
    row_indices: np.ndarray | None
    uniform_shape_code: int | None
    row_shape_codes: np.ndarray | None


class IndexedNpzSource:
    """Compact NPZ row/shape catalogs for filtered or mixed-shape files."""

    capabilities = SourceCapabilities(
        access_mode="random",
        known_length=True,
        exact_distributed_partition=True,
        resumable=True,
        deterministic=True,
    )

    def __init__(self, catalogs, decoder, *, seed, shuffle, sample_rate):
        catalogs = tuple(catalogs)
        if not catalogs:
            raise ValueError("indexed NPZ source requires at least one catalog")
        if not 0.0 <= sample_rate <= 1.0:
            raise ValueError(f"sample_rate must be in [0, 1], got {sample_rate}")
        shape_set = set()
        for catalog in catalogs:
            uniform_shape = catalog.get("uniform_board_size")
            if uniform_shape is not None:
                shape_set.add(tuple(int(value) for value in uniform_shape))
            else:
                shape_set.update(
                    tuple(int(value) for value in shape)
                    for shape in np.asarray(catalog["board_sizes"]).reshape(-1, 2)
                )
        shapes = sorted(shape_set)
        if not shapes:
            raise ValueError("indexed NPZ catalogs contain no logical rows")
        if any(len(shape) != 2 or min(shape) <= 0 for shape in shapes):
            raise ValueError("indexed NPZ board shapes must contain two positive values")
        if len(shapes) > 1 << 16:
            raise ValueError("indexed NPZ source exceeds the uint16 shape-code space")
        self.shape_codes = {
            shape: code for code, shape in enumerate(shapes)
        }
        self._code_shapes = tuple(shapes)
        descriptors = []
        row_begin = 0
        metadata_bytes = 0
        file_ordinals = set()
        for catalog in catalogs:
            physical_row_count = int(catalog["physical_row_count"])
            if not 0 <= physical_row_count < 1 << 64:
                raise ValueError("indexed NPZ physical row count is out of range")
            logical_row_count = int(catalog.get("logical_row_count", -1))
            raw_indices = catalog["row_indices"]
            if raw_indices is None:
                if logical_row_count < 0:
                    logical_row_count = physical_row_count
                if logical_row_count != physical_row_count:
                    raise ValueError(
                        "implicit indexed NPZ rows must cover the physical rows"
                    )
                indices = None
            else:
                indices = np.asarray(raw_indices, dtype=np.int64)
                if indices.ndim != 1:
                    raise ValueError("indexed NPZ row indices must be one-dimensional")
                if logical_row_count < 0:
                    logical_row_count = len(indices)
                if logical_row_count != len(indices):
                    raise ValueError("indexed NPZ logical row count is inconsistent")
            if logical_row_count < 0 or row_begin + logical_row_count >= 1 << 64:
                raise ValueError("indexed NPZ logical row count is out of range")
            if logical_row_count == 0:
                continue
            compact_indices = None
            if indices is not None:
                if np.any(indices < 0) or np.any(indices[1:] <= indices[:-1]):
                    raise ValueError(
                        "indexed NPZ physical rows must increase strictly"
                    )
                if int(indices[-1]) >= physical_row_count:
                    raise ValueError("indexed NPZ physical row is out of range")
            identity_indices = (
                indices is None
                or (
                    logical_row_count == physical_row_count
                    and int(indices[0]) == 0
                    and int(indices[-1]) == logical_row_count - 1
                )
            )
            if not identity_indices:
                index_dtype = (
                    np.uint32 if physical_row_count <= 1 << 32 else np.uint64
                )
                compact_indices = np.ascontiguousarray(indices, dtype=index_dtype)
                compact_indices.flags.writeable = False
                metadata_bytes += compact_indices.nbytes
            row_shape_codes = None
            uniform_shape = catalog.get("uniform_board_size")
            if uniform_shape is not None:
                uniform_shape_code = self.shape_codes[
                    tuple(int(value) for value in uniform_shape)
                ]
            else:
                board_sizes = np.asarray(catalog["board_sizes"], dtype=np.int64)
                if board_sizes.shape != (logical_row_count, 2):
                    raise ValueError(
                        "indexed NPZ catalog row/shape dimensions disagree"
                    )
                code_dtype = np.uint8 if len(self.shape_codes) <= 256 else np.uint16
                codes = np.fromiter(
                    (self.shape_codes[tuple(shape)] for shape in board_sizes),
                    dtype=code_dtype,
                    count=logical_row_count,
                )
                uniform_shape_code = (
                    int(codes[0]) if np.all(codes == codes[0]) else None
                )
                if uniform_shape_code is None:
                    row_shape_codes = np.ascontiguousarray(codes)
                    row_shape_codes.flags.writeable = False
                    metadata_bytes += row_shape_codes.nbytes
            file_ordinal = int(catalog["file_ordinal"])
            if file_ordinal < 0 or file_ordinal in file_ordinals:
                raise ValueError("indexed NPZ file ordinals must be unique and non-negative")
            file_ordinals.add(file_ordinal)
            file_sha256 = bytes(catalog["file_sha256"])
            if len(file_sha256) != 32:
                raise ValueError("indexed NPZ file digest must contain 32 bytes")
            descriptors.append(
                IndexedNpzFileDescriptor(
                    path=str(catalog["path"]),
                    file_ordinal=file_ordinal,
                    file_sha256=file_sha256,
                    logical_row_count=logical_row_count,
                    global_row_begin=row_begin,
                    row_indices=compact_indices,
                    uniform_shape_code=uniform_shape_code,
                    row_shape_codes=row_shape_codes,
                )
            )
            row_begin += logical_row_count
        if not descriptors:
            raise ValueError("indexed NPZ source has no admitted files")
        self.descriptors = tuple(descriptors)
        self.decoder = decoder
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.sample_rate = float(sample_rate)
        self.logical_row_count = row_begin
        self.metadata_bytes = metadata_bytes
        self._row_begins = tuple(item.global_row_begin for item in self.descriptors)
        self._file_identity = hashlib.sha256(
            b"".join(item.file_sha256 for item in self.descriptors)
        ).digest()
        self._active_epoch = 0
        self._world_size = 1
        self._rank = 0
        self._rank_sharded = False

    @staticmethod
    def _physical_row(descriptor, logical_row):
        return (
            logical_row
            if descriptor.row_indices is None
            else int(descriptor.row_indices[logical_row])
        )

    @staticmethod
    def _shape_code(descriptor, logical_row):
        return (
            descriptor.uniform_shape_code
            if descriptor.row_shape_codes is None
            else int(descriptor.row_shape_codes[logical_row])
        )

    def manifest_state(self):
        return {
            "schema": INDEXED_NPZ_SOURCE_SCHEMA,
            "decoder": self.decoder.signature_state(),
            "shuffle": self.shuffle,
            "sample_rate": self.sample_rate,
            "logical_row_count": self.logical_row_count,
            "shape_codes": [
                [list(shape), code]
                for shape, code in sorted(self.shape_codes.items())
            ],
            "files": [
                {
                    "path": item.path,
                    "file_ordinal": item.file_ordinal,
                    "file_sha256": item.file_sha256.hex(),
                    "logical_row_count": item.logical_row_count,
                    "global_row_begin": item.global_row_begin,
                    "row_index_digest": (
                        None
                        if item.row_indices is None
                        else hashlib.sha256(item.row_indices.tobytes()).hexdigest()
                    ),
                    "uniform_shape_code": item.uniform_shape_code,
                    "row_shape_digest": (
                        None
                        if item.row_shape_codes is None
                        else hashlib.sha256(
                            item.row_shape_codes.tobytes()
                        ).hexdigest()
                    ),
                }
                for item in self.descriptors
            ],
        }

    def start_epoch(self, epoch, rank):
        return self.start_cycle(epoch, 0, rank)

    def start_cycle(self, epoch, cycle, rank):
        if type(epoch) is not int or epoch < 0 or cycle < 0:
            raise ValueError("indexed NPZ epoch/cycle must be non-negative")
        if rank != self._rank or not 0 <= rank < self._world_size:
            raise ValueError("indexed NPZ rank differs from its configuration")
        order = tuple(range(len(self.descriptors)))
        if self.shuffle:
            order = tuple(
                deterministic_permutation(
                    len(order),
                    self.seed,
                    "indexed_npz_file_order",
                    (
                        (epoch, self._file_identity)
                        if cycle == 0
                        else (epoch, cycle, self._file_identity)
                    ),
                )
            )
        self._active_epoch = epoch
        if hasattr(self.decoder, "set_epoch"):
            self.decoder.set_epoch(epoch)
        cursor_rank = rank if self._rank_sharded else 0
        return DenseNpzCursor(epoch, cycle, cursor_rank, order, 0, 0)

    @staticmethod
    def _record_key(descriptor, physical_row, cycle):
        address = (physical_row,) if cycle == 0 else (cycle, physical_row)
        return (
            "root",
            descriptor.file_sha256,
            "npz-row",
            address,
        )

    def next_envelope(self, cursor):
        envelopes, cursor = self.next_envelopes(cursor, 1)
        return (envelopes[0] if envelopes else None), cursor

    def next_envelopes(self, cursor, limit):
        if type(limit) is not int or limit <= 0:
            raise ValueError("indexed NPZ chunk limit must be positive")
        file_position = cursor.file_position
        row_position = cursor.row_position
        envelopes = []
        while len(envelopes) < limit and file_position < len(cursor.file_order):
            descriptor = self.descriptors[cursor.file_order[file_position]]
            if self._rank_sharded and row_position < descriptor.logical_row_count:
                global_row = descriptor.global_row_begin + row_position
                row_position += (cursor.rank - global_row) % self._world_size
            if row_position >= descriptor.logical_row_count:
                file_position += 1
                row_position = 0
                continue
            logical_row = row_position
            row_position += self._world_size if self._rank_sharded else 1
            physical_row = self._physical_row(descriptor, logical_row)
            record_key = self._record_key(
                descriptor,
                physical_row,
                cursor.cycle,
            )
            if not sample_is_admitted(
                self.sample_rate, self.seed, cursor.epoch, record_key
            ):
                continue
            envelopes.append(
                RecordEnvelope(
                    source_id=descriptor.file_ordinal,
                    record_key=record_key,
                    shape_code=self._shape_code(descriptor, logical_row),
                    payload=descriptor.global_row_begin + logical_row,
                )
            )
        while file_position < len(cursor.file_order) and row_position >= self.descriptors[
            cursor.file_order[file_position]
        ].logical_row_count:
            file_position += 1
            row_position = 0
        if file_position == len(cursor.file_order):
            row_position = 0
        return tuple(envelopes), DenseNpzCursor(
            cursor.epoch,
            cursor.cycle,
            cursor.rank,
            cursor.file_order,
            file_position,
            row_position,
        )

    def next_packed_records(self, cursor, limit):
        if type(limit) is not int or limit <= 0:
            raise ValueError("indexed NPZ chunk limit must be positive")
        file_position = cursor.file_position
        row_position = cursor.row_position
        parts = []
        count = 0
        while count < limit and file_position < len(cursor.file_order):
            descriptor = self.descriptors[cursor.file_order[file_position]]
            if row_position >= descriptor.logical_row_count:
                file_position += 1
                row_position = 0
                continue
            take = min(limit - count, descriptor.logical_row_count - row_position)
            if self.sample_rate == 1.0:
                begin = descriptor.global_row_begin + row_position
                parts.append(np.arange(begin, begin + take, dtype=np.uint64))
                row_position += take
                count += take
                continue
            admitted = []
            stop = row_position + take
            while row_position < stop:
                logical_row = row_position
                row_position += 1
                physical_row = self._physical_row(descriptor, logical_row)
                if sample_is_admitted(
                    self.sample_rate,
                    self.seed,
                    cursor.epoch,
                    self._record_key(descriptor, physical_row, cursor.cycle),
                ):
                    admitted.append(descriptor.global_row_begin + logical_row)
            if admitted:
                values = np.asarray(admitted, dtype=np.uint64)
                parts.append(values)
                count += len(values)
        while file_position < len(cursor.file_order) and row_position >= self.descriptors[
            cursor.file_order[file_position]
        ].logical_row_count:
            file_position += 1
            row_position = 0
        if file_position == len(cursor.file_order):
            row_position = 0
        record_ids = (
            np.empty(0, dtype=np.uint64)
            if not parts
            else parts[0]
            if len(parts) == 1
            else np.concatenate(parts)
        )
        return PackedRecordBlock(
            record_ids,
            default_cycle=cursor.cycle,
        ), DenseNpzCursor(
            cursor.epoch,
            cursor.cycle,
            cursor.rank,
            cursor.file_order,
            file_position,
            row_position,
        )

    def envelopes_from_record_ids(self, record_ids):
        envelopes = []
        for record_id in record_ids:
            descriptor, logical_row = self._resolve(int(record_id))
            physical_row = self._physical_row(descriptor, logical_row)
            envelopes.append(
                RecordEnvelope(
                    source_id=descriptor.file_ordinal,
                    record_key=self._record_key(descriptor, physical_row, 0),
                    shape_code=self._shape_code(descriptor, logical_row),
                    payload=int(record_id),
                )
            )
        return tuple(envelopes)

    def _resolve(self, record_id):
        record_id = int(record_id)
        descriptor_index = bisect_right(self._row_begins, record_id) - 1
        if descriptor_index < 0:
            raise ValueError("indexed NPZ record ID is out of range")
        descriptor = self.descriptors[descriptor_index]
        logical_row = record_id - descriptor.global_row_begin
        if not 0 <= logical_row < descriptor.logical_row_count:
            raise ValueError("indexed NPZ record ID is out of range")
        return descriptor, logical_row

    def materialize_batch(self, envelopes):
        refs = []
        if isinstance(envelopes, PackedEnvelopeBatch):
            _validate_packed_batch_owner(self, envelopes)
            record_ids = envelopes.record_ids
        else:
            record_ids = (envelope.payload for envelope in envelopes)
        envelope_iter = (
            None
            if isinstance(envelopes, PackedEnvelopeBatch)
            else iter(envelopes)
        )
        for record_id in record_ids:
            envelope = None if envelope_iter is None else next(envelope_iter)
            descriptor, logical_row = self._resolve(int(record_id))
            physical_row = self._physical_row(descriptor, logical_row)
            shape_code = self._shape_code(descriptor, logical_row)
            if envelope is not None and (
                envelope.source_id != descriptor.file_ordinal
                or envelope.shape_code != shape_code
                or envelope.record_key
                != self._record_key(
                    descriptor,
                    physical_row,
                    self._record_cycle(envelope.record_key),
                )
            ):
                raise RuntimeError("indexed NPZ envelope identity is inconsistent")
            board_size = self._code_shapes[shape_code]
            record_key = (
                self._record_key(descriptor, physical_row, 0)
                if envelope is None
                else envelope.record_key
            )
            refs.append(
                _DenseDecodeRef(
                    descriptor.path,
                    (physical_row,),
                    board_size,
                    record_key,
                    struct.pack("<Q", physical_row),
                )
            )
        decoded = (
            self.decoder.decode_batch(refs)
            if hasattr(self.decoder, "decode_batch")
            else None
        )
        return (
            decoded
            if decoded is not None
            else collate_sample_dicts(
                [self.decoder.decode_one(ref) for ref in refs],
                validate_core_fields=True,
            )
        )

    def sample_keys_for_batch(self, envelopes):
        if not isinstance(envelopes, PackedEnvelopeBatch):
            return tuple(envelope.record_key for envelope in envelopes)
        _validate_packed_batch_owner(self, envelopes)
        keys = []
        for record_id in envelopes.record_ids:
            descriptor, logical_row = self._resolve(int(record_id))
            keys.append(
                self._record_key(
                    descriptor,
                    self._physical_row(descriptor, logical_row),
                    0,
                )
            )
        return tuple(keys)

    def save_cursor(self, cursor):
        state = DenseNpzSource.save_cursor(self, cursor)
        state["schema"] = INDEXED_NPZ_SOURCE_SCHEMA
        return state

    def restore_cursor(self, state):
        copied = dict(state)
        if copied.get("schema") != INDEXED_NPZ_SOURCE_SCHEMA:
            raise ValueError("indexed NPZ cursor schema changed")
        copied["schema"] = DENSE_NPZ_SOURCE_SCHEMA
        return DenseNpzSource.restore_cursor(self, copied)

    update_record_key_digest = staticmethod(DenseNpzSource.update_record_key_digest)
    update_batch_record_digest = staticmethod(
        DenseNpzSource.update_batch_record_digest
    )
    _record_cycle = staticmethod(DenseNpzSource._record_cycle)
    configure_distributed = DenseNpzSource.configure_distributed
