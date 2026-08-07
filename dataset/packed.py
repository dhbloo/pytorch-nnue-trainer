from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
import operator

import numpy as np

from .core import deterministic_permutation, rng_u64
from .shuffle import RESERVOIR_ALGORITHM, ReservoirStats


PACKED_RESERVOIR_ALGORITHM = RESERVOIR_ALGORITHM + "-uint64"


def _readonly_uint64(values, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != np.dtype(np.uint64):
        raise TypeError(f"{name} must have dtype uint64")
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not array.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    array.flags.writeable = False
    return array


@dataclass(frozen=True, slots=True)
class PackedRecordBlock:
    """A source-native block of fixed-width logical record handles."""

    record_ids: np.ndarray
    default_cycle: int = 0
    cycles: np.ndarray | None = None
    shape_codes: np.ndarray | None = None
    source_ids: np.ndarray | None = None
    resident_sizes: np.ndarray | None = None

    def __post_init__(self) -> None:
        record_ids = _readonly_uint64(self.record_ids, name="record_ids")
        object.__setattr__(self, "record_ids", record_ids)
        if type(self.default_cycle) is not int or self.default_cycle < 0:
            raise ValueError("packed default_cycle must be a non-negative integer")
        expected = len(record_ids)
        optional_dtypes = {
            "cycles": (np.uint32, np.uint64),
            "shape_codes": (np.uint8, np.uint16, np.uint32),
            "source_ids": (np.uint8, np.uint16, np.uint32),
            "resident_sizes": (np.uint32, np.uint64),
        }
        for name, allowed in optional_dtypes.items():
            value = getattr(self, name)
            if value is None:
                continue
            array = np.asarray(value)
            if array.dtype.type not in allowed:
                choices = ", ".join(np.dtype(dtype).name for dtype in allowed)
                raise TypeError(f"packed {name} must use one of: {choices}")
            if array.ndim != 1 or len(array) != expected:
                raise ValueError(f"packed {name} must match record_ids length")
            if not array.flags.c_contiguous:
                raise ValueError(f"packed {name} must be C-contiguous")
            array.flags.writeable = False
            object.__setattr__(self, name, array)

    def __len__(self) -> int:
        return len(self.record_ids)


@dataclass(frozen=True, slots=True)
class PackedReservoirState:
    algorithm: str
    seed: int
    epoch: int
    stream_key: tuple
    capacity: int
    slots_le: bytes
    rng_counter: int
    offered: int
    emitted: int
    peak_occupancy: int
    closed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.slots_le, (bytes, bytearray, memoryview)):
            raise TypeError("packed reservoir slots must be bytes-like")
        object.__setattr__(self, "slots_le", bytes(self.slots_le))

    @property
    def occupancy(self) -> int:
        return len(self.slots_le) // 8

    def slot_array(self) -> np.ndarray:
        slots = np.frombuffer(self.slots_le, dtype="<u8")
        slots.flags.writeable = False
        return slots


@dataclass(frozen=True, slots=True)
class PackedReadyState:
    record_ids_le: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.record_ids_le, (bytes, bytearray, memoryview)):
            raise TypeError("packed ready IDs must be bytes-like")
        values = bytes(self.record_ids_le)
        if len(values) % 8:
            raise ValueError("packed ready state has a truncated ID buffer")
        object.__setattr__(self, "record_ids_le", values)

    def __len__(self) -> int:
        return len(self.record_ids_le) // 8

    def record_ids(self) -> np.ndarray:
        values = np.frombuffer(self.record_ids_le, dtype="<u8")
        values.flags.writeable = False
        return values


class PackedUInt64ReadyBuffer:
    """Chunked uint64 FIFO with no per-record Python objects."""

    def __init__(self) -> None:
        self._chunks: deque[np.ndarray] = deque()
        self._head = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def clear(self) -> None:
        self._chunks.clear()
        self._head = 0
        self._size = 0

    def extend(self, values) -> None:
        values = _readonly_uint64(values, name="ready record_ids")
        if len(values):
            self._chunks.append(values)
            self._size += len(values)

    def pop(self, count: int) -> np.ndarray:
        if type(count) is not int or not 0 <= count <= self._size:
            raise ValueError("packed ready pop count is out of range")
        if count == 0:
            values = np.empty(0, dtype=np.uint64)
            values.flags.writeable = False
            return values
        first = self._chunks[0]
        available = len(first) - self._head
        if count <= available:
            values = first[self._head : self._head + count]
            self._head += count
            if self._head == len(first):
                self._chunks.popleft()
                self._head = 0
        else:
            parts = []
            remaining = count
            while remaining:
                chunk = self._chunks[0]
                take = min(remaining, len(chunk) - self._head)
                parts.append(chunk[self._head : self._head + take])
                self._head += take
                remaining -= take
                if self._head == len(chunk):
                    self._chunks.popleft()
                    self._head = 0
            values = np.concatenate(parts)
        self._size -= count
        values.flags.writeable = False
        return values

    def state(self) -> PackedReadyState:
        if not self._size:
            return PackedReadyState(b"")
        chunks = tuple(self._chunks)
        values = (
            chunks[0][self._head :]
            if len(chunks) == 1
            else np.concatenate((chunks[0][self._head :], *chunks[1:]))
        )
        return PackedReadyState(values.astype("<u8", copy=False).tobytes())

    def restore(self, state: PackedReadyState) -> None:
        self.clear()
        self.extend(state.record_ids())


class PackedEnvelopeBatch(Sequence):
    """Packed IDs with a compatibility envelope view resolved only on demand."""

    __slots__ = ("record_ids", "_resolve", "identity")

    def __init__(self, record_ids, resolve, identity: str) -> None:
        if not callable(resolve):
            raise TypeError("packed batch resolver must be callable")
        if not isinstance(identity, str) or not identity:
            raise ValueError("packed batch identity must be a non-empty string")
        self.record_ids = _readonly_uint64(record_ids, name="batch record_ids")
        self._resolve = resolve
        self.identity = identity

    def __len__(self) -> int:
        return len(self.record_ids)

    def __iter__(self):
        return iter(self._resolve(self.record_ids))

    def __getitem__(self, index):
        if isinstance(index, slice):
            return PackedEnvelopeBatch(
                np.ascontiguousarray(self.record_ids[index]),
                self._resolve,
                self.identity,
            )
        index = operator.index(index)
        normalized = index if index >= 0 else len(self) + index
        if not 0 <= normalized < len(self):
            raise IndexError("packed batch index out of range")
        return self._resolve(self.record_ids[normalized : normalized + 1])[0]

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, PackedEnvelopeBatch)
            and self.identity == other.identity
            and np.array_equal(self.record_ids, other.record_ids)
        )

    def __repr__(self) -> str:
        return (
            f"PackedEnvelopeBatch(count={len(self)}, identity={self.identity!r})"
        )


class PackedUInt64ShuffleReservoir:
    """Exact reservoir-v2 storage backed by a native uint64 core."""

    def __init__(
        self,
        capacity: int,
        *,
        seed: int,
        epoch: int,
        stream_key: tuple = (),
    ) -> None:
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("packed reservoir capacity must be a positive integer")
        if type(epoch) is not int or epoch < 0:
            raise ValueError("packed reservoir epoch must be a non-negative integer")
        if type(stream_key) is not tuple:
            raise TypeError("packed reservoir stream_key must be a tuple")
        try:
            from dataset_planner_cpp import PackedUInt64Reservoir
        except ImportError as exc:
            raise RuntimeError(
                "dataset_planner_cpp is required; run `python setup.py build_ext --inplace`"
            ) from exc

        self.capacity = capacity
        self.seed = int(seed)
        self.epoch = epoch
        self.stream_key = stream_key
        self._rng_base = rng_u64(
            self.seed,
            "shuffle_reservoir",
            (RESERVOIR_ALGORITHM, self.epoch, self.stream_key),
        )
        self._core = PackedUInt64Reservoir(capacity, self._rng_base)
        self._state_cache: PackedReservoirState | None = None

    def __len__(self) -> int:
        return self._core.occupancy

    @property
    def closed(self) -> bool:
        return self._core.closed

    @property
    def stats(self) -> ReservoirStats:
        return ReservoirStats(
            offered=self._core.offered,
            emitted=self._core.emitted,
            occupancy=self._core.occupancy,
            resident_bytes=0,
            peak_occupancy=self._core.peak_occupancy,
            peak_resident_bytes=0,
        )

    def offer_block(self, block: PackedRecordBlock) -> np.ndarray:
        # The first integration supports one cycle per packed reservoir.
        # Per-record cycle blocks are admitted after tagged handles land.
        if block.cycles is not None:
            raise ValueError("packed reservoir v1 does not accept per-record cycles")
        if block.default_cycle != 0:
            raise ValueError("packed reservoir v1 requires default_cycle=0")
        emitted = self._core.offer(block.record_ids)
        if len(block):
            self._state_cache = None
        emitted.flags.writeable = False
        return emitted

    def drain(self) -> np.ndarray:
        if self.closed:
            return np.empty(0, dtype=np.uint64)
        order = np.asarray(
            deterministic_permutation(
                len(self),
                self.seed,
                "shuffle_reservoir_drain",
                (
                    RESERVOIR_ALGORITHM,
                    self.epoch,
                    self.stream_key,
                    self._core.offered,
                    self._core.rng_counter,
                ),
            ),
            dtype=np.uint64,
        )
        emitted = self._core.drain(order)
        self._state_cache = None
        emitted.flags.writeable = False
        return emitted

    def state(self) -> PackedReservoirState:
        if self._state_cache is None:
            slots = self._core.slots()
            self._state_cache = PackedReservoirState(
                algorithm=PACKED_RESERVOIR_ALGORITHM,
                seed=self.seed,
                epoch=self.epoch,
                stream_key=self.stream_key,
                capacity=self.capacity,
                slots_le=slots.astype("<u8", copy=False).tobytes(),
                rng_counter=self._core.rng_counter,
                offered=self._core.offered,
                emitted=self._core.emitted,
                peak_occupancy=self._core.peak_occupancy,
                closed=self._core.closed,
            )
        return self._state_cache

    def restore(self, state: PackedReservoirState) -> None:
        state = PackedReservoirState(
            algorithm=state.algorithm,
            seed=state.seed,
            epoch=state.epoch,
            stream_key=state.stream_key,
            capacity=state.capacity,
            slots_le=state.slots_le,
            rng_counter=state.rng_counter,
            offered=state.offered,
            emitted=state.emitted,
            peak_occupancy=state.peak_occupancy,
            closed=state.closed,
        )
        expected = (
            PACKED_RESERVOIR_ALGORITHM,
            self.seed,
            self.epoch,
            self.stream_key,
            self.capacity,
        )
        actual = (
            state.algorithm,
            state.seed,
            state.epoch,
            state.stream_key,
            state.capacity,
        )
        if actual != expected:
            raise ValueError(
                "packed reservoir state configuration changed: "
                f"expected {expected}, got {actual}"
            )
        if len(state.slots_le) % 8:
            raise ValueError("packed reservoir state has a truncated slot buffer")
        if state.occupancy > self.capacity:
            raise ValueError("packed reservoir state exceeds configured capacity")
        if min(
            state.rng_counter,
            state.offered,
            state.emitted,
            state.peak_occupancy,
        ) < 0:
            raise ValueError("packed reservoir state contains negative counters")
        if state.offered != state.emitted + state.occupancy:
            raise ValueError("packed reservoir state violates exact-once accounting")
        if not state.occupancy <= state.peak_occupancy <= self.capacity:
            raise ValueError("packed reservoir state has invalid peak occupancy")
        if state.closed and state.occupancy:
            raise ValueError("closed packed reservoir state still contains slots")
        self._core.restore(
            state.slot_array(),
            state.rng_counter,
            state.offered,
            state.emitted,
            state.peak_occupancy,
            state.closed,
        )
        self._state_cache = state
