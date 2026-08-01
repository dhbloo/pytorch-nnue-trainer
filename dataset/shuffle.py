from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from .core import (
    deterministic_permutation,
    rng_u64,
)


RESERVOIR_ALGORITHM = "streaming-shuffle-reservoir-v2"
_U64_MASK = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15

ItemT = TypeVar("ItemT")


@dataclass(frozen=True, slots=True)
class ReservoirStats:
    offered: int
    emitted: int
    occupancy: int
    resident_bytes: int
    peak_occupancy: int
    peak_resident_bytes: int


@dataclass(frozen=True, slots=True)
class ReservoirState(Generic[ItemT]):
    algorithm: str
    seed: int
    epoch: int
    stream_key: tuple
    capacity: int
    max_bytes: int | None
    slots: tuple[ItemT, ...]
    slot_sizes: tuple[int, ...]
    resident_bytes: int
    rng_counter: int
    offered: int
    emitted: int
    peak_occupancy: int
    peak_resident_bytes: int
    closed: bool


class StreamingShuffleReservoir(Generic[ItemT]):
    """Deterministic bounded-memory shuffle for a stream of opaque items."""

    __slots__ = (
        "capacity",
        "max_bytes",
        "seed",
        "epoch",
        "stream_key",
        "_slots",
        "_slot_sizes",
        "_resident_bytes",
        "_rng_base",
        "_rng_counter",
        "_offered",
        "_emitted",
        "_peak_occupancy",
        "_peak_resident_bytes",
        "_closed",
        "_state_cache",
    )

    def __init__(
        self,
        capacity: int,
        *,
        seed: int,
        epoch: int,
        stream_key: tuple = (),
        max_bytes: int | None = None,
    ) -> None:
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("reservoir capacity must be a positive integer")
        if max_bytes is not None and (
            type(max_bytes) is not int or max_bytes <= 0
        ):
            raise ValueError("reservoir max_bytes must be a positive integer")
        if type(epoch) is not int or epoch < 0:
            raise ValueError("reservoir epoch must be a non-negative integer")
        if type(stream_key) is not tuple:
            raise TypeError("reservoir stream_key must be a tuple")

        self.capacity = capacity
        self.max_bytes = max_bytes
        self.seed = int(seed)
        self.epoch = epoch
        self.stream_key = stream_key
        self._slots: list[ItemT] = []
        self._slot_sizes: list[int] = []
        self._resident_bytes = 0
        self._rng_base = rng_u64(
            self.seed,
            "shuffle_reservoir",
            (RESERVOIR_ALGORITHM, self.epoch, self.stream_key),
        )
        self._rng_counter = 0
        self._offered = 0
        self._emitted = 0
        self._peak_occupancy = 0
        self._peak_resident_bytes = 0
        self._closed = False
        self._state_cache = None

    def __len__(self) -> int:
        return len(self._slots)

    @property
    def resident_bytes(self) -> int:
        return self._resident_bytes

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def stats(self) -> ReservoirStats:
        return ReservoirStats(
            offered=self._offered,
            emitted=self._emitted,
            occupancy=len(self._slots),
            resident_bytes=self._resident_bytes,
            peak_occupancy=self._peak_occupancy,
            peak_resident_bytes=self._peak_resident_bytes,
        )

    def _fits(self, size_bytes: int) -> bool:
        return self.max_bytes is None or (
            self._resident_bytes + size_bytes <= self.max_bytes
        )

    def _random_slot(self) -> int:
        upper = len(self._slots)
        limit = (1 << 64) - ((1 << 64) % upper)
        while True:
            if not 0 <= self._rng_counter < (1 << 63):
                raise ValueError("reservoir RNG counter is outside [0, 2**63)")
            value = (
                self._rng_base + self._rng_counter * _SPLITMIX_GAMMA
            ) & _U64_MASK
            self._rng_counter += 1
            value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _U64_MASK
            value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _U64_MASK
            value ^= value >> 31
            if value < limit:
                return value % upper

    def _evict_slot(self, index: int) -> ItemT:
        item = self._slots[index]
        self._resident_bytes -= self._slot_sizes[index]
        last_item = self._slots.pop()
        last_size = self._slot_sizes.pop()
        if index < len(self._slots):
            self._slots[index] = last_item
            self._slot_sizes[index] = last_size
        self._emitted += 1
        return item

    def offer(self, item: ItemT, *, size_bytes: int = 0) -> tuple[ItemT, ...]:
        """Admit one item and return zero or more deterministically evicted items."""
        if self._closed:
            raise RuntimeError("cannot offer an item after the reservoir was drained")
        if type(size_bytes) is not int or size_bytes < 0:
            raise ValueError("reservoir item size must be a non-negative integer")
        if self.max_bytes is not None and size_bytes > self.max_bytes:
            raise ValueError(
                f"reservoir item requires {size_bytes} bytes, exceeding the "
                f"{self.max_bytes}-byte limit"
            )

        self._state_cache = None
        emitted: list[ItemT] = []
        while self._slots and (
            len(self._slots) >= self.capacity or not self._fits(size_bytes)
        ):
            emitted.append(self._evict_slot(self._random_slot()))

        self._slots.append(item)
        self._slot_sizes.append(size_bytes)
        self._resident_bytes += size_bytes
        self._offered += 1
        self._peak_occupancy = max(self._peak_occupancy, len(self._slots))
        self._peak_resident_bytes = max(
            self._peak_resident_bytes, self._resident_bytes
        )
        return tuple(emitted)

    def offer_many(
        self,
        items,
    ) -> tuple[ItemT, ...]:
        """Admit an ordered bounded chunk while amortizing Python call overhead."""
        if self._closed:
            raise RuntimeError("cannot offer items after the reservoir was drained")
        prepared = tuple(items)
        for _, size_bytes in prepared:
            if type(size_bytes) is not int or size_bytes < 0:
                raise ValueError("reservoir item size must be a non-negative integer")
            if self.max_bytes is not None and size_bytes > self.max_bytes:
                raise ValueError(
                    f"reservoir item requires {size_bytes} bytes, exceeding the "
                    f"{self.max_bytes}-byte limit"
                )

        if prepared:
            self._state_cache = None
        emitted = []
        peak_occupancy = self._peak_occupancy
        peak_resident_bytes = self._peak_resident_bytes
        for item, size_bytes in prepared:
            while self._slots and (
                len(self._slots) >= self.capacity or not self._fits(size_bytes)
            ):
                emitted.append(self._evict_slot(self._random_slot()))
            self._slots.append(item)
            self._slot_sizes.append(size_bytes)
            self._resident_bytes += size_bytes
            self._offered += 1
            peak_occupancy = max(peak_occupancy, len(self._slots))
            peak_resident_bytes = max(
                peak_resident_bytes,
                self._resident_bytes,
            )
        self._peak_occupancy = peak_occupancy
        self._peak_resident_bytes = peak_resident_bytes
        return tuple(emitted)

    def offer_zero_sized(self, items) -> tuple[ItemT, ...]:
        """Admit a zero-resident-byte chunk without temporary size pairs."""
        if self._closed:
            raise RuntimeError("cannot offer items after the reservoir was drained")

        prepared = items if isinstance(items, tuple) else tuple(items)
        if not prepared:
            return ()
        self._state_cache = None
        slots = self._slots
        slot_sizes = self._slot_sizes
        capacity = self.capacity
        emitted = []
        for item in prepared:
            if len(slots) >= capacity:
                emitted.append(self._evict_slot(self._random_slot()))
            slots.append(item)
            slot_sizes.append(0)
        self._offered += len(prepared)
        self._peak_occupancy = max(self._peak_occupancy, len(slots))
        return tuple(emitted)

    def drain(self) -> tuple[ItemT, ...]:
        """Close the reservoir and return its remaining items in random order."""
        if self._closed:
            return ()
        self._state_cache = None
        order = deterministic_permutation(
            len(self._slots),
            self.seed,
            "shuffle_reservoir_drain",
            (
                RESERVOIR_ALGORITHM,
                self.epoch,
                self.stream_key,
                self._offered,
                self._rng_counter,
            ),
        )
        emitted = tuple(self._slots[index] for index in order)
        self._emitted += len(emitted)
        self._slots.clear()
        self._slot_sizes.clear()
        self._resident_bytes = 0
        self._closed = True
        return emitted

    def state(self) -> ReservoirState[ItemT]:
        if self._state_cache is None:
            self._state_cache = ReservoirState(
                algorithm=RESERVOIR_ALGORITHM,
                seed=self.seed,
                epoch=self.epoch,
                stream_key=self.stream_key,
                capacity=self.capacity,
                max_bytes=self.max_bytes,
                slots=tuple(self._slots),
                slot_sizes=tuple(self._slot_sizes),
                resident_bytes=self._resident_bytes,
                rng_counter=self._rng_counter,
                offered=self._offered,
                emitted=self._emitted,
                peak_occupancy=self._peak_occupancy,
                peak_resident_bytes=self._peak_resident_bytes,
                closed=self._closed,
            )
        return self._state_cache

    def restore(self, state: ReservoirState[ItemT]) -> None:
        """Restore an exact cursor after validating logical and memory invariants."""
        if state.algorithm != RESERVOIR_ALGORITHM:
            raise ValueError(f"unsupported reservoir algorithm {state.algorithm!r}")
        expected = (
            self.seed,
            self.epoch,
            self.stream_key,
            self.capacity,
            self.max_bytes,
        )
        actual = (
            state.seed,
            state.epoch,
            state.stream_key,
            state.capacity,
            state.max_bytes,
        )
        if actual != expected:
            raise ValueError(
                "reservoir state configuration changed: "
                f"expected {expected}, got {actual}"
            )
        if len(state.slots) != len(state.slot_sizes):
            raise ValueError("reservoir state has mismatched slots and sizes")
        if len(state.slots) > self.capacity:
            raise ValueError("reservoir state exceeds the configured capacity")
        if any(type(size) is not int or size < 0 for size in state.slot_sizes):
            raise ValueError("reservoir state contains an invalid item size")
        if sum(state.slot_sizes) != state.resident_bytes:
            raise ValueError("reservoir state resident byte count is inconsistent")
        if min(
            state.resident_bytes,
            state.rng_counter,
            state.offered,
            state.emitted,
            state.peak_occupancy,
            state.peak_resident_bytes,
        ) < 0:
            raise ValueError("reservoir state contains a negative counter")
        if self.max_bytes is not None and state.resident_bytes > self.max_bytes:
            raise ValueError("reservoir state exceeds the configured byte limit")
        if state.offered != state.emitted + len(state.slots):
            raise ValueError("reservoir state violates the exact-once invariant")
        if state.closed and state.slots:
            raise ValueError("closed reservoir state must not retain items")
        if not 0 <= state.rng_counter < (1 << 63):
            raise ValueError("reservoir state RNG counter is out of range")
        if state.peak_occupancy < len(state.slots):
            raise ValueError("reservoir peak occupancy is inconsistent")
        if state.peak_occupancy > self.capacity:
            raise ValueError("reservoir peak occupancy exceeds capacity")
        if state.peak_resident_bytes < state.resident_bytes:
            raise ValueError("reservoir peak byte count is inconsistent")

        self._slots = list(state.slots)
        self._slot_sizes = list(state.slot_sizes)
        self._resident_bytes = state.resident_bytes
        self._rng_counter = state.rng_counter
        self._offered = state.offered
        self._emitted = state.emitted
        self._peak_occupancy = state.peak_occupancy
        self._peak_resident_bytes = state.peak_resident_bytes
        self._closed = state.closed
        self._state_cache = state
