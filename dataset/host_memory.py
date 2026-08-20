"""Thread-safe logical memory accounting for host-side dataset pipelines.

This module deliberately separates memory accounting from dataset semantics.
It never changes shuffle policy, source order, or any other semantic setting;
callers may only use reservation failure to apply performance backpressure.

The budget is a hard cap over bytes explicitly charged by callers.  It does
not attempt to measure process RSS, allocator fragmentation, or operating
system page-cache residency.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import threading
from types import MappingProxyType
from typing import Mapping


__all__ = [
    "HostMemoryBudget",
    "HostMemoryBudgetExceeded",
    "HostMemoryCategory",
    "HostMemoryLease",
    "HostMemoryReservation",
    "HostMemorySnapshot",
]


class HostMemoryCategory(str, Enum):
    """Stable accounting categories for dataset-owned host memory."""

    SEMANTIC_FIXED = "semantic_fixed"
    DECODED_CACHE = "decoded_cache"
    INFLIGHT_TRANSIENT = "inflight_transient"
    READY_PAGEABLE = "ready_pageable"
    PINNED = "pinned"

    @property
    def is_semantic(self) -> bool:
        return self is HostMemoryCategory.SEMANTIC_FIXED


@dataclass(frozen=True, slots=True)
class HostMemorySnapshot:
    """Immutable point-in-time accounting and high-water telemetry."""

    total_bytes: int
    used_bytes: int
    available_bytes: int
    high_water_bytes: int
    used_by_category: Mapping[str, int]
    high_water_by_category: Mapping[str, int]
    charged_reservations: int
    retired_reservations: int
    active_leases: int
    successful_reservations: int
    failed_reservations: int
    backpressure_events: int

    def as_dict(self) -> dict:
        """Return a JSON-friendly copy suitable for telemetry emission."""

        return {
            "total_bytes": self.total_bytes,
            "used_bytes": self.used_bytes,
            "available_bytes": self.available_bytes,
            "high_water_bytes": self.high_water_bytes,
            "used_by_category": dict(self.used_by_category),
            "high_water_by_category": dict(self.high_water_by_category),
            "charged_reservations": self.charged_reservations,
            "retired_reservations": self.retired_reservations,
            "active_leases": self.active_leases,
            "successful_reservations": self.successful_reservations,
            "failed_reservations": self.failed_reservations,
            "backpressure_events": self.backpressure_events,
        }


class HostMemoryBudgetExceeded(RuntimeError):
    """Raised when a fail-fast reservation cannot fit in the hard cap."""

    def __init__(
        self,
        *,
        category: HostMemoryCategory,
        requested_bytes: int,
        label: str | None,
        reason: str,
        snapshot: HostMemorySnapshot,
    ) -> None:
        self.category = category
        self.requested_bytes = requested_bytes
        self.label = label
        self.reason = reason
        self.snapshot = snapshot
        breakdown = ", ".join(
            f"{name}={value}"
            for name, value in snapshot.used_by_category.items()
        )
        label_detail = "" if label is None else f", label={label!r}"
        super().__init__(
            "host-data memory budget exceeded: "
            f"{reason}; requested={requested_bytes}, category={category.value}"
            f"{label_detail}; used={snapshot.used_bytes}, "
            f"available={snapshot.available_bytes}, total={snapshot.total_bytes}; "
            f"breakdown: {breakdown}. The budget does not alter semantic "
            "dataset parameters automatically."
        )


@dataclass(slots=True)
class _ChargeState:
    reservation_id: int
    category: HostMemoryCategory
    nbytes: int
    label: str | None
    owner_active: bool
    retired: bool
    lease_ids: set[int]


def _validate_byte_count(name: str, value: int, *, positive: bool) -> int:
    if type(value) is not int:
        qualifier = "positive" if positive else "non-negative"
        raise TypeError(f"{name} must be a {qualifier} integer")
    if (positive and value <= 0) or (not positive and value < 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


def _coerce_category(category: HostMemoryCategory | str) -> HostMemoryCategory:
    if isinstance(category, HostMemoryCategory):
        return category
    if not isinstance(category, str):
        raise TypeError("host-memory category must be a HostMemoryCategory or string")
    try:
        return HostMemoryCategory(category)
    except ValueError as exc:
        choices = ", ".join(item.value for item in HostMemoryCategory)
        raise ValueError(
            f"unknown host-memory category {category!r}; expected one of {choices}"
        ) from exc


def _validate_label(label: str | None) -> str | None:
    if label is not None and not isinstance(label, str):
        raise TypeError("host-memory reservation label must be a string or null")
    return label


class HostMemoryBudget:
    """A non-blocking, byte-based hard cap for logical host-data memory.

    ``reserve`` fails immediately with a detailed exception. ``try_reserve``
    returns ``None`` for temporary pressure so an ordered producer can stop
    submitting work and let its consumer make progress.  The primitive never
    waits and never invokes user callbacks while holding its lock.
    """

    def __init__(self, total_bytes: int) -> None:
        self._total_bytes = _validate_byte_count(
            "host-memory total_bytes", total_bytes, positive=True
        )
        self._lock = threading.Lock()
        self._used_bytes = 0
        self._high_water_bytes = 0
        self._used_by_category = {
            category: 0 for category in HostMemoryCategory
        }
        self._high_water_by_category = {
            category: 0 for category in HostMemoryCategory
        }
        self._charges: dict[int, _ChargeState] = {}
        self._next_reservation_id = 1
        self._next_lease_id = 1
        self._successful_reservations = 0
        self._failed_reservations = 0
        self._backpressure_events = 0

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    def reserve(
        self,
        category: HostMemoryCategory | str,
        nbytes: int,
        *,
        label: str | None = None,
    ) -> "HostMemoryReservation":
        """Reserve bytes immediately or raise ``HostMemoryBudgetExceeded``."""

        normalized = _coerce_category(category)
        requested = _validate_byte_count(
            "host-memory reservation bytes", nbytes, positive=False
        )
        label = _validate_label(label)
        failure = None
        with self._lock:
            if requested > self._total_bytes:
                reason = "request exceeds the total hard cap even when empty"
                self._failed_reservations += 1
                failure = (reason, self._snapshot_locked())
            elif self._used_bytes + requested > self._total_bytes:
                reason = "insufficient currently available bytes"
                self._failed_reservations += 1
                failure = (reason, self._snapshot_locked())
            else:
                return self._allocate_locked(normalized, requested, label)
        assert failure is not None
        reason, snapshot = failure
        raise HostMemoryBudgetExceeded(
            category=normalized,
            requested_bytes=requested,
            label=label,
            reason=reason,
            snapshot=snapshot,
        )

    def try_reserve(
        self,
        category: HostMemoryCategory | str,
        nbytes: int,
        *,
        label: str | None = None,
    ) -> "HostMemoryReservation | None":
        """Try a reservation without blocking.

        ``None`` means existing reservations caused temporary backpressure.  A
        request larger than the entire hard cap is impossible and therefore
        raises immediately instead of inviting an infinite retry loop.
        """

        normalized = _coerce_category(category)
        requested = _validate_byte_count(
            "host-memory reservation bytes", nbytes, positive=False
        )
        label = _validate_label(label)
        failure = None
        with self._lock:
            if requested > self._total_bytes:
                reason = "request exceeds the total hard cap even when empty"
                self._failed_reservations += 1
                failure = (reason, self._snapshot_locked())
            elif self._used_bytes + requested > self._total_bytes:
                self._backpressure_events += 1
                return None
            else:
                return self._allocate_locked(normalized, requested, label)
        assert failure is not None
        reason, snapshot = failure
        raise HostMemoryBudgetExceeded(
            category=normalized,
            requested_bytes=requested,
            label=label,
            reason=reason,
            snapshot=snapshot,
        )

    def snapshot(self) -> HostMemorySnapshot:
        with self._lock:
            return self._snapshot_locked()

    def _allocate_locked(
        self,
        category: HostMemoryCategory,
        nbytes: int,
        label: str | None,
    ) -> "HostMemoryReservation":
        reservation_id = self._next_reservation_id
        self._next_reservation_id += 1
        state = _ChargeState(
            reservation_id=reservation_id,
            category=category,
            nbytes=nbytes,
            label=label,
            owner_active=True,
            retired=False,
            lease_ids=set(),
        )
        self._charges[reservation_id] = state
        self._used_bytes += nbytes
        self._used_by_category[category] += nbytes
        self._high_water_bytes = max(self._high_water_bytes, self._used_bytes)
        self._high_water_by_category[category] = max(
            self._high_water_by_category[category],
            self._used_by_category[category],
        )
        self._successful_reservations += 1
        return HostMemoryReservation(
            self,
            state,
        )

    def _snapshot_locked(self) -> HostMemorySnapshot:
        used = {
            category.value: self._used_by_category[category]
            for category in HostMemoryCategory
        }
        high_water = {
            category.value: self._high_water_by_category[category]
            for category in HostMemoryCategory
        }
        return HostMemorySnapshot(
            total_bytes=self._total_bytes,
            used_bytes=self._used_bytes,
            available_bytes=self._total_bytes - self._used_bytes,
            high_water_bytes=self._high_water_bytes,
            used_by_category=MappingProxyType(used),
            high_water_by_category=MappingProxyType(high_water),
            charged_reservations=len(self._charges),
            retired_reservations=sum(
                state.retired for state in self._charges.values()
            ),
            active_leases=sum(
                len(state.lease_ids) for state in self._charges.values()
            ),
            successful_reservations=self._successful_reservations,
            failed_reservations=self._failed_reservations,
            backpressure_events=self._backpressure_events,
        )

    def _acquire_lease(self, state: _ChargeState) -> "HostMemoryLease":
        with self._lock:
            if (
                self._charges.get(state.reservation_id) is not state
                or state.retired
                or not state.owner_active
            ):
                raise RuntimeError("cannot acquire a lease from a retired reservation")
            lease_id = self._next_lease_id
            self._next_lease_id += 1
            state.lease_ids.add(lease_id)
            return HostMemoryLease(
                self,
                state,
                lease_id,
            )

    def _retire_reservation(self, state: _ChargeState) -> bool:
        with self._lock:
            if (
                self._charges.get(state.reservation_id) is not state
                or not state.owner_active
            ):
                return False
            state.owner_active = False
            state.retired = True
            if not state.lease_ids:
                self._drop_charge_locked(state)
            return True

    def _release_lease(self, state: _ChargeState, lease_id: int) -> bool:
        with self._lock:
            if (
                self._charges.get(state.reservation_id) is not state
                or lease_id not in state.lease_ids
            ):
                return False
            state.lease_ids.remove(lease_id)
            if state.retired and not state.owner_active and not state.lease_ids:
                self._drop_charge_locked(state)
            return True

    def _reclassify_reservation(
        self,
        state: _ChargeState,
        category: HostMemoryCategory,
    ) -> bool:
        with self._lock:
            if (
                self._charges.get(state.reservation_id) is not state
                or state.retired
                or not state.owner_active
            ):
                raise RuntimeError("cannot reclassify a retired reservation")
            if state.category.is_semantic != category.is_semantic:
                raise ValueError(
                    "cannot reclassify host memory across the semantic/performance "
                    "boundary"
                )
            if state.category is category:
                return False
            self._used_by_category[state.category] -= state.nbytes
            state.category = category
            self._used_by_category[category] += state.nbytes
            self._high_water_by_category[category] = max(
                self._high_water_by_category[category],
                self._used_by_category[category],
            )
            return True

    def _resize_reservation(
        self,
        state: _ChargeState,
        nbytes: int,
    ) -> bool:
        failure = None
        with self._lock:
            if (
                self._charges.get(state.reservation_id) is not state
                or state.retired
                or not state.owner_active
            ):
                raise RuntimeError("cannot resize a retired reservation")
            delta = nbytes - state.nbytes
            if delta == 0:
                return False
            if nbytes > self._total_bytes:
                reason = "resized object exceeds the total hard cap even when empty"
                self._failed_reservations += 1
                failure = (reason, state.category, state.label, self._snapshot_locked())
            elif delta > 0 and self._used_bytes + delta > self._total_bytes:
                reason = "insufficient currently available bytes for resize"
                self._failed_reservations += 1
                failure = (reason, state.category, state.label, self._snapshot_locked())
            else:
                state.nbytes = nbytes
                self._used_bytes += delta
                self._used_by_category[state.category] += delta
                self._high_water_bytes = max(
                    self._high_water_bytes, self._used_bytes
                )
                self._high_water_by_category[state.category] = max(
                    self._high_water_by_category[state.category],
                    self._used_by_category[state.category],
                )
                return True
        assert failure is not None
        reason, category, label, snapshot = failure
        raise HostMemoryBudgetExceeded(
            category=category,
            requested_bytes=nbytes,
            label=label,
            reason=reason,
            snapshot=snapshot,
        )

    def _drop_charge_locked(self, state: _ChargeState) -> None:
        removed = self._charges.pop(state.reservation_id, None)
        if removed is None:
            return
        self._used_bytes -= state.nbytes
        self._used_by_category[state.category] -= state.nbytes

    def _reservation_active(self, state: _ChargeState) -> bool:
        with self._lock:
            return bool(
                self._charges.get(state.reservation_id) is state
                and state.owner_active
            )

    def _reservation_lease_count(self, state: _ChargeState) -> int:
        with self._lock:
            return (
                len(state.lease_ids)
                if self._charges.get(state.reservation_id) is state
                else 0
            )

    def _lease_active(self, state: _ChargeState, lease_id: int) -> bool:
        with self._lock:
            return bool(
                self._charges.get(state.reservation_id) is state
                and lease_id in state.lease_ids
            )

    def _charge_details(
        self, state: _ChargeState
    ) -> tuple[HostMemoryCategory, int, str | None]:
        with self._lock:
            return state.category, state.nbytes, state.label


class HostMemoryReservation:
    """Owner of one charged object and factory for consumer leases.

    Releasing or retiring the owner prevents new leases.  Existing leases keep
    the full byte charge alive until the final lease is explicitly released.
    """

    __slots__ = (
        "_budget",
        "_state",
    )

    def __init__(
        self,
        budget: HostMemoryBudget,
        state: _ChargeState,
    ) -> None:
        self._budget = budget
        self._state = state

    @property
    def category(self) -> HostMemoryCategory:
        category, _, _ = self._budget._charge_details(self._state)
        return category

    @property
    def nbytes(self) -> int:
        _, nbytes, _ = self._budget._charge_details(self._state)
        return nbytes

    @property
    def label(self) -> str | None:
        _, _, label = self._budget._charge_details(self._state)
        return label

    @property
    def released(self) -> bool:
        return not self._budget._reservation_active(self._state)

    @property
    def retired(self) -> bool:
        return self.released

    @property
    def active_leases(self) -> int:
        return self._budget._reservation_lease_count(self._state)

    def acquire_lease(self) -> "HostMemoryLease":
        return self._budget._acquire_lease(self._state)

    def reclassify(self, category: HostMemoryCategory | str) -> bool:
        """Atomically move the full charge to another compatible category.

        Semantic/fixed memory cannot be reclassified as performance memory or
        vice versa.  Existing leases observe the new category and remain
        charged throughout the transition.
        """

        return self._budget._reclassify_reservation(
            self._state,
            _coerce_category(category),
        )

    def resize(self, nbytes: int) -> bool:
        """Atomically adjust the charge to an actual byte count.

        Growth respects the same hard cap as a new reservation. Shrinking
        releases bytes immediately. A failed growth leaves all accounting
        unchanged.
        """

        requested = _validate_byte_count(
            "host-memory reservation bytes", nbytes, positive=False
        )
        return self._budget._resize_reservation(self._state, requested)

    def release(self) -> bool:
        """Retire the owner; safe to call repeatedly or concurrently."""

        return self._budget._retire_reservation(self._state)

    def retire(self) -> bool:
        """Cache-oriented alias for ``release``."""

        return self.release()

    def __enter__(self) -> "HostMemoryReservation":
        if self.released:
            raise RuntimeError("cannot enter a released host-memory reservation")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.release()
        return False


class HostMemoryLease:
    """Explicit consumer reference keeping a retired object charged."""

    __slots__ = (
        "_budget",
        "_state",
        "_lease_id",
    )

    def __init__(
        self,
        budget: HostMemoryBudget,
        state: _ChargeState,
        lease_id: int,
    ) -> None:
        self._budget = budget
        self._state = state
        self._lease_id = lease_id

    @property
    def category(self) -> HostMemoryCategory:
        category, _, _ = self._budget._charge_details(self._state)
        return category

    @property
    def nbytes(self) -> int:
        _, nbytes, _ = self._budget._charge_details(self._state)
        return nbytes

    @property
    def label(self) -> str | None:
        _, _, label = self._budget._charge_details(self._state)
        return label

    @property
    def released(self) -> bool:
        return not self._budget._lease_active(
            self._state, self._lease_id
        )

    def release(self) -> bool:
        """Release this reference; safe to call repeatedly or concurrently."""

        return self._budget._release_lease(
            self._state, self._lease_id
        )

    def __enter__(self) -> "HostMemoryLease":
        if self.released:
            raise RuntimeError("cannot enter a released host-memory lease")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.release()
        return False
