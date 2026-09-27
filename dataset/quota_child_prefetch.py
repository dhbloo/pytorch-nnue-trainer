"""Exact quota-child packages from one bounded spawn worker per rank."""

from __future__ import annotations

import ctypes
import hashlib
import multiprocessing as mp
import os
import select
import time
import traceback
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .composite_source import MIXING_BLOCK_ROWS
from .core import canonical_pipeline_state_bytes


PACKAGE_BLOCKS = 16
MAX_CURSOR_SESSIONS = 2
RESULT_WAIT_SECONDS = 30.0
CURSOR_CLOSE_SECONDS = 1.0
STARTUP_WAIT_SECONDS = 60.0


@dataclass
class _Slot:
    request: object
    remaining: object
    children: object
    counts: object
    sizes: object
    response: object


@dataclass
class _Session:
    cursor: object
    slot: int
    epoch: int
    source_cycle: int
    next_index: int
    next_remaining: tuple[int, ...]
    ready: deque = field(default_factory=deque)
    pending: bool = False
    generation: int = 0


def _generate_package(seed: int, source_count: int, slot: _Slot) -> None:
    request = np.frombuffer(slot.request, dtype=np.int64)
    remaining = np.frombuffer(slot.remaining, dtype=np.int64).copy()
    children_out = np.frombuffer(slot.children, dtype=np.uint16)
    counts_out = np.frombuffer(slot.counts, dtype=np.uint16).reshape(
        PACKAGE_BLOCKS, source_count
    )
    sizes_out = np.frombuffer(slot.sizes, dtype=np.uint16)
    response = np.frombuffer(slot.response, dtype=np.int64)
    generation, epoch, source_cycle, first_index = map(int, request)
    response[:] = (generation, 0, 0)
    if np.any(remaining < 0):
        raise ValueError("negative quota in worker request")
    for offset in range(PACKAGE_BLOCKS):
        total = int(remaining.sum())
        if not total:
            break
        index = first_index + offset
        digest = hashlib.sha256(canonical_pipeline_state_bytes(
            (seed, epoch, source_cycle, index)
        )).digest()
        rng = np.random.Generator(np.random.PCG64(int.from_bytes(digest[:16], "little")))
        size = min(MIXING_BLOCK_ROWS, total)
        ranks = rng.choice(total, size=size, replace=False)
        children = np.searchsorted(np.cumsum(remaining, dtype=np.int64), ranks, side="right")
        counts = np.bincount(children, minlength=source_count)
        begin = offset * MIXING_BLOCK_ROWS
        children_out[begin:begin + size] = children.astype(np.uint16, copy=False)
        counts_out[offset] = counts.astype(np.uint16, copy=False)
        sizes_out[offset] = size
        remaining -= counts
        response[1] = offset + 1


def _worker_main(seed, source_count, slots, request_rx, result_tx) -> None:
    os.write(result_tx.fileno(), b"\xfe")
    while True:
        token = os.read(request_rx.fileno(), 1)
        if not token or token == b"\xff":
            return
        index = token[0]
        if index >= len(slots):
            return
        slot = slots[index]
        try:
            _generate_package(seed, source_count, slot)
        except BaseException:
            traceback.print_exc()
            np.frombuffer(slot.response, dtype=np.int64)[2] = 1
            os.write(result_tx.fileno(), bytes((index,)))
            return
        os.write(result_tx.fileno(), bytes((index,)))


class QuotaChildPrefetcher:
    """Prefetch exact source IDs without serializing speculative worker state."""

    @property
    def worker_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def __init__(self, seed: int, source_count: int):
        if not 0 < source_count <= 65536:
            raise ValueError("quota child IDs require at most 65536 sources")
        self.seed = int(seed)
        self.source_count = int(source_count)
        self._context = mp.get_context("spawn")
        self._slots = None
        self._request_tx = None
        self._result_rx = None
        self._responses = set()
        self._process = None
        self._sessions = {}
        self._generation = 0
        self._failed = False
        self.wait_ns = 0
        self.completed_packages = 0

    def _start_worker(self):
        if self._failed:
            raise RuntimeError("quota child worker has failed")
        if self._process is not None:
            return
        context = self._context
        self._slots = tuple(
            _Slot(
                context.RawArray(ctypes.c_int64, 4),
                context.RawArray(ctypes.c_int64, self.source_count),
                context.RawArray(ctypes.c_uint16, PACKAGE_BLOCKS * MIXING_BLOCK_ROWS),
                context.RawArray(ctypes.c_uint16, PACKAGE_BLOCKS * self.source_count),
                context.RawArray(ctypes.c_uint16, PACKAGE_BLOCKS),
                context.RawArray(ctypes.c_int64, 3),
            )
            for _ in range(MAX_CURSOR_SESSIONS)
        )
        request_rx, request_tx = context.Pipe(duplex=False)
        result_rx, result_tx = context.Pipe(duplex=False)
        process = context.Process(
            target=_worker_main,
            args=(self.seed, self.source_count, self._slots, request_rx, result_tx),
            daemon=True,
        )
        try:
            process.start()
        except BaseException:
            self._slots = None
            request_rx.close()
            request_tx.close()
            result_rx.close()
            result_tx.close()
            raise
        request_rx.close()
        result_tx.close()
        self._process = process
        self._request_tx = request_tx
        self._result_rx = result_rx
        try:
            self._wait_token(254, STARTUP_WAIT_SECONDS)
        except BaseException:
            self._failed = True
            self._stop_worker()
            raise

    def _wait_token(self, expected: int, timeout: float):
        if expected in self._responses:
            self._responses.remove(expected)
            return
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("quota child worker did not finish before the deadline")
            process = self._process
            readable, _, _ = select.select(
                (self._result_rx.fileno(), process.sentinel), (), (), remaining
            )
            if self._result_rx.fileno() in readable:
                token = os.read(self._result_rx.fileno(), 1)
                if not token:
                    raise RuntimeError("quota child worker closed its result pipe")
                number = token[0]
                if number == expected:
                    return
                if number not in range(MAX_CURSOR_SESSIONS):
                    raise RuntimeError("quota child worker sent an invalid token")
                if number in self._responses:
                    raise RuntimeError("quota child worker sent a duplicate token")
                self._responses.add(number)
                continue
            if process.sentinel in readable:
                raise RuntimeError("quota child worker exited before publishing a result")

    def _submit(self, session: _Session):
        if session.pending:
            raise RuntimeError("quota child package is already pending")
        if not any(session.next_remaining):
            return
        self._start_worker()
        slot = self._slots[session.slot]
        if session.slot in self._responses:
            raise RuntimeError("quota child slot was reused before acknowledgement")
        self._generation += 1
        session.generation = self._generation
        np.frombuffer(slot.request, dtype=np.int64)[:] = (
            session.generation, session.epoch, session.source_cycle, session.next_index
        )
        np.frombuffer(slot.remaining, dtype=np.int64)[:] = session.next_remaining
        session.pending = True
        try:
            # At most two requests and one stop byte can be outstanding.
            if os.write(self._request_tx.fileno(), bytes((session.slot,))) != 1:
                raise RuntimeError("quota child request was not submitted")
        except BaseException:
            self._failed = True
            self._stop_worker()
            raise

    def _session_for(self, cursor):
        identity = id(cursor)
        session = self._sessions.get(identity)
        if session is not None:
            if session.cursor is not cursor:
                raise RuntimeError("quota child cursor identity was reused")
            return session
        used = {item.slot for item in self._sessions.values()}
        free = next((index for index in range(MAX_CURSOR_SESSIONS) if index not in used), None)
        if free is None:
            raise RuntimeError("too many live quota child cursors")
        session = _Session(
            cursor, free, cursor.epoch, cursor.source_cycle, cursor.cycle_index,
            tuple(int(value) for value in cursor.remaining),
        )
        self._sessions[identity] = session
        try:
            self._submit(session)
        except BaseException:
            self._sessions.pop(identity, None)
            raise
        return session

    def _await_done(self, slot_index: int, timeout: float):
        started = time.perf_counter_ns()
        try:
            self._wait_token(slot_index, timeout)
        finally:
            self.wait_ns += time.perf_counter_ns() - started

    def _collect(self, session: _Session):
        if not session.pending:
            raise RuntimeError("quota child producer ended before the cursor")
        slot = self._slots[session.slot]
        try:
            self._await_done(session.slot, RESULT_WAIT_SECONDS)
            generation, block_count, status = map(
                int, np.frombuffer(slot.response, dtype=np.int64)
            )
            if generation != session.generation or status or not 0 < block_count <= PACKAGE_BLOCKS:
                raise RuntimeError("quota child worker returned an invalid package")
            sizes = np.frombuffer(slot.sizes, dtype=np.uint16)
            children = np.frombuffer(slot.children, dtype=np.uint16)
            counts = np.frombuffer(slot.counts, dtype=np.uint16).reshape(
                PACKAGE_BLOCKS, self.source_count
            )
            remaining = list(session.next_remaining)
            ready = []
            for offset in range(block_count):
                total = sum(remaining)
                size = min(MIXING_BLOCK_ROWS, total)
                if int(sizes[offset]) != size:
                    raise RuntimeError("quota child package size mismatch")
                block = children[offset * MIXING_BLOCK_ROWS:offset * MIXING_BLOCK_ROWS + size].copy()
                decrements = counts[offset].astype(np.int64).tolist()
                if int(block.max(initial=0)) >= self.source_count or sum(decrements) != size:
                    raise RuntimeError("quota child package counts mismatch")
                key = (session.epoch, session.source_cycle, session.next_index + offset,
                       tuple(remaining), size)
                remaining = [value - count for value, count in zip(remaining, decrements)]
                if any(value < 0 for value in remaining):
                    raise RuntimeError("quota child package exceeded a source quota")
                ready.append((key, block, decrements))
            self.completed_packages += 1
            session.next_index += block_count
            session.next_remaining = tuple(remaining)
            session.ready.extend(ready)
        except BaseException:
            self._failed = True
            self._stop_worker()
            raise
        finally:
            session.pending = False
        self._submit(session)

    def next_children(self, cursor):
        if self._failed:
            raise RuntimeError("quota child worker has failed")
        if not any(cursor.remaining):
            return np.empty(0, dtype=np.uint16)
        session = self._session_for(cursor)
        if not session.ready:
            self._collect(session)
        expected, children, decrements = session.ready.popleft()
        actual = (cursor.epoch, cursor.source_cycle, cursor.cycle_index,
                  tuple(cursor.remaining), min(MIXING_BLOCK_ROWS, sum(cursor.remaining)))
        if expected != actual:
            self._failed = True
            self._stop_worker()
            raise RuntimeError("quota child cursor key mismatch")
        cursor.remaining = [value - count for value, count in zip(cursor.remaining, decrements)]
        cursor.cycle_index += 1
        return children

    def close_cursor(self, cursor):
        session = self._sessions.get(id(cursor))
        if session is None:
            return
        if session.cursor is not cursor:
            raise RuntimeError("quota child cursor identity changed during close")
        self._sessions.pop(id(cursor))
        if session.pending and self._slots is not None:
            try:
                self._await_done(session.slot, CURSOR_CLOSE_SECONDS)
            except BaseException:
                self._failed = True
                self._stop_worker()
                raise
            finally:
                session.pending = False
        if not self._sessions:
            self._stop_worker()

    def _stop_worker(self):
        process = self._process
        if process is None:
            return
        try:
            os.write(self._request_tx.fileno(), b"\xff")
        except (BrokenPipeError, OSError):
            pass
        process.join(timeout=1.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=2.0)
        alive = process.is_alive()
        if alive:
            self._failed = True
            raise RuntimeError("quota child worker survived termination")
        process.close()
        self._request_tx.close()
        self._result_rx.close()
        self._process = None
        self._slots = None
        self._request_tx = None
        self._result_rx = None
        self._responses.clear()

    def close(self):
        errors = []
        for session in tuple(self._sessions.values()):
            try:
                self.close_cursor(session.cursor)
            except BaseException as exc:
                errors.append(exc)
        if self._process is not None:
            try:
                self._stop_worker()
            except BaseException as exc:
                errors.append(exc)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise ExceptionGroup("quota child producer close failed", errors)
