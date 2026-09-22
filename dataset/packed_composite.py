"""Fixed-width record routing for mixed, uniform-shape processed NPZ sources."""

import hashlib
from collections.abc import Sequence

import numpy as np

from .composite_source import CompositeRecordSource
from .core import canonical_pipeline_state_bytes
from .npz_source import DenseNpzSource
from .packed import PackedEnvelopeBatch, PackedRecordBlock


class _CompositeSampleKeys(Sequence):
    def __init__(self, child_id, keys):
        self.child_id = child_id
        self.keys = keys

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        return ("composite", self.child_id, self.keys[index])



class PackedCompositeRecordSource(CompositeRecordSource):
    """Mix deterministic child streams without per-record Python state."""

    thread_safe_materialization = True
    _child_shift = 48
    _row_mask = (1 << _child_shift) - 1

    @staticmethod
    def supports(sources):
        return bool(sources) and len(sources) < 65536 and all(
            type(source) is DenseNpzSource
            and len(source.shape_codes) == 1
            and sum(d.logical_row_count for d in source.descriptors) < (1 << 48)
            for source in sources
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.supports(self.child_sources):
            raise TypeError("packed mixing requires uniform dense NPZ children")
        self._child_shapes = np.asarray(
            [self.shape_codes[next(iter(source.shape_codes))] for source in self.child_sources],
            dtype=np.int32,
        )
        self._child_identities = tuple(
            hashlib.sha256(canonical_pipeline_state_bytes(source.manifest_state())).hexdigest()
            for source in self.child_sources
        )
        self._identity = hashlib.sha256(
            canonical_pipeline_state_bytes(self.manifest_state())
        ).hexdigest()

    def manifest_state(self):
        return {**super().manifest_state(), "packed_mixing": 2}

    def _fill_cycles(self, cursor, cycle_count):
        children = []
        for child, (source, weight) in enumerate(zip(self.child_sources, self.weights)):
            if cursor.exhausted[child]:
                values = np.empty(0, dtype=np.uint64)
            else:
                block, child_cursor = source.next_packed_records(
                    cursor.child_cursors[child], weight * cycle_count
                )
                cursor.child_cursors[child] = child_cursor
                values = block.record_ids
                if len(values) < weight * cycle_count:
                    cursor.exhausted[child] = True
            children.append(values)
        cycles = min(
            len(values) // weight for values, weight in zip(children, self.weights)
        )
        if not cycles:
            cursor.terminal = True
            cursor.pending = np.empty(0, dtype=np.uint64)
            cursor.pending_position = 0
            return False
        grid = np.empty((cycles, len(self.schedule)), dtype=np.uint64)
        schedule = np.asarray(self.schedule)
        for child, (values, weight) in enumerate(zip(children, self.weights)):
            columns = np.flatnonzero(schedule == child)
            grid[:, columns] = values[:cycles * weight].reshape(cycles, weight) | np.uint64(
                child << self._child_shift
            )
        cursor.pending = grid.reshape(-1)
        cursor.pending_position = 0
        cursor.cycle_index += cycles
        return True

    def _fill_quota_block(self, cursor):
        children = self._draw_quota_sources(cursor)
        cursor.pending_position = 0
        if not len(children):
            cursor.terminal = True
            cursor.pending = np.empty(0, dtype=np.uint64)
            return False
        pending = np.empty(len(children), dtype=np.uint64)
        for child in np.unique(children):
            child = int(child)
            positions = np.flatnonzero(children == child)
            block, child_cursor = self.child_sources[child].next_packed_records(
                cursor.child_cursors[child], len(positions)
            )
            if len(block.record_ids) != len(positions):
                raise RuntimeError("source exhausted before its declared mixing quota")
            cursor.child_cursors[child] = child_cursor
            pending[positions] = block.record_ids | np.uint64(child << self._child_shift)
        cursor.pending = pending
        return True

    def next_packed_records(self, cursor, limit):
        if type(limit) is not int or limit <= 0:
            raise ValueError("composite chunk limit must be positive")
        parts = []
        count = 0
        while count < limit and not cursor.terminal:
            if cursor.pending_position >= len(cursor.pending):
                if self.quotas is not None:
                    filled = self._fill_quota_block(cursor)
                else:
                    cycles = max(
                        1, (limit - count + len(self.schedule) - 1) // len(self.schedule)
                    )
                    filled = self._fill_cycles(cursor, cycles)
                if not filled:
                    break
            take = min(limit - count, len(cursor.pending) - cursor.pending_position)
            parts.append(cursor.pending[cursor.pending_position:cursor.pending_position + take])
            cursor.pending_position += take
            count += take
        values = np.concatenate(parts) if parts else np.empty(0, dtype=np.uint64)
        return PackedRecordBlock(values), cursor

    def next_envelopes(self, cursor, limit):
        block, cursor = self.next_packed_records(cursor, limit)
        return self.envelopes_from_record_ids(block.record_ids), cursor

    def next_envelope(self, cursor):
        values, cursor = self.next_envelopes(cursor, 1)
        return (values[0] if values else None), cursor

    def shape_codes_for_record_ids(self, record_ids):
        children = np.asarray(record_ids, dtype=np.uint64) >> np.uint64(self._child_shift)
        if np.any(children >= len(self.child_sources)):
            raise ValueError("invalid packed composite child")
        return self._child_shapes[children.astype(np.intp)]

    def envelopes_from_record_ids(self, record_ids):
        output = []
        for value in record_ids:
            child = int(value) >> self._child_shift
            if not 0 <= child < len(self.child_sources):
                raise ValueError("invalid packed composite child")
            values = np.asarray([int(value) & self._row_mask], dtype=np.uint64)
            envelope = self.child_sources[child].envelopes_from_record_ids(values)[0]
            output.append(self._wrap(child, envelope))
        return tuple(output)

    def _child_batch(self, envelopes):
        if envelopes.identity != self._identity:
            raise RuntimeError("packed composite batch belongs to a different source")
        children = envelopes.record_ids >> np.uint64(self._child_shift)
        child = int(children[0])
        if not 0 <= child < len(self.child_sources) or np.any(children != child):
            return None
        source = self.child_sources[child]
        values = np.ascontiguousarray(envelopes.record_ids & np.uint64(self._row_mask))
        return child, PackedEnvelopeBatch(
            values, source.envelopes_from_record_ids, self._child_identities[child]
        )

    def materialize_batch_with_keys(self, envelopes):
        routed = (
            self._child_batch(envelopes)
            if isinstance(envelopes, PackedEnvelopeBatch)
            else None
        )
        if routed is None:
            return super().materialize_batch(envelopes), tuple(e.record_key for e in envelopes)
        child, batch = routed
        data, keys = self.child_sources[child].materialize_batch_with_keys(batch)
        # Child decoding already applies its deterministic symmetry. Composite
        # batch transforms need the composite identity, not a child-only key.
        return data, _CompositeSampleKeys(self.child_ids[child], keys)

    def materialize_batch(self, envelopes):
        return self.materialize_batch_with_keys(envelopes)[0]

    def materialize_batches_with_keys(self, batches):
        return [self.materialize_batch_with_keys(batch) for batch in batches]

    def save_cursor(self, cursor):
        pending = cursor.pending
        position = cursor.pending_position
        cursor.pending = []
        cursor.pending_position = 0
        try:
            state = super().save_cursor(cursor)
        finally:
            cursor.pending = pending
            cursor.pending_position = position
        state["packed_pending"] = np.asarray(pending[position:], dtype="<u8").tobytes().hex()
        return state

    def restore_cursor(self, state):
        cursor = super().restore_cursor(state)
        values = np.frombuffer(bytes.fromhex(state["packed_pending"]), dtype="<u8").copy()
        self.shape_codes_for_record_ids(values)
        cursor.pending = values
        return cursor
