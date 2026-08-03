from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data.dataset import IterableDataset

from .core import DatasetCapabilities
from .stream import BatchEnvelope


_PLANNED_BATCH_CAPABILITIES = DatasetCapabilities(True, True, True, True)


class PlannedBatchDataset(IterableDataset):
    """Shared lifecycle and capabilities for planner-backed datasets."""

    YIELDS_BATCHES = True

    def __init__(self):
        super().__init__()
        self._partitioned_stream = None
        self._planned_decoder = None
        self._record_source = None

    @property
    def yields_batches(self):
        return True

    @property
    def capabilities(self):
        return _PLANNED_BATCH_CAPABILITIES

    @property
    def is_fixed_side_input(self):
        return self.fixed_side_input

    @property
    def is_internal_shuffleable(self):
        return True

    def __iter__(self):
        if self._partitioned_stream is None:
            self._build_partitioned_stream()
        yield from self._planned_decoder


class SourceBatchDataset:
    """Materialize v2 planner batches while preserving trainer transactions."""

    def __init__(
        self,
        planner,
        source,
        *,
        finalize_batch=None,
        prefetch_workers: int = 0,
        prefetch_batches: int = 32,
        finalize_in_prefetch: bool = False,
    ):
        self.planner = planner
        self.source = source
        self.finalize_batch = finalize_batch or (lambda data: data)
        if type(prefetch_workers) is not int or prefetch_workers < 0:
            raise ValueError("prefetch_workers must be a non-negative integer")
        if type(prefetch_batches) is not int or prefetch_batches <= 0:
            raise ValueError("prefetch_batches must be a positive integer")
        if type(finalize_in_prefetch) is not bool:
            raise ValueError("finalize_in_prefetch must be a boolean")
        pipeline_prefetch_disabled = (
            prefetch_workers > 0 and planner.pipeline_composer is not None
        )
        effective_prefetch_workers = (
            0 if pipeline_prefetch_disabled else prefetch_workers
        )
        if effective_prefetch_workers > 0 and not getattr(
            source,
            "thread_safe_materialization",
            False,
        ):
            raise ValueError(
                "prefetch_workers requires a source with thread-safe "
                "materialization"
        )
        self.prefetch_workers = prefetch_workers
        self.prefetch_batches = prefetch_batches
        self._effective_prefetch_workers = effective_prefetch_workers
        self._finalize_in_prefetch = bool(
            finalize_in_prefetch and effective_prefetch_workers > 0
        )

    def _decode_batch(self, batch):
        envelopes, mask = self.planner.local_slice(batch)
        if hasattr(self.source, "materialize_batch_with_keys"):
            data, sample_keys = self.source.materialize_batch_with_keys(envelopes)
        else:
            sample_keys = self._sample_keys(envelopes)
            data = self.source.materialize_batch(envelopes)
        return data, mask, sample_keys

    def _sample_keys(self, envelopes):
        if hasattr(self.source, "sample_keys_for_batch"):
            return self.source.sample_keys_for_batch(envelopes)
        return tuple(envelope.record_key for envelope in envelopes)

    def _decode_batches(self, batches):
        local_batches = [self.planner.local_slice(batch) for batch in batches]
        envelope_batches = tuple(envelopes for envelopes, _ in local_batches)
        if hasattr(self.source, "materialize_batches_with_keys"):
            materialized = tuple(
                self.source.materialize_batches_with_keys(envelope_batches)
            )
            if len(materialized) != len(batches):
                raise RuntimeError(
                    "source materialize_batches_with_keys changed the batch count"
                )
            decoded = tuple(
                (data, mask, sample_keys)
                for (data, sample_keys), (_, mask) in zip(
                    materialized, local_batches
                )
            )
            return self._finalize_prefetched(decoded)
        if hasattr(self.source, "materialize_batches"):
            data_batches = tuple(self.source.materialize_batches(envelope_batches))
            if len(data_batches) != len(batches):
                raise RuntimeError(
                    "source materialize_batches changed the batch count"
                )
        else:
            data_batches = tuple(
                self.source.materialize_batch(envelopes)
                for envelopes in envelope_batches
            )
        decoded = tuple(
            (
                data,
                mask,
                self._sample_keys(envelopes),
            )
            for data, (envelopes, mask) in zip(data_batches, local_batches)
        )
        return self._finalize_prefetched(decoded)

    def _finalize_prefetched(self, decoded_batches):
        if not self._finalize_in_prefetch:
            return decoded_batches
        return tuple(
            (self.finalize_batch(data), mask, sample_keys)
            for data, mask, sample_keys in decoded_batches
        )

    def _finalize(self, batch, token, decoded):
        data, mask, sample_keys = decoded
        if not self._finalize_in_prefetch:
            data = self.finalize_batch(data)
        data, token = self.planner.prepare_pipeline_batch(
            batch,
            token,
            data,
            sample_keys,
        )
        return data, token, mask, sample_keys

    def _planned_transactions(self):
        current = self.planner.next_transactional_batch()
        if current is None:
            raise RuntimeError(
                "stream epoch produced no global batch; reduce the batch size "
                "or increase the sampling rate"
            )
        while current is not None:
            batch, token = current
            if batch.is_last:
                following = None
            else:
                following = self.planner.next_transactional_batch()
                if following is None:
                    batch, token = self.planner.finalize_terminal_token(token)
            yield batch, token
            if batch.is_last:
                return
            current = following

    def _publish(self, batch, token, decoded):
        data, token, mask, sample_keys = self._finalize(
            batch,
            token,
            decoded,
        )
        if self.planner.runtime_context.mode != "train":
            self.planner.commit_batch(token)
        return BatchEnvelope(
            data=data,
            token=token,
            is_real=mask,
            sample_keys=sample_keys,
        )

    def _iter_synchronous(self):
        current = self.planner.next_transactional_batch()
        if current is None:
            raise RuntimeError(
                "stream epoch produced no global batch; reduce the batch size "
                "or increase the sampling rate"
            )
        while current is not None:
            batch, token = current
            decoded = self._decode_batch(batch)
            data, token, mask, sample_keys = self._finalize(
                batch,
                token,
                decoded,
            )
            if batch.is_last:
                following = None
            else:
                following = self.planner.next_transactional_batch()
                if following is None:
                    batch, token = self.planner.finalize_terminal_token(token)
            if self.planner.runtime_context.mode != "train":
                self.planner.commit_batch(token)
            yield BatchEnvelope(data, token, mask, sample_keys)
            if batch.is_last:
                return
            current = following

    def _iter_prefetched(self):
        planned = iter(self._planned_transactions())
        pending = deque()
        pending_batches = 0
        exhausted = False
        chunk_size = max(
            1,
            self.prefetch_batches // max(2, self._effective_prefetch_workers),
        )

        def submit_chunk(executor):
            nonlocal exhausted, pending_batches
            if exhausted or pending_batches >= self.prefetch_batches:
                return
            items = []
            capacity = self.prefetch_batches - pending_batches
            for _ in range(min(chunk_size, capacity)):
                try:
                    items.append(next(planned))
                except StopIteration:
                    exhausted = True
                    break
            if not items:
                return
            pending.append(
                (
                    tuple(items),
                    executor.submit(
                        self._decode_batches,
                        tuple(batch for batch, _ in items),
                    ),
                )
            )
            pending_batches += len(items)

        with ThreadPoolExecutor(
            max_workers=self._effective_prefetch_workers
        ) as executor:
            while pending_batches < self.prefetch_batches and not exhausted:
                submit_chunk(executor)
            while pending:
                items, future = pending.popleft()
                decoded_batches = future.result()
                if len(decoded_batches) != len(items):
                    raise RuntimeError("prefetch worker changed the batch count")
                for (batch, token), decoded in zip(items, decoded_batches):
                    pending_batches -= 1
                    yield self._publish(batch, token, decoded)
                while pending_batches < self.prefetch_batches and not exhausted:
                    submit_chunk(executor)

    def __iter__(self):
        if self.planner.finished:
            self.planner.begin_next_epoch()
        completed = False
        try:
            iterator = (
                self._iter_prefetched()
                if self._effective_prefetch_workers > 0
                else self._iter_synchronous()
            )
            yield from iterator
            completed = True
        finally:
            if not completed:
                if self.source.capabilities.resumable:
                    self.planner.rollback_uncommitted()
                elif hasattr(self.source, "close_cursor"):
                    self.source.close_cursor(self.planner._source_cursor)
