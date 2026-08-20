from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data.dataset import IterableDataset

from .core import DatasetCapabilities, PreparedPipelineBatch
from .host_memory import HostMemoryBudget, HostMemoryCategory
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
        prefetch_chunk_batches: int | None = None,
        finalize_in_prefetch: bool = False,
        host_memory_budget: HostMemoryBudget | None = None,
        output_batch_bytes: int = 0,
        planner_token_bytes: int = 0,
        output_is_pinned: bool = False,
    ):
        self.planner = planner
        self.source = source
        self.finalize_batch = finalize_batch or (lambda data: data)
        if type(prefetch_workers) is not int or prefetch_workers < 0:
            raise ValueError("prefetch_workers must be a non-negative integer")
        if type(prefetch_batches) is not int or prefetch_batches <= 0:
            raise ValueError("prefetch_batches must be a positive integer")
        if prefetch_chunk_batches is not None and (
            type(prefetch_chunk_batches) is not int
            or prefetch_chunk_batches <= 0
        ):
            raise ValueError(
                "prefetch_chunk_batches must be null or a positive integer"
            )
        if (
            prefetch_chunk_batches is not None
            and prefetch_chunk_batches > prefetch_batches
        ):
            raise ValueError(
                "prefetch_chunk_batches cannot exceed prefetch_batches"
            )
        if type(finalize_in_prefetch) is not bool:
            raise ValueError("finalize_in_prefetch must be a boolean")
        if host_memory_budget is not None and not isinstance(
            host_memory_budget, HostMemoryBudget
        ):
            raise TypeError("host_memory_budget must be a HostMemoryBudget or null")
        for name, value in (
            ("output_batch_bytes", output_batch_bytes),
            ("planner_token_bytes", planner_token_bytes),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if type(output_is_pinned) is not bool:
            raise ValueError("output_is_pinned must be a boolean")
        composer = planner.pipeline_composer
        pipeline_prefetch_disabled = (
            prefetch_workers > 0
            and composer is not None
            and not composer.is_parallel_stateless
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
        self.prefetch_chunk_batches = prefetch_chunk_batches
        self._effective_prefetch_workers = effective_prefetch_workers
        self._finalize_in_prefetch = bool(
            finalize_in_prefetch and effective_prefetch_workers > 0
        )
        self.host_memory_budget = host_memory_budget
        self.output_batch_bytes = output_batch_bytes
        self.planner_token_bytes = planner_token_bytes
        self.output_is_pinned = output_is_pinned

    @property
    def active_prefetch_workers(self) -> int:
        return self._effective_prefetch_workers

    @property
    def maximum_prefetch_workers(self) -> int:
        return self._effective_prefetch_workers

    @property
    def active_prefetch_batches(self) -> int:
        return self.prefetch_batches

    @property
    def active_prefetch_chunk_batches(self) -> int:
        if self.prefetch_chunk_batches is not None:
            return self.prefetch_chunk_batches
        return max(
            1,
            self.active_prefetch_batches
            // max(2, self.active_prefetch_workers),
        )

    def _try_reserve_batch_memory(self):
        budget = self.host_memory_budget
        if budget is None:
            return ()
        reservations = []
        try:
            if self.planner_token_bytes:
                reservation = budget.try_reserve(
                    HostMemoryCategory.SEMANTIC_FIXED,
                    self.planner_token_bytes,
                    label="queued planner token",
                )
                if reservation is None:
                    return None
                reservations.append(reservation)
            if self.output_batch_bytes:
                reservation = budget.try_reserve(
                    (
                        HostMemoryCategory.PINNED
                        if self.output_is_pinned
                        else HostMemoryCategory.READY_PAGEABLE
                    ),
                    self.output_batch_bytes,
                    label="queued materialized batch",
                )
                if reservation is None:
                    for held in reservations:
                        held.release()
                    return None
                reservations.append(reservation)
            return tuple(reservations)
        except BaseException:
            for reservation in reservations:
                reservation.release()
            raise

    @staticmethod
    def _release_reservations(reservations) -> None:
        for reservation in reservations:
            reservation.release()

    @staticmethod
    def _lease_reservations(reservations):
        leases = []
        try:
            for reservation in reservations:
                leases.append(reservation.acquire_lease())
            for reservation in reservations:
                reservation.retire()
        except BaseException:
            for lease in leases:
                lease.release()
            for reservation in reservations:
                reservation.release()
            raise
        semantic = tuple(
            lease
            for lease in leases
            if lease.category is HostMemoryCategory.SEMANTIC_FIXED
        )
        host = tuple(
            lease
            for lease in leases
            if lease.category is not HostMemoryCategory.SEMANTIC_FIXED
        )
        return semantic, host

    def _run_decode_batches(self, batches):
        return self._decode_batches(batches)

    @staticmethod
    def _await_prefetch(future):
        return future.result()

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
            return self._finalize_prefetched(
                self._prepare_prefetched_pipelines(batches, decoded)
            )
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
        return self._finalize_prefetched(
            self._prepare_prefetched_pipelines(batches, decoded)
        )

    def _prepare_prefetched_pipelines(self, batches, decoded_batches):
        composer = self.planner.pipeline_composer
        if composer is None or not composer.is_parallel_stateless:
            return decoded_batches
        return tuple(
            (
                self.planner.prepare_parallel_pipeline_batch(
                    batch,
                    data,
                    sample_keys,
                ),
                mask,
                sample_keys,
            )
            for batch, (data, mask, sample_keys) in zip(batches, decoded_batches)
        )

    def _finalize_prefetched(self, decoded_batches):
        if not self._finalize_in_prefetch:
            return decoded_batches
        finalized = []
        for data, mask, sample_keys in decoded_batches:
            if isinstance(data, PreparedPipelineBatch):
                data = PreparedPipelineBatch(
                    self.finalize_batch(data.data),
                    data.composite_blob,
                )
            else:
                data = self.finalize_batch(data)
            finalized.append((data, mask, sample_keys))
        return tuple(finalized)

    def _finalize(self, batch, token, decoded):
        data, mask, sample_keys = decoded
        if isinstance(data, PreparedPipelineBatch):
            if not self._finalize_in_prefetch:
                data = PreparedPipelineBatch(
                    self.finalize_batch(data.data),
                    data.composite_blob,
                )
            data, token = self.planner.accept_parallel_pipeline_batch(
                batch,
                token,
                data,
            )
            return data, token, mask, sample_keys
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

    def _publish(self, batch, token, decoded, reservations=()):
        try:
            data, token, mask, sample_keys = self._finalize(
                batch,
                token,
                decoded,
            )
        except BaseException:
            self._release_reservations(reservations)
            raise
        return self._publish_finalized(
            data,
            token,
            mask,
            sample_keys,
            reservations,
        )

    def _publish_finalized(
        self,
        data,
        token,
        mask,
        sample_keys,
        reservations=(),
    ):
        semantic_leases, host_leases = self._lease_reservations(reservations)
        try:
            if self.planner.runtime_context.mode != "train":
                self.planner.commit_batch(token)
        except BaseException:
            for lease in (*semantic_leases, *host_leases):
                lease.release()
            raise
        return BatchEnvelope(
            data=data,
            token=token,
            is_real=mask,
            sample_keys=sample_keys,
            semantic_memory_leases=semantic_leases,
            host_memory_leases=host_leases,
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
            reservations = self._try_reserve_batch_memory()
            if reservations is None:
                raise RuntimeError(
                    "host-data memory budget cannot admit one synchronous batch"
                )
            try:
                decoded = self._decode_batch(batch)
                data, token, mask, sample_keys = self._finalize(
                    batch,
                    token,
                    decoded,
                )
            except BaseException:
                self._release_reservations(reservations)
                raise
            if batch.is_last:
                following = None
            else:
                following = self.planner.next_transactional_batch()
                if following is None:
                    batch, token = self.planner.finalize_terminal_token(token)
            yield self._publish_finalized(
                data,
                token,
                mask,
                sample_keys,
                reservations,
            )
            if batch.is_last:
                return
            current = following

    def _iter_prefetched(self):
        planned = iter(self._planned_transactions())
        pending = deque()
        pending_batches = 0
        exhausted = False
        active_reservation_batches = deque()

        def submit_chunk(executor):
            nonlocal exhausted, pending_batches
            limit = self.active_prefetch_batches
            if exhausted or pending_batches >= limit:
                return False
            items = []
            reservation_batches = []
            capacity = limit - pending_batches
            chunk_size = min(self.active_prefetch_chunk_batches, capacity)
            for _ in range(chunk_size):
                reservations = self._try_reserve_batch_memory()
                if reservations is None:
                    break
                try:
                    items.append(next(planned))
                except StopIteration:
                    self._release_reservations(reservations)
                    exhausted = True
                    break
                except BaseException:
                    self._release_reservations(reservations)
                    for held in reservation_batches:
                        self._release_reservations(held)
                    raise
                reservation_batches.append(reservations)
            if not items:
                return False
            try:
                future = executor.submit(
                    self._run_decode_batches,
                    tuple(batch for batch, _ in items),
                )
            except BaseException:
                for reservations in reservation_batches:
                    self._release_reservations(reservations)
                raise
            pending.append(
                (
                    tuple(items),
                    future,
                    tuple(reservation_batches),
                )
            )
            pending_batches += len(items)
            return True

        try:
            with ThreadPoolExecutor(
                max_workers=self.maximum_prefetch_workers
            ) as executor:
                while pending_batches < self.active_prefetch_batches and not exhausted:
                    if not submit_chunk(executor):
                        break
                if not pending and not exhausted:
                    raise RuntimeError(
                        "host-data memory budget cannot admit one prefetched batch"
                    )
                while pending:
                    items, future, reservation_batches = pending.popleft()
                    try:
                        decoded_batches = self._await_prefetch(future)
                    except BaseException:
                        for reservations in reservation_batches:
                            self._release_reservations(reservations)
                        raise
                    if len(decoded_batches) != len(items):
                        for reservations in reservation_batches:
                            self._release_reservations(reservations)
                        raise RuntimeError("prefetch worker changed the batch count")
                    active_reservation_batches.extend(reservation_batches)
                    for (batch, token), decoded in zip(items, decoded_batches):
                        reservations = active_reservation_batches.popleft()
                        pending_batches -= 1
                        envelope = self._publish(
                            batch,
                            token,
                            decoded,
                            reservations,
                        )
                        yield envelope
                    while (
                        pending_batches < self.active_prefetch_batches
                        and not exhausted
                    ):
                        if not submit_chunk(executor):
                            break
                if not exhausted:
                    raise RuntimeError(
                        "host-data memory budget stalled the ordered prefetch queue"
                    )
        finally:
            while active_reservation_batches:
                self._release_reservations(
                    active_reservation_batches.popleft()
                )
            while pending:
                _items, _future, reservation_batches = pending.popleft()
                for reservations in reservation_batches:
                    self._release_reservations(reservations)

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
