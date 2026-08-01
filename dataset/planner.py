from __future__ import annotations

import hashlib
import struct
from collections import deque
from dataclasses import asdict, dataclass, replace

import numpy as np

from .core import (
    DatasetCapabilities,
    DatasetRuntimeContext,
    PipelineStateComposer,
    canonical_pipeline_state_bytes,
)
from .packed import (
    PACKED_RESERVOIR_ALGORITHM,
    PackedEnvelopeBatch,
    PackedReadyState,
    PackedReservoirState,
    PackedUInt64ReadyBuffer,
    PackedUInt64ShuffleReservoir,
)
from .shuffle import ReservoirState, StreamingShuffleReservoir
from .source import RecordEnvelope, RecordSource, SOURCE_CURSOR_SCHEMA


PLANNER_ALGORITHM = "dataset-planner-v2"
SOURCE_CHUNK_SIZE = 1024


@dataclass(frozen=True, slots=True)
class PlannerConfig:
    shuffle: bool
    shuffle_buffer_size: int = 32768
    shuffle_buffer_bytes: int | None = None
    max_shape_queues: int = 64
    steps_per_epoch: int | None = None

    def __post_init__(self) -> None:
        if type(self.shuffle) is not bool:
            raise TypeError("planner shuffle must be a bool")
        if type(self.shuffle_buffer_size) is not int or self.shuffle_buffer_size <= 0:
            raise ValueError("shuffle_buffer_size must be a positive integer")
        if self.shuffle_buffer_bytes is not None and (
            type(self.shuffle_buffer_bytes) is not int
            or self.shuffle_buffer_bytes <= 0
        ):
            raise ValueError("shuffle_buffer_bytes must be a positive integer")
        if type(self.max_shape_queues) is not int or self.max_shape_queues <= 0:
            raise ValueError("max_shape_queues must be a positive integer")
        if self.steps_per_epoch is not None and (
            type(self.steps_per_epoch) is not int or self.steps_per_epoch <= 0
        ):
            raise ValueError("steps_per_epoch must be a positive integer")


@dataclass(frozen=True, slots=True)
class PlannedEnvelopeBatch:
    envelopes: tuple[RecordEnvelope, ...] | PackedEnvelopeBatch
    is_real: tuple[bool, ...]
    epoch: int
    batch_index: int
    is_last: bool


@dataclass(frozen=True, slots=True)
class PlannerState:
    schema: str
    algorithm: str
    manifest_digest: str
    config: PlannerConfig
    epoch: int
    batch_index: int
    source_cycle: int
    source_cursor: dict
    reservoir: ReservoirState[RecordEnvelope] | PackedReservoirState
    ready: tuple[RecordEnvelope, ...] | PackedReadyState
    shape_queues: tuple[tuple[int, tuple[RecordEnvelope, ...]], ...]
    source_exhausted: bool
    finished: bool


@dataclass(frozen=True, slots=True)
class PlannerBatchToken:
    epoch: int
    batch_index: int
    before_digest: str
    after_digest: str
    batch: PlannedEnvelopeBatch
    before_state: PlannerState | None
    after_state: PlannerState | None
    before_pipeline_blob: bytes
    after_pipeline_blob: bytes
    distributed_policy: str

    @property
    def coordination_descriptor(self) -> tuple:
        if self.distributed_policy == "rank-sharded-budget-v1":
            return (
                self.distributed_policy,
                self.epoch,
                self.batch_index,
                self.batch.is_last,
            )
        return (
            self.distributed_policy,
            self.epoch,
            self.batch_index,
            self.before_digest,
            self.after_digest,
        )


@dataclass(frozen=True, slots=True)
class PreparedPlannerCommit:
    before_digest: str
    after_digest: str
    after_state: PlannerState | None
    after_pipeline_blob: bytes
    token_count: int
    coordination_descriptor: tuple


class DatasetPlanner:
    """Mutable v2 planner whose explicit state is snapshotted only when needed."""

    def __init__(
        self,
        source: RecordSource,
        runtime_context: DatasetRuntimeContext,
        config: PlannerConfig,
        pipeline_composer: PipelineStateComposer | None = None,
    ) -> None:
        if not source.capabilities.deterministic:
            raise ValueError("dataset planner requires a deterministic record source")
        self.source = source
        self.runtime_context = runtime_context
        self.config = config
        self._rank_sharded = (
            runtime_context.world_size > 1
            and not source.capabilities.exact_distributed_partition
        )
        self.distributed_policy = (
            "rank-sharded-budget-v1"
            if self._rank_sharded
            else "replicated-global-plan-v1"
        )
        if self._rank_sharded:
            if runtime_context.mode != "train":
                raise ValueError(
                    "distributed evaluation requires a source with exact "
                    "distributed partitioning"
                )
            if config.steps_per_epoch is None:
                raise ValueError(
                    "distributed unknown-length training requires steps_per_epoch"
                )
            if not hasattr(source, "start_cycle"):
                raise TypeError(
                    "rank-sharded source must implement deterministic cycle restart"
                )
        if hasattr(source, "configure_distributed"):
            source.configure_distributed(
                runtime_context.world_size,
                runtime_context.rank,
                rank_sharded=self._rank_sharded,
            )
        self.pipeline_composer = pipeline_composer
        self._committed_pipeline_blob = (
            pipeline_composer.initial_blob if pipeline_composer is not None else b""
        )
        self._yield_pipeline_blob = self._committed_pipeline_blob
        self._manifest_digest = hashlib.sha256(
            canonical_pipeline_state_bytes(source.manifest_state())
        ).hexdigest()
        shape_codes = getattr(source, "shape_codes", None)
        self._uniform_shape_code = (
            next(iter(shape_codes.values()))
            if isinstance(shape_codes, dict) and len(shape_codes) == 1
            else None
        )
        self._packed_uniform = (
            self._uniform_shape_code is not None
            and not self._rank_sharded
            and callable(getattr(source, "next_packed_records", None))
            and callable(getattr(source, "envelopes_from_record_ids", None))
        )
        self._epoch = 0
        self._batch_index = 0
        self._source_cycle = 0
        self._source_cursor = None
        self._reservoir = None
        self._ready: deque[RecordEnvelope] = deque()
        self._packed_ready = PackedUInt64ReadyBuffer()
        self._shape_queues: dict[int, list[RecordEnvelope]] = {}
        self._source_exhausted = False
        self._finished = False
        self.start_epoch(0)
        self._committed_state = self.state() if source.capabilities.resumable else None
        self._yield_state = self._committed_state
        self._committed_digest = self._initial_digest(0)
        self._yield_digest = self._committed_digest

    @property
    def capabilities(self):
        return DatasetCapabilities(
            yields_batches=True,
            resumable=self.source.capabilities.resumable,
            deterministic=True,
            supports_batch_pipeline=True,
        )

    @property
    def shuffle_window_size(self) -> int:
        return self.config.shuffle_buffer_size

    @shuffle_window_size.setter
    def shuffle_window_size(self, value: int) -> None:
        value = int(value)
        if value == self.config.shuffle_buffer_size:
            return
        if self.batch_index or self.reservoir_stats.offered:
            raise RuntimeError("cannot resize an active planner shuffle reservoir")
        self.config = replace(self.config, shuffle_buffer_size=value)
        self.start_epoch(self.epoch)
        self._committed_state = (
            self.state() if self.source.capabilities.resumable else None
        )
        self._yield_state = self._committed_state
        self._committed_digest = self._initial_digest(self.epoch)
        self._yield_digest = self._committed_digest

    @property
    def resume_unsupported_reason(self):
        if self.source.capabilities.resumable:
            return None
        return "dataset source does not support exact resume"

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def batch_index(self) -> int:
        return self._batch_index

    @property
    def committed_epoch(self) -> int:
        if self._committed_state is not None:
            return self._committed_state.epoch
        return self._epoch

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def reservoir_stats(self):
        return self._reservoir.stats

    def start_epoch(self, epoch: int) -> None:
        if type(epoch) is not int or epoch < 0:
            raise ValueError("planner epoch must be a non-negative integer")
        previous_cursor = getattr(self, "_source_cursor", None)
        if previous_cursor is not None and hasattr(self.source, "close_cursor"):
            self.source.close_cursor(previous_cursor)
        self._epoch = epoch
        self._batch_index = 0
        self._source_cycle = 0
        self._source_cursor = self.source.start_epoch(
            epoch, self.runtime_context.rank
        )
        self._reservoir = self._new_reservoir(0)
        self._ready.clear()
        self._packed_ready.clear()
        self._shape_queues.clear()
        self._source_exhausted = False
        self._finished = False

    def _new_reservoir(self, source_cycle: int):
        capacity = self.config.shuffle_buffer_size if self.config.shuffle else 1
        stream_key = (
            (self._manifest_digest,)
            if source_cycle == 0
            else (self._manifest_digest, "source-cycle", source_cycle)
        )
        if self._packed_uniform:
            return PackedUInt64ShuffleReservoir(
                capacity,
                seed=self.runtime_context.seed,
                epoch=self._epoch,
                stream_key=stream_key,
            )
        return StreamingShuffleReservoir(
            capacity,
            seed=self.runtime_context.seed,
            epoch=self._epoch,
            stream_key=stream_key,
            max_bytes=self.config.shuffle_buffer_bytes,
        )

    def _ready_count(self) -> int:
        return len(self._packed_ready) if self._packed_uniform else len(self._ready)

    def _clear_ready(self) -> None:
        self._packed_ready.clear()
        self._ready.clear()

    @property
    def planning_batch_size(self) -> int:
        return (
            self.runtime_context.local_batch_size
            if self._rank_sharded
            else self.runtime_context.global_batch_size
        )

    def _restart_source_cycle(self) -> None:
        if (
            not self._rank_sharded
            or not self._source_exhausted
            or self._ready_count()
        ):
            raise RuntimeError("cannot restart an active planner source cycle")
        if hasattr(self.source, "close_cursor"):
            self.source.close_cursor(self._source_cursor)
        self._source_cycle += 1
        self._source_cursor = self.source.start_cycle(
            self._epoch,
            self._source_cycle,
            self.runtime_context.rank,
        )
        self._reservoir = self._new_reservoir(self._source_cycle)
        self._source_exhausted = False

    def _finish_budget_epoch(self) -> None:
        if not self._reservoir.closed:
            self._reservoir.drain()
        self._clear_ready()
        self._shape_queues.clear()
        self._source_exhausted = True
        self._finished = True

    def _initial_digest(self, epoch: int) -> str:
        return hashlib.sha256(
            canonical_pipeline_state_bytes(
                {
                    "algorithm": PLANNER_ALGORITHM,
                    "manifest_digest": self._manifest_digest,
                    "runtime": asdict(self.runtime_context.global_contract),
                    "config": asdict(self.config),
                    "epoch": epoch,
                    "kind": "epoch-start",
                    "pipeline_state": self._committed_pipeline_blob,
                }
            )
        ).hexdigest()

    def _token_digest(
        self,
        before_digest: str,
        batch: PlannedEnvelopeBatch,
        pipeline_blob: bytes,
    ) -> str:
        digest = hashlib.sha256()
        digest.update(b"NNUE-dataset-planner-token-v2\0")
        digest.update(bytes.fromhex(before_digest))
        digest.update(
            struct.pack(
                "<qq?Q",
                batch.epoch,
                batch.batch_index,
                batch.is_last,
                len(batch.envelopes),
            )
        )
        if hasattr(self.source, "update_batch_record_digest"):
            self.source.update_batch_record_digest(
                digest,
                batch.envelopes,
                batch.is_real,
            )
        else:
            for envelope, is_real in zip(batch.envelopes, batch.is_real):
                if hasattr(self.source, "update_record_key_digest"):
                    self.source.update_record_key_digest(digest, envelope.record_key)
                else:
                    key = canonical_pipeline_state_bytes(envelope.record_key)
                    digest.update(struct.pack("<Q", len(key)))
                    digest.update(key)
                digest.update(b"\x01" if is_real else b"\x00")
        digest.update(struct.pack("<Q", len(pipeline_blob)))
        digest.update(pipeline_blob)
        return digest.hexdigest()

    def _next_emitted(self) -> RecordEnvelope | None:
        self._refill_ready()
        return self._ready.popleft() if self._ready else None

    def _refill_ready(self) -> None:
        if self._packed_uniform:
            self._refill_packed_ready()
            return
        while not self._ready and not self._source_exhausted:
            if hasattr(self.source, "next_envelopes"):
                envelopes, self._source_cursor = self.source.next_envelopes(
                    self._source_cursor,
                    SOURCE_CHUNK_SIZE,
                )
                if not envelopes:
                    self._ready.extend(self._reservoir.drain())
                    self._source_exhausted = True
                    break
                if getattr(self.source, "zero_resident_envelopes", False):
                    self._ready.extend(
                        self._reservoir.offer_zero_sized(envelopes)
                    )
                else:
                    self._ready.extend(
                        self._reservoir.offer_many(
                            (envelope, envelope.resident_bytes)
                            for envelope in envelopes
                        )
                    )
                continue
            envelope, self._source_cursor = self.source.next_envelope(
                self._source_cursor
            )
            if envelope is None:
                self._ready.extend(self._reservoir.drain())
                self._source_exhausted = True
                break
            self._ready.extend(
                self._reservoir.offer(
                    envelope, size_bytes=envelope.resident_bytes
                )
            )

    def _refill_packed_ready(self) -> None:
        while not self._packed_ready and not self._source_exhausted:
            block, self._source_cursor = self.source.next_packed_records(
                self._source_cursor,
                SOURCE_CHUNK_SIZE,
            )
            if not len(block):
                self._packed_ready.extend(self._reservoir.drain())
                self._source_exhausted = True
                break
            self._packed_ready.extend(self._reservoir.offer_block(block))

    def _next_uniform_batch(self) -> PlannedEnvelopeBatch | None:
        if self._packed_uniform:
            return self._next_packed_uniform_batch()
        batch_size = self.planning_batch_size
        envelopes = []
        empty_cycles = 0
        while len(envelopes) < batch_size:
            self._refill_ready()
            if not self._ready:
                if (
                    self._rank_sharded
                    and self.config.steps_per_epoch is not None
                    and self._batch_index < self.config.steps_per_epoch
                ):
                    empty_cycles = (
                        empty_cycles + 1
                        if self._reservoir.stats.offered == 0
                        else 0
                    )
                    if empty_cycles > 16:
                        raise RuntimeError(
                            "rank-sharded source produced no admitted records in "
                            "16 consecutive cycles"
                        )
                    self._restart_source_cycle()
                    continue
                if self.runtime_context.mode == "train":
                    self._finished = True
                    return None
                if not envelopes:
                    self._finished = True
                    return None
                real_count = len(envelopes)
                while len(envelopes) < batch_size:
                    envelopes.append(
                        envelopes[(len(envelopes) - real_count) % real_count]
                    )
                self._finished = True
                return self._make_batch(
                    tuple(envelopes),
                    (True,) * real_count + (False,) * (batch_size - real_count),
                    is_last=True,
                )

            take = min(batch_size - len(envelopes), len(self._ready))
            pop_ready = self._ready.popleft
            append = envelopes.append
            for _ in range(take):
                envelope = pop_ready()
                if envelope.shape_code != self._uniform_shape_code:
                    raise RuntimeError(
                        "source emitted a record outside its declared uniform shape"
                    )
                append(envelope)

        return self._make_batch(
            tuple(envelopes),
            (True,) * batch_size,
            is_last=False,
        )

    def _next_packed_uniform_batch(self) -> PlannedEnvelopeBatch | None:
        batch_size = self.planning_batch_size
        parts = []
        count = 0
        while count < batch_size:
            self._refill_packed_ready()
            if not self._packed_ready:
                if self.runtime_context.mode == "train":
                    self._finished = True
                    return None
                if not count:
                    self._finished = True
                    return None
                values = np.concatenate(parts) if len(parts) > 1 else parts[0]
                real_count = len(values)
                values = np.resize(values, batch_size)
                self._finished = True
                envelopes = PackedEnvelopeBatch(
                    values,
                    self.source.envelopes_from_record_ids,
                    self._manifest_digest,
                )
                return self._make_batch(
                    envelopes,
                    (True,) * real_count + (False,) * (batch_size - real_count),
                    is_last=True,
                )
            take = min(batch_size - count, len(self._packed_ready))
            parts.append(self._packed_ready.pop(take))
            count += take
        values = np.concatenate(parts) if len(parts) > 1 else parts[0]
        envelopes = PackedEnvelopeBatch(
            values,
            self.source.envelopes_from_record_ids,
            self._manifest_digest,
        )
        return self._make_batch(
            envelopes,
            (True,) * batch_size,
            is_last=False,
        )

    def _append_shape(self, envelope: RecordEnvelope) -> list[RecordEnvelope]:
        queue = self._shape_queues.get(envelope.shape_code)
        if queue is None:
            if len(self._shape_queues) >= self.config.max_shape_queues:
                raise RuntimeError(
                    "dataset source exceeded the configured shape queue limit"
                )
            queue = self._shape_queues[envelope.shape_code] = []
        queue.append(envelope)
        return queue

    def _make_batch(
        self,
        envelopes: tuple[RecordEnvelope, ...] | PackedEnvelopeBatch,
        is_real: tuple[bool, ...],
        *,
        is_last: bool,
    ) -> PlannedEnvelopeBatch:
        budget_last = (
            self.config.steps_per_epoch is not None
            and self._batch_index + 1 >= self.config.steps_per_epoch
        )
        batch = PlannedEnvelopeBatch(
            envelopes=envelopes,
            is_real=is_real,
            epoch=self._epoch,
            batch_index=self._batch_index,
            is_last=is_last or budget_last,
        )
        self._batch_index += 1
        if budget_last:
            self._finish_budget_epoch()
        return batch

    def next_batch(self) -> PlannedEnvelopeBatch | None:
        if self._finished:
            return None
        if self._uniform_shape_code is not None:
            return self._next_uniform_batch()
        batch_size = self.planning_batch_size
        empty_cycles = 0
        while True:
            envelope = self._next_emitted()
            if envelope is None:
                if (
                    self._rank_sharded
                    and self.config.steps_per_epoch is not None
                    and self._batch_index < self.config.steps_per_epoch
                ):
                    empty_cycles = (
                        empty_cycles + 1
                        if self._reservoir.stats.offered == 0
                        else 0
                    )
                    if empty_cycles > 16:
                        raise RuntimeError(
                            "rank-sharded source produced no admitted records in "
                            "16 consecutive cycles"
                        )
                    self._restart_source_cycle()
                    continue
                if self.runtime_context.mode == "train":
                    self._shape_queues.clear()
                    self._finished = True
                    return None
                nonempty = sorted(
                    shape for shape, queue in self._shape_queues.items() if queue
                )
                if not nonempty:
                    self._finished = True
                    return None
                shape_code = nonempty[0]
                real = self._shape_queues.pop(shape_code)
                padded = list(real)
                while len(padded) < batch_size:
                    padded.append(real[(len(padded) - len(real)) % len(real)])
                is_last = not any(self._shape_queues.values())
                if is_last:
                    self._finished = True
                return self._make_batch(
                    tuple(padded),
                    (True,) * len(real) + (False,) * (batch_size - len(real)),
                    is_last=is_last,
                )

            queue = self._append_shape(envelope)
            if len(queue) == batch_size:
                envelopes = tuple(queue)
                self._shape_queues[envelope.shape_code] = []
                return self._make_batch(
                    envelopes,
                    (True,) * batch_size,
                    is_last=False,
                )

    def local_slice(
        self, batch: PlannedEnvelopeBatch
    ) -> tuple[tuple[RecordEnvelope, ...], np.ndarray]:
        if self._rank_sharded:
            envelopes = batch.envelopes
            mask = np.asarray(batch.is_real, dtype=np.bool_)
        else:
            identity = self.runtime_context.rank_local_identity
            envelopes = batch.envelopes[
                identity.slice_start : identity.slice_stop
            ]
            mask = np.asarray(
                batch.is_real[identity.slice_start : identity.slice_stop],
                dtype=np.bool_,
            )
        if len(envelopes) != self.runtime_context.local_batch_size:
            raise RuntimeError(
                f"planner produced {len(envelopes)} local rows, expected "
                f"{self.runtime_context.local_batch_size}"
            )
        return envelopes, mask

    def state(self) -> PlannerState:
        return PlannerState(
            schema=SOURCE_CURSOR_SCHEMA,
            algorithm=PLANNER_ALGORITHM,
            manifest_digest=self._manifest_digest,
            config=self.config,
            epoch=self._epoch,
            batch_index=self._batch_index,
            source_cycle=self._source_cycle,
            source_cursor=self.source.save_cursor(self._source_cursor),
            reservoir=self._reservoir.state(),
            ready=(
                self._packed_ready.state()
                if self._packed_uniform
                else tuple(self._ready)
            ),
            shape_queues=tuple(
                (shape, tuple(queue))
                for shape, queue in sorted(self._shape_queues.items())
                if queue
            ),
            source_exhausted=self._source_exhausted,
            finished=self._finished,
        )

    def restore(self, state: PlannerState) -> None:
        if state.schema != SOURCE_CURSOR_SCHEMA:
            raise ValueError(f"unsupported planner cursor schema {state.schema!r}")
        if state.algorithm != PLANNER_ALGORITHM:
            raise ValueError(f"unsupported planner algorithm {state.algorithm!r}")
        if state.manifest_digest != self._manifest_digest:
            raise ValueError("cannot restore planner after the source manifest changed")
        if state.config != self.config:
            raise ValueError("cannot restore planner after its configuration changed")
        if state.batch_index < 0:
            raise ValueError("planner state has a negative batch index")
        if self.config.steps_per_epoch is not None:
            if state.batch_index > self.config.steps_per_epoch:
                raise ValueError("planner state exceeds its epoch step budget")
            if (
                state.batch_index == self.config.steps_per_epoch
                and not state.finished
            ):
                raise ValueError(
                    "planner state reached its epoch step budget without finishing"
                )
        if type(state.source_exhausted) is not bool or type(state.finished) is not bool:
            raise ValueError("planner state contains a malformed lifecycle flag")
        packed_state = isinstance(state.reservoir, PackedReservoirState)
        if packed_state != self._packed_uniform:
            raise ValueError("planner packed storage contract changed")
        if self._packed_uniform and not isinstance(state.ready, PackedReadyState):
            raise ValueError("planner packed ready state is malformed")
        if not self._packed_uniform and not isinstance(state.ready, tuple):
            raise ValueError("planner object ready state is malformed")
        if state.source_exhausted != state.reservoir.closed:
            raise ValueError("planner and reservoir exhaustion state disagree")
        ready_nonempty = bool(len(state.ready))
        if state.finished and (
            not state.source_exhausted or ready_nonempty or state.shape_queues
        ):
            raise ValueError("finished planner state still contains active work")
        if len(state.shape_queues) > self.config.max_shape_queues:
            raise ValueError("planner state exceeds the configured shape queue limit")
        shapes = [shape for shape, _ in state.shape_queues]
        if len(shapes) != len(set(shapes)):
            raise ValueError("planner state contains duplicate shape queues")
        batch_size = self.planning_batch_size
        if any(
            not queue or len(queue) >= batch_size
            for _, queue in state.shape_queues
        ):
            raise ValueError("planner state contains an invalid shape queue")
        if any(
            envelope.shape_code != shape
            for shape, queue in state.shape_queues
            for envelope in queue
        ):
            raise ValueError("planner state shape queue contains a mismatched record")

        self.start_epoch(state.epoch)
        self._batch_index = state.batch_index
        if state.source_cycle < 0:
            raise ValueError("planner state has a negative source cycle")
        self._source_cycle = state.source_cycle
        if hasattr(self.source, "close_cursor"):
            self.source.close_cursor(self._source_cursor)
        self._source_cursor = self.source.restore_cursor(state.source_cursor)
        self._reservoir = self._new_reservoir(self._source_cycle)
        self._reservoir.restore(state.reservoir)
        if self._packed_uniform:
            self._packed_ready.restore(state.ready)
        else:
            self._ready.extend(state.ready)
        self._shape_queues.update(
            (shape, list(queue)) for shape, queue in state.shape_queues
        )
        self._source_exhausted = state.source_exhausted
        self._finished = state.finished

    def next_transactional_batch(
        self,
    ) -> tuple[PlannedEnvelopeBatch, PlannerBatchToken] | None:
        before_state = self._yield_state
        before_digest = self._yield_digest
        before_pipeline_blob = self._yield_pipeline_blob
        batch = self.next_batch()
        if batch is None:
            return None
        after_state = self.state() if self.source.capabilities.resumable else None
        self._yield_state = after_state
        after_digest = self._token_digest(
            before_digest, batch, before_pipeline_blob
        )
        token = PlannerBatchToken(
            epoch=batch.epoch,
            batch_index=batch.batch_index,
            before_digest=before_digest,
            after_digest=after_digest,
            batch=batch,
            before_state=before_state,
            after_state=after_state,
            before_pipeline_blob=before_pipeline_blob,
            after_pipeline_blob=before_pipeline_blob,
            distributed_policy=self.distributed_policy,
        )
        self._yield_digest = after_digest
        return batch, token

    def prepare_pipeline_batch(
        self,
        batch: PlannedEnvelopeBatch,
        token: PlannerBatchToken,
        data: dict,
        sample_keys: tuple,
    ) -> tuple[dict, PlannerBatchToken]:
        if batch != token.batch:
            raise RuntimeError("pipeline batch does not match its planner token")
        if self.pipeline_composer is None:
            return data, token
        if token.after_digest != self._yield_digest:
            raise RuntimeError("pipeline token is not the latest planned batch")
        prepared = self.pipeline_composer.prepare_batch(
            data,
            sample_keys=sample_keys,
            rng_keys=tuple(
                (batch.epoch, key, occurrence)
                for occurrence, key in enumerate(sample_keys)
            ),
            current_blob=token.before_pipeline_blob,
        )
        after_digest = self._token_digest(
            token.before_digest, batch, prepared.composite_blob
        )
        token = replace(
            token,
            after_digest=after_digest,
            after_pipeline_blob=prepared.composite_blob,
        )
        self._yield_pipeline_blob = prepared.composite_blob
        self._yield_digest = after_digest
        return prepared.data, token

    def finalize_terminal_token(
        self, token: PlannerBatchToken
    ) -> tuple[PlannedEnvelopeBatch, PlannerBatchToken]:
        if not self.finished:
            raise RuntimeError("cannot finalize a non-terminal planner token")
        if token.after_digest != self._yield_digest:
            raise RuntimeError("terminal planner token is not the latest yielded token")
        batch = replace(token.batch, is_last=True)
        after_state = self.state() if self.source.capabilities.resumable else None
        self._yield_state = after_state
        after_digest = self._token_digest(
            token.before_digest, batch, token.after_pipeline_blob
        )
        finalized = replace(
            token,
            after_digest=after_digest,
            batch=batch,
            after_state=after_state,
        )
        self._yield_digest = after_digest
        return batch, finalized

    def prepare_commit(self, tokens) -> PreparedPlannerCommit:
        tokens = tuple(tokens)
        if not tokens:
            raise ValueError("planner commit transaction must contain at least one token")
        digest = self._committed_digest
        pipeline_blob = self._committed_pipeline_blob
        expected_state = self._committed_state
        expected_epoch = (
            self._committed_state.epoch
            if self._committed_state is not None
            else tokens[0].epoch
        )
        expected_batch = (
            self._committed_state.batch_index
            if self._committed_state is not None
            else tokens[0].batch_index
        )
        for token in tokens:
            if token.before_digest != digest:
                raise RuntimeError("non-contiguous planner token digest")
            if (token.epoch, token.batch_index) != (
                expected_epoch,
                expected_batch,
            ):
                raise RuntimeError("non-contiguous planner batch token")
            if token.before_pipeline_blob != pipeline_blob:
                raise RuntimeError("planner token pipeline state is non-contiguous")
            if (
                self.source.capabilities.resumable
                and token.before_state != expected_state
            ):
                raise RuntimeError("planner token does not start at the committed cursor")
            if self.pipeline_composer is not None:
                self.pipeline_composer._decode_blob(token.after_pipeline_blob)
            elif token.after_pipeline_blob:
                raise RuntimeError("planner token contains unexpected pipeline state")
            if (
                not isinstance(token.after_digest, str)
                or len(token.after_digest) != 64
            ):
                raise RuntimeError("planner token digest is malformed")
            digest = token.after_digest
            expected_batch += 1
            expected_state = token.after_state
            pipeline_blob = token.after_pipeline_blob
        coordination_descriptor = (
            (
                self.distributed_policy,
                len(tokens),
                tokens[0].epoch,
                tokens[0].batch_index,
                tokens[-1].epoch,
                tokens[-1].batch_index,
                tokens[-1].batch.is_last,
            )
            if self._rank_sharded
            else (
                self.distributed_policy,
                self._committed_digest,
                digest,
                len(tokens),
            )
        )
        return PreparedPlannerCommit(
            before_digest=self._committed_digest,
            after_digest=digest,
            after_state=tokens[-1].after_state,
            after_pipeline_blob=pipeline_blob,
            token_count=len(tokens),
            coordination_descriptor=coordination_descriptor,
        )

    def commit_prepared(self, candidate: PreparedPlannerCommit) -> None:
        if candidate.before_digest != self._committed_digest:
            raise RuntimeError("prepared planner commit no longer matches state")
        self._committed_digest = candidate.after_digest
        self._committed_pipeline_blob = candidate.after_pipeline_blob
        if self.source.capabilities.resumable:
            if candidate.after_state is None:
                raise RuntimeError("resumable planner token is missing cursor state")
            self._committed_state = candidate.after_state

    def commit_batch(self, token: PlannerBatchToken) -> None:
        self.commit_prepared(self.prepare_commit([token]))

    def begin_next_epoch(self) -> None:
        if not self.finished:
            raise RuntimeError("cannot advance an unfinished planner epoch")
        if self._yield_digest != self._committed_digest:
            raise RuntimeError("cannot advance with uncommitted planner batches")
        self.start_epoch(self.epoch + 1)
        self._committed_state = (
            self.state() if self.source.capabilities.resumable else None
        )
        self._yield_state = self._committed_state
        self._committed_digest = self._initial_digest(self.epoch)
        self._yield_digest = self._committed_digest
        self._yield_pipeline_blob = self._committed_pipeline_blob

    def rollback_uncommitted(self) -> None:
        if not self.source.capabilities.resumable:
            raise RuntimeError("cannot roll back a non-resumable source")
        self.restore(self._committed_state)
        self._yield_state = self._committed_state
        self._yield_digest = self._committed_digest
        self._yield_pipeline_blob = self._committed_pipeline_blob

    def state_dict(self) -> dict:
        if not self.source.capabilities.resumable:
            raise RuntimeError("dataset source does not support exact resume")
        return {
            "version": 3,
            "planner_algorithm": PLANNER_ALGORITHM,
            "signature": self.signature_state(),
            "cursor": self._state_to_serializable(self._committed_state),
            "cursor_digest": self._committed_digest,
            "pipeline_state": self._committed_pipeline_blob.hex(),
        }

    def coordination_state_dict(self) -> dict | None:
        """Return rank-independent committed progress for sharded streams."""
        if not self._rank_sharded:
            return None
        state = self._committed_state
        if state is None:
            raise RuntimeError("rank-sharded stream does not support exact resume")
        return {
            "version": 1,
            "distributed_policy": self.distributed_policy,
            "signature": self.signature_state(),
            "epoch": state.epoch,
            "batch_index": state.batch_index,
            "finished": state.finished,
        }

    def _envelope_to_state(self, envelope: RecordEnvelope) -> dict:
        payload = (
            self.source.save_payload(envelope.payload)
            if hasattr(self.source, "save_payload")
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
            return tuple(DatasetPlanner._tuple_tree(item) for item in value)
        if isinstance(value, tuple):
            return tuple(DatasetPlanner._tuple_tree(item) for item in value)
        return value

    def _envelope_from_state(self, state: dict) -> RecordEnvelope:
        payload = (
            self.source.restore_payload(state["payload"])
            if hasattr(self.source, "restore_payload")
            else state["payload"]
        )
        return RecordEnvelope(
            source_id=int(state["source_id"]),
            record_key=self._tuple_tree(state["record_key"]),
            shape_code=int(state["shape_code"]),
            payload=payload,
            resident_bytes=int(state["resident_bytes"]),
        )

    def _state_to_serializable(self, state: PlannerState) -> dict:
        reservoir = state.reservoir
        if isinstance(reservoir, PackedReservoirState):
            return {
                "schema": state.schema,
                "algorithm": state.algorithm,
                "manifest_digest": state.manifest_digest,
                "config": asdict(state.config),
                "epoch": state.epoch,
                "batch_index": state.batch_index,
                "source_cycle": state.source_cycle,
                "source_cursor": state.source_cursor,
                "reservoir": {
                    "algorithm": reservoir.algorithm,
                    "seed": reservoir.seed,
                    "epoch": reservoir.epoch,
                    "stream_key": reservoir.stream_key,
                    "capacity": reservoir.capacity,
                    "slots_le": reservoir.slots_le.hex(),
                    "rng_counter": reservoir.rng_counter,
                    "offered": reservoir.offered,
                    "emitted": reservoir.emitted,
                    "peak_occupancy": reservoir.peak_occupancy,
                    "closed": reservoir.closed,
                },
                "ready_ids_le": state.ready.record_ids_le.hex(),
                "shape_queues": [],
                "source_exhausted": state.source_exhausted,
                "finished": state.finished,
            }
        return {
            "schema": state.schema,
            "algorithm": state.algorithm,
            "manifest_digest": state.manifest_digest,
            "config": asdict(state.config),
            "epoch": state.epoch,
            "batch_index": state.batch_index,
            "source_cycle": state.source_cycle,
            "source_cursor": state.source_cursor,
            "reservoir": {
                "algorithm": reservoir.algorithm,
                "seed": reservoir.seed,
                "epoch": reservoir.epoch,
                "stream_key": reservoir.stream_key,
                "capacity": reservoir.capacity,
                "max_bytes": reservoir.max_bytes,
                "slots": [self._envelope_to_state(item) for item in reservoir.slots],
                "slot_sizes": list(reservoir.slot_sizes),
                "resident_bytes": reservoir.resident_bytes,
                "rng_counter": reservoir.rng_counter,
                "offered": reservoir.offered,
                "emitted": reservoir.emitted,
                "peak_occupancy": reservoir.peak_occupancy,
                "peak_resident_bytes": reservoir.peak_resident_bytes,
                "closed": reservoir.closed,
            },
            "ready": [self._envelope_to_state(item) for item in state.ready],
            "shape_queues": [
                [shape, [self._envelope_to_state(item) for item in queue]]
                for shape, queue in state.shape_queues
            ],
            "source_exhausted": state.source_exhausted,
            "finished": state.finished,
        }

    def _state_from_serializable(self, state: dict) -> PlannerState:
        try:
            reservoir = state["reservoir"]
            if reservoir["algorithm"] == PACKED_RESERVOIR_ALGORITHM:
                if state["shape_queues"] != []:
                    raise ValueError(
                        "packed planner state cannot contain shape queues"
                    )
                reservoir_state = PackedReservoirState(
                    algorithm=str(reservoir["algorithm"]),
                    seed=int(reservoir["seed"]),
                    epoch=int(reservoir["epoch"]),
                    stream_key=self._tuple_tree(reservoir["stream_key"]),
                    capacity=int(reservoir["capacity"]),
                    slots_le=bytes.fromhex(reservoir["slots_le"]),
                    rng_counter=int(reservoir["rng_counter"]),
                    offered=int(reservoir["offered"]),
                    emitted=int(reservoir["emitted"]),
                    peak_occupancy=int(reservoir["peak_occupancy"]),
                    closed=reservoir["closed"],
                )
                return PlannerState(
                    schema=str(state["schema"]),
                    algorithm=str(state["algorithm"]),
                    manifest_digest=str(state["manifest_digest"]),
                    config=PlannerConfig(**state["config"]),
                    epoch=int(state["epoch"]),
                    batch_index=int(state["batch_index"]),
                    source_cycle=int(state.get("source_cycle", 0)),
                    source_cursor=state["source_cursor"],
                    reservoir=reservoir_state,
                    ready=PackedReadyState(
                        bytes.fromhex(state["ready_ids_le"])
                    ),
                    shape_queues=(),
                    source_exhausted=state["source_exhausted"],
                    finished=state["finished"],
                )
            reservoir_state = ReservoirState(
                algorithm=str(reservoir["algorithm"]),
                seed=int(reservoir["seed"]),
                epoch=int(reservoir["epoch"]),
                stream_key=self._tuple_tree(reservoir["stream_key"]),
                capacity=int(reservoir["capacity"]),
                max_bytes=(
                    None
                    if reservoir["max_bytes"] is None
                    else int(reservoir["max_bytes"])
                ),
                slots=tuple(
                    self._envelope_from_state(item) for item in reservoir["slots"]
                ),
                slot_sizes=tuple(int(value) for value in reservoir["slot_sizes"]),
                resident_bytes=int(reservoir["resident_bytes"]),
                rng_counter=int(reservoir["rng_counter"]),
                offered=int(reservoir["offered"]),
                emitted=int(reservoir["emitted"]),
                peak_occupancy=int(reservoir["peak_occupancy"]),
                peak_resident_bytes=int(reservoir["peak_resident_bytes"]),
                closed=reservoir["closed"],
            )
            return PlannerState(
                schema=str(state["schema"]),
                algorithm=str(state["algorithm"]),
                manifest_digest=str(state["manifest_digest"]),
                config=PlannerConfig(**state["config"]),
                epoch=int(state["epoch"]),
                batch_index=int(state["batch_index"]),
                source_cycle=int(state.get("source_cycle", 0)),
                source_cursor=state["source_cursor"],
                reservoir=reservoir_state,
                ready=tuple(
                    self._envelope_from_state(item) for item in state["ready"]
                ),
                shape_queues=tuple(
                    (
                        int(shape),
                        tuple(self._envelope_from_state(item) for item in queue),
                    )
                    for shape, queue in state["shape_queues"]
                ),
                source_exhausted=state["source_exhausted"],
                finished=state["finished"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("cannot restore malformed planner cursor state") from exc

    def load_state_dict(self, state: dict) -> None:
        expected = {
            "version": 3,
            "planner_algorithm": PLANNER_ALGORITHM,
            "signature": self.signature_state(),
        }
        actual = {key: state.get(key) for key in expected}
        if actual != expected:
            raise RuntimeError("cannot restore planner after its contract changed")
        cursor = self._state_from_serializable(state.get("cursor"))
        digest = state.get("cursor_digest")
        if not isinstance(digest, str) or len(digest) != 64:
            raise RuntimeError("cannot restore malformed planner cursor digest")
        try:
            pipeline_blob = bytes.fromhex(state.get("pipeline_state", ""))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("cannot restore malformed pipeline state") from exc
        if self.pipeline_composer is not None:
            self.pipeline_composer._decode_blob(pipeline_blob)
        elif pipeline_blob:
            raise RuntimeError("cannot restore pipeline state without pipelines")
        self.restore(cursor)
        self._committed_state = cursor
        self._yield_state = cursor
        self._committed_digest = digest
        self._yield_digest = digest
        self._committed_pipeline_blob = pipeline_blob
        self._yield_pipeline_blob = pipeline_blob

    def signature_state(self) -> dict:
        return {
            "algorithm": PLANNER_ALGORITHM,
            "source_manifest_digest": self._manifest_digest,
            "runtime": asdict(self.runtime_context.global_contract),
            "config": asdict(self.config),
            "distributed_policy": self.distributed_policy,
            "pipelines": (
                [
                    {
                        "pipeline_id": pipeline.pipeline_id,
                        "schema_version": pipeline.schema_version,
                        "signature": pipeline.signature_state(),
                        "input_fields": [
                            asdict(field) for field in pipeline.input_fields
                        ],
                        "output_fields": [
                            asdict(field) for field in pipeline.output_fields
                        ],
                    }
                    for pipeline in self.pipeline_composer.pipelines
                ]
                if self.pipeline_composer is not None
                else []
            ),
        }

    def close(self) -> None:
        cursor = getattr(self, "_source_cursor", None)
        if cursor is not None and hasattr(self.source, "close_cursor"):
            self.source.close_cursor(cursor)
        self._source_cursor = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
