# Data pipeline

This page is the maintained reference for the training data pipeline. It describes the implemented architecture,
the behavior developers and users can rely on, the portable resource policy, and the relevant configuration.

The central design rule is to unify scheduling semantics without forcing every file format into the same physical
record representation. Each `RecordSource` owns deterministic admission, compact identity, and format-specific
I/O. `DatasetPlanner` owns bounded shuffle, shape-aware batching, distributed policy, and transactional progress.

## Scope and guarantees

The built-in iterable pipeline provides:

- no per-record Python metadata: Dense sources use `O(file count)` metadata and indexed sources use compact
  fixed-width NumPy maps only when their format requires one;
- shuffle and prefetch memory bounded by configured active work, with decode memory proportional to a bounded
  number of active source files and materialized batches rather than total dataset size;
- deterministic sampling, shuffle, augmentation keys, batching, and supported resume;
- one planner contract across random-access NPZ, sequential binary, and composite sources;
- rank-consistent global batches for known-length random-access data;
- explicit step-bounded rank sharding for unknown-length sequential training;
- ordered prefetch whose completion timing cannot change sample order;
- no required per-entry sidecars, indexes, or decompressed mirrors for binary streams;
- exact capability checks instead of silently emulating unsupported behavior.

The pipeline does not promise a uniform global permutation. It implements a deterministic bounded reservoir,
which provides long-lived mixing while keeping memory independent of dataset size. It also does not support old
object-record checkpoints, arbitrary random access into continuous binary/LZ4 streams, or exact distributed
complete-dataset evaluation for a source whose cardinality is unknown.

Dataset files are immutable for the lifetime of a run. Source manifests include file identity and all semantics
that affect logical records. Restore rejects changed manifests or schema versions.

Configured paths are canonicalized before manifest construction. Duplicate canonical paths and symbolic aliases
are rejected, as are hard-link aliases when the filesystem exposes device/inode identity. Validation uses one
identity lookup per path, is linear in file count, and does not perform pairwise filesystem probes.

## Architecture

```text
Trainer / DataLoader
        |
        v
SourceBatchDataset
  - bounded ordered prefetch
  - optional stateless worker finalization
  - ordered batch publication
        |
        v
DatasetPlanner
  - deterministic reservoir shuffle
  - shape-aware global batches
  - rank-local slices and evaluation masks
  - transactional cursor and pipeline state
        |
        v
RecordSource
  +----------------------+----------------------+------------------+
  |                      |                      |                  |
  v                      v                      v                  v
Dense/Indexed NPZ   Sequential binary     Composite source   Map evaluation
  compact row IDs     bounded raw entry     routed payloads    direct indexing
  format-specific I/O  interleaved readers  child ownership    sampler-owned
```

The main implementation boundaries are:

- [`dataset/source.py`](../dataset/source.py): source capabilities and logical envelopes;
- [`dataset/planner.py`](../dataset/planner.py): planning, batching, transactions, and checkpoint state;
- [`dataset/packed.py`](../dataset/packed.py): fixed-width packed reservoir and ready storage;
- [`dataset/npz_source.py`](../dataset/npz_source.py): compact Dense and Indexed NPZ sources;
- [`dataset/sequential_source.py`](../dataset/sequential_source.py): interleaved sequential readers;
- [`dataset/source_dataset.py`](../dataset/source_dataset.py): decode prefetch and ordered publication.

Map-style datasets retain direct indexing for complete evaluation and sampler-owned sampling. Built-in iterable
datasets use the planner and receive a required `DatasetRuntimeContext` from the trainer.

| Public dataset type | Execution path |
| --- | --- |
| `katago_numpy`, `processed_katago_numpy`, `multi` | Map-style indexing and sampler-owned sampling |
| `iterative_katago_numpy`, `iterative_processed_katago_numpy` | Planned NPZ with Dense or Indexed identities |
| `batched_processed_katago_numpy` | Dense NPZ with a packed uniform-shape fast path and internal decode prefetch |
| `sparse_numpy`, `iterative_sparse_numpy` | Planned Indexed NPZ |
| `simple_binary`, `packed_binary` | Interleaved sequential source |
| `iterative_multi` | Composite of native child sources |

## Core contracts

### Source capabilities

A source declares behavior that the planner may depend on:

```python
@dataclass(frozen=True)
class SourceCapabilities:
    access_mode: Literal["random", "sequential"]
    known_length: bool
    exact_distributed_partition: bool
    resumable: bool
    deterministic: bool
```

Capabilities are enforced. For example, distributed training over an unknown-length sequential source requires
`steps_per_epoch`, and exact distributed evaluation rejects a source that cannot partition a known cardinality.
Single-rank evaluation may exhaust an unknown-length deterministic stream and mask its padded final batch.

### Logical envelopes and packed IDs

The generic planner boundary is a bounded logical envelope:

```python
@dataclass(frozen=True)
class RecordEnvelope:
    source_id: int
    record_key: object
    shape_code: int
    payload: object
    resident_bytes: int = 0
```

`record_key` is stable logical identity. `shape_code` supports shape-homogeneous batches. `payload` is opaque to
the planner, and `resident_bytes` accounts for owned sequential payload retained by the reservoir.

Uniform Dense and Indexed NPZ sources use a faster physical backend: contiguous `uint64` record IDs flow through
the native reservoir, ready FIFO, batch slicing, digesting, and materialization without constructing envelopes.
A compatibility envelope is created only if external or debugging code explicitly iterates the packed view.
Mixed-shape and sequential sources use generic bounded envelopes.

No built-in source retains one Python object per logical record in the dataset. Indexed inspection may construct
temporary Python values while building its compact NumPy maps, but those values are not part of steady-state
source or planner memory.

### Runtime context

`DatasetRuntimeContext` supplies global and local batch size, world size, rank, seed, and execution mode. This
keeps rank policy, batch ownership, and RNG domains explicit. Iterable datasets do not fall back to process-global
random state or infer distributed identity from ambient state.

## Planning and batching

### Deterministic admission

`sample_rate` is applied before reservoir insertion with repository-owned deterministic RNG. Admission is keyed
by source identity, logical address, seed, epoch, and cycle as appropriate. Rejected records consume neither
shuffle capacity nor decode work.

### Streaming reservoir

For a capacity `B`, the planner fills `B` slots, then replaces one deterministic random slot for every admitted
input and emits the displaced record. At source exhaustion it deterministically permutes and drains the remaining
slots. A record in a full reservoir has eviction probability `1 / B` per new arrival and expected residence of
approximately `B` arrivals, with a decreasing long tail.

The reservoir is bounded by record count and, for owned payloads, an optional byte ceiling. Fixed-width NPZ IDs
use the count bound. Sequential binary entries use both bounds because their raw bytes must survive until the
record is emitted.

Packed reservoir checkpoints own immutable little-endian ID bytes and the exact RNG counter. Generic checkpoints
serialize bounded envelopes and source-owned payload state.

### Shape-aware global batches

The planner forms batches by `shape_code` before taking rank-local slices. Uniform sources bypass shape queues.
At a source boundary:

- training drops an incomplete global shape batch;
- evaluation pads it deterministically and publishes an `is_real` mask;
- every rank observes compatible batch and transaction boundaries.

### Transactions and stateful pipelines

Planning produces a batch and a token containing before/after digests and supported cursor state. The trainer
commits the token only after the corresponding optimizer step succeeds. A failure or cancellation rolls a
resumable stream back to the last committed boundary.

Stateful batch pipelines run in emitted order and participate in the transaction digest. Decode prefetch is
disabled for a stateful pipeline because planning and publication state cannot safely advance out of order. This
resolution is exposed in the prefetch audit.

## Physical sources

### Dense and Indexed NPZ

Dense NPZ metadata is proportional to file count. A Dense descriptor retains file identity, logical row count,
global row prefix, and board shape. One global `uint64` ID resolves as:

```text
record ID -> prefix search -> file descriptor + row -> decoded arrays
```

Indexed NPZ uses the same planner contract but may retain compact NumPy row maps when filtering or physical
layout prevents a dense identity. The row map belongs to the source, not the planner.

For processed Dense deflated NPZ files without value-dependent filtering, manifest construction reads and
validates the embedded NPY headers without inflating array payloads. It checks required fields, dtype, dimensions,
row counts, board shape, and declared payload bounds, then hashes the physical file. Filtered/channel-selected
and ZIP-stored mmap paths retain full inspection where values or mapping validation are required.

The processed Dense fast path resolves a packed batch once, groups rows by physical file, gathers one file at a
time, and scatters results back into planner order. In the primary adaptive mode, decoded arrays, in-flight
loads, ready batches, and pinned batches are charged to one hard rank-local host-memory budget. Compatibility
mode retains the older fixed entry/byte limits. Both modes keep ZIP-stored read-only mappings in a separately
bounded cache.

One-file-at-a-time gather prevents a prefetch chunk from retaining all touched files as temporary arrays. Cache
limits are independent of total dataset file count.

On that same path, packed sample keys are a lazy sequence over read-only file-index and row arrays. Canonical
Python key tuples are created only for a stateful pipeline, fallback RNG, audit, or compatibility caller. Normal
power-of-two symmetry uses the versioned `processed-splitmix-v1` vectorized stream directly over packed file and
row identity. Non-power-of-two symmetry groups use the deterministic rejection fallback.

Indexed NPZ uses its compact maps to construct the required logical references and currently materializes rows
through the generic decoder path. It does not claim the Dense path's chunked threaded materialization, shared
multi-file caches, lazy packed keys, or vectorized symmetry fast path.

### Sequential binary

`simple_binary` and `packed_binary` preserve sequential I/O. They do not build a global entry index or sidecar.
At an epoch boundary the source deterministically orders files, opens at most `sequential_active_streams`, and
reads `sequential_read_quantum` logical entries from one reader before rotating to the next. Reader completion
order never controls logical order.

The reservoir retains the smallest complete raw entry needed for later decode. Packed-game move samples share
the entry bytes conceptually but carry their own subrecord identity. Memory is bounded by active readers,
`shuffle_window_size`, `shuffle_buffer_bytes`, shape remainders, and prefetch state.

Sequential file identity uses path, size, modification time, ordinal, and versioned source semantics. It avoids
a mandatory content-hash pre-scan. Files therefore must not change during a run.

Uncompressed seekable binary streams support exact resume with reader byte offsets, entry counters, pending raw
subrecords, and serialized reservoir payloads. Continuous LZ4 streams are deterministic but declare resume
unsupported because their decompressor state cannot be reconstructed from a normal file seek.

### Composite sources

`iterative_multi` combines native child sources without converting their physical handles. A deterministic
integer-ratio schedule selects children, and composite identity includes the child ID. Materialization groups by
child when necessary and restores planner order.

The composite is resumable only when every child is resumable. `sync_length=True` emits complete ratio cycles
and stops when the next cycle cannot be satisfied; other modes follow their configured quota policy.

## Prefetch and device handoff

`SourceBatchDataset` plans a bounded number of batches and submits numbered decode chunks to sources that declare
thread-safe materialization. Workers may finish out of order, but results are published in planner order and
never exceed `prefetch_batches`.

The processed NPZ path can decode several neighboring batches in one call. Its stateless NumPy-to-pinned-Tensor
conversion also runs in the workers. Planner state, stateful pipelines, publication, and transaction commits stay
on the owning thread. In normal training, Accelerate performs the pinned-host-to-device copies while
fetching the next loader batch.

Training can optionally set the top-level `cuda_prefetch_batches` option from `1` through `4`. Any positive
value delivers training batches through `StaticSlotLoaderWrapper`: the first batch fixes a schema of keys,
shapes, and dtypes (batches must be flat `dict`s of tensors, which the planned-batch datasets guarantee),
and every later batch is copied H2D into one persistent set of device tensors on a dedicated copy stream.
Each slot is marked `torch._dynamo.mark_static_address`, so under Inductor CUDA graphs the compiled step
replays against fixed input pointers and skips the per-tensor input-stabilization DtoD copies (and their
host-side latency) that fresh device addresses would otherwise force every step. Ordering uses two events per
step: the compute stream waits for "copy ready", and the next refill waits for an event recorded behind the
consumer's forward/backward launches, so a slot is never overwritten while its contents can still be read. A
fresh iterator (epoch restart) orders its first refill behind the compute stream's current tail, covering the
previous epoch's final step still executing against the slots. A batch whose schema does not match the
established one raises immediately. The configured depth only gates engagement; the slot path holds one batch
on device regardless of the value.

Setting the `NNUE_FORCE_CUDA_PREFETCH` environment variable restores the previous deep-lookahead prefetcher
(`CudaPrefetchLoaderWrapper`): while the compute stream consumes one batch, a dedicated CUDA stream copies up
to `cuda_prefetch_batches` upcoming batches from pinned host memory into a bounded device queue. CUDA events
preserve stream ordering without a host synchronization, and every device tensor records the consuming stream
so allocator reuse remains safe. Host allocations remain retained until their copy event completes, and both
queued device batches and retired host batches have fixed count bounds. It remains the fallback for any future
dataset shape or compile regime where static-address slots misbehave.

The default is `0`, which selects the original device-loader class and, with observability off, adds no
stream, event, queue, clock, or per-batch branch to the normal path (observability mode selects the
instrumented loader variant instead; see below). Device handoff applies to the training loader only;
validation retains its
existing device-placement behavior. It supports built-in iterable streams and exact-resume map loaders with
`num_worker: 0`. Other map loaders leave device placement to Accelerate and reject this option instead of silently
ignoring it. Prefetched `BatchEnvelope` tokens remain uncommitted until the optimizer transaction consumes them.
Closing or failing the iterator drains pending copies and then invokes the planner's existing rollback path
(a pending device error surfaces from that drain first), so checkpoint identity depends only on consumed
batches.

The generic `cuda_prefetch_batches` default stays `0` because the slot path presumes a constant batch schema and
CUDA training. Enable it only after representative end-to-end validation shows a benefit.

Batch-yielding and resumable built-in streams require DataLoader `num_worker: 0`; loader processes are rejected
because they would duplicate planner ownership and checkpoint state. Parallelism comes from the dataset's
internal ordered decode workers. The adaptive controller selects their active count; compatibility mode uses
the legacy `prefetch_threads` setting.

## Adaptive resource control

The primary resource interface is the top-level singular `data_pipeline` mapping. It is separate from
`data_pipelines`, the older list of semantic batch transforms. The adaptive interface supports
`dataset_type: batched_processed_katago_numpy` and requires `num_worker: 0`. It can compose `data_pipelines`
whose registered transforms declare themselves parallel and stateless; other semantic transforms remain on the
compatibility path and are rejected when adaptive control is explicitly requested.

For the standard training entry point, omitting `data_pipeline` enables its default adaptive policy when those
requirements hold and no fixed `prefetch_threads`, `prefetch_batches`, `pin_memory`, or loader performance alias is
present. Existing fixed configurations, nonzero loader-worker configurations, and semantic `data_pipelines` remain
on their compatibility path. Supplying an explicit `data_pipeline` mapping is strict: incompatible options are
reported as configuration errors instead of silently disabling adaptation.

For `iterative_multi`, opt in explicitly with `data_pipeline: {}` and keep
`num_worker: 0`. This path supports dense, unfiltered
`batched_processed_katago_numpy` children, each with one explicit board size,
and no composite batch transforms. It retains the record-level `blend_ratio`,
`sample_rate`, and `sync_length` semantics. Packed record IDs are shuffled and
bucketed by shape before the global batch is partitioned across ranks; each
rank therefore receives the same board size at each step. Queued shape buckets
and source cursors are included in exact checkpoint/rollback state.

The mixed path prepares one shared decoded cache over all child files, and uses
one parent memory budget, bounded parallel decoding, pinning, and telemetry.
It does not allocate an independent adaptive budget for every child. Keep
`dataloader_args.batch_by_boardsize: true` for mixed sizes. With
`cuda_prefetch_batches: 1`, mixed shapes automatically select the existing
shape-flexible CUDA lookahead loader; uniform streams retain static input slots.
Mixed datasets without an explicit adaptive mapping retain
the generic compatibility path. Old generic planner checkpoints cannot be
resumed into the packed mixed path; start a new run when enabling it.

### Portable resource resolution

Each training process collects an immutable snapshot of host memory and CPU availability. Probing uses
standard-library interfaces first. Isolated platform adapters then provide Windows native memory/affinity
fallbacks and Linux procfs/cgroup v1/v2 refinements; cgroup discovery is best-effort rather than the primary source
of host capacity.
Malformed values and unlimited sentinels are ignored instead of becoming artificial limits.
Native Windows execution is best-effort rather than a tier-one support target.

Automatic memory selection requires a safe effective-available-memory observation. It starts from:

```text
50% of effective available memory
```

When effective total memory is also available, 25% of that total is an additional ceiling. Total capacity alone is
never used to guess current headroom: if effective available memory cannot be established safely, automatic memory
sizing fails with guidance to set `host_memory_budget` explicitly. An explicit budget remains a user maximum, but
a medium- or high-confidence effective-available observation may reduce it to 80% of current headroom. Automatic
CPU selection floors the effective affinity/quota count and falls back to the logical CPU count. An explicit CPU
budget is likewise a maximum and is reduced when a smaller observed CPU ceiling exists. Automatic CPU selection
fails with an actionable request for an explicit value when no usable CPU fact is available.

Reports are grouped by opaque node identity so each node total is divided by its actual number of local ranks.
Every report is resolved independently, then all ranks select the conservative global minima for per-rank memory
and CPU. This gives every distributed rank identical performance settings even when nodes differ or concurrent
probes vary slightly. Node identities are not included in serialized resource metadata.

### Run-scoped decoded storage

Automatic processed-NPZ runs keep the source files and their format unchanged. At startup, one leader per node
streams the fixed NPY members into a unique temporary directory. Local ranks then open the same read-only files
with NumPy mmap, so compressed members are expanded once per node rather than once per rank. The operating system
shares resident pages and evicts them under ordinary memory pressure; the trainer does not reserve the full
expanded size as private rank memory.

The cache is intentionally ephemeral. It is never reused by another run, does not add a content hash or payload
scan, and is deleted only after all local decoders have closed. If the temporary directory is not visible to every
local rank, any node lacks space, or load-time filtering/channel transforms require private arrays, activation is
disabled globally and every rank uses the existing bounded private cache. `manual` plans also retain the private
cache so their explicit byte allocation keeps its literal meaning.

### Bounded automatic adaptation

`continuous` starts with a throughput-oriented plan that fits the resolved memory and CPU limits. The private-cache
working set is estimated from dataset size, batch size, and shuffle lookahead. Decode concurrency starts at half the
portable CPU ceiling; chunk size and queue depth are balanced around that count. Workers, chunk size, and queue
depth form one layout: the controller never adjusts one value while leaving the other two in an unrelated
intermediate state.

The controller has only two adaptive decisions:

- two consecutive wait-heavy windows with private-cache reloads grow the cache geometrically, by at least one
  decoded file and never beyond the resolved memory limit; cache capacity never shrinks during a run;
- two consistent windows may start one adjacent layout trial. During initial calibration, healthy operation with
  at least 1.5x producer headroom can halve worker concurrency and rebalance the layout. Under starvation, a
  candidate instead doubles workers, reduces a head-of-line-blocked chunk, or doubles chunk and queue depth together
  to amortize source-tail work. The candidate skips one settling window, is measured for two windows, and is either
  accepted as the new layout or rejected in favor of the previously accepted layout. Acceptance requires producer
  headroom without a material throughput or prefetch-wait regression; larger-granularity trials must additionally
  improve throughput by at least 3%.

Layout rejection is an online performance choice, not a data rollback. It changes only workers, chunk size, and
queue depth; it never rewinds samples, planner state, model state, or optimizer state. The controller never runs
independent healthy-path shrink actions for cache, queue, or chunk. Epoch-transition windows are excluded from
signals and trials, and a rejected complete layout is not retried within the same process. `continuous` can still
try unseen layouts after a later sustained data bottleneck, while `manual` validates and freezes the supplied
`advanced` plan immediately.

The controller changes performance capacity only. It never changes sample admission, shuffle/reservoir size,
record order, source mixture, batch size, augmentation, or any other semantic setting.

### Hard logical-memory accounting and backpressure

The selected per-rank budget is a hard cap over dataset-owned allocations explicitly charged in these categories:

- `semantic_fixed`: live planner, transactional, and queued-token state;
- `decoded_cache`: retained decoded NPZ arrays;
- `inflight_transient`: temporary bytes required while loading a file;
- `ready_pageable`: decoded pageable batches awaiting consumption;
- `pinned`: pinned batches awaiting consumption, including bounded device-copy lookahead.

Device lookahead is also reserved as fixed capacity headroom when the controller validates a plan, including
pending copies and the bounded set of host transfer owners whose completion events have not retired yet. The
lookahead wrapper transfers each output lease to that event-backed owner and releases it only after the copy is
complete; transaction-token leases continue with the delivered batch.
The trainer likewise reports its gradient-accumulation depth to the runtime, which reserves output and
transaction-token headroom for every micro-batch retained until the optimizer transaction commits. Both values
are derived runtime facts, not additional user tuning parameters.

This logical budget is deliberately not a claim about whole-process RSS, allocator fragmentation, or operating-
system page cache. Run-scoped mmap payloads therefore remain outside the rank-private logical budget. A temporary
queue reservation that does not fit applies non-blocking backpressure: the ordered producer stops submitting work
until the consumer releases bytes. A single request that cannot fit even in an empty budget fails clearly.
Semantic memory cannot be reclassified as performance memory, and pressure never causes semantic settings to be
changed automatically.

### Metrics and runtime state

TensorBoard records a compact iteration-axis view under `data_pipeline/...`:

- exposed data wait, H2D wait, producer headroom, and the worst-rank prefetch and source-tail wait fractions;
- the worst-rank cache-reload count;
- worst-rank logical-memory use and backpressure events, with high-water fraction emitted initially and only when
  it changes;
- decoder workers, chunk size, ready-queue depth, and decoded-cache capacity when those settings initialize or
  change.

Mean process RSS is recorded as `running_stat/process_rss_gib_mean` because it covers the whole training process,
not only the data pipeline.

The controller's raw window totals and throughput inputs remain internal and are discarded after each decision.
Pipeline metrics are not duplicated under `data_pipeline_rows/...`; global sample throughput remains available as
`running_stat/entry/s`. The JSONL log uses the same compact public pipeline view, while controller decisions remain
available as `data_pipeline_tuning` events.

Resolved settings and decision reasons are non-semantic runtime state; incompatible runtime state is ignored without
changing data-resume semantics. Rank coordination and checkpoints retain only the selected settings, the accepted
layout boundary, the controller phase, the frozen flag, and the latest event. Incomplete trial measurements are
discarded on checkpoint restore: the accepted layout is installed immediately, and fresh calibration may later
select another candidate from new windows instead of persisting timing noise.

Automatic CPU or memory probe drift alone does not invalidate a compatible saved layout; restore accepts it when it
still fits the current effective resource limits. Cumulative elapsed time, epoch, and consumed rows continue from
the checkpoint. JSONL records beyond that checkpoint are removed before appending the resumed trajectory.
For runs created by this version, TensorBoard keeps iteration-axis and selected train/validation row-axis tags in
one `log` stream; an abnormal resume can therefore leave a short overlapping tail in this observational view
rather than risk purging one axis with the other axis's scale. Existing split-layout event directories are not
migrated.

The distributed source-tail fraction subtracts nested decoded-prefetch wait from source wait on each rank before
taking the worst rank. This avoids mixing maxima from different ranks and keeps H2D, CUDA, or trainer time from
masquerading as synchronous planner or task-submission work.

## Distributed behavior

Known-length random-access sources produce one rank-independent global plan and disjoint equal rank-local slices.
Their record digests and transaction descriptors can be compared across ranks.

Unknown-length sequential sources cannot provide that plan without a forbidden full scan. Distributed training
therefore requires `steps_per_epoch`. Files are sorted by descending byte size and greedily assigned to the
least-loaded rank with stable tie-breaking. Each rank owns its shard for the run and produces full local batches
until the common optimizer-step budget is met. Repeated source cycles include the cycle in admission and
augmentation identity.

For rank-sharded sequential streams, source cursors, reservoir contents, and pipeline state are rank-local.
Cross-rank coordination compares the common policy, epoch, batch index, terminal flag, and optimizer-transaction
range. Exact complete-dataset distributed evaluation remains limited to known-length sources.

## Checkpoint and resume

A resumable checkpoint contains only active state:

- epoch, cycle, batch index, and source scheduler cursor;
- reservoir contents and RNG counters;
- ready and shape queues;
- pipeline state;
- composite child state;
- active sequential readers and pending entries;
- versioned source manifest identity.

In-memory prefetch tokens reuse immutable reservoir snapshots until mutation, so planning ahead does not copy
the entire reservoir per token. Serialized checkpoints are self-contained. Restore validates source identity,
schema, distributed policy, batch geometry, and pipeline state before accepting the cursor.

Old object-record checkpoints are intentionally incompatible. Non-resumable sources fail clearly instead of
producing a partial continuation.

## Configuration

The recommended processed-NPZ configuration omits the adaptive mapping because the standard training entry point
selects its portable defaults:

```yaml
dataset_type: batched_processed_katago_numpy
dataset_args:
  shuffle_window_size: 32768
  apply_symmetry: true
```

The four normal `data_pipeline` options are:

| Option | Meaning | Default |
| --- | --- | --- |
| `host_memory_budget` | Maximum host-data memory for the whole node; `auto` requires safe effective available memory | `auto` |
| `data_wait_budget` | Maximum tolerated data-wait fraction; accepts a fraction or percentage | `0.01` (1%) |
| `data_cpu_budget` | Maximum decode CPU count for the whole node; `auto` uses effective/logical CPU facts | `auto` |
| `adaptation` | Automatic `continuous` or fixed `manual` control | `continuous` |

Memory and CPU values are node totals, not per-rank allowances. The coordinator divides them across local ranks
and then applies the most conservative per-rank result globally. Adding ranks therefore does not multiply the
configured resource cap. Byte values accept positive integers or strings such as `512 MiB` and `2.5 GiB`.
`data_wait_budget` is a controller target, not a per-window guarantee. Each observation window spans
`log_interval`, which therefore also controls adaptation cadence and the amount of short-term noise in a decision.
`continuous` is the default because available resources and producer demand can change during a long run, while
the controller changes only bounded performance capacity. `manual` is reserved for an explicitly fixed,
hardware-specific plan.

Adaptive mode rejects fixed performance keys in `dataset_args`: `prefetch_threads`, `prefetch_batches`, and
`pin_memory`. It also rejects nonempty `data_pipelines`, because stateful semantic transforms are
not yet supported by the adaptive runtime. Format, filtering, admission, target, and augmentation options remain
under `dataset_args` and keep their existing semantics. Training shuffle is enabled by default and disabled with
top-level `no_shuffle: true`; only its semantic window belongs in `dataset_args`. Loader aliases `dataloader_args.pin_memory`
and `dataloader_args.shuffle_buffer_size` are rejected as well: pinning belongs to the adaptive plan, while the
semantic shuffle window must be configured once as `dataset_args.shuffle_window_size` before runtime sizing.

For a completely fixed performance plan, use `adaptation: manual` with all five `advanced` values:

```yaml
data_pipeline:
  host_memory_budget: auto
  data_wait_budget: 1%
  data_cpu_budget: auto
  adaptation: manual
  advanced:
    decoded_cache_bytes: 2 GiB
    decode_workers: 4
    decode_chunk_batches: 2
    ready_queue_batches: 8
    pin_memory: true
```

`advanced.decoded_cache_bytes` and `advanced.decode_workers` are node totals. Decode workers must divide evenly
across local ranks and provide at least one worker per rank. Chunk and ready-queue counts are rank-local, the ready
queue must be at least as deep as the chunk, and the entire plan must fit the resolved memory/CPU caps. Pinned
memory must be supported by the runtime.

### Fixed compatibility configuration

When the adaptive mapping is omitted, explicit legacy performance controls, a nonzero `num_worker`, or nonempty
semantic `data_pipelines` keep the existing fixed path. This preserves older configurations and direct dataset use:

| Option | Meaning | Typical/default value |
| --- | --- | --- |
| `no_shuffle` (top level) | Disable the trainer's deterministic reservoir shuffle | `false` |
| `sample_rate` | Deterministic pre-reservoir admission rate | `1.0` |
| `shuffle_window_size` | Maximum active reservoir records | `32768` |
| `shuffle_buffer_bytes` | Additional payload byte ceiling | binary default: 256 MiB |
| `steps_per_epoch` | Optimizer-step budget; required for distributed unknown-length streams | unset |
| `sequential_active_streams` | Simultaneously open sequential readers | `2` |
| `sequential_read_quantum` | Entries read before rotating readers | `256` |
| `prefetch_threads` | Internal ordered decode workers for batched processed NPZ | `2` |
| `prefetch_batches` | Maximum submitted but unpublished batches | `32` |
| `pin_memory` | Convert decoded NumPy batches to pinned tensors | CUDA availability |
| `observability` | Enable grouped pipeline metrics | `false` |

A fixed configuration example is:

```yaml
dataset_type: batched_processed_katago_numpy
num_worker: 0
dataset_args:
  prefetch_threads: 2
  prefetch_batches: 32
  pin_memory: true
```

These controls do not form a second runtime controller. New adaptive configurations should use the top-level
interface. Unknown or removed options are rejected during dataset construction.

## Memory model

| Source | Retained dataset metadata | Active shuffle state | Required sidecar |
| --- | --- | --- | --- |
| Dense NPZ | `O(file count)` descriptors and prefix arrays | packed `uint64` IDs | no |
| Indexed NPZ | descriptors plus compact NumPy row maps | packed IDs when shape is uniform | no |
| Simple binary | file descriptors and active readers | bounded raw entries | no |
| Packed binary | file descriptors and active readers | bounded raw entries/subrecords | no |
| Composite | sum of child metadata | one bounded planner state | no |

Automatic processed-NPZ runs normally use one run-scoped mmap cache per node, leaving ready batches and planner
state as the dominant private rank allocations. Private fallback and manual modes retain the bounded decoded NPZ
LRU. In either backend, Python control-plane memory does not grow with rows or batches processed.

## Performance validation

Absolute throughput and memory use depend on storage, compression, board shape, filtering, process count, and
model demand. Treat automatic settings as safe starting points rather than portable performance guarantees.
Validate materially different workloads end to end, using the emitted producer, wait, cache, queue, and logical-
memory metrics to distinguish data-pipeline limits from model-side limits. The benchmark methodology is described
in [Training performance](performance.md).

## Operational checks

When changing the pipeline or deploying a materially different workload, verify:

1. decoded fields, masks, sample order, and augmentation choices for fixed seed and epoch;
2. exact supported resume at a committed batch boundary;
3. rank-local disjointness and common transaction ranges for distributed runs;
4. cold manifest time, steady decode throughput, and peak RSS;
5. cache entry/byte bounds and prefetch queue depth;
6. end-to-end training throughput and loss trajectory.

Short semantic and scale tests should precede long training runs. Performance changes are accepted only when a
representative end-to-end workload improves or when a documented memory/correctness benefit justifies a neutral
throughput result.

## Adding a source format

A new source must answer:

1. Is its natural access random or sequential?
2. Is logical length known without a complete scan?
3. What stable identity is available without per-record retained objects?
4. How is output shape represented compactly?
5. What payload must survive reservoir residence?
6. Can materialization operate on complete batches?
7. What manifest fields detect data or decoder-semantic changes?
8. Can exact resume be implemented without defeating the format's storage model?
9. Which distributed and evaluation modes can it support honestly?

The source implements the common capability, cursor, identity, and materialization contracts. Format-specific
path, offset, row, or subrecord logic remains behind the source interface; it must not be added to the planner.

## Maintained invariants and limitations

- Logical order is determined by explicit seed, epoch/cycle, source state, and planner counters.
- Prefetch timing never determines sample order.
- Stateful pipelines observe committed emitted order.
- Dense-source metadata is bounded by file count; unavoidable Indexed maps use compact fixed-width arrays rather
  than per-record Python objects. Active shuffle and prefetch memory is bounded by configuration. Decode memory
  is bounded by active file and batch counts, not row count, but its byte peak must accommodate a whole decoded
  source file and may exceed the nominal cache byte cap for one oversize file.
- Normal binary I/O remains sequential and requires no sidecar.
- Unsupported resume, partitioning, or evaluation modes fail explicitly.
- Source-specific physical details stay behind `RecordSource`.
- Dense uniform NPZ has the strongest production performance coverage; other formats retain the same semantic
  and memory contracts but should be profiled on their representative real data before performance claims.
