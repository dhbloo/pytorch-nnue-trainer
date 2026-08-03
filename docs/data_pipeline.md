# Data pipeline

This page is the maintained reference for the training data pipeline. It describes the implemented architecture,
the behavior developers and users can rely on, the relevant configuration, and the current representative
performance.

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
| `iterative_sparse_numpy` | Planned Indexed NPZ |
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
time, and scatters results back into planner order. Its decode workers share two independent bounded caches:

- deflated arrays: at most six files and 1.125 GiB, while always permitting one oversize file;
- ZIP-stored read-only mappings: at most 16 files.

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
on the owning thread. In normal Mix9S training, Accelerate performs the pinned-host-to-device copies while
fetching the next loader batch.

Training can optionally set the top-level `cuda_prefetch_batches` option from `1` through `4`. A value of `1`
provides double buffering: while the compute stream consumes one batch, a dedicated CUDA stream copies the next
batch from pinned host memory. CUDA events preserve stream ordering without a host synchronization, and every
device tensor records the consuming stream so allocator reuse remains safe. Host allocations remain retained
until their copy event completes. Both queued device batches and retired host batches have fixed count bounds.

The default is `0`, which selects the original device-loader class and adds no stream, event, queue, clock, or
per-batch branch to the normal path. CUDA lookahead applies to the training loader only; validation retains its
existing device-placement behavior. It supports built-in iterable streams and exact-resume map loaders with
`num_worker: 0`. Other map loaders leave device placement to Accelerate and reject this option instead of silently
ignoring it. Prefetched `BatchEnvelope` tokens remain uncommitted until the optimizer transaction consumes them.
Closing or failing the iterator discards lookahead and invokes the planner's existing rollback path, so checkpoint
identity depends only on consumed batches.

On the maintained RTX 4080 SUPER Mix9S workload, depths `0`, `1`, and `2` produced the same byte-level digest
for 512 decoded batches, masks, sample keys, and transaction descriptors. The final uninstrumented audit used
two interleaved 5,000-step pairs. Depth `0` sustained 73.438 and 76.543 it/s, while depth `1` sustained 70.123
and 73.827 it/s; the pooled result was 74.990 versus 71.975 it/s (-4.02%). Raw production H2D work is only about
0.115 ms/batch and already finishes before the first training kernel while host-side pre-forward work runs, so
dedicated-stream bookkeeping is a net cost for this workload.

A controlled end-to-end stress test kept the real data, Mix9S model, ResNetV2 teacher, loss, and optimizer but
added one unused 16 MiB pinned payload per batch. The payload takes about 10.35 ms to copy on the reference host.
After removing single-process error collectives that serialized the copy engine, three 2,000-step pairs measured
positive depth-1 changes of +21.44%, +21.19%, and +3.46% under varying GPU load. A final trace reduced median
step time from 24.23 to 21.02 ms. This verifies that the implementation overlaps material transfers when the
workload exposes them. Keep the default `0` for the current Mix9S workload; enable lookahead only after an
end-to-end measurement demonstrates a benefit for a workload with larger transfers or less host work before its
first kernel.

Batch-yielding and resumable built-in streams require DataLoader `num_worker: 0`; loader processes are rejected
because they would duplicate planner ownership and checkpoint state. The production Mix9S path gets its
parallelism from the dataset's two internal ordered decode threads.

### Observability and automatic tuning

The batched processed-NPZ path has three construction-time execution modes:

- `off` (the default) keeps the original uninstrumented dataset, decoder, and device-loader classes. It performs
  no per-batch metric branches, counter updates, clocks, locks, or tuning callbacks;
- `metrics` selects separate instrumented classes that aggregate only at batch, decode-chunk, cache, and file
  boundaries;
- `autotune` enables the same metrics plus a bounded controller for timing-only runtime parameters.

The dataset layer exposes numeric interval snapshots but does not depend on a logging backend. At the trainer's
normal `log_interval`, the trainer aggregates the snapshots across ranks and writes them alongside existing
metrics under the independent `data_pipeline/...` and `data_pipeline_rows/...` TensorBoard namespaces. The same
flat metric map is stored under `data_pipeline` in `training_log.jsonl`. Metrics cover producer capacity,
source-wait and H2D submission latency, prefetch queue depth and starvation, decoded-cache behavior, manifest
cost, and retained memory. With CUDA lookahead enabled, CUDA events additionally report actual device-copy time
and compute-stream exposed wait. Distributed summaries use the slowest producer and largest exposed wait.

Automatic tuning may adjust active decode workers, ordered-prefetch depth, and processed-NPZ decoded-cache
limits within configured CPU limits and cache budgets. The decoder still permits one file larger than its byte
budget so oversized datasets remain usable. Tuning never changes sample admission, reservoir size, record order,
batch size, source mixture, augmentation, or another value that affects training semantics. One option changes
at a time, decisions have a cooldown, and tuning freezes after it meets the configured producer-headroom and
data-wait targets or reaches its bounded tuning window. When excess producer headroom exists, it may deliberately
reduce raw scan throughput to release CPU and queue memory while preserving the configured margin over training
demand. Explicit prefetch values are locked by default.

The current resolved settings and decision reasons are written atomically to `pipeline_tuning.json` in the run
directory and are included as non-semantic checkpoint state. A checkpoint made on a different runtime ignores
an incompatible tuning state without affecting data resume. After the controller freezes, a small metadata-only
profile is also stored in the operating system's user cache directory. An exact hardware, software, dataset,
batch, and pipeline fingerprint reuses it after a short health check; compatible reuse is optional. Dataset
payloads, file paths, and decoded cache contents are never written to the tuning-profile cache.

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

Common iterable options are:

| Option | Meaning | Typical/default value |
| --- | --- | --- |
| `shuffle` | Enable deterministic reservoir shuffle | training-dependent |
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
| `autotune` | Enable conservative runtime tuning and profile reuse | `false` |

A representative processed-NPZ configuration is:

```yaml
dataset_type: batched_processed_katago_numpy
num_worker: 0
cuda_prefetch_batches: 0
dataset_args:
  shuffle: true
  sample_rate: 1.0
  shuffle_window_size: 32768
  prefetch_threads: 2
  prefetch_batches: 32
  pin_memory: true
  apply_symmetry: true
  observability: true
  autotune:
    reuse: exact
    respect_explicit: false
    warmup_iterations: 1000
    verify_iterations: 500
    decision_interval: 500
    freeze_after: 10000
    target_producer_headroom: 1.5
    max_data_wait_fraction: 0.02
    max_prefetch_threads: 4
    max_prefetch_batches: 64
    host_cache_budget_bytes: 2147483648
```

Format-specific filtering and target options remain under `dataset_args`. Unknown or removed options are rejected
during dataset construction.

## Memory model

| Source | Retained dataset metadata | Active shuffle state | Required sidecar |
| --- | --- | --- | --- |
| Dense NPZ | `O(file count)` descriptors and prefix arrays | packed `uint64` IDs | no |
| Indexed NPZ | descriptors plus compact NumPy row maps | packed IDs when shape is uniform | no |
| Simple binary | file descriptors and active readers | bounded raw entries | no |
| Packed binary | file descriptors and active readers | bounded raw entries/subrecords | no |
| Composite | sum of child metadata | one bounded planner state | no |

The decoded NPZ LRU is the dominant retained host allocation on the production Mix9S path. It is deliberately
bounded to avoid decompression thrash while remaining independent of the total number of files. After the cache
is full, Python control-plane memory does not grow with rows or batches processed.

## Current performance

The following snapshot was measured on 2026-08-02 with an AMD Ryzen 9 5950X, an RTX 4080 SUPER, batch size 128,
two internal decode threads, a 32-batch prefetch bound, symmetry enabled, and the real Mix9S training files.
Filesystem cache, storage, compression ratio, board size, and filtering materially affect absolute numbers.

| Measurement | Current result |
| --- | ---: |
| Packed planning without materialization | 14.7–16.2M rows/s |
| Transactional packed planning | 6.83M rows/s |
| 231-file manifest, 30,811,590 rows | 0.97 s, zero decoded cache entries |
| Full 231-file pinned scan | 30,811,520 rows in 243.75 s; 126.4K rows/s |
| Full-scan peak process RSS | 2,052,072 KiB |
| Mix9S steady training consumption | approximately 9.6K rows/s |
| Mix9S training process RSS | approximately 3.05–3.12 GiB |
| Main-thread source publication | approximately 0.014 ms/batch |
| Loader region including H2D | approximately 0.7 ms/batch |

The full scan reached the six-entry decoded-cache bound and did not slow as file count increased. A post-warmup
allocation window processed another 524,288 rows with no increase in traced live Python memory. The pipeline
supplies roughly 15 times the rows consumed by training on this host.

The latest end-to-end Mix9S KD baseline audit sustained roughly 73–77 iterations/s. Useful end-to-end MFU was
29.26%; profiling attributes the remaining gap to teacher forward, student forward/backward, optimizer work,
and required numerical synchronization rather than data starvation. See
[Training performance](performance.md) for model-side methodology and results.

These are reference measurements, not portable guarantees. New hardware, formats, filters, worker/process
counts, cache policies, or storage should be measured end to end.

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
