# Data pipeline

This page is the maintained reference for the training data pipeline. It describes the implemented architecture,
the behavior developers and users can rely on, the portable resource policy, and the relevant configuration.

The central design rule is to unify scheduling semantics without forcing every file format into the same physical
record representation. Each `RecordSource` owns deterministic admission, compact identity, and format-specific
I/O. `DatasetPlanner` owns bounded shuffle, shape-aware batching, distributed policy, and transactional progress.

## Scope and guarantees

The built-in iterable pipeline provides:

- no dataset-sized per-record Python metadata: Dense sources use `O(file count)` metadata and indexed sources use compact
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
  +----------------------+----------------------+
  |                      |                      |
  v                      v                      v
Dense/Indexed NPZ   Sequential binary     Composite source
  compact row IDs     bounded raw entry     routed payloads
  format-specific I/O  interleaved readers  child ownership

Map dataset / DataLoader
  - direct indexing and sampler-owned training
  - EvaluationBatchPlannerDataset for complete evaluation
```

The main implementation boundaries are:

- [`dataset/source.py`](../dataset/source.py): source capabilities and logical envelopes;
- [`dataset/planner.py`](../dataset/planner.py): planning, batching, transactions, and checkpoint state;
- [`dataset/packed.py`](../dataset/packed.py): fixed-width packed reservoir and ready storage;
- [`dataset/npz_source.py`](../dataset/npz_source.py): compact Dense and Indexed NPZ sources;
- [`dataset/sequential_source.py`](../dataset/sequential_source.py): interleaved sequential readers;
- [`dataset/source_dataset.py`](../dataset/source_dataset.py): decode prefetch and ordered publication;
- [`dataset/execution.py`](../dataset/execution.py): physical backend selection, resource-policy capabilities,
  and shared execution lifecycle.

Map-style datasets retain direct indexing for complete evaluation and sampler-owned sampling. Built-in iterable
datasets use the planner and receive a required `DatasetRuntimeContext` from the trainer.

The common contract does not imply one physical hot path. Built-in iterable sources share `DatasetPlanner`
and `SourceBatchDataset` (or its observed subclass); map datasets retain a separate indexing/sampler path.

| Public dataset type | Backend | Physical work |
| --- | --- | --- |
| `katago_numpy`, `processed_katago_numpy`, `multi` | `map` | Eager arrays, direct indexing, sampler-owned sampling |
| `iterative_processed_katago_numpy` | `packed` | Dense IDs and batch decoding; explicit policy selects the budgeted batched adapter |
| `batched_processed_katago_numpy`, `batched_katago_numpy` | `packed` | Dense processed/native NPZ, vectorized decoding, bounded caches and ordered internal prefetch |
| `iterative_katago_numpy`, `sparse_numpy`, `iterative_sparse_numpy` | `indexed` | Row maps, serial materialization and a one-file cache; batch decoding still prepares rows individually |
| `simple_binary`, `packed_binary` | `sequential` | Interleaved readers, bounded raw entries, ordered per-record decoding |
| `iterative_multi` | `packed` or `generic` | Packed dense routing when eligible; otherwise generic child routing, including saved compatibility layouts |

Backend names describe source execution, not every planner operation: an indexed source can also use packed
planner IDs. Eligible packed composites require each child to be a uniform-shape dense NPZ source; different
children may have different shapes. Indexed, sequential, and map support is not evidence of equivalent decode
parallelism, caching, memory enforcement, or throughput.

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
Eligible packed composites also retain packed IDs when different children have different shapes. Non-packed
mixed sources and sequential sources use generic bounded envelopes.

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

Packed reservoir draining builds its permutation in native code, preserving the
same counter-based hash draws and sample order. It avoids a Python integer list
and per-element interpreter loop, and periodically releases the GIL so decoder
threads can continue while the permutation is built.

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

### Composite sources and mixing modes

`iterative_multi` combines native child sources. Each entry in `dataset_dict`
provides its own format, paths, board sizes, and optional rule declaration.
`multi` remains the map-style concatenation counterpart; the mixing modes below
apply to `iterative_multi`.

Use one `mixing` mapping to select both the source distribution and epoch
completion policy. Omitting it selects `balanced`.

| `mixing.mode` | Source distribution | Epoch completion |
|---|---|---|
| `balanced` | Equal source counts | Last complete equal-weight cycle |
| `weighted` | Each child's positive `blend_ratio` (default `1.0`) | Last complete weighted cycle |
| `natural` | Random interleaving proportional to remaining source rows | Every source row supplied once |
| `tempered` | Size weights `N_i ** size_power` | Maximum feasible proportional quotas, rounded down to whole rows |

`balanced` and `weighted` supply deterministic integer-ratio cycles to the
planner's bounded shuffle. `natural` and interior `tempered` modes randomly
interleave remaining source quotas in bounded blocks without replacement.
Within each child, the existing file order and bounded sample shuffle still
apply: this is not a globally uniform permutation of every stored position.
Sources with the same output shape can share a batch. Different shapes retain
the existing shape buckets. Source weights are not per-batch quotas, and a
short prefix can fluctuate.

For example, combine this data fragment with a model and optimizer recipe:

```yaml
dataset_type: iterative_multi
data_pipeline: {}
no_shuffle: false
num_worker: 0
dataset_args:
  mixing:
    mode: natural
  shuffle_window_size: 262144
  dataset_dict:
    freestyle_source:
      dataset_type: batched_processed_katago_numpy
      data_paths: [data/freestyle/train]
      boardsizes: 15
      rule: freestyle
    standard_source:
      dataset_type: batched_processed_katago_numpy
      data_paths: [data/standard/train]
      boardsizes: 15
      rule: standard
    renju_source:
      dataset_type: batched_katago_numpy
      data_paths: [data/renju/train]
      boardsizes: 15
      rule: renju
dataloader_args:
  batch_by_boardsize: true
```

Choose equal source supply with `mixing: {mode: balanced}`. To configure a
`2:1:1` source mix, use `mixing: {mode: weighted}` and set `blend_ratio: 2.0`
on the first child and `blend_ratio: 1.0` on the other two. These weights balance
sources, not rule classes: two freestyle sources and one source for each other
rule have a `2:1:1` rule mix at equal source weights. `blend_ratio` is rejected
outside `weighted`, even when explicitly set to `1.0`.

To transition between equal source counts and the original size distribution:

```yaml
mixing:
  mode: tempered
  size_power: 0.5
```

`size_power` is required in `tempered`, must be finite and in `[0, 1]`, and is
rejected in other modes. Zero is exactly `balanced`; one is exactly `natural`,
including order and completion behavior for the same seed. For an interior
power `alpha`, let `N_min` be the smallest child count. The epoch quota is
`floor(N_min * (N_i / N_min) ** alpha)` for each child; limiting sources retain
all `N_min` rows. This differs from the continuous ideal by less than one row
per child. The remaining rows of larger sources are not visited in that epoch.
Known-length, globally planned sources are not oversampled within an epoch.
Unknown-length rank-sharded streams retain their existing step-budget cycle policy.

`N_i` counts logical rows after decoder-side filters. `natural` and interior
`tempered` require known counts and `sample_rate: 1` on every child; sources
with probabilistic admission or unknown length are rejected instead of using
incorrect estimated counts. Natural mixing skips valid empty sources and
rejects an entirely empty mixture. Balanced, weighted, and interior tempered
mixing fail when a source cannot supply its required positive contribution.
Binary/sequential streams can still use balanced or weighted mixing under
their existing planner constraints.

Completion refers to the source stream. Training drops incomplete global
shape batches, so yielded training counts can be slightly smaller; evaluation
pads shape tails and supplies an `is_real` mask instead. Use `natural` on the
validation composite to cover all held-out sources. A fixed training step
limit may also stop partway through an epoch.

Keep `no_shuffle: false` in training recipes; direct `build_dataset` callers
must pass `shuffle=True` to enable the planner shuffle. Natural/tempered source
selection itself is seeded independently of that planner switch. Source
selection is rank-independent for globally planned datasets. Mode, quotas,
seed, pending source records, and remaining counts participate in exact resume;
changing the mixing contract rejects an old cursor.

`sync_length` has been removed. Migrate equal-source recipes to `balanced`,
custom ratios to `weighted`, and full-source coverage to `natural`. The old
weighted-then-drain ordering is not retained. Old composite runtime checkpoints
are incompatible with the new source schema; use the original code to resume
those runs or start a new data stream with the new configuration.

The example uses the same adaptive runtime for processed and native raw NPZ.
Native `batched_katago_numpy` normalizes full-board arrays once per cache miss,
then shares vectorized gathering, symmetry, packed mixing and ordered prefetch
with `batched_processed_katago_numpy`. Its decoded cache is bounded CPU memory;
it does not convert the dataset or write normalized files. Raw sources disable
the optional processed-only disk cache for the mixture. Native raw filters and
padded board masks require `iterative_katago_numpy`; the batched reader rejects
them instead of automatically falling back to indexed decoding. Arbitrary channel
selection is not a supported native raw option. Prefer the batched reader when
its full-board contract fits the input. Rules remain attached. See [Rule annotations for NPZ
sources](#rule-annotations-for-npz-sources) for the field contract.

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
internal ordered decode workers where supported. Budgeted packed execution selects their active count;
batched compatibility layouts retain `prefetch_threads`, while serial backends do not gain decode workers.

## Adaptive resource control

The primary resource interface is the top-level singular `data_pipeline` mapping. It is separate from
`data_pipelines`, the list of semantic batch transforms. All built-in dataset families participate in execution
selection, but their resource capabilities differ:

- `continuous` adjusts supported worker/cache/queue capacities; `manual` fixes a supported budgeted layout.
  Dense and indexed NPZ can reserve host-data memory, although indexed execution remains serial.
- `observed` is a memory-accounting mode, not an adaptation value. Map, binary, binary-containing composites,
  and implicit compatibility layouts report observations without enforcing a host-data hard cap or tuning their
  layout. They accept the automatic continuous policy, not explicit memory/CPU caps or manual layouts.
- `legacy_fixed` is the execution policy when no runtime spec is passed, as in direct dataset use and current
  validation setup. Iterable sources still use the common planner/executor. The trainer can preserve that same
  layout under an implicit continuous request with passive observation; the reported policy alone does not
  establish that tuning or reserved memory accounting is active.

In the training entry point, omitting `data_pipeline` enables budgeted defaults for eligible batched NPZ sources.
Fresh `iterative_multi` runs also use that default when every child is an unfiltered, single-shape batched dense
NPZ source without channel selectors, and the output shape is guaranteed by `fixed_board_size` or one concrete
NPZ file per child. The children may have different shapes. Other `iterative_multi` and formerly fixed
`iterative_*`/sparse sources retain their compatibility layouts. Legacy prefetch/pinning/loader settings or
transforms without parallel-stateless support also preserve the implicit layout. An explicit mapping, including
`{}`, opts into the selected backend's supported policy and rejects incompatible options. Budgeted execution
requires parallel-stateless batch transforms; observed map and
standalone binary execution retain ordered transform support. Built-in iterable sources require `num_worker: 0`
in either policy; map loaders can use DataLoader workers.

Validation currently constructs its dataset without the training resource spec, so it uses fixed defaults even
when training explicitly enables adaptive control. Map evaluation uses its evaluation planner wrapper. Training
resource-policy settings therefore do not establish validation memory or throughput behavior.

For other supported `iterative_multi` layouts, opt in explicitly with `data_pipeline: {}`. Dense, indexed raw,
and sparse NPZ children can share a budget; binary-containing mixtures use observed generic execution.
Parallel-stateless transforms belong on the composite, not its children. Eligible dense mixtures use the packed
backend; other supported mixtures use generic routing. Format-specific filtering and shape restrictions still
apply. The selected `mixing` policy
and supported record-level `sample_rate` semantics are retained. Packed record IDs are shuffled and
bucketed by shape before the global batch is partitioned across ranks; each
rank therefore receives the same board size at each step. Queued shape buckets
and source cursors are included in exact checkpoint/rollback state.

Mixed dense NPZ batches retain packed IDs through decoding. Requests are
grouped by child across each bounded prefetch chunk, decoded as arrays, and
scattered back into the original batch positions. Sample keys remain lazy;
rule labels and deterministic child transforms follow the same row routing.
This avoids reconstructing per-row envelopes and dictionaries for numeric
mixed batches. The generic mixed path also combines compatible numeric child
arrays directly; tensor and variable-length fields retain compatibility
collation. Field schemas and batch-shared values are still checked.

The budgeted mixed path uses one parent memory budget for decoded RAM caches,
materialized batches, pinning, and telemetry; parallel decoding is available on the packed path.
An eligible all-processed mixture may also prepare a shared mapped cache; native raw mixtures use only the RAM caches.
Its output reservations include the temporary coexistence of child decode
arrays and merged output, plus routing and lazy-key metadata. It does not
allocate an independent adaptive budget for every child. Keep
`dataloader_args.batch_by_boardsize: true` for mixed sizes. With
`cuda_prefetch_batches: 1`, mixed shapes automatically select the existing
shape-flexible CUDA lookahead loader; uniform streams retain static input slots.
Fresh eligible dense mixtures now select the same packed path when the mapping is omitted. All other omitted
mixtures retain the generic compatibility path. Resume preserves the saved generic or packed backend rather than
silently changing sample order. This fresh-run default can change the sample order compared with older omitted
configurations. Enabling an explicit policy does not automatically convert an existing generic checkpoint into
packed execution.

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

Eligible budgeted continuous processed-NPZ runs keep the source files and their format unchanged. At startup, one leader per node
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

For budgeted packed NPZ, `continuous` starts with a plan that fits the resolved memory and CPU limits. The private-cache
working set is estimated from dataset size, batch size, and shuffle lookahead. Decode concurrency starts at half the
portable CPU ceiling; chunk size and queue depth are balanced around that count. Workers, chunk size, and queue
depth form one layout: the controller never adjusts one value while leaving the other two in an unrelated
intermediate state.

Waiting decode calls enter the concurrency limit in FIFO order. A worker that
finishes a later chunk cannot repeatedly take the next slot ahead of an older
waiting call. This keeps the ordered consumer's next chunk from being starved
by speculative work, without increasing worker concurrency, cache capacity, or
the ready queue. Batch publication and committed sample order remain unchanged.

After the initial queue fill, the consumer plans replacement batches incrementally
between training steps and accumulates them into the existing decode chunks.
A bounded extra admission catches up when queue capacity grows. Staged batches
count toward the same memory and queue limits; epoch ends and an otherwise empty
queue flush partial chunks. This avoids periodic large planning bursts while
preserving grouped file reads and batch decoding.

Compatible adaptive NPZ training also prefetches across epoch boundaries. Once
the current epoch is fully planned, its remaining batches and the next epoch's
first batches share one bounded queue and decode executor. The next planner is
adopted only after the current epoch is committed; checkpoints still describe
consumed training data. Augmentation uses each batch's epoch even when workers
decode two epochs concurrently. Both planner states are included in the host
memory capacity calculation. If the selected layout cannot fit that overlap,
the loader retains finite-epoch prefetch. This is automatic and adds no public
configuration option; stateful transforms and evaluation keep their existing
epoch lifecycle.

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
| `host_memory_budget` | Node-total logical host-data budget on reserved paths; observed paths accept only `auto` without enforcing a cap | `auto` |
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

Explicit policies reject fixed performance keys in `dataset_args`: `prefetch_threads`, `prefetch_batches`, and
`pin_memory`. Budgeted execution accepts parallel-stateless `data_pipelines`; unsupported transforms require
the compatibility layout. Format, filtering, admission, target, and augmentation options remain
under `dataset_args` and keep their existing semantics. Training shuffle is enabled by default and disabled with
top-level `no_shuffle: true`; only its semantic window belongs in `dataset_args`. Explicit iterable policies reject
loader aliases `dataloader_args.pin_memory` and `dataloader_args.shuffle_buffer_size`: pinning belongs to the
execution layout, while the semantic shuffle window belongs in `dataset_args.shuffle_window_size`. Observed map
loaders retain their loader pinning option.

For a fixed dense-NPZ budgeted plan, use `adaptation: manual` with all five `advanced` values:

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

When the adaptive mapping is omitted, formerly fixed iterable dataset types, explicit legacy performance
controls, or transforms without parallel-stateless support retain the existing layout. Nonzero `num_worker`
does not enable a fallback for built-in iterable sources; it is rejected. Compatibility controls include:

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

Eligible budgeted continuous processed-NPZ runs use one run-scoped mmap cache per node, leaving ready batches and planner
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
- Dense uniform NPZ has the strongest production performance coverage. Other formats share semantic contracts
  where their capabilities allow, but memory accounting and execution differ; profile representative real data
  before making performance claims.

### Possible simplification order

Keep public names as compatibility aliases while recommending fewer entry points for new configurations. Next,
consolidate duplicated adapter setup where source identity, ordering, and cursor schemas remain unchanged. Only
optimize or retire a decoder path after checking its real callers, format/filter coverage, complete-epoch field
and sample-key parity, old-cursor continuation, and representative end-to-end performance. In particular, the
indexed raw reader still uses the map reader internally, and dense native NPZ does not cover every indexed raw
input. Retain format backends and saved execution layouts until those dependencies have a validated replacement;
a shared contract does not require one universal decoder.

## Rule annotations for NPZ sources

Raw and processed KataGo NPZ datasets accept an optional `rule` declaration:
`freestyle`, `standard`, or `renju`. `Rule.from_string` in `utils/data_utils.py`
uses the existing stable training indices `0`, `1`, and `2`. These indices are
not the binary format's enum values (Renju's enum value is `4`). The declaration
applies to every sample in that source; it does not infer or validate game rules
from board positions, filenames, or stored targets.

When declared, the loader adds `rule_index` with sample shape `(1,)` and NumPy
`int64` dtype. Batches have shape `(B, 1)` and become `torch.int64` tensors during
normal loader conversion. Scalar, vectorized, packed, and cached decode paths
retain the annotation through shuffling, augmentation, and distributed slicing.
Changing the declaration changes source identity for exact resume. Omitting it
preserves the previous output and identity. A composite rejects children that
mix annotated and unannotated output schemas, even with different board sizes.

The singular `rule` annotates a source. The existing plural `rules` option does
not create NPZ annotations and must not be used as a replacement. Models receive
`rule_index` in the batch dictionary. ResNet-family models can consume it with
`input_type: rule`; see [Rule-conditioned model inputs](model_inputs.md) for
configuration, side splitting, tensor shapes, compilation, and checkpoints.

Rule labels and mixing are independent: different sources may declare the same
rule, and no mode automatically balances rule classes. Configure each child's
`rule` alongside its `data_paths` and choose the source distribution through
`mixing`, as shown in [Composite sources and mixing modes](#composite-sources-and-mixing-modes).

Rule-free input types continue to ignore the metadata. Selecting a rule-conditioned
input requires every source and inference caller to provide the annotation.
