# Training performance

This page is the maintained reference for the performance work in this repository. It records the
measurement contract, the optimizations that remain enabled, the current hardware results, and the checks
required before accepting another performance change.

## Measurement contract

The maintained references cover multiple hardware regimes. Each section records its software stack, workload,
and MFU convention; results from different systems are context, not direct A/B comparisons. Synthetic inputs remain
on the GPU so model-only benchmarks exclude data decoding, host-to-device copies, logging, teacher inference, and
cold compilation.

Results are steady-state medians after warm-up. Comparisons must keep the model arguments, batch shape,
precision, compiler policy, optimizer, and starting state fixed. GPU jobs run serially. For bounded benchmark
runs, use a per-process allocator ceiling and reduce the batch size after a recoverable OOM; never raise the
ceiling to make an oversized workload fit. The reference limits are 0.25 for focused checks and 0.50 for
full-model profiling.
Compiled multi-process results must also record the source commit and verify that DDP remains inside the live
compiled wrapper. A valid check uses distinct rank-local inputs and requires identical parameters after the
optimizer step, a nonzero parameter update, and confirmation that the trainer's forward target is the DDP object;
a recorded world size alone does not establish synchronized training.
Replicated multi-process training also checks the optimizer-owned parameters after the first non-skipped update
and before every permanent or final checkpoint. The check compares bounded rank-zero reference chunks without
modifying live weights; temporary snapshots stay off this path. Sharded strategies require their own replica-group
semantics and therefore do not use this whole-parameter equality check.
Normal training leaves `max_memory_fraction` unset, which applies no PyTorch allocator ceiling. The option is an
explicit allocator guard based on total visible VRAM; it neither reserves memory nor tracks current free memory,
and it does not constrain every non-PyTorch allocation.

## Current results

The reference measurements use the configurations described below. Throughput can vary with compiler cache
and system load, so these values are reference points rather than portable constants.

### RTX 4080 SUPER model references

These measurements use PyTorch 2.8.0, CUDA 12.8, cuDNN 9.10.2, BF16 autocast, TorchInductor `max-autotune`,
performance level 2, and fused AdamW unless stated otherwise. MFU uses the measured 97.5 TFLOP/s dense BF16
roofline and forward-plus-backward FLOPs. It is meaningful for dense ResNet workloads. For VQ, embedding,
depthwise, fake-quantized, and launch-bound models, kernel time and end-to-end throughput remain the primary
metrics because useful work is not represented by dense FLOPs alone.

| Workload | Batch | Median step | Throughput | Main remaining limit |
| --- | ---: | ---: | ---: | --- |
| Mix9 | 128 | 6.39 ms | 20,018 samples/s | balanced convolution, quantization/reduction, and GEMM |
| Mix9s | 128 | 6.89 ms | 18,576 samples/s | balanced convolution, quantization/reduction, and GEMM |
| Mix10 | 128 | 6.49 ms | 19,728 samples/s | balanced convolution, quantization/reduction, and GEMM |
| Mix9sVQ, 65,536 codes | 512 | 61.81 ms | 8,284 samples/s | VQ search and VQ-adjacent grouping/EMA work |
| Flat V4 cosine VQ, 65,536 codes | 2048 | 24.99 ms | 81,949 samples/s | cosine search, followed by convolution |

The three non-VQ MixNet rows were measured after the optimizer and mapping-lowering changes described below.
Absolute values drift between sessions by more than the effect being measured, so comparisons should use the
same session configuration. The VQ rows use an earlier validated configuration and are carried forward
unmodified.

The uniform 34-model profile gives the following structural picture:

- ResNet v1/v2 spend 82–84% of GPU kernel time in dense convolution and sustain about 79–81% MFU. ResNet
  v3 reaches about 71.5% MFU; its masked normalization/reduction path remains the material difference.
- Non-VQ MixNet backward is 60–68% of step time. Convolution contributes roughly 39–45% of GPU kernel time,
  pointwise/quantization/reductions 36–40%, and GEMM/BMM 10–21%. There is no dominant isolated kernel.
- Mix9sVQ previously attributed about 19% of GPU kernel time to coarse search. Grouping, rotation, norms,
  perplexity, and EMA make the wider VQ pipeline the main remaining opportunity.
- MobileNet spends 50–60% of kernel time in training normalization and about 70% of the step in backward.
  Its low dense MFU is a bandwidth/reduction property rather than unused dense-compute capacity.
- Pattern models are dominated by repeated-index embedding backward, scatter/sort, and small depthwise work.
  Linear and the smallest Flat models are launch- and optimizer-bound.

Exact model-only, same-batch H200 context reinforces that the ratios are workload-dependent rather than a
portable hardware multiplier:

| Workload | Batch | H200 median step | H200 throughput | Relative to the 4080 SUPER row |
| --- | ---: | ---: | ---: | ---: |
| Mix9s | 128 | 7.036 ms | 18,192 samples/s | 0.979x |
| Mix10 | 128 | 7.044 ms | 18,170 samples/s | 0.921x |
| Mix9sVQ, 65,536 codes | 512 | 31.880 ms | 16,060 samples/s | 1.939x |

The measured small non-VQ batches were launch-bound and did not benefit from the larger accelerator, while the
matching VQ workload nearly doubled throughput. Larger production batches are measured separately below.

#### End-to-end ResNet reference (600k acceptance, 2026-08-07)

Full 600,000-iteration runs per reference config on the RTX 4080 SUPER, recipe identical between
codebases (seed 42, weight decay 1e-7, save every 50k / rolling temp save every 5k / validation
every 50k). The `perf` runs use the `batched_processed_katago_numpy` pipeline (4 prefetch threads,
64-batch queue) with static input slots, `cuda_prefetch_batches: 1`, BF16 autocast, TorchInductor
`max-autotune`. Rates derive from TensorBoard wall times: **effective** is the whole-run rate
including validation and checkpoint pauses; **clean** is the median per-interval speed with
pause-containing intervals excluded. `old` is the historical codebase, `master` the previous
reference implementation. Final validation loss establishes training-quality parity (the old↔master
spacing per config is itself up to ±0.008 at this horizon).

| Workload | Old eff | Master eff | `perf` eff | vs master | Final val (perf / master) |
| --- | ---: | ---: | ---: | ---: | --- |
| ResNet v1 2b32 | 88.3 | 159.9 | 224.6 | +40.5% | 1.9622 / 1.9641 |
| ResNet v1 4b64 | 85.5 | 148.1 | 189.8 | +28.2% | 1.7456 / 1.7465 |
| ResNet v1 6b96 | 46.7 | 120.3 | 178.5 | +48.4% | 1.6336 / 1.6332 |
| ResNet v1 10b128 | 41.2 | 76.2 | 109.9 | +44.1% | 1.5545 / 1.5572 |
| ResNet v1 15b192 | 16.1 | 32.3 | 37.7 | +16.8% | 1.5025 / 1.5059 |
| ResNet v1 20b256 | 7.7 | 17.3 | 18.9 | +9.4% | 1.4649 / 1.4608 |
| ResNet v2 4b64 | 120.4 | 139.3 | 197.2 | +41.6% | 1.7440 / 1.7417 |
| ResNet v2 6b96 | 110.5 | 114.8 | 175.4 | +52.8% | 1.6321 / 1.6324 |
| ResNet v2 10b128 | 80.7 | 78.2 | 105.5 | +34.9% | 1.5708 / 1.5729 |
| ResNet v2 15b192 | 33.2 | 30.5 | 37.1 | +21.4% | 1.5173 / 1.5136 |
| ResNet v2 20b256 | 17.7 | 16.8 | 18.6 | +10.5% | 1.5011 / 1.4978 |

Clean steady-state rates (it/s): v1 233.8 / 199.0 / 185.4 / 113.0 / 38.3 / 19.1 and v2 — / 205.0 /
181.2 / 108.8 / 37.7 / 19.0. Estimated end-to-end MFU at clean steady state (forward+backward FLOPs
over the 97.5 TFLOP/s dense BF16 roofline): ~59% for the 10b128 class, ~79% for the 20b256 class.
The 2b32-class models are launch-bound at batch 128 and are tracked by step time rather than MFU.

### Four-H200 synchronized MixNet reference (2026-08-17)

These real-data brackets use the pre-history-consolidation synchronized snapshot captured on 2026-08-17,
PyTorch 2.10.0, CUDA 13.0, cuDNN 9.15.1, BF16 autocast,
TorchInductor `max-autotune`, one optimizer step per iteration, and 300 warm-up plus 1,200 measured steps. The
live wrapper was `OptimizedModule -> DistributedDataParallel`, rank-local inputs differed, and post-step parameter
spread was zero. The brackets retain the production data, KD teacher, loss, clipping, optimizer, and finite-value
paths while omitting logging, validation, and checkpoints.

Nominal MFU divides represented aggregate work by four times the 989 TFLOP/s dense-BF16 roofline. Represented
work includes student forward/backward and the frozen teacher forward, but not trainer reductions or custom VQ
work; VQ MFU is therefore only a lower bound. The GPU-busy values come from the sampler's trailing process window,
which can extend past the measured bracket, so they are context only and are not tensor-core MFU.

| Workload | Global batch | Mean step | Throughput | Max-rank data wait | Trailing GPU busy | Nominal MFU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mix9s | 8,192 | 54.930 ms | 149,134 samples/s | 2.18 ms | 69.92% | 11.20% |
| Mix9s | 16,384 | 88.333 ms | 185,480 samples/s | 2.50 ms | 84.00% | 13.94% |
| Mix9s | 32,768 | 164.090 ms | 199,696 samples/s | 3.47 ms | 92.23% | 15.00% |
| Mix10, 256 middle / 128 feature | 8,192 | 85.520 ms | 95,791 samples/s | 1.37 ms | 87.57% | 19.10% |
| Mix10, 256 middle / 128 feature | 16,384 | 158.731 ms | 103,219 samples/s | 1.59 ms | 93.80% | 20.58% |
| Mix9sVQ, 16,384 codes | 8,192 | 75.656 ms | 108,279 samples/s | 2.45 ms | 72.00% | >=8.14% |
| Mix9sVQ, 16,384 codes | 12,288 | 102.022 ms | 120,444 samples/s | 2.62 ms | 76.77% | >=9.05% |
| Mix9sVQ, 16,384 codes | 14,336 | 127.335 ms | 112,585 samples/s | 3.03 ms | 79.46% | >=8.46% |

Data wait remains small at every measured point. Mix9s gains 24.4% throughput from batch 8,192 to 16,384, but
only another 7.7% at 32,768 while nearly doubling step time and reserved memory. Mix10 similarly gains only 7.8%
from doubling 8,192. Mix9sVQ peaks at 12,288; increasing to 14,336 reduces throughput by 6.5%. Its estimated
one-time warm-up overhead was about 38, 735, and 928 seconds respectively. This subtracts the projected
steady steps from the complete warm-up and is not an isolated compilation timer; repeat measurements should record
cache state and report cold start separately. These models can show high GPU busy while sustaining low dense MFU
because depthwise, pointwise, quantization, reduction, and communication work is not equivalent to large dense
Tensor Core work.

The profiler adds enough overhead that traced wall times are not throughput measurements, so the table above is
authoritative for speed and the traces are used only for attribution. At the selected throughput knees, Mix9s
(batch 16,384) spent 52.9% of GPU kernel time in pointwise work, 39.2% in GEMM/convolution, and 5.7% in
reductions; Mix10 (batch 8,192) spent 50.8%, 41.4%, and 5.2% respectively. NCCL accounted for less than 0.3% in
both. For Mix9sVQ (batch 12,288), VQ-specific grouping, search, and EMA kernels together accounted for 8.1% of
the real step wall time, while its largest custom search kernel accounted for only 2.8%. H2D copies were
sub-millisecond in every selected profile. The remaining gap is therefore broad operator efficiency and fusion,
not a single data, transfer, optimizer, clipping, or communication bottleneck. None of these workloads approaches
80% dense-BF16 MFU; doing so would require substantial model or compiler-lowering changes rather than another
input-pipeline tuning knob.

#### Equal-wall-time Mix9s selection

A six-arm screen compared global batches 8,192, 16,384, and 32,768 with both the existing learning rate and a
batch-scaled rate. Although larger batches increased samples per second and GPU busy, batch 8,192 produced the
lowest validation loss after the same trainer elapsed time. The scaled rates for 16,384 and 32,768 became
unstable. The two batch-8,192 finalists were then repeated with seeds 42-45 in balanced AB/BA order. Each pair was
interpolated to the shorter run's final validation time:

| Seed | LR 0.002 loss | LR 0.001 loss | Lower loss |
| ---: | ---: | ---: | --- |
| 42 | 0.627276 | 0.620444 | LR 0.001 |
| 43 | 0.608115 | 0.624786 | LR 0.002 |
| 44 | 0.609969 | 0.615764 | LR 0.002 |
| 45 | 0.615468 | 0.626228 | LR 0.002 |
| Mean | 0.615207 | 0.621805 | LR 0.002 |

LR 0.002 won three of four pairs and reduced mean total loss by 0.006599 (1.06%); the population standard
deviation of the paired difference was 0.008658. The resulting long-run recipe uses global batch 8,192 and
initial LR 0.002 with the production StepLR schedule (50,000-step interval, gamma 0.9). Only the batch and initial
rate are transferred from the short constant-rate experiment; the long-run result must be reported separately.

## Optimizations retained in the code

### Runtime and input pipeline

- The trainer compiles the forward-and-loss region while keeping unsupported control flow outside the graph.
  Model-scoped Inductor options can refine lowering without changing unrelated models.
- Vanilla DDP uses gradient bucket views, so gradients reuse reducer storage instead of being copied into
  communication buckets. Diagnostic multi-process brackets suggested about 5% lower fixed-batch step time at
  global batch 2,048 and 1.5-2.0% on the larger Mix9s, Mix10, and Mix9sVQ checks. Those A/B artifacts predate the
  commit/topology provenance gate above, so the exact deltas are contextual rather than acceptance references.
  Compile-time unwrapping must preserve the live DDP wrapper.
- CUDA AdamW runs as one Triton launch over each parameter group (`utils/fused_adamw.py`). PyTorch's
  `fused=True` kernel passes its hyperparameters as `double` and recomputes `pow(beta, step)` in double
  precision per thread; sm_89 runs FP64 at 1/64 rate, and its 36-tensor launch cap means every small bias
  tensor pays that cost too, so the optimizer was a fixed 0.479 ms on every model here regardless of size.
  Evaluating the bias correction once on the host removes all FP64 from the GPU: 512.5 -> 12.0 us of
  CUDA-graph replay time over the Mix9s parameter set. State stays per-parameter, so `state_dict` remains
  interchangeable with `torch.optim.AdamW` in both directions. Muon accepts non-contiguous convolution
  gradients, batches same-shape updates, and updates persistent momentum buffers in place.
- KataGo input can be decoded and collated as complete batches, with bounded producer concurrency and
  asynchronous device preparation. The loader explicitly marks batch ownership to avoid double batching.
  The 2026-08-07 RTX 4080 acceptance used four prefetch threads and a 64-batch queue. Eligible processed-NPZ
  training now enables portable adaptive sizing when the `data_pipeline` mapping is omitted; device lookahead
  remains an explicit workload choice.
- Training batches are delivered through persistent device slots marked `mark_static_address` when
  `cuda_prefetch_batches > 0` (`StaticSlotLoaderWrapper`). With Inductor CUDA graphs this eliminates the
  per-tensor input-stabilization DtoD copies and their host latency; copy ordering uses a two-event
  handshake on a dedicated stream, and a fresh iterator (epoch restart) orders its first refill behind
  the compute stream's current tail, so slots are never overwritten while a previous step can still
  read them. `NNUE_FORCE_CUDA_PREFETCH=1` restores the ring prefetcher.
- Loss, metric, schema, and rank-agreement failures synchronize before backward. On an unscaled single-device
  final micro-step, one combined loss/metric/gradient-norm result uses the one-step-late pinned readback.
  Vanilla CUDA/NCCL DDP instead copies the scalar norm of its already-synchronized gradients to pinned host memory
  without another collective and settles that copy in the same optimizer boundary. Plugin and non-NCCL backends
  retain the general synchronous distributed finite check. Validation, checkpoints, and the final boundary drain
  every queued check; divergence still reports the poisoned step's own iteration, and at most one poisoned optimizer
  mutation can occur before the abort. Profiling retains stricter same-iteration readback.
- Single-device gradient clipping uses a direct `_foreach_norm -> vector_norm -> clamp -> _foreach_mul_`
  chain (`utils/training_utils.clip_grad_norm`), bitwise-identical to the stock implementation, skipping
  its per-step regrouping and dispatcher overhead; multi-device/dtype layouts and multi-process runs fall
  back to the stock path.
- TensorBoard event writes pass through a buffer shim below the TFRecord framing
  (`utils/tb_writer.py`); on latency-bound filesystems (e.g. drvfs mounts) the stock per-scalar framing
  could block the training loop for hundreds of milliseconds per log interval.
- Periodic checkpoints serialize off the training loop (`utils/async_checkpoint.py`). Submitting a
  save clones every payload tensor on a dedicated copy stream behind the compute tail and blocks the
  compute stream only until that device-resident snapshot completes (~2 ms for a 238 MiB state); a
  single writer thread then pickles (its DtoH copies overlap training), commits each file through an
  atomic tmp+rename — model files before the training-state file, so a state file never references
  unwritten model weights — and prunes older snapshots as a post-write hook. A failed write surfaces
  at the next save or at shutdown and fails the run exactly like a synchronous failure; an OOM during
  staging drains the copy stream and falls back to the historic inline path. The only behavioural
  delta is that checkpoint bytes reach the filesystem shortly after the save iteration instead of
  before it: a crash inside that window resumes from the previous completed checkpoint, and torn
  partial writes remain invisible (verified with a mid-write SIGKILL + resume test).
  `NNUE_SYNC_CHECKPOINT=1` forces the old inline path. Measured +6.4% throughput on the 10b128
  ResNet with saves every 250 iterations; the win is ~0.3-0.4% at the 5000-iteration reference
  cadence.
- `easyrun.sh` creates an Accelerate configuration when none exists, saving BF16 and TorchInductor defaults
  there instead of injecting them on every launch. Existing Accelerate configurations are not overwritten.
- An explicit `max_memory_fraction` is applied before datasets and models allocate CUDA tensors. The default is no
  allocator cap; `0.9` is a reasonable explicit guard on an otherwise exclusive GPU, while smaller benchmark caps
  remain useful for controlled OOM recovery.

### Shared model operators

- The old monolithic `model/blocks.py` is split into reusable `model/layers` and `model/ops` modules.
  MixNet-specific composition remains in `model/mixnet_components.py`; primitive operations stay reusable.
- ResNet convolution weights and gradients retain channels-last layout where it selects faster cuDNN
  training kernels. Masked normalization uses a compact closed-form backward.
- MixNet reuses batched directional operations, GEMM-based diagonal three-tap mappings, optimized mixed-dtype
  pixelwise depthwise gradients, and model-scoped 1x1-to-GEMM lowering for Mix9, Mix9s, and Mix10. The VQ
  subclass intentionally keeps normal convolution lowering because the GEMM hint regressed its full graph.
- The mapping trunk's pointwise stages are expressed as matmuls rather than 1x1 convolutions. Their previous
  lowering to `aten.convolution_backward` was an Inductor fallback with `constrain_to_fx_strides`, which
  denied the backward a Triton epilogue and pinned its neighbours' strides. As matmuls, 14 of the 22 SiLU
  backwards fuse into the dgrad template and the bias gradient becomes a freely scheduled sum
  (258.4 -> 71.8 us). SiLU was already at the memory roofline, so removing the round trip was the only
  available lever. Parameter shapes, `state_dict` keys and the exported layout are unchanged.
- Small, heavily reused pattern tables keep native grouped embedding backward opaque to Inductor. PatNet v2
  retains channels-last embedding output for its depthwise stages. Large embedding tables do not use this
  boundary.

### Vector quantization

- Supported L2 and cosine searches use BF16 Tensor Core coarse candidates followed by FP32 refinement and
  deterministic tie handling. Specialized 32/64/96/128-dimensional paths cover measured 16,384- and
  65,536-code workloads; unsupported devices, layouts, shapes, or sizes fall back to KeOps.
- Single-rank EMA updates avoid a codebook-sized temporary. Cosine EMA normalization, perplexity reduction,
  repeated cluster quantiles, initialization-state synchronization, and conservative dead-code checks each
  remove measured memory traffic or synchronization. Distributed EMA keeps its original global-sum path.
- The accelerated search is an empirically validated shortlist, not a mathematical guarantee for every
  adversarial near-duplicate codebook. Set `accelerated_search: false` to require the exact KeOps search.
- INT8 VQ search is not retained. It was faster on static inputs but developed millions of assignment
  mismatches after EMA produced tightly clustered codes. FP8 is also not enabled: the representative Ada
  matrix shapes failed the operator throughput gate before stability testing was justified.

## Validation and acceptance

A performance patch is accepted only when it improves repeated, interleaved whole-model measurements. An
isolated kernel win is supporting evidence, not sufficient evidence. The minimum gates are:

1. compare state layout, outputs, auxiliary losses, gradients, and mutable state against a saved reference;
2. benchmark the targeted operator with representative shapes and both forward and backward where relevant;
3. repeat complete compiled training steps after warm-up, reporting median latency, throughput, and peak
   allocator memory;
4. for numerical changes, replay a deterministic multi-step trajectory and run a real-data stability soak
   with finite-state, loss-trend, checkpoint-resume, and VQ-assignment checks as applicable;
5. for compiled multi-process training, record the source commit, inspect the retained DDP topology and actual
   trainer forward target, feed distinct rank-local data, and require both a nonzero parameter update and zero
   post-step parameter spread.

The retained VQ path completed long compiled BF16 runs with finite model, optimizer, codebook, and EMA state.
Earlier 100,000-update stability claims remain scoped to their original recorded process topology and must not be
treated as synchronized multi-rank evidence. All registered model types passed CPU forward/backward smoke
coverage; optimized GPU paths additionally passed compiled numerical and representative checkpoint validation.

## Reproducing measurements

Benchmark and profile a complete model:

```bash
python -m tools.benchmark_model \
  --model-type mix9s \
  --model-args '{dim_middle: 128, dim_feature: 64, dim_policy: 32, dim_value: 64, dim_dwconv: 32}' \
  --batch-size 128 --board-size 15 --precision bf16 \
  --warmup-steps 10 --steps 50 --peak-tflops 97.5 \
  --max-memory-fraction 0.50 --output /tmp/mix9s-benchmark.json

python -m tools.profile_model \
  --model-type mix9s --batch-size 128 --board-size 15 --precision bf16 \
  --warmup-steps 5 --steps 5 --max-memory-fraction 0.50 \
  --output /tmp/mix9s-profile.json
```

Measure the production trainer ceiling with a fixed device-resident batch:

```bash
accelerate launch --config_file path/to/accelerate.yaml \
  tools/benchmark_trainer_ceiling.py \
  --config path/to/training.yaml \
  --board-size 15 --warmup-steps 300 --steps 1200 \
  --output trainer-ceiling.json
```

The trainer-ceiling artifact records its Git provenance and aborts before measurement unless warm-up changes an
optimizer-owned parameter, the live forward target retains DDP on every rank, and the resulting replicas agree.
Keep the artifact with the reported result; a `world_size` field by itself is not sufficient evidence.

When moving to an unmeasured accelerator architecture, rerun shape sweeps and profiles before changing kernels. Batch
optima, cuDNN algorithms, compiler schedules, FP8 economics, and VQ tile choices are hardware-specific. Keep
the portable fallbacks and numerical gates unchanged, then accept architecture-specific specializations only
after they improve the complete training step on that platform.
