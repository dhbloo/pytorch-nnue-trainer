# Training performance

This page is the maintained reference for the performance work in this repository. It records the
measurement contract, the optimizations that remain enabled, the current hardware results, and the checks
required before accepting another performance change.

## Measurement contract

The current reference system is an NVIDIA GeForce RTX 4080 SUPER with PyTorch 2.8.0, CUDA 12.8, and cuDNN
9.10.2. Model benchmarks use BF16 autocast, TorchInductor `max-autotune`, performance level 2, and fused
AdamW unless stated otherwise. Synthetic inputs remain on the GPU so model benchmarks exclude data decoding,
host-to-device copies, logging, teacher inference, and cold compilation.

Results are steady-state medians after warm-up. Comparisons must keep the model arguments, batch shape,
precision, compiler policy, optimizer, and starting state fixed. GPU jobs run serially. Use a per-process
allocator ceiling and reduce the batch size after a recoverable OOM; never raise the ceiling to make an
oversized workload fit. The reference limits are 0.25 for focused checks and 0.50 for full-model profiling.

MFU uses the measured 97.5 TFLOP/s dense BF16 roofline and forward-plus-backward FLOPs. It is meaningful for
dense ResNet workloads. For VQ, embedding, depthwise, fake-quantized, and launch-bound models, kernel time and
end-to-end throughput are the primary metrics because useful work is not represented by dense FLOPs alone.

## Current results

The reference measurements use the configurations described below. Throughput can vary with compiler cache
and system load, so these values are reference points rather than portable constants.

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

### End-to-end ResNet reference (600k acceptance, 2026-08-07)

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

## Optimizations retained in the code

### Runtime and input pipeline

- The trainer compiles the forward-and-loss region while keeping unsupported control flow outside the graph.
  Model-scoped Inductor options can refine lowering without changing unrelated models.
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
  Since the 2026-08-07 acceptance, the gomoku ResNet reference configs run this pipeline with four
  prefetch threads, a 64-batch queue, and `cuda_prefetch_batches: 1`.
- Training batches are delivered through persistent device slots marked `mark_static_address` when
  `cuda_prefetch_batches > 0` (`StaticSlotLoaderWrapper`). With Inductor CUDA graphs this eliminates the
  per-tensor input-stabilization DtoD copies and their host latency; copy ordering uses a two-event
  handshake on a dedicated stream, and a fresh iterator (epoch restart) orders its first refill behind
  the compute stream's current tail, so slots are never overwritten while a previous step can still
  read them. `NNUE_FORCE_CUDA_PREFETCH=1` restores the ring prefetcher.
- Single-process training handles rank-local phase errors on the CPU and performs one combined
  finite-value check after backward. The check is one stack->isfinite->all chain whose result is
  copied asynchronously into a ping-pong pair of pinned slots and read one iteration later, so the
  host never waits on the GPU pipeline for the finite result. Validation passes and checkpoints
  first drain every queued check including the current step's, preserving the invariant that neither
  can observe or commit the product of a step whose finite result has not landed. A divergence abort
  still reports the poisoned step's own iteration number; at most one extra (already-poisoned)
  optimizer step can be applied before the abort, and the profiling loop keeps the stricter
  same-iteration readback.
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
- `max_memory_fraction` is applied before datasets and models allocate CUDA tensors. This is a safety boundary
  on machines where VRAM oversubscription causes host paging.

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
   with finite-state, loss-trend, checkpoint-resume, and VQ-assignment checks as applicable.

The retained VQ path completed long compiled BF16 runs with finite model, optimizer, codebook, and EMA state.
ResNetV2 Muon and production Mix9s/Mix10 paths completed 100,000-update guarded stability runs and resumed
full training state. All registered model types passed CPU forward/backward smoke coverage; optimized GPU
paths additionally passed compiled equivalence and representative checkpoint tests.

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

Capture and replay behavior around a change:

```bash
python -m tools.check_optimizer_equivalence

python -m tools.check_model_equivalence snapshot /tmp/mix9s-reference.pt \
  --model-type mix9s --batch-size 2 --board-size 15
python -m tools.check_model_equivalence compare /tmp/mix9s-reference.pt

python -m tools.check_training_trajectory snapshot /tmp/mix9s-trajectory.pt \
  --model-type mix9s --batch-size 16 --steps 100 --device cuda \
  --precision bf16 --compile --max-memory-fraction 0.25
python -m tools.check_training_trajectory compare /tmp/mix9s-trajectory.pt \
  --device cuda --max-memory-fraction 0.25
```

When moving to H200 or another architecture, rerun shape sweeps and profiles before changing kernels. Batch
optima, cuDNN algorithms, compiler schedules, FP8 economics, and VQ tile choices are hardware-specific. Keep
the portable fallbacks and numerical gates unchanged, then accept architecture-specific specializations only
after they improve the complete training step on that platform.
