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

### End-to-end Mix9s knowledge-distillation reference

The latest real-data end-to-end reference was measured on 2026-08-03 from commit `f534671` on the reference
RTX 4080 SUPER. It used the Mix9s model, batch size 128, BF16 autocast, TorchInductor
`max-autotune`, the `batched_processed_katago_numpy` input pipeline with four prefetch threads and a
512-batch queue, and the ResNetV2 KD teacher. The run used the real training and validation streams,
validation every 50,000 iterations, and model/trainer-state checkpoints every 50,000 iterations.

| Run | Iterations | Trainer elapsed | Effective rate | Clean steady-state rate | Final validation loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current implementation (`f534671`) | 1,000,000 | 3:30:05.844 | 79.33 it/s, 10,154 samples/s | 84.93 median, 84.43 harmonic | 0.612722 |
| Previous reference (`bf96979`) | 1,000,000 | 4:35:52.262 | 60.41 it/s, 7,733 samples/s | 66.12 median, 66.05 harmonic | 0.614139 |

Trainer elapsed includes the training loop, validation, and checkpoint costs through the final validation.
Clean steady-state rates cover iterations 50,000 through 1,000,000 and exclude the first 500-step window
following each periodic state save; the harmonic mean retains the cost of remaining slow windows. Relative to
the previous reference, the current implementation reduced trainer time by 23.8%, increased clean harmonic
throughput by 27.8%, and ended with a validation loss lower by 0.001417.

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
- Single-process training handles rank-local phase errors on the CPU and performs one combined CUDA finite-value
  check after backward. It does not construct and synchronously copy six success flags to CUDA every step. This
  also prevents those pageable scalar copies from serializing an optional dedicated-stream H2D prefetch.
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
