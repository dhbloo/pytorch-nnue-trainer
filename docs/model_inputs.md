# Rule-conditioned model inputs

ResNet, ResNetV2, and ResNetV3 accept an optional rule-conditioned input. Source
mixing and source annotations remain data-pipeline settings; the model chooses
how to encode those annotations. See [Data pipeline](data_pipeline.md#rule-annotations-for-npz-sources)
for the source contract and [mixing modes](data_pipeline.md#composite-sources-and-mixing-modes)
for balanced, natural, weighted, and tempered sampling.

## Configuration

For a ResNetV2 with two board channels and four rule channels:

```yaml
model_type: resnetv2
model_args:
  num_blocks: 10
  dim_feature: 128
  input_type: rule
  input_args:
    base: basicns
    split_by_side: [renju]
  final_norm: bn
  value_bias: true
```

Every child dataset must declare `rule: freestyle`, `rule: standard`, or
`rule: renju`. Choosing `input_type: rule` requires these labels at both training
and inference; labels are never guessed or defaulted. Existing input types and
checkpoints keep their previous behavior when the new input is not selected.

`base` selects an existing input type. `basicns` supplies the two board planes;
`basic` also includes the STM plane. Pattern embedding inputs can be wrapped in
the same way. For ResNetV3, use `base: mask` or `base: maskns` to retain its
`(planes, board_mask)` interface.

`split_by_side` lists the rules whose black and white sides need distinct
conditions. It defaults to `[renju]`; `[]` gives three conditions and
`[freestyle, standard, renju]` gives six. Class order always follows canonical
rule order, then black/white, regardless of list order. Unknown or duplicate
names are rejected. Splitting a rule requires its data to retain actual STM:
a source whose stored STM is zero cannot be split by side.

## Tensor and semantic contract

| Field | Batch shape | Dtype / meaning |
| --- | --- | --- |
| `board_input` | `(B, 2, H, W)` | Existing board planes, converted to float32 by the base input |
| `rule_index` | `(B, 1)` or `(B,)` | int64 or int32: freestyle = 0, standard = 1, renju = 2 |
| `stm_input` | `(B, 1)` or `(B,)` | float32: black = -1, white = +1; zero allowed only for unsplit rules |

The normal dataset path emits `rule_index` as int64 `(B, 1)` and STM as float32
`(B, 1)`. Tensors and model must be on the same device. Invalid IDs, invalid STM,
missing labels, and incompatible shapes fail explicitly. CPU eager execution
raises an error immediately; CUDA uses a device assertion without synchronizing
the host on every forward. As with other CUDA device assertions, invalid input
requires restarting the affected CUDA process.

The default condition vector has shape `(B, 4)` and these one-hot columns:

| Rule | STM | Condition column |
| --- | --- | --- |
| freestyle | -1, 0, +1 | 0 |
| standard | -1, 0, +1 | 1 |
| renju | -1 | 2 |
| renju | +1 | 3 |

`basicns` plus this vector gives a float32 `(B, 6, H, W)` convolution input;
`basic` gives `(B, 7, H, W)`. Training autocast still controls convolution
precision. Dataset rules use `Rule.index`, not binary enum values: the Renju
enum value is 4, but its training index is 2.

The dataset's `fixed_side_input` changes board perspective while retaining
absolute STM, which the encoder uses normally. The input API's optional
`inv_side=True` follows the existing relative-perspective convention: swap board
planes and negate STM, including the split condition. Do not use it as an
invariant color-swap augmentation for asymmetric rules, or apply it as a
fixed-side perspective transformation.

## Composition and compilation

`RuleConditionEncoder` owns the semantic mapping and produces a small feature
vector independently of board layout. `RuleConditionedInput` wraps an existing
plane input and broadcasts the features spatially. Other model architectures
can reuse the encoder without adopting spatial planes.

The encoder uses a fixed device lookup table. It does not build one-hot tensors
from intermediate class IDs, transfer tables on every forward, or loop over
samples. Input components are concatenated once; spatial broadcasting uses
`expand`, without an intermediate repeated board-sized tensor. The final
concatenation necessarily materializes the convolution input.

Compile the whole model (and training loss), as the existing trainer does, to
allow fusion across Python module boundaries. Module separation itself does not
create graph breaks. Label values may change without recompilation; different
batch or board shapes may require separate graphs. A single graph is not a
promise of a single GPU kernel: validation reductions, convolutions, and other
operators may still require separate launches.

The default adds four channels only to the first convolution. It therefore has
some unavoidable additional convolution work compared with a rule-free model;
measure that separately from encoding overhead. Immutable rule buffers preserve
the trainer's optimization that defers ordinary BatchNorm buffer broadcasts to
evaluation, including models with no BatchNorm buffers.

## Checkpoints and inference

The canonical class table is stored as tensor metadata in the model checkpoint.
Loading a checkpoint with a missing or different mapping fails, even with
`strict=False`, instead of silently changing class meanings. EMA remains tensor
state throughout. Changing the split policy or enabling conditioning requires
a new compatible model/run; it is not a drop-in resume of a rule-free model.

Direct PyTorch inference and `torch.compile` receive the same labeled data
dictionary as training. The shared model performance tools generate valid
synthetic rule labels automatically. Current engine export `ModelIOv1` has no
rule input and explicitly rejects conditioned models; ONNX/TorchScript engine
export needs a separately designed I/O contract. Interactive callers must also
provide `rule_index` before using a conditioned model.

When mixing dense raw and processed NPZ sources, use `batched_katago_numpy`
and `batched_processed_katago_numpy` children with `data_pipeline: {}` to share
the adaptive batch runtime and bounded decoded RAM caches. Keep `dataloader_args.batch_by_boardsize: true` for mixed board
sizes. Balanced mixing assigns equal source shares, so multiple sources with
the same rule contribute multiple shares; side splitting does not rebalance
samples. Validation should use held-out directories and its own mixing mode.
