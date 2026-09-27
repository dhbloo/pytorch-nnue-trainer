import difflib
import math
from fractions import Fraction
from functools import reduce
from math import gcd, lcm
from torch.utils.data.dataset import Dataset, IterableDataset
from utils.file_utils import make_file_list
from utils.misc_utils import Registry, import_submodules
from .core import DatasetRuntimeContext
from .pipeline import BasePipeline, build_data_pipeline, warp_dataset_with_pipeline
from .source_dataset import PlannedBatchDataset
from .mixing import MixingConfig

DATASETS = Registry("dataset")
import_submodules(__name__, recursive=False)


def _read_multi_dataset(
    file_list: list[str],
    dataset_dict: dict[str, dict],
    *,
    runtime_context: DatasetRuntimeContext | None = None,
    allow_blend_ratio: bool = False,
    budgeted_npz_alias: bool = False,
    wrap_map_evaluation: bool = True,
    **kwargs,
):
    datasets = []
    blend_ratios = []
    for dataset_name, dataset_args in dataset_dict.items():
        dataset_args = dict(dataset_args)  # never mutate the caller's config dict
        dataset_type = dataset_args.pop("dataset_type")
        if budgeted_npz_alias and dataset_type == "iterative_processed_katago_numpy":
            dataset_type = "batched_processed_katago_numpy"
        if dataset_type not in DATASETS:
            raise ValueError(
                f"invalid dataset type in {dataset_name}: {dataset_type}"
            )
        dataset_cls = DATASETS[dataset_type]

        data_paths = dataset_args.pop("data_paths", None)
        if data_paths is None:
            data_paths = file_list
        elif isinstance(data_paths, str):
            data_paths = [data_paths]
        else:
            if not isinstance(data_paths, list):
                raise TypeError(
                    f"data_paths in {dataset_name} must be a list of strings"
                )

        if "blend_ratio" in dataset_args and not allow_blend_ratio:
            raise ValueError("blend_ratio is only valid with mixing.mode: weighted")
        ratio = dataset_args.pop("blend_ratio", 1.0)
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
            raise TypeError("blend_ratio must be a positive number")
        blend_ratio = float(ratio)
        blend_ratios.append(blend_ratio)

        # Per-child values take precedence, but run the same public option
        # validation and migration path as a top-level dataset.
        resolved_args = {**kwargs, **dataset_args}
        if isinstance(resolved_args.get("rules"), (set, frozenset)):
            resolved_args["rules"] = sorted(resolved_args["rules"])
        if isinstance(resolved_args.get("boardsizes"), (set, frozenset)):
            resolved_args["boardsizes"] = sorted(resolved_args["boardsizes"])
        if runtime_context is not None:
            dataset = build_dataset(
                dataset_type,
                data_paths,
                runtime_context=runtime_context,
                wrap_map_evaluation=wrap_map_evaluation,
                **resolved_args,
            )
        else:
            flist = make_file_list(data_paths, dataset_cls.FILE_EXTS)
            dataset = dataset_cls(file_list=flist, **resolved_args)
        datasets.append(dataset)
    
    def emits_rule_index(dataset):
        # Pipeline and evaluation wrappers preserve the underlying fields.
        wrapped = getattr(dataset, "dataset", None)
        if wrapped is not None:
            return emits_rule_index(wrapped)
        children = getattr(dataset, "datasets", None)
        if children is not None:
            return all(emits_rule_index(child) for child in children)
        if isinstance(dataset, (DATASETS["simple_binary"], DATASETS["packed_binary"])):
            return not dataset.drop_extra
        return getattr(dataset, "rule_index", None) is not None

    labelled = [emits_rule_index(dataset) for dataset in datasets]
    if any(labelled) and not all(labelled):
        raise ValueError(
            "mixed datasets must either all provide rule_index or all omit it; "
            "set rule on every NPZ child when using rule annotations"
        )
    return datasets, blend_ratios


@DATASETS.register("multi")
class MultiDataset(Dataset):
    """MultiDataset combines all (random accessable) datasets into one dataset."""

    def __init__(
        self,
        file_list: list[str],
        dataset_dict: dict[str, dict],
        rules: set[str],
        boardsizes: set[tuple[int, int]],
        fixed_side_input: bool = False,
        fixed_board_size: tuple[int, int] | None = None,
        apply_symmetry: bool | str = False,
        shuffle: bool = False,
        runtime_context: DatasetRuntimeContext | None = None,
    ) -> None:
        super().__init__()
        self.fixed_side_input = fixed_side_input
        self.datasets, _ = _read_multi_dataset(
            file_list,
            dataset_dict,
            rules=rules,
            boardsizes=boardsizes,
            fixed_side_input=fixed_side_input,
            fixed_board_size=fixed_board_size,
            apply_symmetry=apply_symmetry,
            shuffle=shuffle,
            runtime_context=runtime_context,
            wrap_map_evaluation=False,
        )
        # check if all datasets have __len__ and __getitem__ method
        if not all(
            hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__")
            for dataset in self.datasets
        ):
            raise TypeError("all multi children must be random-access datasets")

    @property
    def is_fixed_side_input(self):
        return self.fixed_side_input

    def close(self):
        runtime = getattr(self, "_adaptive_pipeline_runtime", None)
        if runtime is not None:
            runtime.close()
            self._adaptive_pipeline_runtime = None
        for dataset in self.datasets:
            close = getattr(dataset, "close", None)
            if close is not None:
                close()

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, index):
        for dataset in self.datasets:
            if index < len(dataset):
                return dataset[index]
            index -= len(dataset)
        raise IndexError("Index out of range")

    def map_record_ref(self, index):
        from .stream import MapRecordRef

        original_index = int(index)
        for child, dataset in enumerate(self.datasets):
            if index < len(dataset):
                ref = dataset.map_record_ref(index)
                return MapRecordRef(
                    f"{type(self).__name__}:{child}:{ref.dataset_id}",
                    original_index,
                    (
                        "map-child",
                        ref.sample_key[1],
                        ref.sample_key[2],
                        (child, *ref.sample_key[3]),
                    ),
                    ref.output_shape,
                )
            index -= len(dataset)
        raise IndexError("Index out of range")

    def map_output_shape(self, index):
        for dataset in self.datasets:
            if index < len(dataset):
                shape_of = getattr(dataset, "map_output_shape", None)
                return (
                    shape_of(index)
                    if shape_of is not None
                    else dataset.map_record_ref(index).output_shape
                )
            index -= len(dataset)
        raise IndexError("Index out of range")

    def map_sample_key(self, index):
        for child, dataset in enumerate(self.datasets):
            if index < len(dataset):
                key_of = getattr(dataset, "map_sample_key", None)
                key = (
                    key_of(index)
                    if key_of is not None
                    else dataset.map_record_ref(index).sample_key
                )
                return (
                    "map-child",
                    key[1],
                    key[2],
                    (child, *key[3]),
                )
            index -= len(dataset)
        raise IndexError("Index out of range")


from .mixed_runtime import AdaptiveMultiMixin


@DATASETS.register("iterative_multi")
class MultiIterativeDataset(AdaptiveMultiMixin, PlannedBatchDataset):
    """MultiDataset combines all iterative datasets into one dataset."""

    def __init__(
        self,
        file_list: list[str],
        dataset_dict: dict[str, dict],
        rules: set[str],
        boardsizes: set[tuple[int, int]],
        fixed_side_input: bool = False,
        fixed_board_size: tuple[int, int] | None = None,
        apply_symmetry: bool | str = False,
        sample_rate: float = 1.0,
        shuffle: bool = False,
        mixing: dict | None = None,
        batch_size: int | None = None,
        batch_pipelines=(),
        shuffle_window_size: int = 32768,
        shuffle_buffer_bytes: int | None = None,
        steps_per_epoch: int | None = None,
        runtime_context: DatasetRuntimeContext | None = None,
        adaptive_pipeline=None,
        resume_execution_hint=None,
        quota_child_prefetch: bool = False,
    ) -> None:
        super().__init__()
        self.batch_pipelines = tuple(batch_pipelines)
        self.fixed_side_input = fixed_side_input
        self.shuffle = shuffle
        self.sample_rate = sample_rate
        self.shuffle_window_size = shuffle_window_size
        self.shuffle_buffer_bytes = shuffle_buffer_bytes
        self.steps_per_epoch = steps_per_epoch
        self.mixing = MixingConfig.parse(mixing)
        self.runtime_context = runtime_context
        self.resume_execution_hint = resume_execution_hint
        if type(quota_child_prefetch) is not bool:
            raise TypeError("quota_child_prefetch must be a boolean")
        self.quota_child_prefetch = quota_child_prefetch
        self.child_ids = tuple(dataset_dict)
        child_runtime = None
        if runtime_context is not None:
            from .core import single_process_dataset_context

            child_runtime = single_process_dataset_context(
                1,
                seed=runtime_context.seed,
                mode=runtime_context.mode,
            )
        self.datasets, self.blend_ratios = _read_multi_dataset(
            file_list,
            dataset_dict,
            rules=rules,
            boardsizes=boardsizes,
            fixed_side_input=fixed_side_input,
            fixed_board_size=fixed_board_size,
            apply_symmetry=apply_symmetry,
            sample_rate=sample_rate,
            shuffle=shuffle,
            runtime_context=child_runtime,
            allow_blend_ratio=self.mixing.mode == "weighted",
            budgeted_npz_alias=(
                adaptive_pipeline is not None
                and not any(
                    args.get("dataset_type") in {"simple_binary", "packed_binary"}
                    for args in dataset_dict.values()
                )
            ),
        )
        if not self.datasets:
            raise ValueError("iterative_multi requires at least one child dataset")
        self._init_adaptive_multi(adaptive_pipeline)
        # check if all datasets have __iter__ method
        if not all(hasattr(dataset, "__iter__") for dataset in self.datasets):
            raise TypeError("all iterative_multi children must be iterable")

    @property
    def is_internal_shuffleable(self):
        return all(map(lambda d: d.is_internal_shuffleable, self.datasets))

    @property
    def capabilities(self):
        from .core import DatasetCapabilities

        children = [
            getattr(dataset, "capabilities", None) for dataset in self.datasets
        ]
        return DatasetCapabilities(
            yields_batches=True,
            resumable=all(
                capability is not None and capability.resumable
                for capability in children
            ),
            deterministic=all(
                capability is not None and capability.deterministic
                for capability in children
            ),
            supports_batch_pipeline=all(
                capability is not None and capability.supports_batch_pipeline
                for capability in children
            ),
        )

    def _integer_ratio_weights(self):
        fractions = []
        for index, ratio in enumerate(self.blend_ratios):
            if not math.isfinite(ratio) or ratio <= 0:
                raise ValueError(
                    f"iterative_multi child {index} blend_ratio must be finite and positive, "
                    f"got {ratio}"
                )
            fraction = Fraction(str(ratio))
            if fraction.denominator > 1_000_000:
                raise ValueError(
                    f"iterative_multi child {index} blend_ratio denominator "
                    f"{fraction.denominator} exceeds 1000000"
                )
            fractions.append(fraction)
        denominator = reduce(lcm, (value.denominator for value in fractions), 1)
        weights = [
            value.numerator * (denominator // value.denominator)
            for value in fractions
        ]
        divisor = reduce(gcd, weights)
        return tuple(weight // divisor for weight in weights)

    def _build_partitioned_stream(self):
        runtime_context = getattr(self, "runtime_context", None)
        if runtime_context is None:
            raise RuntimeError("iterative_multi requires a DatasetRuntimeContext")
        from .composite_source import CompositeRecordSource
        from .core import PipelineStateComposer
        from .planner import DatasetPlanner, PlannerConfig
        from .execution import build_execution, resolve_execution
        from .stream import reject_duplicate_physical_files

        structural_defaults = {
            "fixed_side_input": False,
            "fixed_board_size": None,
            "has_pass_move": False,
            "drop_extra": False,
            "board_input_channels": None,
            "stm_input_channel": None,
            "value_target_channels": None,
        }
        structural_states = []
        for dataset in self.datasets:
            extra = getattr(dataset, "extra_kwargs", {})
            structural_states.append(
                {
                    name: getattr(dataset, name, extra.get(name, default))
                    for name, default in structural_defaults.items()
                }
            )
        for name in structural_defaults:
            values = [
                state[name]
                for dataset, state in zip(self.datasets, structural_states)
                if name != "drop_extra"
                or hasattr(dataset, name)
                or name in getattr(dataset, "extra_kwargs", {})
            ]
            if not values:
                continue
            if any(value != values[0] for value in values[1:]):
                raise ValueError(
                    f"iterative_multi children have incompatible structural "
                    f"option {name!r}: {values}"
                )
        child_paths = [
            path
            for dataset in self.datasets
            for path in getattr(dataset, "file_list", ())
        ]
        if child_paths:
            reject_duplicate_physical_files(child_paths)
        self._configure_adaptive_children()
        child_sources = []
        for child, dataset in enumerate(self.datasets):
            if not hasattr(dataset, "_build_partitioned_stream"):
                raise TypeError(
                    f"iterative_multi child {child} does not expose a record planner"
                )
            planner = dataset._build_partitioned_stream()
            source = getattr(dataset, "_record_source", None)
            try:
                if planner is None or source is None or planner.source is not source:
                    raise TypeError(
                        f"iterative_multi child {self.child_ids[child]!r} does not "
                        "expose a native record source"
                    )
                if planner.pipeline_composer is not None:
                    raise ValueError(
                        f"iterative_multi child {self.child_ids[child]!r} declares "
                        "batch_pipelines; pipelines must be configured on the composite"
                    )
                child_sources.append(source)
            finally:
                if planner is not None and hasattr(planner, "close"):
                    planner.close()
        weights = self._integer_ratio_weights()
        from .packed_composite import PackedCompositeRecordSource
        composer = (
            PipelineStateComposer(self.batch_pipelines)
            if self.batch_pipelines
            else None
        )
        self.execution_decision = resolve_execution(
            child_sources,
            requested_policy=self.adaptive_pipeline,
            pipeline_composer=composer,
            resume_hint=self.resume_execution_hint,
            observational_generic=self._observed_composite,
        )
        source_cls = (
            PackedCompositeRecordSource
            if self.execution_decision.backend == "packed"
            else CompositeRecordSource
        )
        if self.quota_child_prefetch and (
            self.adaptive_pipeline is None
            or runtime_context.mode != "train"
            or self.mixing.mode != "natural"
            or source_cls is not PackedCompositeRecordSource
            or not PackedCompositeRecordSource.supports(child_sources)
        ):
            raise ValueError(
                "quota_child_prefetch requires a train-mode adaptive packed "
                "natural mixture of uniform dense NPZ children"
            )
        self._record_source = source_cls(
            child_sources,
            self.child_ids,
            weights,
            mixing=self.mixing,
            seed=runtime_context.seed,
            **({"quota_child_prefetch": True} if self.quota_child_prefetch else {}),
        )
        self._partitioned_stream = DatasetPlanner(
            self._record_source,
            runtime_context,
            PlannerConfig(
                shuffle=self.shuffle,
                shuffle_buffer_size=self.shuffle_window_size,
                shuffle_buffer_bytes=self.shuffle_buffer_bytes,
                steps_per_epoch=self.steps_per_epoch,
            ),
            pipeline_composer=composer,
        )
        if self.adaptive_pipeline is None:
            self._planned_decoder = build_execution(
                self._partitioned_stream,
                self._record_source,
                decision=self.execution_decision,
            )
        try:
            self._install_adaptive_multi()
        except BaseException:
            self.close()
            raise
        self.composite_audit = {
            "mixing": self._record_source.manifest_state()["mixing"],
            "quotas": self._record_source.quotas,
            "weights": weights,
            "schedule": self._record_source.schedule,
            "capabilities": self._record_source.capabilities,
            "child_manifests": tuple(
                source.manifest_state() for source in child_sources
            ),
        }
        return self._partitioned_stream

def build_dataset(
    dataset_type: str,
    data_paths: list[str],
    *,
    runtime_context: DatasetRuntimeContext,
    rules: None | list[str] = None,
    boardsizes: None | int | tuple[int, int] | list[tuple[int, int]] = None,
    fixed_side_input: bool = False,
    fixed_board_size: None | int | tuple[int, int] = None,
    shuffle: bool=False,
    pipeline_args: None | dict = None,
    adaptive_pipeline=None,
    resume_execution_hint=None,
    wrap_map_evaluation: bool = True,
    **kwargs,
) -> Dataset | IterableDataset:
    if not isinstance(runtime_context, DatasetRuntimeContext):
        raise TypeError("build_dataset requires a DatasetRuntimeContext")
    if dataset_type not in DATASETS:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    dataset_cls = DATASETS[dataset_type]
    observed_map = adaptive_pipeline is not None and dataset_type in {
        "katago_numpy", "processed_katago_numpy", "multi"
    }
    if adaptive_pipeline is not None and dataset_type == "iterative_processed_katago_numpy":
        # Both public names construct the same dense NPZ source and planner.
        # The batched adapter adds budgeted cache/prefetch ownership.
        dataset_cls = DATASETS["batched_processed_katago_numpy"]
    if (
        adaptive_pipeline is not None
        and dataset_type not in {
            "batched_processed_katago_numpy", "batched_katago_numpy",
            "iterative_processed_katago_numpy", "iterative_katago_numpy",
            "iterative_sparse_numpy", "sparse_numpy", "iterative_multi",
            "simple_binary", "packed_binary",
            "katago_numpy", "processed_katago_numpy", "multi",
        }
    ):
        raise ValueError(
            "adaptive_pipeline requires a supported built-in dataset"
        )
    explicit_shuffle_window_size = "shuffle_window_size" in kwargs
    explicit_pin_memory = "pin_memory" in kwargs
    if adaptive_pipeline is not None:
        legacy_pipeline_options = {
            "prefetch_threads",
            "prefetch_batches",
            "pin_memory",
        }
        conflicts = sorted(legacy_pipeline_options.intersection(kwargs))
        if conflicts:
            raise ValueError(
                "adaptive_pipeline cannot be combined with legacy option(s): "
                + ", ".join(conflicts)
            )

    common_keys = {
        "rules", "boardsizes", "fixed_side_input", "fixed_board_size",
        "apply_symmetry", "sample_rate", "shuffle",
    }
    format_keys = {
        "katago_numpy": {
            "rule",
            "has_pass_move", "filter_stm", "filter_condition", "value_td_level",
        },
        "iterative_katago_numpy": {
            "rule",
            "has_pass_move", "filter_stm", "filter_condition", "value_td_level",
            "shuffle_window_size", "shuffle_buffer_bytes", "steps_per_epoch",
        },
        "processed_katago_numpy": {
            "rule",
            "has_pass_move", "filter_stm", "filter_condition",
            "board_input_channels", "stm_input_channel", "value_target_channels",
        },
        "iterative_processed_katago_numpy": {
            "rule",
            "has_pass_move", "filter_stm", "filter_condition",
            "board_input_channels", "stm_input_channel", "value_target_channels",
            "shuffle_window_size", "shuffle_buffer_bytes", "steps_per_epoch",
        },
        "batched_katago_numpy": {
            "rule", "has_pass_move", "value_td_level",
            "prefetch_threads", "prefetch_batches", "pin_memory", "observability",
            "shuffle_window_size", "shuffle_buffer_bytes", "steps_per_epoch",
        },
        "batched_processed_katago_numpy": {
            "rule",
            "has_pass_move", "filter_stm", "filter_condition",
            "board_input_channels", "stm_input_channel", "value_target_channels",
            "prefetch_threads", "prefetch_batches", "pin_memory",
            "observability",
            "shuffle_window_size", "shuffle_buffer_bytes", "steps_per_epoch",
        },
        "iterative_sparse_numpy": {
            "drop_extra", "shuffle_window_size", "shuffle_buffer_bytes",
            "steps_per_epoch",
        },
        "sparse_numpy": {
            "drop_extra", "shuffle_window_size", "shuffle_buffer_bytes",
            "steps_per_epoch",
        },
        "simple_binary": {
            "drop_extra", "shuffle_window_size", "shuffle_buffer_bytes",
            "sequential_active_streams", "sequential_read_quantum",
            "steps_per_epoch",
        },
        "packed_binary": {
            "drop_extra", "has_pass_move", "value_td_lambda",
            "dynamic_value_lambda", "multipv_temperature", "use_mate_multipv",
            "winrate_model_args",
            "shuffle_window_size", "shuffle_buffer_bytes",
            "sequential_active_streams", "sequential_read_quantum",
            "steps_per_epoch",
        },
        "multi": {"dataset_dict"},
        "iterative_multi": {
            "dataset_dict", "mixing", "shuffle_window_size",
            "shuffle_buffer_bytes", "steps_per_epoch", "quota_child_prefetch",
        },
    }
    accepted = common_keys | format_keys.get(dataset_type, set())
    if dataset_type in {"katago_numpy", "processed_katago_numpy", "multi"} and "sample_rate" in kwargs:
        raise ValueError(
            f"dataset {dataset_type!r} rejects sample_rate because map-style sampling "
            "is sampler-owned"
        )
    unknown = sorted(set(kwargs) - accepted) if dataset_type in format_keys else []
    if "max_partition_per_file" in unknown:
        raise ValueError(
            f"dataset {dataset_type!r} option 'max_partition_per_file' was removed; "
            "the global stream planner now owns partitioning"
        )
    if unknown:
        suggestions = {
            key: difflib.get_close_matches(key, sorted(accepted), n=1)
            for key in unknown
        }
        detail = ", ".join(
            f"{key!r}" + (f" (did you mean {matches[0]!r}?)" if matches else "")
            for key, matches in suggestions.items()
        )
        raise ValueError(
            f"dataset {dataset_type!r} has unknown option(s): {detail}; "
            f"valid options are {', '.join(sorted(accepted))}"
        )

    if dataset_cls == MultiDataset or dataset_cls == MultiIterativeDataset:
        file_list = data_paths
    else:
        file_list = make_file_list(data_paths, dataset_cls.FILE_EXTS)

    rules = rules or ["freestyle", "standard", "renju"]
    if boardsizes is None:
        boardsizes = [(s, s) for s in range(9, 22)]
    elif isinstance(boardsizes, int):
        boardsizes = [(boardsizes, boardsizes)]
    elif isinstance(boardsizes, tuple):
        boardsizes = [boardsizes]
    if not isinstance(boardsizes, list):
        raise TypeError(
            f"boardsizes must be a list of tuples, got {boardsizes!r}"
        )

    if isinstance(fixed_board_size, int):
        fixed_board_size = (fixed_board_size, fixed_board_size)
    if fixed_board_size is not None and not isinstance(
        fixed_board_size, tuple
    ):
        raise TypeError(
            "fixed_board_size must be None, an integer, or a tuple"
        )

    if getattr(dataset_cls, "YIELDS_BATCHES", False):
        # batch-yielding datasets collate internally and need the batch size
        kwargs["batch_size"] = runtime_context.local_batch_size
        if pipeline_args is not None:
            batch_pipelines = build_data_pipeline(pipeline_args)
            unsupported = [
                type(p).__name__
                for p in batch_pipelines
                if not hasattr(p, "process_batch")
                or not getattr(p, "pipeline_id", "")
                or not getattr(p, "input_fields", ())
                or not getattr(p, "output_fields", ())
                or type(p).signature_state is BasePipeline.signature_state
            ]
            if unsupported:
                raise ValueError(
                    "pipelines do not implement the deterministic batch descriptor: "
                    + ", ".join(unsupported)
                )
            kwargs["batch_pipelines"] = batch_pipelines
            pipeline_args = None
    map_runtime = None
    if adaptive_pipeline is not None:
        from .pipeline_runtime import AdaptivePipelineRuntimeSpec

        if not isinstance(adaptive_pipeline, AdaptivePipelineRuntimeSpec):
            raise TypeError(
                "adaptive_pipeline must be AdaptivePipelineRuntimeSpec or null"
            )
        if observed_map:
            from .execution import ObservedExecutionRuntime

            map_runtime = ObservedExecutionRuntime(adaptive_pipeline)
        else:
            kwargs["adaptive_pipeline"] = adaptive_pipeline
    if resume_execution_hint is not None:
        if dataset_cls is not MultiIterativeDataset:
            raise ValueError("resume execution hint requires iterative_multi")
        kwargs["resume_execution_hint"] = resume_execution_hint
    if dataset_cls in {MultiDataset, MultiIterativeDataset}:
        kwargs["runtime_context"] = runtime_context

    dataset = dataset_cls(
        file_list=file_list,
        rules=set(rules),
        boardsizes=set(boardsizes),
        fixed_side_input=fixed_side_input,
        fixed_board_size=fixed_board_size,
        shuffle=shuffle,
        **kwargs,
    )
    dataset.runtime_context = runtime_context
    dataset._explicit_shuffle_window_size = explicit_shuffle_window_size
    dataset._explicit_pin_memory = explicit_pin_memory

    if pipeline_args is not None:
        dataset = warp_dataset_with_pipeline(dataset, pipeline_args)

    if observed_map:
        from .execution import attach_observed_map_execution

        dataset = attach_observed_map_execution(
            dataset, adaptive_pipeline, map_runtime
        )

    if (
        wrap_map_evaluation
        and runtime_context.mode in {"validate", "test"}
        and isinstance(dataset, Dataset)
        and not isinstance(dataset, IterableDataset)
    ):
        from .stream import EvaluationBatchPlannerDataset

        dataset = EvaluationBatchPlannerDataset(dataset, runtime_context)

    return dataset
