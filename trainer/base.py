"""Base trainer: accelerator setup, data, model, optimizer, checkpoint, logging, and training loop."""

import torch
import numpy as np
import random
import numbers
from torch.utils.data import IterableDataset
from accelerate import (
    Accelerator,
    DataLoaderConfiguration,
    DistributedDataParallelKwargs,
)
from accelerate.data_loader import BatchSamplerShard
from accelerate.utils import (
    DynamoBackend,
    GradientAccumulationPlugin,
    gather_object,
    send_to_device,
)
from accelerate.utils.other import is_compiled_module
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass
import inspect
import json
import hashlib
import time
import os
import re
import stat

from tqdm.auto import tqdm

from dataset import build_dataset
from dataset.core import (
    DatasetRuntimeContext,
    Maximum,
    RNG_ALGORITHM,
    SufficientStats,
    SumCount,
    canonical_pipeline_state_bytes,
)
from model import build_model
from model.vq import (
    VectorQuantize,
    clear_ddp_preinit_gradients,
)
from trainer.profiler import NULL_PROFILER, build_profiler
from utils.compile_utils import model_inductor_config, with_inductor_options
from utils.cuda_utils import configure_cuda_memory_limit
from utils.training_utils import (
    build_lr_scheduler,
    build_optimizer,
    build_data_loader,
    resolve_weight_clipping,
    apply_weight_clipping,
    state_dict_drop_size_unmatched,
    DeviceLoaderWrapper,
    ObservedDeviceLoaderWrapper,
    CudaPrefetchLoaderWrapper,
    ResumableSampler,
)
from utils.misc_utils import (
    seed_everything,
    set_performance_level,
    add_dict_to,
    log_value_dict,
    format_time,
)
from utils.file_utils import (
    make_dir,
    save_torch_ckpt,
    load_torch_ckpt,
    find_latest_ckpt,
    get_iteration_from_ckpt_filename,
)


def _devices_match(actual, expected):
    """Compare devices using PyTorch's current-device semantics.

    An unindexed CUDA device (``cuda``) means the process's current CUDA
    device, while tensors always report an explicit index (for example,
    ``cuda:0``).  CPU indices are not meaningful for placement checks.
    """
    actual = torch.device(actual)
    expected = torch.device(expected)
    if actual.type != expected.type:
        return False
    if actual.type == "cpu":
        return True
    if actual.type == "cuda":
        current_index = None

        def resolved_index(device):
            nonlocal current_index
            if device.index is not None:
                return device.index
            if current_index is None:
                current_index = torch.cuda.current_device()
            return current_index

        return resolved_index(actual) == resolved_index(expected)
    if actual.index is None or expected.index is None:
        return True
    return actual.index == expected.index


_RUNTIME_SIDECAR_RE = re.compile(
    r"runtime_(?P<iteration>(?:[0-9]{7}|[1-9][0-9]{7,}))_"
    r"(?P<generation>[0-9a-f]{32})"
    r"_rank_(?P<rank>(?:0|[1-9][0-9]*))\.pt"
)
_RUNTIME_TEMP_RE = re.compile(
    r"\.runtime-tmp_(?P<iteration>(?:[0-9]{7}|[1-9][0-9]{7,}))_"
    r"(?P<generation>[0-9a-f]{32})"
    r"_rank_(?P<rank>(?:0|[1-9][0-9]*))_"
    r"(?P<pid>(?:0|[1-9][0-9]*))_(?P<nonce>[0-9a-f]{32})\.pt"
)
_RUNTIME_SIDECAR_VERSION = 1
_RUNTIME_MANIFEST_VERSION = 1
_RUNTIME_SERIALIZATION_ALLOWANCE = 1024 * 1024
_RUNTIME_DTYPE_BYTES = {
    "torch.float16": 2,
    "torch.bfloat16": 2,
    "torch.float32": 4,
    "torch.float64": 8,
}
_RUNTIME_AUTOCAST_DTYPES = {
    "bf16": "torch.bfloat16",
    "fp16": "torch.float16",
}


def _runtime_serialization_allowance(raw_bytes):
    return max(
        _RUNTIME_SERIALIZATION_ALLOWANCE,
        (raw_bytes + 99) // 100,
    )


def _expected_vq_runtime_dtype(descriptor, count, autocast_mode):
    if autocast_mode not in {"no", *_RUNTIME_AUTOCAST_DTYPES}:
        raise RuntimeError(
            f"unsupported VQ autocast mode {autocast_mode!r}"
        )
    if descriptor["convert_to_fp32"]:
        return "torch.float32"
    if count == 0 or autocast_mode == "no":
        return descriptor["parameter_dtype"]
    return _RUNTIME_AUTOCAST_DTYPES[autocast_mode]


_LEGACY_MAP_CONTENT_ALLOWLIST = {
    # Exact CPU fixture generated from repository 33fb246. Legacy map
    # checkpoints did not store content hashes, so compatibility is limited to
    # this independently hashed artifact.
    "4ac501466c59a784e13f8fcbf1882322d4a3ee1211c456604689a6daeafac721",
}


@dataclass
class TrainingState:
    """Mutable counters that survive checkpoint round-trips."""

    iteration: int = 0
    epoch: int = 0
    rows: int = 0


class BaseTrainer:
    """Base class for training with Accelerate.

    Handles accelerator setup, dataset/dataloader construction, model and optimizer
    creation, checkpoint save/load, TensorBoard + JSONL logging, and the main training
    loop.  Subclasses must implement :meth:`train_step` and :meth:`validate_step` to
    supply the forward/backward logic.

    All parameters are keyword-only.

    Args:
        # Run
        rundir: Directory for checkpoints, logs, and TensorBoard events.
        iterations: Total number of training iterations (optimizer steps).
        batch_size: Global batch size (split across processes).
        gradient_accumulation_steps: Number of micro-batches accumulated per
            optimizer step.  Each iteration consumes this many batches, so the
            effective batch size is ``batch_size * gradient_accumulation_steps``.
        random_seed: Seed for all random number generators.
        performance_level: Torch performance level (0 = safe, 2 = fast).
        max_memory_fraction: Optional per-process CUDA allocator ceiling in
            ``(0, 1]``. Applied before datasets and models allocate tensors.
        use_cpu: Force training on CPU even if CUDA is available.
        profiler_args: Profiler configuration, or ``None`` to disable (zero
            overhead).  Keys: ``timing`` (log per-region iteration times at
            every ``log_interval``), ``trace_at`` (iterations at which to
            record a torch.profiler trace window into
            ``rundir/profile_trace``), ``trace_iters`` (active iterations per
            trace window, default 30).
        # Data
        dataset_type: Registered dataset class name (passed to ``build_dataset``).
        train_datas: Paths to training data files.
        val_datas: Paths to validation data files, or ``None`` to skip validation.
        dataset_args: Extra keyword arguments forwarded to the dataset constructor.
        val_dataset_type: Separate dataset class for validation, or ``None`` to reuse
            ``dataset_type``.
        val_dataset_args: Extra keyword arguments for the validation dataset, or
            ``None`` to reuse ``dataset_args``.
        dataloader_args: Extra keyword arguments forwarded to ``build_data_loader``.
        data_pipelines: Optional list of pipeline configurations for data transforms.
        num_worker: Number of dataloader worker processes.
        cuda_prefetch_batches: Bounded training-batch lookahead on a dedicated
            CUDA H2D stream. ``0`` disables it; ``1`` is double buffering.
        no_shuffle: Disable training data shuffling.
        eval_bs_multipler: Multiplier applied to ``batch_size`` for validation.
        # Model
        model_type: Registered model class name (passed to ``build_model``).
        model_args: Extra keyword arguments forwarded to the model constructor.
        load_from: Path to a pretrained checkpoint to load before training, or
            ``None`` to skip.
        # Optimizer
        optim_type: Optimizer name (e.g. ``"adamw"``).
        optim_args: Extra keyword arguments forwarded to the optimizer constructor.
        learning_rate: Initial learning rate.
        weight_decay: Weight decay coefficient.
        # LR scheduler
        lr_scheduler_type: LR scheduler name (e.g. ``"constant"``, ``"cosine"``).
        lr_scheduler_args: Extra keyword arguments forwarded to the scheduler.
        # Gradient clipping
        clip_grad_norm: Max gradient norm for ``clip_grad_norm_``, or ``None``.
        clip_grad_value: Max gradient value for ``clip_grad_value_``, or ``None``.
        # Intervals
        log_interval: Iterations between TensorBoard / JSONL log writes.
        show_interval: Iterations between stdout progress prints.
        save_interval: Iterations between permanent checkpoints.
        val_interval: Iterations between validation runs.
        temp_save_interval: Iterations between temporary (rolling) checkpoints.
        state_save_interval: Iterations between training-state saves, or ``None``
            to save state at every checkpoint save.  State is only ever written
            at iterations where model checkpoints are also saved, so that a
            state file always has matching model files to resume from.
        num_keep_states: Number of most recent training-state files to keep,
            or ``-1`` to keep all.
        # Distributed
        find_unused_parameters: Enable ``find_unused_parameters`` in DDP wrapper.
    """

    def __init__(
        self,
        *,
        # Run
        rundir: str,
        iterations: int = 1_000_000,
        batch_size: int = 128,
        gradient_accumulation_steps: int = 1,
        random_seed: int = 42,
        performance_level: int = 2,
        max_memory_fraction: float | None = None,
        use_cpu: bool = False,
        profiler_args: dict | None = None,
        # Data
        dataset_type: str,
        train_datas: list[str],
        val_datas: list[str] | None = None,
        dataset_args: dict | None = None,
        val_dataset_type: str | None = None,
        val_dataset_args: dict | None = None,
        dataloader_args: dict | None = None,
        data_pipelines: list | None = None,
        num_worker: int = 4,
        cuda_prefetch_batches: int = 0,
        no_shuffle: bool = False,
        eval_bs_multipler: int = 1,
        # Model
        model_type: str,
        model_args: dict | None = None,
        load_from: str | None = None,
        # Optimizer
        optim_type: str = "adamw",
        optim_args: dict | None = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
        # LR scheduler
        lr_scheduler_type: str = "constant",
        lr_scheduler_args: dict | None = None,
        # Gradient clipping
        clip_grad_norm: float | None = None,
        clip_grad_value: float | None = None,
        # Intervals
        log_interval: int = 500,
        show_interval: int = 1000,
        save_interval: int = 100_000,
        val_interval: int = 50_000,
        temp_save_interval: int = 5000,
        state_save_interval: int | None = None,
        num_keep_states: int = 1,
        # Distributed
        find_unused_parameters: bool = False,
    ):
        # Store all parameters
        self.rundir = rundir
        self.iterations = iterations
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.random_seed = random_seed
        self.performance_level = performance_level
        self.max_memory_fraction = max_memory_fraction
        self.use_cpu = use_cpu
        self.profiler_args = profiler_args or {}

        self.dataset_type = dataset_type
        self.train_datas = train_datas
        self.val_datas = val_datas
        self.dataset_args = dataset_args or {}
        self.val_dataset_type = val_dataset_type
        self.val_dataset_args = val_dataset_args or {}
        self.dataloader_args = dataloader_args or {}
        self.data_pipelines = data_pipelines
        self.num_worker = num_worker
        if (
            type(cuda_prefetch_batches) is not int
            or not 0 <= cuda_prefetch_batches
            <= CudaPrefetchLoaderWrapper.MAX_PREFETCH_BATCHES
        ):
            raise ValueError("cuda_prefetch_batches must be an integer in [0, 4]")
        self.cuda_prefetch_batches = cuda_prefetch_batches
        self.no_shuffle = no_shuffle
        self.eval_bs_multipler = eval_bs_multipler

        self.model_type = model_type
        self.model_args = model_args or {}
        self.load_from = load_from

        self.optim_type = optim_type
        self.optim_args = optim_args or {}
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.lr_scheduler_type = lr_scheduler_type
        self.lr_scheduler_args = lr_scheduler_args or {}

        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_value = clip_grad_value

        self.log_interval = log_interval
        self.show_interval = show_interval
        self.save_interval = save_interval
        self.val_interval = val_interval
        self.temp_save_interval = temp_save_interval
        self.state_save_interval = state_save_interval
        self.num_keep_states = num_keep_states

        self.find_unused_parameters = find_unused_parameters

        self.state = TrainingState()
        self._pending_scheduler_states = None
        self._pending_runtime_lrs = None
        self._pending_resume_state = None
        self._pending_resume_version = None
        self._resume_states_for_save = None
        self._last_state_save_iteration = None
        self._checkpoint_attempt_failed = False
        self._checkpoint_safe = True
        # Iterations of temporary (non-permanent) snapshots saved by this session;
        # cleanup only ever deletes snapshots recorded here.
        self._temp_snapshot_iters = set()
        self._setup_accelerator()
        if (
            self.cuda_prefetch_batches > 0
            and self.accelerator.device.type != "cuda"
        ):
            raise ValueError("cuda_prefetch_batches requires CUDA training")
        self._setup_cuda_memory_limit()
        self._setup_logging()
        self._setup_profiler()
        self._setup_data()
        self._init_models()
        self._setup_optimizer()
        self._optimizer_config = self._current_optimizer_config()
        self._load_checkpoint()
        self._setup_scheduler()
        self._prepare_for_training()

    # ── Alternate constructor for evaluation ───────────────────────

    @classmethod
    def init_for_evaluation(
        cls,
        *,
        checkpoint: str,
        model_type: str,
        model_args: dict | None = None,
        test_datas: list[str],
        dataset_type: str,
        dataset_args: dict | None = None,
        dataloader_args: dict | None = None,
        batch_size: int = 128,
        eval_bs_multipler: int = 1,
        num_worker: int = 4,
        use_cpu: bool = False,
        random_seed: int = 42,
        performance_level: int = 0,
        max_memory_fraction: float | None = None,
        find_unused_parameters: bool = False,
        **extra,
    ):
        """Create a trainer instance configured for evaluation only.

        Uses ``object.__new__`` to skip ``__init__``, then runs only the
        setup steps needed for inference: accelerator, test data, model,
        checkpoint loading, and distributed preparation.

        Args:
            checkpoint: Path to the checkpoint file to evaluate.
            model_type: Registered model class name (passed to ``build_model``).
            model_args: Extra keyword arguments forwarded to the model constructor.
            test_datas: Paths to test data files.
            dataset_type: Registered dataset class name (passed to ``build_dataset``).
            dataset_args: Extra keyword arguments forwarded to the dataset constructor.
            dataloader_args: Extra keyword arguments forwarded to ``build_data_loader``.
            batch_size: Global batch size (split across processes).
            eval_bs_multipler: Multiplier applied to ``batch_size`` for evaluation.
            num_worker: Number of dataloader worker processes.
            use_cpu: Force evaluation on CPU even if CUDA is available.
            random_seed: Seed for all random number generators.
            performance_level: Torch performance level (0 = safe, 2 = fast).
            max_memory_fraction: Optional per-process CUDA allocator ceiling.
            find_unused_parameters: Enable ``find_unused_parameters`` in DDP wrapper.
            **extra: Additional keyword arguments forwarded to ``_init_eval_attrs``.
        """
        self = object.__new__(cls)

        self.checkpoint = checkpoint
        self.model_type = model_type
        self.model_args = model_args or {}
        self.test_datas = test_datas
        self.dataset_type = dataset_type
        self.dataset_args = dataset_args or {}
        self.dataloader_args = dataloader_args or {}
        self.batch_size = batch_size
        self.eval_bs_multipler = eval_bs_multipler
        self.num_worker = num_worker
        self.use_cpu = use_cpu
        self.random_seed = random_seed
        self.performance_level = performance_level
        self.max_memory_fraction = max_memory_fraction
        self.find_unused_parameters = find_unused_parameters

        self._apply_init_defaults()
        self._init_eval_attrs(**extra)
        self._setup_accelerator()
        self._setup_cuda_memory_limit()
        self._setup_test_data()
        self._init_models()
        self._load_eval_checkpoint()
        self._prepare_for_evaluation()
        return self

    def _apply_init_defaults(self):
        """Set every defaulted ``__init__`` parameter (across the MRO) as an
        instance attribute, unless already set.

        Relies on the convention that ``__init__`` parameter names equal
        attribute names.  This lets :meth:`init_for_evaluation` skip
        ``__init__`` without each subclass mirroring its defaults in
        :meth:`_init_eval_attrs`; that hook only needs true overrides.
        """
        for klass in type(self).__mro__:
            init = klass.__dict__.get("__init__")
            if init is None:
                continue
            for name, param in inspect.signature(init).parameters.items():
                if param.default is inspect.Parameter.empty:
                    continue
                if not hasattr(self, name):
                    default = param.default
                    # Match __init__'s `x or {}` normalization for dict params
                    if default is None and name.endswith("_args"):
                        default = {}
                    setattr(self, name, default)

    def _init_eval_attrs(self, **extra):
        """Hook for subclasses to store extra attributes before the eval setup chain.

        Defaults for all ``__init__`` parameters are already applied by
        :meth:`_apply_init_defaults`; only set attributes that need
        eval-specific values.
        """

    # ── Setup helpers (called from __init__) ──────────────────────

    def _setup_accelerator(self):
        """Create the Accelerator with DDP config, set random seed and performance level."""
        dataloader_config = DataLoaderConfiguration(dispatch_batches=False, non_blocking=True)
        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=self.find_unused_parameters
        )
        # sync_with_dataloader=False: our micro-batch loop restarts the data
        # iterator across epochs, so accumulation must not sync on dataloader
        # exhaustion or the one-call-one-step invariant of _run_train_step breaks.
        ga_plugin = GradientAccumulationPlugin(
            num_steps=self.gradient_accumulation_steps,
            sync_with_dataloader=False,
        )
        self.accelerator = Accelerator(
            cpu=self.use_cpu,
            dataloader_config=dataloader_config,
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_plugin=ga_plugin,
        )
        seed_everything(self.random_seed)
        set_performance_level(self.performance_level)
        self.profiler = NULL_PROFILER

    def _setup_cuda_memory_limit(self):
        """Apply the configured allocator ceiling before data/model allocation."""
        self.cuda_memory_limit_bytes = configure_cuda_memory_limit(
            self.accelerator.device,
            self.max_memory_fraction,
        )
        if self.cuda_memory_limit_bytes is not None:
            limit_gib = self.cuda_memory_limit_bytes / (1024**3)
            self.accelerator.print(
                f"CUDA allocator limit: {self.max_memory_fraction:.1%} ({limit_gib:.2f} GiB)"
            )

    def _setup_profiler(self):
        """Build the training profiler (a zero-overhead null object when disabled)."""
        self.profiler = build_profiler(
            self.profiler_args,
            rundir=self.rundir,
            is_main_process=self.accelerator.is_main_process,
            use_cpu=self.use_cpu,
        )

    def _setup_logging(self):
        """Open TensorBoard writer and JSONL log file (main process only)."""
        if self.accelerator.is_main_process:
            self.tb_logger = SummaryWriter(os.path.join(self.rundir, "log"))
            self.log_file = open(os.path.join(self.rundir, "training_log.jsonl"), "a")
        else:
            self.tb_logger, self.log_file = None, None

    def _collective_dataset_call(self, phase, callback):
        """Run rank-local dataset I/O and exchange status before later collectives."""
        value = None
        local_error = None
        try:
            value = callback()
        except BaseException as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        errors = (
            [local_error]
            if self.accelerator.num_processes == 1
            else gather_object([local_error])
        )
        failures = [
            f"rank {rank}: {error}"
            for rank, error in enumerate(errors)
            if error is not None
        ]
        if failures:
            raise RuntimeError(
                f"{phase} failed collectively: {'; '.join(failures)}"
            )
        return value

    def _validate_stream_signature_collectively(self, stream, phase):
        if stream is None:
            return
        signature = self._canonical_resume_value(stream.signature_state())
        payload = json.dumps(
            signature, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        digests = (
            [digest]
            if self.accelerator.num_processes == 1
            else gather_object([digest])
        )
        if any(value != digests[0] for value in digests[1:]):
            raise RuntimeError(
                f"{phase} stream signature differs across ranks: "
                + "; ".join(
                    f"rank {rank}={value}" for rank, value in enumerate(digests)
                )
            )

    def _setup_data(self):
        """Build train and validation datasets and their dataloaders."""
        self._set_batch_size_per_process()
        shuffle = not self.no_shuffle
        train_dataset_args = self._training_dataset_args()

        self.train_dataset = self._collective_dataset_call(
            "training dataset setup",
            lambda: build_dataset(
            self.dataset_type,
            self.train_datas,
            runtime_context=DatasetRuntimeContext(
                global_batch_size=self.batch_size,
                local_batch_size=self.batch_size_per_process,
                world_size=self.accelerator.num_processes,
                rank=self.accelerator.process_index,
                seed=self.random_seed,
                mode="train",
            ),
            shuffle=shuffle,
            pipeline_args=self.data_pipelines,
            **train_dataset_args,
            ),
        )
        self.train_dataset._cuda_prefetch_batches = self.cuda_prefetch_batches
        stream_capabilities = getattr(self.train_dataset, "capabilities", None)
        self._resume_stream = None
        if (
            stream_capabilities is not None
            and hasattr(self.train_dataset, "_build_partitioned_stream")
        ):
            self._resume_stream = self._collective_dataset_call(
                "training stream manifest inspection",
                self.train_dataset._build_partitioned_stream,
            )
            self._validate_stream_signature_collectively(
                self._resume_stream, "training"
            )
        custom_sampling = {
            key
            for key in ("sampler", "batch_sampler", "generator")
            if key in self.dataloader_args
        }
        resume_reason = None
        if isinstance(self.train_dataset, IterableDataset) and self._resume_stream is None:
            resume_reason = "iterable datasets do not expose an efficient exact cursor"
        elif self.num_worker != 0:
            resume_reason = "multi-worker loaders do not expose exact worker RNG state"
        elif custom_sampling:
            resume_reason = (
                "custom loader sampling is opaque: "
                + ", ".join(sorted(custom_sampling))
            )
        elif (
            self._resume_stream is not None
            and not self._resume_stream.capabilities.resumable
        ):
            resume_reason = self._resume_stream.resume_unsupported_reason
        self._exact_resume_reason = resume_reason
        self._resume_sampler = None
        train_loader_args = dict(self.dataloader_args)
        loader_shuffle = shuffle
        if resume_reason is None and self._resume_stream is None:
            self._resume_sampler = ResumableSampler(
                self.train_dataset,
                shuffle=shuffle,
                seed=self.random_seed,
                rank=self.accelerator.process_index,
                world_size=self.accelerator.num_processes,
            )
            train_loader_args["sampler"] = self._resume_sampler
            # DataLoader creates a base seed whenever an iterator is built,
            # even with zero workers.  Keep that bookkeeping off the global
            # torch RNG so rebuilding an iterator after resume is transparent
            # to stochastic dataset transforms and the model.
            train_loader_args["generator"] = torch.Generator().manual_seed(
                self.random_seed
            )
            loader_shuffle = False
        self.train_loader = build_data_loader(
            self.train_dataset,
            self.batch_size_per_process,
            num_workers=self.num_worker,
            shuffle=loader_shuffle,
            **train_loader_args,
        )
        self._validate_cuda_prefetch_support(self.train_loader)
        self._data_stream_signature = self._build_data_stream_signature()
        if hasattr(self.train_dataset, "attach_pipeline_run_dir"):
            self.train_dataset.attach_pipeline_run_dir(
                self.rundir if self.accelerator.is_main_process else None
            )
        if getattr(self.train_dataset, "pipeline_stats", None) is not None:
            self._log_metrics = self._log_metrics_observed
        self._synchronize_pipeline_tuning_state()

        if self.val_datas or self.val_dataset_type:
            val_dataset_args = self._evaluation_dataset_args(
                self.val_dataset_args
                if self.val_dataset_type
                else self.dataset_args
            )
            self.val_dataset = self._collective_dataset_call(
                "validation dataset setup",
                lambda: build_dataset(
                self.val_dataset_type or self.dataset_type,
                self.val_datas,
                runtime_context=DatasetRuntimeContext(
                    global_batch_size=self.batch_size * self.eval_bs_multipler,
                    local_batch_size=self.batch_size_per_process * self.eval_bs_multipler,
                    world_size=self.accelerator.num_processes,
                    rank=self.accelerator.process_index,
                    seed=self.random_seed,
                    mode="validate",
                ),
                shuffle=False,
                pipeline_args=self.data_pipelines,
                **val_dataset_args,
                ),
            )
            val_capabilities = getattr(self.val_dataset, "capabilities", None)
            if (
                val_capabilities is not None
                and hasattr(self.val_dataset, "_build_partitioned_stream")
            ):
                val_stream = self._collective_dataset_call(
                    "validation stream manifest inspection",
                    self.val_dataset._build_partitioned_stream,
                )
                self._validate_stream_signature_collectively(
                    val_stream, "validation"
                )
            self.val_loader = build_data_loader(
                self.val_dataset,
                self.batch_size_per_process * self.eval_bs_multipler,
                num_workers=self.num_worker,
                shuffle=False,
                **self.dataloader_args,
            )
        else:
            self.val_dataset, self.val_loader = None, None

    def _training_dataset_args(self):
        """Translate optimizer-step options to the data-stream contract."""
        dataset_args = dict(self.dataset_args)
        steps_per_epoch = dataset_args.get("steps_per_epoch")
        if steps_per_epoch is not None:
            if type(steps_per_epoch) is not int or steps_per_epoch <= 0:
                raise ValueError("steps_per_epoch must be a positive integer")
            dataset_args["steps_per_epoch"] = (
                steps_per_epoch * self.gradient_accumulation_steps
            )
        return dataset_args

    def _synchronize_pipeline_tuning_state(self):
        state_method = getattr(
            getattr(self, "train_dataset", None),
            "pipeline_tuning_state_dict",
            None,
        )
        if state_method is None:
            return
        state = state_method() if self.accelerator.is_main_process else None
        if self.accelerator.num_processes > 1:
            values = [state]
            torch.distributed.broadcast_object_list(values, src=0)
            state = values[0]
        if state is not None:
            self.train_dataset.load_pipeline_tuning_state_dict(state)

    @staticmethod
    def _evaluation_dataset_args(configured_args):
        """Remove training-only stream limits from evaluation datasets."""
        dataset_args = dict(configured_args)
        dataset_args.pop("steps_per_epoch", None)
        dataset_args.pop("autotune", None)
        dataset_args.pop("observability", None)
        return dataset_args

    def _setup_test_data(self):
        """Build test dataset and dataloader for evaluation."""
        self._set_batch_size_per_process()
        test_dataset_args = self._evaluation_dataset_args(self.dataset_args)

        self.test_dataset = self._collective_dataset_call(
            "test dataset setup",
            lambda: build_dataset(
            self.dataset_type,
            self.test_datas,
            runtime_context=DatasetRuntimeContext(
                global_batch_size=self.batch_size * self.eval_bs_multipler,
                local_batch_size=self.batch_size_per_process * self.eval_bs_multipler,
                world_size=self.accelerator.num_processes,
                rank=self.accelerator.process_index,
                seed=self.random_seed,
                mode="test",
            ),
            shuffle=False,
            **test_dataset_args,
            ),
        )
        test_capabilities = getattr(self.test_dataset, "capabilities", None)
        if (
            test_capabilities is not None
            and hasattr(self.test_dataset, "_build_partitioned_stream")
        ):
            test_stream = self._collective_dataset_call(
                "test stream manifest inspection",
                self.test_dataset._build_partitioned_stream,
            )
            self._validate_stream_signature_collectively(test_stream, "test")
        self.test_loader = build_data_loader(
            self.test_dataset,
            self.batch_size_per_process * self.eval_bs_multipler,
            num_workers=self.num_worker,
            shuffle=False,
            **self.dataloader_args,
        )

    def _set_batch_size_per_process(self):
        """Validate and derive the per-process batch size from the global batch size."""
        num_processes = self.accelerator.num_processes
        if self.batch_size % num_processes != 0:
            raise ValueError(
                f"Global batch size {self.batch_size} must be divisible by"
                f" process count {num_processes}."
            )
        self.batch_size_per_process = self.batch_size // num_processes

    @staticmethod
    def _canonical_resume_value(value):
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, dict):
            return {
                str(key): BaseTrainer._canonical_resume_value(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, (list, tuple)):
            return [BaseTrainer._canonical_resume_value(item) for item in value]
        if isinstance(value, set):
            items = [BaseTrainer._canonical_resume_value(item) for item in value]
            return sorted(items, key=repr)
        raise TypeError(f"non-canonical value {type(value).__name__}")

    def _training_file_identity(self):
        identities = []
        for configured_path in self.train_datas:
            path = os.path.abspath(os.fspath(configured_path))
            if os.path.isfile(path):
                candidates = [path]
            elif os.path.isdir(path):
                candidates = []
                for root, dirs, files in os.walk(path):
                    dirs.sort()
                    candidates.extend(
                        os.path.join(root, filename) for filename in sorted(files)
                    )
            else:
                candidates = [path]
            for candidate in candidates:
                if os.path.isfile(candidate):
                    stat = os.stat(candidate)
                    digest = hashlib.sha256()
                    with open(candidate, "rb") as stream:
                        while True:
                            chunk = stream.read(1024 * 1024)
                            if not chunk:
                                break
                            digest.update(chunk)
                    identities.append(
                        [
                            candidate,
                            int(stat.st_size),
                            int(stat.st_mtime_ns),
                            digest.hexdigest(),
                        ]
                    )
                else:
                    identities.append([candidate, None, None, None])
        return identities

    def _build_data_stream_signature(self):
        if getattr(self, "_resume_stream", None) is not None:
            raw = self._resume_stream.signature_state()
            raw["gradient_accumulation_steps"] = self.gradient_accumulation_steps
            return self._canonical_resume_value(raw)
        raw = {
            "batch_size": self.batch_size,
            "batch_size_per_process": self.batch_size_per_process,
            "dataset_type": self.dataset_type,
            "dataset_args": self.dataset_args,
            "data_pipelines": self.data_pipelines,
            "train_datas": [os.path.abspath(os.fspath(p)) for p in self.train_datas],
            "file_identity": self._training_file_identity(),
            "shuffle": not self.no_shuffle,
            "num_worker": self.num_worker,
            "dataloader_args": self.dataloader_args,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "world_size": self.accelerator.num_processes,
            "map_rng_algorithm": RNG_ALGORITHM,
        }
        try:
            return self._canonical_resume_value(raw)
        except TypeError as exc:
            if self._exact_resume_reason is None:
                self._exact_resume_reason = (
                    f"data-stream configuration is not canonical: {exc}"
                )
            return None

    def _data_stream_signature_matches(self, saved, *, legacy_map_state=False):
        if saved == self._data_stream_signature:
            return True
        if not legacy_map_state:
            return False
        current = self._data_stream_signature
        if not isinstance(saved, dict) or not isinstance(current, dict):
            return False
        legacy_projection = dict(current)
        legacy_projection.pop("map_rng_algorithm", None)
        current_identities = current.get("file_identity")
        if not isinstance(current_identities, list):
            return False
        hashes = []
        projected_identities = []
        for identity in current_identities:
            if not isinstance(identity, list) or len(identity) != 4:
                return False
            projected_identities.append(identity[:3])
            hashes.append(identity[3])
        legacy_projection["file_identity"] = projected_identities
        return (
            saved == legacy_projection
            and bool(hashes)
            and all(
                digest in _LEGACY_MAP_CONTENT_ALLOWLIST
                for digest in hashes
            )
        )

    def _init_models(self):
        """Instantiate models. Override in subclasses to add extra models.

        Trained models go into ``self.models`` — they are included in the optimizer,
        wrapped with DDP by ``_prepare_for_training``, and saved/loaded in checkpoints.

        Frozen / helper models go into ``self.aux_models`` — they are only placed on the
        correct device via ``accelerator.prepare_model(..., evaluation_mode=True)`` and
        are never optimized or checkpointed.

        Both dicts are keyed by ``model.name``.  The primary trained model's name is
        stored in ``self.model_name`` and accessible via the ``self.model`` property.
        """
        self.models = {}
        self.aux_models = {}
        model = build_model(self.model_type, **self.model_args)
        self._primary_inductor_config = model_inductor_config(model)
        self._configure_model_compilation(model)
        self.model_name = model.name
        self.models[self.model_name] = model

    @property
    def model(self):
        """The primary model (shortcut into ``self.models``)."""
        return self.models[self.model_name]

    @model.setter
    def model(self, value):
        self.models[self.model_name] = value

    def _setup_optimizer(self):
        """Create optimizers for all trained models, keyed by name.

        The default implementation creates a single ``"main"`` optimizer covering
        all trained models' parameters.  Subclasses may override to create several
        optimizers (e.g. one per model for alternating optimization); all entries
        are prepared, stepped, and checkpointed by the base class.
        """
        trained_models = list(self.models.values())
        model_or_models = trained_models[0] if len(trained_models) == 1 else trained_models
        self.optimizers = {
            "main": build_optimizer(
                self.optim_type,
                model_or_models,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                **self.optim_args,
            )
        }

    @property
    def optimizer(self):
        """The primary optimizer (shortcut into ``self.optimizers``)."""
        return self.optimizers["main"]

    @optimizer.setter
    def optimizer(self, value):
        self.optimizers["main"] = value

    def _setup_scheduler(self):
        """Create schedulers and restore their exact runtime state on resume."""
        self.schedulers = {
            name: build_lr_scheduler(
                opt,
                self.lr_scheduler_type,
                self.iterations,
                last_it=-1,
                **self.lr_scheduler_args,
            )
            for name, opt in self.optimizers.items()
        }
        if self._pending_scheduler_states is not None:
            if set(self._pending_scheduler_states) != set(self.schedulers):
                raise RuntimeError(
                    "Cannot resume: scheduler names changed from"
                    f" {sorted(self._pending_scheduler_states)} to"
                    f" {sorted(self.schedulers)}."
                )
            for name, scheduler_state in self._pending_scheduler_states.items():
                self.schedulers[name].load_state_dict(scheduler_state)
            for name, learning_rates in self._pending_runtime_lrs.items():
                for group, learning_rate in zip(
                    self.optimizers[name].param_groups, learning_rates
                ):
                    group["lr"] = learning_rate
            self._pending_scheduler_states = None
            self._pending_runtime_lrs = None

    def _current_optimizer_config(self):
        return {
            "type": self.optim_type,
            "args": self._canonical_resume_value(self.optim_args),
            "weight_decay": self.weight_decay,
            "optimizers": {
                name: [
                    {
                        "configured_lr": group["lr"],
                        "num_parameters": len(group["params"]),
                    }
                    for group in optimizer.param_groups
                ]
                for name, optimizer in self.optimizers.items()
            },
        }

    def _scheduler_config(self):
        return {
            "type": self.lr_scheduler_type,
            "args": self._canonical_resume_value(self.lr_scheduler_args),
            "iterations": self.iterations,
        }

    @property
    def scheduler(self):
        """The primary LR scheduler (shortcut into ``self.schedulers``)."""
        return self.schedulers["main"]

    @scheduler.setter
    def scheduler(self, value):
        self.schedulers["main"] = value

    # ── Checkpoint ────────────────────────────────────────────────
    #
    # Layout (new format), under ``rundir/ckpts/``:
    #   ckpt_{model_name}_{iteration:07d}.pt   one standard single-model file per
    #                                          checkpointed model (weights + metadata)
    #   state_{iteration:07d}.pt               training state: per-name optimizer
    #                                          states, scaler state, and counters
    #
    # A state file is only written at iterations where model files are also
    # written, so the latest state always has matching model files to resume
    # from.  Legacy runs with merged ``ckpt_*`` files in the rundir root are
    # still loadable (resume then continues in the new format).

    @property
    def ckpt_dir(self):
        """Directory holding checkpoint and training-state files."""
        return os.path.join(self.rundir, "ckpts")

    def _checkpointed_models(self) -> dict:
        """Models to save/load in checkpoints, keyed by name.

        Defaults to all trained models.  Subclasses may override to add
        stateful auxiliary models (e.g. an EMA teacher from ``aux_models``).
        """
        return self.models

    def _vq_runtime_modules(self):
        """Discover canonical VQ modules independently of runtime wrappers."""
        discovered = []
        modules_by_identity = {}
        for model_name, wrapped in self._checkpointed_models().items():
            if not isinstance(model_name, str) or not model_name:
                raise RuntimeError(
                    f"invalid checkpointed model name {model_name!r}"
                )
            for path, module in self._unwrap(wrapped).named_modules(
                remove_duplicate=False
            ):
                if not isinstance(module, VectorQuantize):
                    continue
                identity = id(module)
                if identity not in modules_by_identity:
                    entry = {
                        "model": model_name,
                        "path": path,
                        "aliases": [],
                        "module": module,
                    }
                    modules_by_identity[identity] = entry
                    discovered.append(entry)
                modules_by_identity[identity]["aliases"].append(
                    {"model": model_name, "path": path}
                )
        return discovered

    @staticmethod
    def _vq_runtime_schema(discovered):
        models = {}
        for entry in discovered:
            module = entry["module"]
            initialized = torch.equal(
                module.inited.detach(),
                module.inited.new_ones(()),
            )
            models.setdefault(entry["model"], []).append(
                {
                    "path": entry["path"],
                    "aliases": list(entry["aliases"]),
                    "descriptor": module._vq_init_continuation_descriptor(),
                    "inited": initialized,
                }
            )
        return [
            {"model": model_name, "modules": modules}
            for model_name, modules in models.items()
        ]

    def _vq_runtime_autocast_mode(self):
        mode = getattr(self.accelerator, "mixed_precision", "no")
        if mode is None:
            mode = "no"
        if not isinstance(mode, str) or not mode:
            raise RuntimeError(
                f"invalid accelerator mixed-precision mode {mode!r}"
            )
        return mode

    def _capture_vq_runtime_payload(self):
        discovered = self._vq_runtime_modules()
        schema = self._vq_runtime_schema(discovered)
        autocast_mode = self._vq_runtime_autocast_mode()
        payload_models = {}
        raw_bytes = 0
        maximum_raw_bytes = 0
        has_partial = False
        module_stats = []
        for entry in discovered:
            state = entry["module"]._vq_init_continuation_state()
            samples = state["samples"]
            sample_dtype = state["dtype"]
            expected_dtype = _expected_vq_runtime_dtype(
                state["descriptor"],
                state["count"],
                autocast_mode,
            )
            if (
                sample_dtype not in _RUNTIME_DTYPE_BYTES
                or not samples.is_floating_point()
                or sample_dtype != expected_dtype
            ):
                raise RuntimeError(
                    f"pending VQ dtype {sample_dtype!r} is incompatible with"
                    f" autocast mode {autocast_mode!r} for"
                    f" {entry['model']!r}:{entry['path']!r}"
                )
            sample_bytes = samples.numel() * samples.element_size()
            raw_bytes += sample_bytes
            maximum_raw_bytes += (
                state["descriptor"]["target_samples"]
                * state["descriptor"]["dim_feature"]
                * samples.element_size()
            )
            has_partial |= state["count"] > 0
            module_stats.append(
                {
                    "model": entry["model"],
                    "path": entry["path"],
                    "dtype": state["dtype"],
                    "count": state["count"],
                    "raw_bytes": sample_bytes,
                }
            )
            payload_models.setdefault(entry["model"], {})[entry["path"]] = {
                "aliases": list(entry["aliases"]),
                "state": state,
            }
        return (
            {
                "version": _RUNTIME_SIDECAR_VERSION,
                "models": payload_models,
                "schema": schema,
                "autocast_mode": autocast_mode,
            },
            {
                "version": _RUNTIME_MANIFEST_VERSION,
                "schema": schema,
                "autocast_mode": autocast_mode,
                "modules": module_stats,
                "has_partial": has_partial,
                "raw_bytes": raw_bytes,
                "maximum_bytes": (
                    maximum_raw_bytes
                    + _runtime_serialization_allowance(maximum_raw_bytes)
                ),
            },
        )

    def _runtime_generation_collectively(self):
        envelope = None
        if self.accelerator.is_main_process:
            try:
                generation = os.urandom(16).hex()
                if re.fullmatch(r"[0-9a-f]{32}", generation) is None:
                    raise RuntimeError("generated token has invalid grammar")
                envelope = {"generation": generation, "error": None}
            except BaseException as exc:
                envelope = {
                    "generation": None,
                    "error": (
                        f"rank {self.accelerator.process_index}:"
                        f" {type(exc).__name__}: {exc}"
                    ),
                }
        envelopes = (
            [envelope]
            if self.accelerator.num_processes == 1
            else gather_object([envelope])
        )
        main_envelope = envelopes[0]
        if not isinstance(main_envelope, dict) or main_envelope.get("error"):
            error = (
                main_envelope.get("error")
                if isinstance(main_envelope, dict)
                else f"rank 0: invalid generation envelope {main_envelope!r}"
            )
            raise RuntimeError(
                f"checkpoint runtime generation failed collectively: {error}"
            )
        return main_envelope["generation"]

    @staticmethod
    def _runtime_basename(iteration, generation, rank):
        basename = (
            f"runtime_{iteration:07d}_{generation}_rank_{rank}.pt"
        )
        match = _RUNTIME_SIDECAR_RE.fullmatch(basename)
        if (
            match is None
            or int(match.group("iteration")) != iteration
            or int(match.group("rank")) != rank
            or match.group("generation") != generation
        ):
            raise RuntimeError(f"invalid runtime sidecar basename {basename!r}")
        return basename

    def _open_runtime_file(self, basename):
        if (
            not isinstance(basename, str)
            or _RUNTIME_SIDECAR_RE.fullmatch(basename) is None
            or os.path.basename(basename) != basename
            or "\x00" in basename
        ):
            raise RuntimeError(f"invalid runtime sidecar path {basename!r}")
        directory_fd = os.open(
            self.ckpt_dir,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            file_fd = os.open(
                basename,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
        except BaseException:
            os.close(directory_fd)
            raise
        file_stat = os.fstat(file_fd)
        if not stat.S_ISREG(file_stat.st_mode):
            os.close(file_fd)
            os.close(directory_fd)
            raise RuntimeError(
                f"runtime sidecar {basename!r} is not a regular file"
            )
        return directory_fd, file_fd, file_stat

    def _write_vq_runtime_sidecar(self, iteration, generation):
        rank = self.accelerator.process_index
        basename = self._runtime_basename(iteration, generation, rank)
        payload, manifest = self._capture_vq_runtime_payload()
        payload.update(
            {
                "iteration": iteration,
                "world_size": self.accelerator.num_processes,
                "rank": rank,
                "generation": generation,
            }
        )
        nonce = os.urandom(16).hex()
        temp_basename = (
            f".runtime-tmp_{iteration:07d}_{generation}_rank_{rank}_"
            f"{os.getpid()}_{nonce}.pt"
        )
        if _RUNTIME_TEMP_RE.fullmatch(temp_basename) is None:
            raise RuntimeError(
                f"invalid runtime temporary basename {temp_basename!r}"
            )
        directory_fd = os.open(
            self.ckpt_dir,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        temp_exists = False
        try:
            temp_fd = os.open(
                temp_basename,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=directory_fd,
            )
            temp_exists = True
            with os.fdopen(temp_fd, "wb") as output:
                torch.save(payload, output)
                output.flush()
                os.fsync(output.fileno())
            os.link(
                temp_basename,
                basename,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
            os.unlink(temp_basename, dir_fd=directory_fd)
            temp_exists = False
        finally:
            if temp_exists:
                try:
                    os.unlink(temp_basename, dir_fd=directory_fd)
                except FileNotFoundError:
                    pass
            os.close(directory_fd)

        directory_fd, file_fd, file_stat = self._open_runtime_file(
            basename
        )
        try:
            digest = hashlib.sha256()
            while chunk := os.read(file_fd, 1024 * 1024):
                digest.update(chunk)
        finally:
            os.close(file_fd)
            os.close(directory_fd)
        if (
            file_stat.st_size > manifest["maximum_bytes"]
            or file_stat.st_size
            > manifest["raw_bytes"]
            + _runtime_serialization_allowance(manifest["raw_bytes"])
        ):
            raise RuntimeError(
                f"runtime sidecar {basename} exceeds configured size bound:"
                f" {file_stat.st_size} > {manifest['maximum_bytes']}"
            )
        manifest.update(
            {
                "basename": basename,
                "sha256": digest.hexdigest(),
                "size": file_stat.st_size,
                "iteration": iteration,
                "generation": generation,
                "rank": rank,
            }
        )
        return manifest

    def _capture_vq_runtime_collectively(self, iteration, generation):
        manifest = None
        local_error = None
        try:
            manifest = self._write_vq_runtime_sidecar(
                iteration,
                generation,
            )
        except BaseException as exc:
            local_error = (
                f"rank {self.accelerator.process_index}:"
                f" {type(exc).__name__}: {exc}"
            )
        results = (
            [(manifest, local_error)]
            if self.accelerator.num_processes == 1
            else gather_object([(manifest, local_error)])
        )
        errors = [error for _, error in results if error is not None]
        if errors:
            raise RuntimeError(
                "checkpoint runtime-sidecar capture failed collectively: "
                + "; ".join(errors)
            )
        return manifest

    def _model_ckpt_path(self, name: str, iteration: int) -> str:
        return os.path.join(self.ckpt_dir, f"ckpt_{name}_{iteration:07d}.pt")

    def _state_path(self, iteration: int) -> str:
        return os.path.join(self.ckpt_dir, f"state_{iteration:07d}.pt")

    def _load_checkpoint(self):
        """Resume from the latest training state in *ckpt_dir*, falling back to
        legacy merged checkpoints in the rundir root, then to optional
        pretrained weights."""
        state_filename = find_latest_ckpt(self.ckpt_dir, "state_")

        if state_filename:
            self._load_state_checkpoint(state_filename)
            return

        if os.path.isdir(self.ckpt_dir) and any(
            fn.startswith("ckpt_") for fn in os.listdir(self.ckpt_dir)
        ):
            self.accelerator.print(
                f"Warning: model checkpoints exist in {self.ckpt_dir} but no state file"
                " was found; starting from scratch."
            )

        legacy_filename = find_latest_ckpt(self.rundir, f"ckpt_{self.model_name}")
        if legacy_filename:
            self._load_legacy_checkpoint(legacy_filename)
            return

        self._load_pretrained_weights()

    def _load_state_checkpoint(self, state_filename):
        """Load models, optimizers, and scaler anchored on *state_filename*."""
        iteration = get_iteration_from_ckpt_filename(state_filename)
        if iteration is None:
            raise RuntimeError(f"Cannot parse iteration from state file {state_filename}")

        state = torch.load(state_filename, map_location="cpu", weights_only=True)
        if state.get("format_version") != 2:
            raise RuntimeError(
                f"Cannot resume checkpoint {state_filename}: exact continuation"
                " state format version 2 is required."
            )
        saved_optimizer_config = state.get("optimizer_config")
        if saved_optimizer_config != self._optimizer_config:
            raise RuntimeError(
                f"Cannot resume checkpoint {state_filename}: configured optimizer"
                f" layout/base learning rates changed from {saved_optimizer_config!r}"
                f" to {self._optimizer_config!r}."
            )
        saved_scheduler_config = state.get("scheduler_config")
        current_scheduler_config = self._scheduler_config()
        if saved_scheduler_config != current_scheduler_config:
            raise RuntimeError(
                f"Cannot resume checkpoint {state_filename}: scheduler"
                f" configuration changed from {saved_scheduler_config!r} to"
                f" {current_scheduler_config!r}."
            )
        resume = state.get("resume")
        if (
            not isinstance(resume, dict)
            or resume.get("version") not in {1, 2}
        ):
            raise RuntimeError(
                f"Cannot resume checkpoint {state_filename}: exact data-stream/RNG"
                " continuation state is missing."
            )
        if resume.get("world_size") != self.accelerator.num_processes:
            raise RuntimeError(
                f"Cannot resume checkpoint {state_filename}: world size changed"
                f" from {resume.get('world_size')} to"
                f" {self.accelerator.num_processes}."
            )
        rank_states = resume.get("ranks", [])
        if len(rank_states) != self.accelerator.num_processes:
            raise RuntimeError(
                f"Cannot resume checkpoint {state_filename}: invalid per-rank"
                " continuation state count."
            )
        rank_states, legacy_map_state = self._validate_rank_resume_states(
            rank_states, classify_legacy=True
        )
        rank_state = rank_states[self.accelerator.process_index]
        if not rank_state.get("supported", False):
            raise RuntimeError(
                f"Cannot resume checkpoint {state_filename}:"
                f" {rank_state.get('reason', 'data stream is unsupported')}."
            )
        if not self._data_stream_signature_matches(
            rank_state.get("data_stream_signature"),
            legacy_map_state=legacy_map_state,
        ):
            raise RuntimeError(
                f"Cannot resume checkpoint {state_filename}: data-stream"
                " configuration or file identity changed."
            )
        metadata = state.get("metadata", {})
        if int(metadata.get("iteration", -1)) != iteration:
            raise RuntimeError(
                "Cannot resume: state filename and metadata iteration differ."
            )

        for name, m in self._checkpointed_models().items():
            model_ckpt = self._model_ckpt_path(name, iteration)
            if not os.path.isfile(model_ckpt):
                raise FileNotFoundError(
                    f"Training state {state_filename} refers to iteration {iteration},"
                    f" but model checkpoint {model_ckpt} is missing"
                )
            model_state_dict, _, model_metadata = load_torch_ckpt(model_ckpt)
            if int(model_metadata.get("iteration", -1)) != iteration:
                raise RuntimeError(
                    f"Cannot resume: model {name!r} metadata iteration differs."
                )
            m.load_state_dict(model_state_dict)

        vq_modules = self._vq_runtime_modules()
        if resume["version"] == 1:
            self._validate_v1_vq_compatibility(vq_modules, iteration)
        elif not isinstance(rank_state.get("vq_runtime"), dict):
            raise RuntimeError(
                "Cannot resume version-2 checkpoint: VQ runtime manifest is missing."
            )

        if set(state["optimizers"]) != set(self.optimizers):
            raise RuntimeError(
                f"Cannot resume checkpoint {state_filename}: optimizer names changed."
            )
        for name, opt_state in state["optimizers"].items():
            self.optimizers[name].load_state_dict(opt_state)
        self._pending_runtime_lrs = {
            name: [group["lr"] for group in optimizer.param_groups]
            for name, optimizer in self.optimizers.items()
        }
        self._pending_scheduler_states = state["schedulers"]
        if self.accelerator.scaler is not None and state.get("scaler") is not None:
            self.accelerator.scaler.load_state_dict(state["scaler"])

        self.state.iteration = int(metadata.get("iteration", 0))
        self.state.epoch = int(metadata.get("epoch", 0))
        self.state.rows = int(metadata.get("rows", 0))
        self._pending_resume_state = rank_state
        self._pending_resume_version = resume["version"]
        self._last_state_save_iteration = self.state.iteration
        self.accelerator.print(f"Loaded from {state_filename}")

    @staticmethod
    def _validate_v1_vq_compatibility(vq_modules, iteration):
        initialized = [
            torch.equal(
                entry["module"].inited.detach(),
                entry["module"].inited.new_ones(()),
            )
            for entry in vq_modules
        ]
        uninitialized = [
            entry
            for entry, is_initialized in zip(vq_modules, initialized)
            if not is_initialized
        ]
        if uninitialized and iteration != 0:
            raise RuntimeError(
                "Cannot resume version-1 checkpoint with uninitialized VQ"
                " after iteration zero: partial initialization samples"
                " may be missing."
            )
        if uninitialized and any(initialized):
            raise RuntimeError(
                "Cannot resume version-1 checkpoint with mixed initialized"
                " and uninitialized VQ modules: pending initialization"
                " samples may be missing."
            )

    def _validate_rank_resume_states(
        self, rank_states, *, classify_legacy=False
    ):
        """Validate exact integer rank slots before selecting a local cursor."""
        legacy_keys = {"supported", "data_stream_signature", "sampler", "rng"}
        if all(
            isinstance(state, dict)
            and "cursor_kind" not in state
            and state.get("supported") is True
            and set(state) == legacy_keys
            for state in rank_states
        ):
            result = list(rank_states)
            return (result, True) if classify_legacy else result
        expected_slots = set(range(self.accelerator.num_processes))
        slots = []
        by_slot = {}
        for state in rank_states:
            if not isinstance(state, dict):
                raise RuntimeError("Cannot resume: rank continuation state is not a dictionary.")
            cursor_kind = state.get("cursor_kind")
            if state.get("supported") is True and cursor_kind not in {
                "file_stream_v1",
                "map_sampler_v1",
            }:
                raise RuntimeError(
                    f"Cannot resume: unknown cursor kind {cursor_kind!r}."
                )
            if state.get("supported") is not True and cursor_kind is not None:
                raise RuntimeError(
                    "Cannot resume: unsupported continuation state declares "
                    f"cursor kind {cursor_kind!r}."
                )
            slot = state.get("slot")
            if type(slot) is not int or slot not in expected_slots:
                raise RuntimeError(f"Cannot resume: invalid integer rank slot {slot!r}.")
            if state.get("rank") != slot:
                raise RuntimeError(
                    f"Cannot resume: embedded rank {state.get('rank')!r} does not match slot {slot}."
                )
            if slot in by_slot:
                raise RuntimeError(f"Cannot resume: duplicate rank slot {slot}.")
            local_digest = state.get("local_digest")
            digest_input = dict(state)
            digest_input.pop("local_digest", None)
            expected_local_digest = hashlib.sha256(
                canonical_pipeline_state_bytes(digest_input)
            ).hexdigest()
            if local_digest != expected_local_digest:
                raise RuntimeError(
                    f"Cannot resume: rank slot {slot} local digest is missing or corrupted."
                )
            projection = self._global_resume_projection(state)
            expected_global_digest = hashlib.sha256(
                canonical_pipeline_state_bytes(projection)
            ).hexdigest()
            if state.get("global_digest") != expected_global_digest:
                raise RuntimeError(
                    f"Cannot resume: rank slot {slot} global digest is missing or corrupted."
                )
            slots.append(slot)
            by_slot[slot] = state
        if set(slots) != expected_slots:
            raise RuntimeError(
                f"Cannot resume: rank slots are {sorted(slots)}, expected "
                f"{sorted(expected_slots)}."
            )
        ordered = [by_slot[slot] for slot in range(self.accelerator.num_processes)]
        global_digests = {state["global_digest"] for state in ordered}
        if len(global_digests) != 1:
            raise RuntimeError(
                "Cannot resume: rank-independent continuation state differs across ranks."
            )
        return (ordered, False) if classify_legacy else ordered

    @staticmethod
    def _global_resume_projection(state):
        projection = {
            "supported": state.get("supported"),
            "cursor_kind": state.get("cursor_kind"),
            "data_stream_signature": state.get("data_stream_signature"),
        }
        if "vq_runtime" in state:
            projection["vq_runtime"] = (
                {
                    "version": state["vq_runtime"].get("version"),
                    "schema": state["vq_runtime"].get("schema"),
                    "autocast_mode": state["vq_runtime"].get(
                        "autocast_mode"
                    ),
                }
                if state.get("vq_runtime") is not None
                else None
            )
        if state.get("cursor_kind") == "file_stream_v1":
            coordination = state.get("stream_coordination")
            projection[
                "stream_coordination" if coordination is not None else "stream"
            ] = coordination if coordination is not None else state.get("stream")
        else:
            sampler = dict(state.get("sampler") or {})
            sampler.pop("rank", None)
            projection["sampler"] = sampler
        return projection

    def _load_legacy_checkpoint(self, ckpt_filename):
        """Load a legacy merged checkpoint (models + optimizer in one file)."""
        model_state_dict, training_state_dicts, metadata = load_torch_ckpt(ckpt_filename)
        if int(metadata.get("iteration", 0)) > 0:
            raise RuntimeError(
                f"Cannot resume legacy checkpoint {ckpt_filename}: exact"
                " continuation and configured learning-rate state are missing."
            )
        # Primary model is stored under the "model" key
        self.model.load_state_dict(model_state_dict)
        # Additional trained models are stored under their name
        for name, m in self.models.items():
            if name != self.model_name and name in training_state_dicts:
                m.load_state_dict(training_state_dicts.pop(name))
        self.optimizer.load_state_dict(training_state_dicts["optimizer"])
        if self.accelerator.scaler is not None and "scalar" in training_state_dicts:
            self.accelerator.scaler.load_state_dict(training_state_dicts["scalar"])
        self.accelerator.print(f"Loaded from legacy checkpoint {ckpt_filename}")
        self.state.iteration = int(metadata.get("iteration", 0))
        self.state.epoch = int(metadata.get("epoch", 0))
        self.state.rows = int(metadata.get("rows", 0))

    def _load_pretrained_weights(self):
        """Load optional pretrained weights into the locally initialized model."""
        if self.load_from is not None:
            model_state_dict, _, _ = load_torch_ckpt(self.load_from)
            model_state_dict = state_dict_drop_size_unmatched(self.model, model_state_dict)
            missing_keys, unexpected_keys = self.model.load_state_dict(
                model_state_dict, strict=False
            )
            if unexpected_keys:
                self.accelerator.print(
                    f"unexpected keys in state_dict: {', '.join(unexpected_keys)}"
                )
            if missing_keys:
                self.accelerator.print(
                    f"missing keys in state_dict: {', '.join(missing_keys)}"
                )
            self.accelerator.print(f"Loaded from pretrained: {self.load_from}")

    def _is_save_iteration(self, iteration: int) -> bool:
        """Whether *iteration* is due for a checkpoint save per the configured intervals."""
        return (
            iteration % self.save_interval == 0
            or iteration % self.temp_save_interval == 0
            or (
                self.state_save_interval is not None
                and iteration % self.state_save_interval == 0
            )
        )

    @staticmethod
    def _checkpoint_values_equal(left, right):
        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            if not isinstance(left, torch.Tensor) or not isinstance(
                right, torch.Tensor
            ):
                return False
            if left.shape != right.shape or left.dtype != right.dtype:
                return False
            if left.device != right.device:
                left = left.detach().cpu()
                right = right.detach().cpu()
            return torch.equal(left, right)
        if isinstance(left, dict) or isinstance(right, dict):
            if not isinstance(left, dict) or not isinstance(right, dict):
                return False
            return list(left) == list(right) and all(
                BaseTrainer._checkpoint_values_equal(
                    left[key],
                    right[key],
                )
                for key in left
            )
        if isinstance(left, (list, tuple)) or isinstance(
            right, (list, tuple)
        ):
            if type(left) is not type(right) or len(left) != len(right):
                return False
            return all(
                BaseTrainer._checkpoint_values_equal(a, b)
                for a, b in zip(left, right)
            )
        return left == right

    def _validate_committed_checkpoint_set(self, iteration):
        state_path = self._state_path(iteration)
        state = torch.load(
            state_path,
            map_location="cpu",
            weights_only=True,
        )
        if (
            not isinstance(state, dict)
            or state.get("format_version") != 2
            or int(state.get("metadata", {}).get("iteration", -1))
            != iteration
            or int(state.get("metadata", {}).get("epoch", -1))
            != self.state.epoch
            or int(state.get("metadata", {}).get("rows", -1))
            != self.state.rows
        ):
            raise RuntimeError(
                f"checkpoint state collision at iteration {iteration}:"
                " committed state metadata differs"
            )
        resume = state.get("resume")
        if (
            not isinstance(resume, dict)
            or resume.get("version") not in {1, 2}
            or resume.get("world_size")
            != self.accelerator.num_processes
            or not isinstance(resume.get("ranks"), list)
            or len(resume["ranks"]) != self.accelerator.num_processes
        ):
            raise RuntimeError(
                f"checkpoint state collision at iteration {iteration}:"
                " committed resume metadata differs"
            )
        rank_states = self._validate_rank_resume_states(resume["ranks"])
        if (
            state.get("optimizer_config") != self._optimizer_config
            or state.get("scheduler_config") != self._scheduler_config()
            or not self._checkpoint_values_equal(
                state.get("optimizers"),
                {
                    name: optimizer.state_dict()
                    for name, optimizer in self.optimizers.items()
                },
            )
            or not self._checkpoint_values_equal(
                state.get("schedulers"),
                {
                    name: scheduler.state_dict()
                    for name, scheduler in self.schedulers.items()
                },
            )
        ):
            raise RuntimeError(
                f"checkpoint state collision at iteration {iteration}:"
                " optimizer or scheduler state differs"
            )
        current_scaler = (
            self.accelerator.scaler.state_dict()
            if self.accelerator.scaler is not None
            else None
        )
        if not self._checkpoint_values_equal(
            state.get("scaler"),
            current_scaler,
        ):
            raise RuntimeError(
                f"checkpoint state collision at iteration {iteration}:"
                " scaler state differs"
            )

        local_saved_state = dict(
            rank_states[self.accelerator.process_index]
        )
        local_current_state = self._local_resume_state()
        for value in (local_saved_state, local_current_state):
            value.pop("vq_runtime", None)
            value.pop("global_digest", None)
            value.pop("local_digest", None)
        if not self._checkpoint_values_equal(
            local_saved_state,
            local_current_state,
        ):
            raise RuntimeError(
                f"checkpoint state collision at iteration {iteration}:"
                " rank-local cursor or RNG state differs"
            )

        for name, model in self._checkpointed_models().items():
            model_path = self._model_ckpt_path(name, iteration)
            if not os.path.isfile(model_path):
                raise RuntimeError(
                    f"checkpoint state collision at iteration {iteration}:"
                    f" committed model {name!r} is missing"
                )
            saved_model, _, metadata = load_torch_ckpt(model_path)
            current_model = self._unwrap(model).state_dict()
            if (
                int(metadata.get("iteration", -1)) != iteration
                or not self._checkpoint_values_equal(
                    saved_model,
                    current_model,
                )
            ):
                raise RuntimeError(
                    f"checkpoint state collision at iteration {iteration}:"
                    f" committed model {name!r} differs"
                )
        if resume["version"] == 2:
            current_schema = self._vq_runtime_schema(
                self._vq_runtime_modules()
            )
            local_manifest = None
            for rank, rank_state in enumerate(rank_states):
                manifest = rank_state.get("vq_runtime")
                self._validate_retained_vq_runtime_manifest(
                    manifest,
                    iteration=iteration,
                    rank=rank,
                )
                if manifest.get("schema") != current_schema:
                    raise RuntimeError(
                        f"checkpoint state collision at iteration {iteration}:"
                        " VQ runtime schema differs"
                    )
                if rank == self.accelerator.process_index:
                    local_manifest = manifest
            self._validate_current_vq_runtime_collision(
                local_manifest,
                iteration=iteration,
            )
        else:
            self._validate_v1_vq_compatibility(
                self._vq_runtime_modules(),
                iteration,
            )

    def _prepare_checkpoint_directory_collectively(self):
        local_error = None
        try:
            os.makedirs(self.ckpt_dir, exist_ok=True)
            if not os.path.isdir(self.ckpt_dir):
                raise RuntimeError(
                    f"checkpoint path {self.ckpt_dir!r} is not a directory"
                )
        except BaseException as exc:
            local_error = (
                f"rank {self.accelerator.process_index}:"
                f" {type(exc).__name__}: {exc}"
            )
        errors = (
            [local_error]
            if self.accelerator.num_processes == 1
            else gather_object([local_error])
        )
        failures = [error for error in errors if error is not None]
        if failures:
            raise RuntimeError(
                "checkpoint directory preparation failed collectively: "
                + "; ".join(failures)
            )

    def _save_checkpoint(self, force_state: bool = False):
        """Save a checkpoint inside a collective rank boundary."""
        saves_state = hasattr(self, "state") and (
            force_state
            or self.state_save_interval is None
            or self.state.iteration % self.state_save_interval == 0
        )
        if saves_state and os.path.exists(
            self._state_path(self.state.iteration)
        ):
            local_error = None
            try:
                self._validate_committed_checkpoint_set(
                    self.state.iteration
                )
            except BaseException as exc:
                local_error = (
                    f"rank {self.accelerator.process_index}:"
                    f" {type(exc).__name__}: {exc}"
                )
            errors = (
                [local_error]
                if self.accelerator.num_processes == 1
                else gather_object([local_error])
            )
            failures = [error for error in errors if error is not None]
            if failures:
                self._checkpoint_attempt_failed = True
                raise RuntimeError(
                    "checkpoint collision validation failed collectively: "
                    + "; ".join(failures)
                )
            self._last_state_save_iteration = self.state.iteration
            return
        if saves_state:
            try:
                self._prepare_checkpoint_directory_collectively()
                generation = self._runtime_generation_collectively()
                runtime_manifest = self._capture_vq_runtime_collectively(
                    self.state.iteration,
                    generation,
                )
            except BaseException:
                self._checkpoint_attempt_failed = True
                raise
            local_resume_state = None
            local_capture_error = None
            try:
                local_resume_state = self._local_resume_state(
                    vq_runtime=runtime_manifest,
                )
            except BaseException as exc:
                local_capture_error = (
                    f"rank {self.accelerator.process_index}:"
                    f" {type(exc).__name__}: {exc}"
                )
            capture_results = (
                [(local_resume_state, local_capture_error)]
                if self.accelerator.num_processes == 1
                else gather_object([(local_resume_state, local_capture_error)])
            )
            capture_errors = [
                error
                for _, error in capture_results
                if error is not None
            ]
            if capture_errors:
                self._checkpoint_attempt_failed = True
                raise RuntimeError(
                    "checkpoint continuation-state capture failed collectively:"
                    f" {'; '.join(capture_errors)}"
                )
            self._resume_states_for_save = [
                state for state, _ in capture_results
            ]
            self._resume_states_for_save = self._validate_rank_resume_states(
                self._resume_states_for_save
            )
            local_validation_error = None
            try:
                self._validate_current_vq_runtime_collision(
                    runtime_manifest,
                    iteration=self.state.iteration,
                )
            except BaseException as exc:
                local_validation_error = (
                    f"rank {self.accelerator.process_index}:"
                    f" {type(exc).__name__}: {exc}"
                )
            validation_errors = (
                [local_validation_error]
                if self.accelerator.num_processes == 1
                else gather_object([local_validation_error])
            )
            validation_failures = [
                error
                for error in validation_errors
                if error is not None
            ]
            if validation_failures:
                self._checkpoint_attempt_failed = True
                raise RuntimeError(
                    "checkpoint runtime-sidecar validation failed"
                    " collectively: "
                    + "; ".join(validation_failures)
                )
        try:
            self._run_synchronized_operation(
                "checkpoint commit",
                lambda: self._save_checkpoint_main(force_state),
                main_process_only=True,
            )
        except BaseException:
            self._checkpoint_attempt_failed = True
            raise
        finally:
            self._resume_states_for_save = None
        if saves_state:
            self._last_state_save_iteration = self.state.iteration
        if hasattr(self, "rundir"):
            try:
                self._run_synchronized_operation(
                    "checkpoint cleanup",
                    self._cleanup_checkpoints,
                    main_process_only=True,
                )
            except BaseException as exc:
                raise RuntimeError(
                    "checkpoint committed; cleanup failed"
                ) from exc

    def _save_checkpoint_main(self, force_state: bool = False):
        """Save per-model checkpoint files and (if due) the training state, then
        prune temporary snapshots and old state files.

        Args:
            force_state: Save the training state even if ``state_save_interval``
                does not match (used for the final save at end of training).
        """
        st = self.state
        metadata = {
            "iteration": str(st.iteration),
            "epoch": str(st.epoch),
            "rows": str(st.rows),
        }
        for name, m in self._checkpointed_models().items():
            save_torch_ckpt(
                self._model_ckpt_path(name, st.iteration),
                self._unwrap(m).state_dict(),
                {},
                metadata,
            )
        if st.iteration % self.save_interval != 0:
            self._temp_snapshot_iters.add(st.iteration)

        if (
            force_state
            or self.state_save_interval is None
            or st.iteration % self.state_save_interval == 0
        ):
            self._save_training_state()

    def _run_synchronized_operation(
        self,
        operation: str,
        callback,
        *,
        main_process_only: bool = False,
    ):
        """Run an operation between barriers and propagate any rank's error to all ranks."""
        accelerator = self.accelerator
        accelerator.wait_for_everyone()
        local_exception = None
        local_error = None
        try:
            if not main_process_only or accelerator.is_main_process:
                callback()
        except BaseException as exc:
            local_exception = exc
            local_error = (
                f"rank {accelerator.process_index}:"
                f" {type(exc).__name__}: {exc}"
            )
        finally:
            accelerator.wait_for_everyone()

        errors = [error for error in gather_object([local_error]) if error is not None]
        if errors:
            raise RuntimeError(
                f"{operation} failed collectively: {'; '.join(errors)}"
            ) from local_exception

    def _save_training_state(self):
        """Atomically write optimizer/scaler states and counters to a state file."""
        st = self.state
        state = {
            "format_version": 2,
            "optimizers": {name: opt.state_dict() for name, opt in self.optimizers.items()},
            "schedulers": {
                name: scheduler.state_dict()
                for name, scheduler in self.schedulers.items()
            },
            "optimizer_config": self._optimizer_config,
            "scheduler_config": self._scheduler_config(),
            "resume": {
                "version": 2,
                "world_size": self.accelerator.num_processes,
                "ranks": self._resume_states_for_save,
            },
            "scaler": (
                self.accelerator.scaler.state_dict()
                if self.accelerator.scaler is not None
                else None
            ),
            "metadata": {
                "iteration": st.iteration,
                "epoch": st.epoch,
                "rows": st.rows,
            },
        }
        state_path = self._state_path(st.iteration)
        tmp_path = state_path + ".tmp"
        torch.save(state, tmp_path)
        os.replace(tmp_path, state_path)

    def _capture_rng_state(self):
        np_state = np.random.get_state()
        state = {
            "python": random.getstate(),
            "numpy": {
                "bit_generator": np_state[0],
                "state_u32_le": np_state[1].astype("<u4", copy=False).tobytes(),
                "position": np_state[2],
                "has_gauss": np_state[3],
                "cached_gaussian": np_state[4],
            },
            "torch_cpu": bytes(torch.get_rng_state().tolist()),
            "torch_cuda": None,
        }
        if not self.use_cpu and self.accelerator.device.type == "cuda":
            state["torch_cuda"] = bytes(
                torch.cuda.get_rng_state(self.accelerator.device).cpu().tolist()
            )
        return state

    def _restore_rng_state(self, state):
        random.setstate(state["python"])
        np_state = state["numpy"]
        numpy_words = (
            np.frombuffer(np_state["state_u32_le"], dtype="<u4").copy()
            if "state_u32_le" in np_state
            else np_state["state"].numpy().copy()
        )
        np.random.set_state(
            (
                np_state["bit_generator"],
                numpy_words,
                np_state["position"],
                np_state["has_gauss"],
                np_state["cached_gaussian"],
            )
        )
        cpu_rng = state["torch_cpu"]
        if not isinstance(cpu_rng, torch.Tensor):
            cpu_rng = torch.frombuffer(bytearray(cpu_rng), dtype=torch.uint8).clone()
        torch.set_rng_state(cpu_rng)
        cuda_state = state.get("torch_cuda")
        if cuda_state is not None:
            if self.use_cpu or self.accelerator.device.type != "cuda":
                raise RuntimeError(
                    "Cannot restore CUDA RNG state in a CPU training run."
                )
            if not isinstance(cuda_state, torch.Tensor):
                cuda_state = torch.frombuffer(
                    bytearray(cuda_state), dtype=torch.uint8
                ).clone()
            torch.cuda.set_rng_state(cuda_state, self.accelerator.device)

    def _local_resume_state(self, *, vq_runtime=None):
        reason = getattr(
            self,
            "_exact_resume_reason",
            "trainer continuation state is not initialized",
        )
        if reason is not None:
            state = {
                "supported": False,
                "reason": reason,
                "slot": self.accelerator.process_index,
                "rank": self.accelerator.process_index,
            }
            if vq_runtime is not None:
                state["vq_runtime"] = vq_runtime
            state["global_digest"] = hashlib.sha256(
                canonical_pipeline_state_bytes(
                    self._global_resume_projection(state)
                )
            ).hexdigest()
            state["local_digest"] = hashlib.sha256(
                canonical_pipeline_state_bytes(state)
            ).hexdigest()
            return state
        state = {
            "supported": True,
            "slot": self.accelerator.process_index,
            "rank": self.accelerator.process_index,
            "data_stream_signature": self._data_stream_signature,
            "rng": self._capture_rng_state(),
        }
        if getattr(self, "_resume_stream", None) is not None:
            identity = self.train_dataset.runtime_context.rank_local_identity
            state.update(
                {
                    "cursor_kind": "file_stream_v1",
                    "stream": self._resume_stream.state_dict(),
                    "slice": [identity.slice_start, identity.slice_stop],
                }
            )
            if hasattr(self._resume_stream, "coordination_state_dict"):
                coordination = self._resume_stream.coordination_state_dict()
                if coordination is not None:
                    state["stream_coordination"] = coordination
        else:
            state.update(
                {
                    "cursor_kind": "map_sampler_v1",
                    "sampler": self._resume_sampler.state_dict(),
                }
            )
        if vq_runtime is not None:
            state["vq_runtime"] = vq_runtime
        tuning_state_method = getattr(
            getattr(self, "train_dataset", None),
            "pipeline_tuning_state_dict",
            None,
        )
        if tuning_state_method is not None:
            tuning_state = tuning_state_method()
            if tuning_state is not None:
                state["pipeline_tuning"] = tuning_state
        state["global_digest"] = hashlib.sha256(
            canonical_pipeline_state_bytes(
                self._global_resume_projection(state)
            )
        ).hexdigest()
        state["local_digest"] = hashlib.sha256(
            canonical_pipeline_state_bytes(state)
        ).hexdigest()
        return state

    def _validate_vq_runtime_manifest(
        self,
        manifest,
        *,
        iteration=None,
        rank=None,
    ):
        if not isinstance(manifest, dict):
            raise RuntimeError("VQ runtime manifest must be a dictionary")
        required_keys = {
            "version",
            "schema",
            "autocast_mode",
            "modules",
            "has_partial",
            "raw_bytes",
            "maximum_bytes",
            "basename",
            "sha256",
            "size",
            "iteration",
            "generation",
            "rank",
        }
        if set(manifest) != required_keys:
            raise RuntimeError(
                "VQ runtime manifest has missing or unexpected fields"
            )
        if manifest.get("version") != _RUNTIME_MANIFEST_VERSION:
            raise RuntimeError(
                f"unsupported VQ runtime manifest {manifest.get('version')!r}"
            )
        current_autocast_mode = self._vq_runtime_autocast_mode()
        if manifest.get("autocast_mode") != current_autocast_mode:
            raise RuntimeError(
                "VQ runtime autocast mode differs from checkpoint:"
                f" saved={manifest.get('autocast_mode')!r},"
                f" current={current_autocast_mode!r}"
            )
        if iteration is None:
            iteration = self.state.iteration
        if rank is None:
            rank = self.accelerator.process_index
        if type(iteration) is not int or iteration < 0:
            raise RuntimeError(f"invalid VQ runtime iteration {iteration!r}")
        if type(rank) is not int or rank < 0:
            raise RuntimeError(f"invalid VQ runtime rank {rank!r}")
        generation = manifest.get("generation")
        basename = manifest.get("basename")
        match = (
            _RUNTIME_SIDECAR_RE.fullmatch(basename)
            if isinstance(basename, str)
            else None
        )
        if (
            match is None
            or int(match.group("iteration")) != iteration
            or int(match.group("rank")) != rank
            or match.group("generation") != generation
            or manifest.get("iteration") != iteration
            or manifest.get("rank") != rank
            or re.fullmatch(r"[0-9a-f]{32}", generation or "") is None
            or re.fullmatch(r"[0-9a-f]{64}", manifest.get("sha256") or "")
            is None
        ):
            raise RuntimeError(
                f"VQ runtime manifest path identity is invalid: {basename!r}"
            )

        discovered = self._vq_runtime_modules()
        current_schema = self._vq_runtime_schema(discovered)
        if current_schema != manifest.get("schema"):
            raise RuntimeError(
                "VQ runtime module/configuration schema differs from checkpoint"
            )
        descriptor_by_key = {
            (model["model"], module["path"]): module
            for model in current_schema
            for module in model["modules"]
        }
        module_stats = manifest.get("modules")
        if not isinstance(module_stats, list):
            raise RuntimeError("VQ runtime manifest module statistics are invalid")
        stats_by_key = {}
        raw_bytes = 0
        maximum_raw_bytes = 0
        has_partial = False
        for module_stat in module_stats:
            if (
                not isinstance(module_stat, dict)
                or set(module_stat)
                != {"model", "path", "dtype", "count", "raw_bytes"}
            ):
                raise RuntimeError(
                    "VQ runtime manifest module statistics are invalid"
                )
            key = (module_stat.get("model"), module_stat.get("path"))
            if key in stats_by_key or key not in descriptor_by_key:
                raise RuntimeError(
                    "VQ runtime manifest has duplicate or unexpected modules"
                )
            dtype_bytes = _RUNTIME_DTYPE_BYTES.get(module_stat.get("dtype"))
            count = module_stat.get("count")
            module_raw_bytes = module_stat.get("raw_bytes")
            descriptor = descriptor_by_key[key]
            target = descriptor["descriptor"]["target_samples"]
            dim_feature = descriptor["descriptor"]["dim_feature"]
            expected_dtype = _expected_vq_runtime_dtype(
                descriptor["descriptor"],
                count,
                current_autocast_mode,
            )
            if (
                dtype_bytes is None
                or type(count) is not int
                or not 0 <= count <= target
                or type(module_raw_bytes) is not int
                or module_raw_bytes != count * dim_feature * dtype_bytes
                or (descriptor["inited"] and count != 0)
                or module_stat.get("dtype") != expected_dtype
            ):
                raise RuntimeError(
                    f"VQ runtime manifest statistics are invalid for"
                    f" {key[0]!r}:{key[1]!r}"
                )
            stats_by_key[key] = module_stat
            raw_bytes += module_raw_bytes
            maximum_raw_bytes += target * dim_feature * dtype_bytes
            has_partial |= count > 0
        if set(stats_by_key) != set(descriptor_by_key):
            raise RuntimeError(
                "VQ runtime manifest has missing or unexpected modules"
            )
        expected_maximum = (
            maximum_raw_bytes
            + _runtime_serialization_allowance(maximum_raw_bytes)
        )
        if (
            manifest.get("has_partial") is not has_partial
            or manifest.get("raw_bytes") != raw_bytes
            or manifest.get("maximum_bytes") != expected_maximum
            or type(manifest.get("size")) is not int
            or manifest["size"] < 0
        ):
            raise RuntimeError("VQ runtime manifest storage bounds are invalid")
        return discovered, stats_by_key, expected_maximum

    def _restore_vq_runtime_sidecar(self, manifest):
        iteration = self.state.iteration
        rank = self.accelerator.process_index
        discovered, stats_by_key, expected_maximum = (
            self._validate_vq_runtime_manifest(
                manifest,
                iteration=iteration,
                rank=rank,
            )
        )
        basename = manifest.get("basename")
        directory_fd, file_fd, file_stat = self._open_runtime_file(
            basename
        )
        try:
            expected_size = manifest.get("size")
            maximum_size = manifest.get("maximum_bytes")
            raw_bytes = manifest.get("raw_bytes")
            if (
                type(expected_size) is not int
                or type(maximum_size) is not int
                or type(raw_bytes) is not int
                or file_stat.st_size != expected_size
                or maximum_size != expected_maximum
                or file_stat.st_size > expected_maximum
                or file_stat.st_size
                > raw_bytes + _runtime_serialization_allowance(raw_bytes)
            ):
                raise RuntimeError(
                    f"VQ runtime sidecar {basename} has invalid size"
                )
            digest = hashlib.sha256()
            while chunk := os.read(file_fd, 1024 * 1024):
                digest.update(chunk)
            if digest.hexdigest() != manifest.get("sha256"):
                raise RuntimeError(
                    f"VQ runtime sidecar {basename} SHA-256 differs"
                )
            os.lseek(file_fd, 0, os.SEEK_SET)
            with os.fdopen(os.dup(file_fd), "rb") as input_file:
                payload = torch.load(
                    input_file,
                    map_location="cpu",
                    weights_only=True,
                )
        finally:
            os.close(file_fd)
            os.close(directory_fd)

        if (
            not isinstance(payload, dict)
            or set(payload)
            != {
                "version",
                "models",
                "schema",
                "autocast_mode",
                "iteration",
                "world_size",
                "rank",
                "generation",
            }
            or payload.get("version") != _RUNTIME_SIDECAR_VERSION
            or payload.get("iteration") != iteration
            or payload.get("world_size") != self.accelerator.num_processes
            or payload.get("rank") != rank
            or payload.get("generation") != manifest.get("generation")
            or payload.get("schema") != manifest.get("schema")
            or payload.get("autocast_mode")
            != manifest.get("autocast_mode")
        ):
            raise RuntimeError(
                f"VQ runtime sidecar {basename} metadata differs"
            )
        expected_entries = {
            (entry["model"], entry["path"]): entry
            for entry in discovered
        }
        payload_models = payload.get("models")
        if not isinstance(payload_models, dict):
            raise RuntimeError("VQ runtime sidecar model map is invalid")
        expected_model_names = {
            model_name for model_name, _ in expected_entries
        }
        if set(payload_models) != expected_model_names or any(
            not isinstance(modules, dict)
            for modules in payload_models.values()
        ):
            raise RuntimeError(
                "VQ runtime sidecar has missing or unexpected models"
            )
        actual_keys = {
            (model_name, path)
            for model_name, modules in payload_models.items()
            for path in modules
        }
        if actual_keys != set(expected_entries):
            raise RuntimeError(
                "VQ runtime sidecar has missing or unexpected modules"
            )
        for key, entry in expected_entries.items():
            model_name, path = key
            saved = payload_models[model_name][path]
            if (
                not isinstance(saved, dict)
                or set(saved) != {"aliases", "state"}
                or saved.get("aliases") != entry["aliases"]
            ):
                raise RuntimeError(
                    f"VQ runtime aliases differ for {model_name!r}:{path!r}"
                )
            saved_state = saved.get("state")
            module_stat = stats_by_key[key]
            samples = (
                saved_state.get("samples")
                if isinstance(saved_state, dict)
                else None
            )
            if (
                not isinstance(samples, torch.Tensor)
                or saved_state.get("dtype") != module_stat["dtype"]
                or saved_state.get("count") != module_stat["count"]
                or samples.numel() * samples.element_size()
                != module_stat["raw_bytes"]
            ):
                raise RuntimeError(
                    f"VQ runtime payload statistics differ for"
                    f" {model_name!r}:{path!r}"
                )
            try:
                entry["module"]._load_vq_init_continuation_state(
                    saved_state
                )
            except BaseException as exc:
                raise RuntimeError(
                    f"VQ runtime restore failed for {model_name!r}:{path!r}:"
                    f" {type(exc).__name__}: {exc}"
                ) from exc

    def _restore_vq_runtime_collectively(self, manifest):
        local_error = None
        try:
            self._restore_vq_runtime_sidecar(manifest)
        except BaseException as exc:
            local_error = (
                f"rank {self.accelerator.process_index}:"
                f" {type(exc).__name__}: {exc}"
            )
        errors = (
            [local_error]
            if self.accelerator.num_processes == 1
            else gather_object([local_error])
        )
        failures = [error for error in errors if error is not None]
        if failures:
            raise RuntimeError(
                "VQ runtime restore failed collectively: "
                + "; ".join(failures)
            )

    def _restore_continuation_state(self):
        if getattr(self, "_pending_resume_state", None) is None:
            return
        if getattr(self, "_pending_resume_version", 1) == 2:
            self._restore_vq_runtime_collectively(
                self._pending_resume_state["vq_runtime"]
            )
        cursor_kind = self._pending_resume_state.get("cursor_kind")
        if cursor_kind == "file_stream_v1":
            if self._resume_stream is None:
                raise RuntimeError("Cannot restore file-stream cursor into a non-stream dataset.")
            self._resume_stream.load_state_dict(self._pending_resume_state["stream"])
            identity = self.train_dataset.runtime_context.rank_local_identity
            if self._pending_resume_state.get("slice") != [
                identity.slice_start,
                identity.slice_stop,
            ]:
                raise RuntimeError("Cannot restore file stream: rank-local slice changed.")
            self.state.epoch = self._resume_stream.committed_epoch
        elif cursor_kind in {None, "map_sampler_v1"}:
            self._resume_sampler.load_state_dict(
                self._pending_resume_state["sampler"]
            )
            if self._resume_sampler.epoch != self.state.epoch:
                raise RuntimeError(
                    "Cannot resume: sampler epoch does not match checkpoint metadata."
                )
        else:
            raise RuntimeError(
                f"Cannot restore unknown cursor kind {cursor_kind!r}."
            )
        tuning_state = self._pending_resume_state.get("pipeline_tuning")
        if tuning_state is not None:
            restore_tuning_state = getattr(
                self.train_dataset,
                "restore_pipeline_tuning_state_dict",
                None,
            )
            if restore_tuning_state is not None:
                restore_tuning_state(tuning_state)
        self._restore_rng_state(self._pending_resume_state["rng"])
        self._pending_resume_state = None
        self._pending_resume_version = None

    def _validate_retained_vq_runtime_manifest(
        self,
        manifest,
        *,
        iteration,
        rank,
    ):
        if not isinstance(manifest, dict):
            raise RuntimeError("VQ runtime manifest must be a dictionary")
        required_keys = {
            "version",
            "schema",
            "autocast_mode",
            "modules",
            "has_partial",
            "raw_bytes",
            "maximum_bytes",
            "basename",
            "sha256",
            "size",
            "iteration",
            "generation",
            "rank",
        }
        if set(manifest) != required_keys:
            raise RuntimeError(
                "VQ runtime manifest has missing or unexpected fields"
            )
        generation = manifest.get("generation")
        basename = manifest.get("basename")
        match = (
            _RUNTIME_SIDECAR_RE.fullmatch(basename)
            if isinstance(basename, str)
            else None
        )
        if (
            manifest.get("version") != _RUNTIME_MANIFEST_VERSION
            or match is None
            or int(match.group("iteration")) != iteration
            or int(match.group("rank")) != rank
            or match.group("generation") != generation
            or manifest.get("iteration") != iteration
            or manifest.get("rank") != rank
            or not isinstance(manifest.get("autocast_mode"), str)
            or not manifest["autocast_mode"]
            or re.fullmatch(r"[0-9a-f]{32}", generation or "") is None
            or re.fullmatch(r"[0-9a-f]{64}", manifest.get("sha256") or "")
            is None
        ):
            raise RuntimeError("retained state has malformed VQ runtime identity")

        schema = manifest.get("schema")
        if not isinstance(schema, list):
            raise RuntimeError("retained state has malformed VQ runtime schema")
        descriptor_by_key = {}
        model_names = set()
        descriptor_keys = {
            "codebook_size",
            "dim_feature",
            "kmeans_sample_multiplier",
            "kmeans_iter",
            "use_cosine_sim",
            "use_simvq",
            "convert_to_fp32",
            "parameter_dtype",
            "target_samples",
        }
        for model in schema:
            if (
                not isinstance(model, dict)
                or set(model) != {"model", "modules"}
                or not isinstance(model.get("model"), str)
                or not model["model"]
                or model["model"] in model_names
                or not isinstance(model.get("modules"), list)
            ):
                raise RuntimeError(
                    "retained state has malformed VQ runtime schema"
                )
            model_names.add(model["model"])
            for module in model["modules"]:
                if (
                    not isinstance(module, dict)
                    or set(module)
                    != {"path", "aliases", "descriptor", "inited"}
                    or not isinstance(module.get("path"), str)
                    or not isinstance(module.get("aliases"), list)
                    or not module["aliases"]
                    or module["aliases"][0]
                    != {"model": model["model"], "path": module["path"]}
                    or any(
                        not isinstance(alias, dict)
                        or set(alias) != {"model", "path"}
                        or not isinstance(alias["model"], str)
                        or not alias["model"]
                        or not isinstance(alias["path"], str)
                        for alias in module["aliases"]
                    )
                    or len(
                        {
                            (alias["model"], alias["path"])
                            for alias in module["aliases"]
                            if isinstance(alias, dict)
                            and set(alias) == {"model", "path"}
                        }
                    )
                    != len(module["aliases"])
                    or type(module.get("inited")) is not bool
                    or not isinstance(module.get("descriptor"), dict)
                    or set(module["descriptor"]) != descriptor_keys
                ):
                    raise RuntimeError(
                        "retained state has malformed VQ runtime schema"
                    )
                descriptor = module["descriptor"]
                if (
                    any(
                        type(descriptor[key]) is not int
                        or descriptor[key] < 0
                        for key in (
                            "codebook_size",
                            "dim_feature",
                            "kmeans_sample_multiplier",
                            "kmeans_iter",
                            "target_samples",
                        )
                    )
                    or any(
                        type(descriptor[key]) is not bool
                        for key in (
                            "use_cosine_sim",
                            "use_simvq",
                            "convert_to_fp32",
                        )
                    )
                    or descriptor["parameter_dtype"]
                    not in _RUNTIME_DTYPE_BYTES
                ):
                    raise RuntimeError(
                        "retained state has malformed VQ runtime descriptor"
                    )
                key = (model["model"], module["path"])
                if key in descriptor_by_key:
                    raise RuntimeError(
                        "retained state has duplicate VQ runtime modules"
                    )
                descriptor_by_key[key] = module

        module_stats = manifest.get("modules")
        if not isinstance(module_stats, list):
            raise RuntimeError(
                "retained state has malformed VQ runtime statistics"
            )
        stats_by_key = {}
        raw_bytes = 0
        maximum_raw_bytes = 0
        has_partial = False
        for module_stat in module_stats:
            if (
                not isinstance(module_stat, dict)
                or set(module_stat)
                != {"model", "path", "dtype", "count", "raw_bytes"}
            ):
                raise RuntimeError(
                    "retained state has malformed VQ runtime statistics"
                )
            key = (module_stat.get("model"), module_stat.get("path"))
            descriptor_entry = descriptor_by_key.get(key)
            dtype_bytes = _RUNTIME_DTYPE_BYTES.get(module_stat.get("dtype"))
            count = module_stat.get("count")
            module_raw_bytes = module_stat.get("raw_bytes")
            if descriptor_entry is None or key in stats_by_key:
                raise RuntimeError(
                    "retained state has missing, duplicate, or unexpected"
                    " VQ runtime modules"
                )
            descriptor = descriptor_entry["descriptor"]
            expected_dtype = _expected_vq_runtime_dtype(
                descriptor,
                count,
                manifest["autocast_mode"],
            )
            if (
                dtype_bytes is None
                or type(count) is not int
                or not 0 <= count <= descriptor["target_samples"]
                or type(module_raw_bytes) is not int
                or module_raw_bytes
                != count * descriptor["dim_feature"] * dtype_bytes
                or (descriptor_entry["inited"] and count != 0)
                or module_stat.get("dtype") != expected_dtype
            ):
                raise RuntimeError(
                    "retained state has invalid VQ runtime storage statistics"
                )
            stats_by_key[key] = module_stat
            raw_bytes += module_raw_bytes
            maximum_raw_bytes += (
                descriptor["target_samples"]
                * descriptor["dim_feature"]
                * dtype_bytes
            )
            has_partial |= count > 0
        maximum_bytes = (
            maximum_raw_bytes
            + _runtime_serialization_allowance(maximum_raw_bytes)
        )
        size = manifest.get("size")
        if (
            set(stats_by_key) != set(descriptor_by_key)
            or manifest.get("has_partial") is not has_partial
            or manifest.get("raw_bytes") != raw_bytes
            or manifest.get("maximum_bytes") != maximum_bytes
            or type(size) is not int
            or size < 0
            or size > maximum_bytes
            or size > raw_bytes + _runtime_serialization_allowance(raw_bytes)
        ):
            raise RuntimeError(
                "retained state has invalid VQ runtime storage bounds"
            )

        directory_fd, file_fd, file_stat = self._open_runtime_file(basename)
        try:
            if file_stat.st_size != size:
                raise RuntimeError(
                    f"retained VQ runtime sidecar {basename!r} has wrong size"
                )
            digest = hashlib.sha256()
            while chunk := os.read(file_fd, 1024 * 1024):
                digest.update(chunk)
            if digest.hexdigest() != manifest["sha256"]:
                raise RuntimeError(
                    f"retained VQ runtime sidecar {basename!r} SHA-256 differs"
                )
        finally:
            os.close(file_fd)
            os.close(directory_fd)
        return basename

    def _validate_current_vq_runtime_collision(
        self,
        manifest,
        *,
        iteration,
    ):
        """Require a committed local sidecar to equal the current VQ cache."""
        rank = self.accelerator.process_index
        discovered, _, expected_maximum = self._validate_vq_runtime_manifest(
            manifest,
            iteration=iteration,
            rank=rank,
        )
        basename = manifest["basename"]
        directory_fd, file_fd, file_stat = self._open_runtime_file(
            basename
        )
        try:
            if (
                file_stat.st_size != manifest["size"]
                or file_stat.st_size > expected_maximum
                or file_stat.st_size
                > manifest["raw_bytes"]
                + _runtime_serialization_allowance(manifest["raw_bytes"])
            ):
                raise RuntimeError(
                    f"committed VQ runtime sidecar {basename!r} has wrong size"
                )
            digest = hashlib.sha256()
            while chunk := os.read(file_fd, 1024 * 1024):
                digest.update(chunk)
            if digest.hexdigest() != manifest["sha256"]:
                raise RuntimeError(
                    f"committed VQ runtime sidecar {basename!r} SHA-256 differs"
                )
            os.lseek(file_fd, 0, os.SEEK_SET)
            with os.fdopen(os.dup(file_fd), "rb") as input_file:
                payload = torch.load(
                    input_file,
                    map_location="cpu",
                    weights_only=True,
                )
        finally:
            os.close(file_fd)
            os.close(directory_fd)

        if (
            not isinstance(payload, dict)
            or set(payload)
            != {
                "version",
                "models",
                "schema",
                "autocast_mode",
                "iteration",
                "world_size",
                "rank",
                "generation",
            }
            or payload.get("version") != _RUNTIME_SIDECAR_VERSION
            or payload.get("schema") != manifest["schema"]
            or payload.get("autocast_mode")
            != manifest["autocast_mode"]
            or payload.get("iteration") != iteration
            or payload.get("world_size")
            != self.accelerator.num_processes
            or payload.get("rank") != rank
            or payload.get("generation") != manifest["generation"]
            or not isinstance(payload.get("models"), dict)
        ):
            raise RuntimeError(
                "committed VQ runtime metadata differs from current attempt"
            )

        expected_entries = {
            (entry["model"], entry["path"]): entry
            for entry in discovered
        }
        payload_models = payload["models"]
        if (
            set(payload_models)
            != {model_name for model_name, _ in expected_entries}
            or any(
                not isinstance(modules, dict)
                for modules in payload_models.values()
            )
            or {
                (model_name, path)
                for model_name, modules in payload_models.items()
                for path in modules
            }
            != set(expected_entries)
        ):
            raise RuntimeError(
                "committed VQ runtime modules differ from current attempt"
            )
        for key, entry in expected_entries.items():
            model_name, path = key
            saved = payload_models[model_name][path]
            if (
                not isinstance(saved, dict)
                or set(saved) != {"aliases", "state"}
                or saved.get("aliases") != entry["aliases"]
                or not self._checkpoint_values_equal(
                    saved.get("state"),
                    entry["module"]._vq_init_continuation_state(),
                )
            ):
                raise RuntimeError(
                    f"checkpoint state collision at iteration {iteration}:"
                    f" committed VQ runtime {model_name!r}:{path!r} differs"
                )

    def _cleanup_checkpoints(self):
        """Prune old state files beyond ``num_keep_states`` and temporary model
        snapshots saved by this session that are no longer needed.

        Only snapshots recorded in ``_temp_snapshot_iters`` (i.e. saved by this
        session at a non-``save_interval`` iteration) are ever deleted, so
        pre-existing snapshots survive even if ``save_interval`` changed
        between runs.
        """
        if not os.path.isdir(self.ckpt_dir):
            return

        filenames = os.listdir(self.ckpt_dir)
        state_files = []  # (iteration, path)
        model_groups = {}  # iteration -> [paths]
        for fn in filenames:
            path = os.path.join(self.ckpt_dir, fn)
            if not os.path.isfile(path):
                continue
            iteration = get_iteration_from_ckpt_filename(fn)
            if iteration is None:
                continue
            if fn.startswith("state_"):
                state_files.append((iteration, path))
            elif fn.startswith("ckpt_"):
                model_groups.setdefault(iteration, []).append(path)

        state_files.sort()
        num_keep = len(state_files) if self.num_keep_states < 0 else self.num_keep_states
        num_delete = max(0, len(state_files) - num_keep)
        retained_states = state_files[num_delete:]
        protected_sidecars = set()
        for iteration, path in retained_states:
            try:
                saved = torch.load(
                    path,
                    map_location="cpu",
                    weights_only=True,
                )
                if (
                    not isinstance(saved, dict)
                    or saved.get("format_version") != 2
                    or int(saved.get("metadata", {}).get("iteration", -1))
                    != iteration
                ):
                    raise RuntimeError(
                        "retained state format or iteration is invalid"
                    )
                resume = saved.get("resume", {})
                if (
                    not isinstance(resume, dict)
                    or resume.get("version") not in {1, 2}
                    or resume.get("world_size")
                    != self.accelerator.num_processes
                    or not isinstance(resume.get("ranks"), list)
                    or len(resume["ranks"])
                    != self.accelerator.num_processes
                ):
                    raise RuntimeError(
                        "retained state resume metadata is invalid"
                    )
                rank_states = self._validate_rank_resume_states(
                    resume["ranks"]
                )
                if resume.get("version") == 2:
                    for rank, rank_state in enumerate(rank_states):
                        manifest = rank_state.get("vq_runtime")
                        basename = self._validate_retained_vq_runtime_manifest(
                            manifest,
                            iteration=iteration,
                            rank=rank,
                        )
                        protected_sidecars.add(basename)
            except BaseException as exc:
                raise RuntimeError(
                    f"cannot validate retained checkpoint state {path}: {exc}"
                ) from exc

        malformed_runtime = [
            filename
            for filename in filenames
            if (
                filename.startswith("runtime_")
                and _RUNTIME_SIDECAR_RE.fullmatch(filename) is None
            )
            or (
                filename.startswith(".runtime-tmp_")
                and _RUNTIME_TEMP_RE.fullmatch(filename) is None
            )
        ]
        if malformed_runtime:
            raise RuntimeError(
                "malformed runtime checkpoint files require manual inspection: "
                + ", ".join(sorted(malformed_runtime))
            )

        for _, path in state_files[:num_delete]:
            os.remove(path)
        kept_state_iters = {it for it, _ in retained_states}

        for filename in filenames:
            path = os.path.join(self.ckpt_dir, filename)
            if (
                _RUNTIME_SIDECAR_RE.fullmatch(filename) is not None
                and filename not in protected_sidecars
            ) or _RUNTIME_TEMP_RE.fullmatch(filename) is not None:
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass

        latest_iter = max(model_groups, default=None)
        for iteration in list(self._temp_snapshot_iters):
            if iteration == latest_iter:
                continue
            if iteration in kept_state_iters:
                continue
            for path in model_groups.get(iteration, []):
                os.remove(path)
            self._temp_snapshot_iters.discard(iteration)

    def _load_eval_checkpoint(self):
        """Load only model weights from checkpoint for evaluation (no optimizer/scaler/state)."""
        model_state_dict, _, metadata = load_torch_ckpt(self.checkpoint)
        self.model.load_state_dict(model_state_dict)
        self._ckpt_metadata = metadata
        epoch = metadata.get("epoch", "?")
        it = metadata.get("iteration", "?")
        self.accelerator.print(f"Loaded from {self.checkpoint}, epoch: {epoch}, it: {it}")

    def _prepare_for_evaluation(self):
        """Wrap trained models and test dataloader with accelerator for distributed evaluation."""
        accelerator = self.accelerator
        model_names = list(self.models.keys())
        to_prepare = [self.models[n] for n in model_names]
        prepared = accelerator.prepare(*to_prepare)
        if len(to_prepare) == 1:
            prepared = (prepared,)
        for i, name in enumerate(model_names):
            self.models[name] = prepared[i]
        self.test_loader = self._prepare_data_loader(
            self.test_loader,
            even_batches=False,
        )

    # ── Training loop ─────────────────────────────────────────────

    def _unwrap(self, model):
        """Strip DDP and ``torch.compile`` wrappers from *model*."""
        m = self.accelerator.unwrap_model(model)
        if is_compiled_module(m):
            m = m._orig_mod
        return m

    def _maybe_compile(self, fn, *, inductor_config=None, **overrides):
        """Compile *fn* with the accelerator's dynamo settings.

        Returns *fn* unchanged when dynamo is disabled, so callers can use the
        result unconditionally.  Use this to pull loss computation (and other
        per-iteration tensor work) into the same compiled graph as the model
        forward instead of running it eagerly.  Explicit overrides are intended
        for model subgraphs that require e.g. dynamic shapes or a full graph;
        backend and mode still come from the Accelerate configuration unless a
        caller deliberately replaces them.
        """
        plugin = self.accelerator.state.dynamo_plugin
        if plugin.backend == DynamoBackend.NO:
            return fn
        compile_kwargs = plugin.to_kwargs()
        compile_kwargs.update(overrides)
        if inductor_config is None:
            inductor_config = self._primary_inductor_config
        compile_kwargs = with_inductor_options(compile_kwargs, inductor_config)
        return torch.compile(fn, **compile_kwargs)

    def _configure_model_compilation(self, model):
        """Inject the configured compiler into optional model subgraphs.

        Models keep eager reference callables when used standalone.  A model
        may implement ``configure_compilation(compile_fn)`` to compile an eager
        graph-break region with the active compile policy and optional
        subgraph-specific overrides.  This runs before Accelerate/DDP wraps
        the model and does not add compiled callables to its state dict.
        """
        configure = getattr(model, "configure_compilation", None)
        if configure is not None:
            inductor_config = model_inductor_config(model)

            def compile_fn(fn, **overrides):
                return self._maybe_compile(
                    fn,
                    inductor_config=inductor_config,
                    **overrides,
                )

            configure(compile_fn)

    @property
    def unwrapped_model(self):
        """Return the raw primary model, stripping DDP and ``torch.compile`` wrappers."""
        return self._unwrap(self.model)

    def _prepare_for_training(self):
        """Wrap trained models, optimizers and dataloaders with accelerator for distributed training."""
        accelerator = self.accelerator
        # Prepare all trained models + optimizers + train dataloader together
        # (backends like DeepSpeed require the dataloader in the joint call;
        # batch-yielding loaders cannot go through accelerate and are wrapped
        # separately by _prepare_data_loader)
        iterable_train = isinstance(self.train_loader.dataset, IterableDataset)
        exact_resume_train = self._resume_sampler is not None
        model_names = list(self.models.keys())
        optimizer_names = list(self.optimizers.keys())
        to_prepare = [self.models[n] for n in model_names]
        to_prepare.extend(self.optimizers[n] for n in optimizer_names)
        if not iterable_train and not exact_resume_train:
            to_prepare.append(self.train_loader)
        prepared = accelerator.prepare(*to_prepare)
        for i, name in enumerate(model_names):
            self.models[name] = prepared[i]
        for i, name in enumerate(optimizer_names):
            self.optimizers[name] = prepared[len(model_names) + i]
        if iterable_train or exact_resume_train:
            self.train_loader = self._prepare_data_loader(self.train_loader)
        else:
            self.train_loader = prepared[-1]
        if self.val_loader is not None:
            self.val_loader = self._prepare_data_loader(
                self.val_loader,
                even_batches=False,
            )
        self._setup_weight_clipping()

    def _prepare_data_loader(self, dataloader, *, even_batches=True):
        """Prepare *dataloader* for this process: shard across ranks and move to device.

        Iterable datasets partition work by rank and worker internally, so
        they only need device placement via :class:`DeviceLoaderWrapper`.
        Accelerate remains the sharding owner for map-style datasets.
        Batch-producing built-in streams already plan global batches and expose
        one stable local slice per rank.
        """
        if (
            isinstance(dataloader.dataset, IterableDataset)
            or (
                dataloader is getattr(self, "train_loader", None)
                and getattr(self, "_resume_sampler", None) is not None
            )
        ):
            pipeline_stats = getattr(dataloader.dataset, "pipeline_stats", None)
            is_training_loader = dataloader is getattr(self, "train_loader", None)
            if is_training_loader and self.cuda_prefetch_batches > 0:
                return CudaPrefetchLoaderWrapper(
                    dataloader,
                    self.accelerator.device,
                    prefetch_batches=self.cuda_prefetch_batches,
                    pipeline_stats=pipeline_stats,
                )
            if pipeline_stats is not None:
                return ObservedDeviceLoaderWrapper(
                    dataloader,
                    self.accelerator.device,
                    pipeline_stats,
                )
            return DeviceLoaderWrapper(dataloader, self.accelerator.device)

        previous_even_batches = self.accelerator.even_batches
        self.accelerator.even_batches = even_batches
        try:
            prepared = self.accelerator.prepare_data_loader(dataloader)
        finally:
            self.accelerator.even_batches = previous_even_batches

        if not even_batches and self.accelerator.num_processes > 1:
            if not isinstance(prepared.batch_sampler, BatchSamplerShard):
                raise RuntimeError(
                    "Expected Accelerate to install BatchSamplerShard for a"
                    " distributed evaluation loader."
                )
            # The underlying sampler still enforces the configured sample-level
            # drop_last. The outer shard must retain a complete unpaired batch.
            prepared.batch_sampler.drop_last = False
        return prepared

    def _validate_cuda_prefetch_support(self, dataloader):
        """Reject enabled lookahead where device placement is owned elsewhere."""
        supported = (
            isinstance(dataloader.dataset, IterableDataset)
            or getattr(self, "_resume_sampler", None) is not None
        )
        if self.cuda_prefetch_batches > 0 and not supported:
            reason = getattr(self, "_exact_resume_reason", None)
            detail = f": {reason}" if reason else ""
            raise ValueError(
                "cuda_prefetch_batches requires an iterable dataset or the "
                "exact-resume map loader with num_worker=0; Accelerate owns "
                f"device placement for this loader{detail}"
            )

    def _setup_weight_clipping(self):
        """Resolve per-model weight-clipping groups once for the training loop."""
        self._weight_clip_groups = []
        for m in self.models.values():
            unwrapped = self._unwrap(m)
            if hasattr(unwrapped, "weight_clipping"):
                self._weight_clip_groups.extend(
                    resolve_weight_clipping(unwrapped.named_parameters(), unwrapped.weight_clipping)
                )

    def _all_trained_parameters(self):
        """Yield all parameters from all trained models."""
        from itertools import chain
        return chain(*(m.parameters() for m in self.models.values()))

    def _clip_gradients(self, parameters=None):
        """Measure gradient norm and apply configured clipping."""
        if parameters is None:
            parameters = list(self._all_trained_parameters())
        else:
            parameters = list(parameters)
        max_norm = (
            self.clip_grad_norm
            if self.clip_grad_norm is not None
            else float("inf")
        )
        grad_norm = self.accelerator.clip_grad_norm_(
            parameters,
            max_norm=max_norm,
        )
        if self.clip_grad_value is not None:
            self.accelerator.clip_grad_value_(
                parameters,
                clip_value=self.clip_grad_value,
            )
        return grad_norm

    def _normalize_train_metrics(self, metric_dict, namespace):
        normalized = {}
        schema = []
        errors = []
        if not isinstance(metric_dict, dict):
            return {}, [], [
                f"{namespace} metrics must be a dict, got"
                f" {type(metric_dict).__name__}"
            ]
        for key, value in metric_dict.items():
            if not isinstance(key, str):
                errors.append(f"{namespace} metric key {key!r} is not a string")
                continue
            if isinstance(value, torch.Tensor):
                if (
                    value.ndim != 0
                    or value.dtype == torch.bool
                    or value.is_complex()
                    or not _devices_match(
                        value.device,
                        self.accelerator.device,
                    )
                ):
                    errors.append(
                        f"{namespace}[{key!r}] must be a real scalar tensor,"
                        f" got shape={tuple(value.shape)}, dtype={value.dtype},"
                        f" device={value.device}; expected device"
                        f" {self.accelerator.device}"
                    )
                    continue
                tensor = value
            elif isinstance(value, (SumCount, Maximum, SufficientStats)):
                tensor_values = (
                    [value.sum]
                    if isinstance(value, SumCount)
                    else (
                        [value.value]
                        if isinstance(value, Maximum)
                        else list(value.tensors.values())
                    )
                )
                if any(
                    not isinstance(item, torch.Tensor)
                    or item.dtype == torch.bool
                    or item.is_complex()
                    or not _devices_match(item.device, self.accelerator.device)
                    for item in tensor_values
                ):
                    errors.append(
                        f"{namespace}[{key!r}] has invalid sufficient-statistic tensors"
                    )
                    continue
                normalized[key] = value
                schema.append(
                    (namespace, key, type(value).__name__, value.scope)
                )
                continue
            elif isinstance(value, numbers.Real) and not isinstance(value, bool):
                tensor = torch.as_tensor(value, device=self.accelerator.device)
            else:
                errors.append(
                    f"{namespace}[{key!r}] must be a real number or scalar"
                    f" tensor, got {type(value).__name__}"
                )
                continue
            normalized[key] = tensor
            schema.append((namespace, key, str(tensor.dtype), tensor.device.type))
        return dict(sorted(normalized.items())), sorted(schema), errors

    def _synchronize_step_validity(
        self,
        *,
        loss,
        metric_values,
        schema,
        errors,
        phase,
        require_grad=True,
        defer_finite=False,
    ):
        accelerator = self.accelerator
        expected_schema = getattr(self, "_train_metric_schema", None)
        if expected_schema is None:
            rank_schemas = (
                [schema]
                if accelerator.num_processes == 1
                else gather_object([schema])
            )
            if any(item != rank_schemas[0] for item in rank_schemas[1:]):
                errors.append(
                    "metric schema differs across ranks: "
                    + "; ".join(
                        f"rank {rank}={item!r}"
                        for rank, item in enumerate(rank_schemas)
                    )
                )
            self._train_metric_schema = schema
        elif schema != expected_schema:
            errors.append(
                f"metric schema changed from {expected_schema!r} to {schema!r}"
            )

        valid_loss = (
            isinstance(loss, torch.Tensor)
            and loss.ndim == 0
            and loss.is_floating_point()
            and (loss.requires_grad or not require_grad)
            and _devices_match(loss.device, accelerator.device)
        )
        if not valid_loss:
            errors.append(
                "loss must be a scalar tensor with floating dtype,"
                + (
                    " requires_grad=True,"
                    if require_grad
                    else ""
                )
                + " and accelerator device; got"
                f" {type(loss).__name__}"
                + (
                    (
                        f" shape={tuple(loss.shape)}, dtype={loss.dtype},"
                        f" device={loss.device}"
                    )
                    if isinstance(loss, torch.Tensor)
                    else ""
                )
            )
        if defer_finite:
            if accelerator.num_processes != 1:
                raise RuntimeError(
                    "finite validation may only be deferred in a single-process run"
                )
            if errors:
                self._known_divergent = True
                raise RuntimeError(
                    "Training step validation failed: "
                    f"iteration {self.state.iteration} rank"
                    f" {accelerator.process_index} {phase}: "
                    + "; ".join(errors)
                )
            return

        if accelerator.num_processes == 1 and errors:
            self._known_divergent = True
            raise RuntimeError(
                "Training step validation failed: "
                f"iteration {self.state.iteration} rank"
                f" {accelerator.process_index} {phase}: "
                + "; ".join(errors)
            )

        checks = []
        if accelerator.num_processes > 1:
            checks.append(
                torch.tensor(
                    not errors,
                    dtype=torch.bool,
                    device=accelerator.device,
                )
            )
        if valid_loss:
            checks.append(torch.isfinite(loss.detach()))
        else:
            checks.append(
                torch.tensor(False, dtype=torch.bool, device=accelerator.device)
            )
        checks.extend(torch.isfinite(value.detach()) for value in metric_values)
        local_valid = torch.stack(checks).all().to(dtype=torch.long)
        global_valid_count = (
            local_valid
            if accelerator.num_processes == 1
            else accelerator.reduce(local_valid, reduction="sum")
        )
        if int(global_valid_count.item()) == accelerator.num_processes:
            return
        if not errors:
            errors.append("non-finite loss, metric, or gradient norm")
        rank_error = (
            f"iteration {self.state.iteration} rank"
            f" {accelerator.process_index} {phase}: {'; '.join(errors)}"
        )
        all_errors = (
            [rank_error]
            if accelerator.num_processes == 1
            else [
                item
                for item in gather_object([rank_error if errors else None])
                if item is not None
            ]
        )
        self._known_divergent = True
        raise RuntimeError(
            "Training step validation failed: " + "; ".join(all_errors)
        )

    def _synchronize_phase_errors(self, phase, errors, *, divergent=False):
        """Raise rank-local phase failures at one matching all-rank boundary."""
        accelerator = self.accelerator
        if accelerator.num_processes == 1:
            if not errors:
                return
            if divergent:
                self._known_divergent = True
            iteration = getattr(getattr(self, "state", None), "iteration", "?")
            raise RuntimeError(
                f"{phase} failed collectively: iteration {iteration} rank"
                f" {accelerator.process_index} {phase}: {'; '.join(errors)}"
            )
        local_ok = torch.tensor(
            not errors,
            dtype=torch.long,
            device=accelerator.device,
        )
        global_ok = accelerator.reduce(local_ok, reduction="sum")
        if int(global_ok.item()) == accelerator.num_processes:
            return
        iteration = getattr(getattr(self, "state", None), "iteration", "?")
        rank_error = (
            f"iteration {iteration} rank"
            f" {accelerator.process_index} {phase}: {'; '.join(errors)}"
            if errors
            else None
        )
        all_errors = (
            [rank_error]
            if accelerator.num_processes == 1
            else [
                item
                for item in gather_object([rank_error])
                if item is not None
            ]
        )
        if divergent:
            self._known_divergent = True
        raise RuntimeError(f"{phase} failed collectively: {'; '.join(all_errors)}")

    def _run_train_step(self, data):
        """Execute one training iteration: forward, backward, optimizer step.

        With ``gradient_accumulation_steps > 1``, consumes additional
        micro-batches from the train iterator; the optimizer steps once per
        call, so one call always equals one optimizer step.

        The default implementation assumes a single loss optimized by all
        optimizers in one backward pass.  Paradigms with alternating
        optimization (e.g. GANs) should override this method; helpers like
        :meth:`_clip_gradients` and the hooks remain reusable.
        """
        accelerator = self.accelerator

        # apply weight clipping if needed (groups resolved in _prepare_for_training)
        if self._weight_clip_groups:
            with self.profiler.region("clip"):
                apply_weight_clipping(self._weight_clip_groups)

        total_loss_dict, total_aux_dict = {}, {}
        final_grad_norm = None
        prepared_stream_commit = None
        prepared_sampler_commit = None
        for micro_step in range(self.gradient_accumulation_steps):
            if micro_step > 0:
                with self.profiler.region("data"):
                    data = self._fetch_batch()
            step_errors = []
            extra_kwargs = {}
            with self.profiler.region("pre"):
                try:
                    extra_kwargs = self.on_before_step(data)
                except BaseException as exc:
                    step_errors.append(
                        f"on_before_step raised {type(exc).__name__}: {exc}"
                    )
            self._synchronize_phase_errors(
                f"micro-step {micro_step} pre-step hook",
                step_errors,
            )
            with accelerator.accumulate(*self.models.values()), accelerator.autocast():
                with self.profiler.region("fwd"):
                    loss = torch.zeros(
                        (),
                        device=accelerator.device,
                        requires_grad=True,
                    )
                    loss_dict, aux_dict = {}, {}
                    if not step_errors:
                        try:
                            loss, loss_dict, aux_dict = self.train_step(
                                data,
                                **extra_kwargs,
                            )
                        except BaseException as exc:
                            step_errors.append(
                                f"train_step raised {type(exc).__name__}: {exc}"
                            )
                    loss_dict, loss_schema, loss_errors = (
                        self._normalize_train_metrics(loss_dict, "loss")
                    )
                    aux_dict, aux_schema, aux_errors = (
                        self._normalize_train_metrics(aux_dict, "aux")
                    )
                    metric_values = list(loss_dict.values()) + list(aux_dict.values())
                    metric_schema = loss_schema + aux_schema
                    # On the final unscaled single-rank micro-step, backward
                    # can be queued before one combined finite-result sync.
                    # Structural failures still raise immediately above.
                    defer_finite = (
                        accelerator.num_processes == 1
                        and accelerator.scaler is None
                        and accelerator.sync_gradients
                    )
                    self._synchronize_step_validity(
                        loss=loss,
                        metric_values=metric_values,
                        schema=metric_schema,
                        errors=step_errors + loss_errors + aux_errors,
                        phase=f"micro-step {micro_step} before backward",
                        defer_finite=defer_finite,
                    )
                with self.profiler.region("bwd"):
                    accelerator.backward(loss)

                mutation_errors = []
                with self.profiler.region("opt"):
                    try:
                        if accelerator.sync_gradients:
                            clear_ddp_preinit_gradients(
                                self._all_trained_parameters()
                            )
                            final_grad_norm = self._clip_gradients()
                            if accelerator.scaler is None:
                                self._synchronize_step_validity(
                                    loss=final_grad_norm,
                                    metric_values=(
                                        [loss, *metric_values]
                                        if defer_finite
                                        else []
                                    ),
                                    schema=metric_schema,
                                    errors=[],
                                    phase="after backward",
                                    require_grad=False,
                                )
                            pending_tokens = getattr(
                                self, "_pending_stream_tokens", []
                            )
                            if pending_tokens:
                                prepared_stream_commit = (
                                    self._resume_stream.prepare_commit(pending_tokens)
                                )
                                descriptor = getattr(
                                    prepared_stream_commit,
                                    "coordination_descriptor",
                                    (
                                        prepared_stream_commit.before_digest,
                                        prepared_stream_commit.after_digest,
                                        prepared_stream_commit.token_count,
                                    ),
                                )
                                descriptors = (
                                    [descriptor]
                                    if accelerator.num_processes == 1
                                    else gather_object([descriptor])
                                )
                                if any(
                                    value != descriptors[0]
                                    for value in descriptors[1:]
                                ):
                                    raise RuntimeError(
                                        "stream commit digest differs across ranks: "
                                        + "; ".join(
                                            f"rank {rank}={value!r}"
                                            for rank, value in enumerate(descriptors)
                                        )
                                    )
                            pending_sampler_tokens = getattr(
                                self, "_pending_sampler_tokens", []
                            )
                            if pending_sampler_tokens:
                                prepared_sampler_commit = (
                                    self._resume_sampler.prepare_commit(
                                        pending_sampler_tokens
                                    )
                                )
                                sampler_descriptor = (
                                    prepared_sampler_commit.before_epoch,
                                    prepared_sampler_commit.before_offset,
                                    prepared_sampler_commit.after_epoch,
                                    prepared_sampler_commit.after_offset,
                                    prepared_sampler_commit.token_count,
                                )
                                sampler_descriptors = (
                                    [sampler_descriptor]
                                    if accelerator.num_processes == 1
                                    else gather_object([sampler_descriptor])
                                )
                                if any(
                                    value != sampler_descriptors[0]
                                    for value in sampler_descriptors[1:]
                                ):
                                    raise RuntimeError(
                                        "sampler commit cursor differs across ranks: "
                                        + "; ".join(
                                            f"rank {rank}={value!r}"
                                            for rank, value in enumerate(
                                                sampler_descriptors
                                            )
                                        )
                                    )

                        # step/zero_grad are skipped internally on non-sync micro-batches
                        for opt in self.optimizers.values():
                            self._optimizer_state_mutated = True
                            opt.step()
                        if accelerator.sync_gradients:
                            # keep an LR unchanged when AMP overflow skipped its
                            # optimizer's step; each optimizer decides overflow
                            # independently
                            for name, sched in self.schedulers.items():
                                if not getattr(
                                    self.optimizers[name],
                                    "step_was_skipped",
                                    False,
                                ):
                                    self._optimizer_state_mutated = True
                                    sched.step()
                        for opt in self.optimizers.values():
                            self._optimizer_state_mutated = True
                            opt.zero_grad(set_to_none=True)
                    except BaseException as exc:
                        mutation_errors.append(
                            f"{type(exc).__name__}: {exc}"
                        )
            self._synchronize_phase_errors(
                f"micro-step {micro_step} optimizer mutation",
                mutation_errors,
                divergent=True,
            )
            add_dict_to(total_loss_dict, loss_dict)
            add_dict_to(total_aux_dict, aux_dict)

        if self.gradient_accumulation_steps > 1:
            for metric_dict in (total_loss_dict, total_aux_dict):
                for k in metric_dict:
                    metric_dict[k] = metric_dict[k] / self.gradient_accumulation_steps
        if self.clip_grad_norm is not None and final_grad_norm is not None:
            total_aux_dict["grad_norm"] = final_grad_norm.detach()

        after_errors = []
        try:
            self._optimizer_state_mutated = True
            self.on_after_step(data)
        except BaseException as exc:
            after_errors.append(f"{type(exc).__name__}: {exc}")
        self._synchronize_phase_errors(
            "post-step hook",
            after_errors,
            divergent=True,
        )
        pending_tokens = getattr(self, "_pending_stream_tokens", [])
        if pending_tokens:
            if prepared_stream_commit is None:
                raise RuntimeError("stream tokens were not prepared before optimizer mutation")
            self._resume_stream.commit_prepared(prepared_stream_commit)
            self.state.epoch = self._resume_stream.committed_epoch
            self._pending_stream_tokens = []
        pending_sampler_tokens = getattr(self, "_pending_sampler_tokens", [])
        if pending_sampler_tokens:
            if prepared_sampler_commit is None:
                raise RuntimeError(
                    "sampler tokens were not prepared before optimizer mutation"
                )
            self._resume_sampler.commit_prepared(prepared_sampler_commit)
            self.state.epoch = self._resume_sampler.epoch
            self._pending_sampler_tokens = []

        return total_loss_dict, total_aux_dict

    def _gather_train_metrics(self, loss_dict, aux_dict, count):
        """Average and transfer all scalar training metrics in one batch."""
        loss_keys = list(loss_dict)
        aux_keys = list(aux_dict)
        values = [
            metric_dict[key] / count
            for metric_dict, keys in (
                (loss_dict, loss_keys),
                (aux_dict, aux_keys),
            )
            for key in keys
        ]
        if not values:
            return {}, {}
        local_values = torch.stack(values)
        gathered = self.accelerator.gather(local_values)
        global_values = gathered.reshape(
            self.accelerator.num_processes,
            len(values),
        ).mean(dim=0)
        host_values = global_values.tolist()
        loss_count = len(loss_keys)
        return (
            dict(zip(loss_keys, host_values[:loss_count])),
            dict(zip(aux_keys, host_values[loss_count:])),
        )

    def run(self):
        """Run the full training loop from ``state.iteration`` to ``iterations``."""
        st = self.state
        accelerator = self.accelerator

        primary_exception = None
        self._checkpoint_safe = False
        self._known_divergent = False
        try:
            accelerator.print(
                f"Start training from iteration {st.iteration}, epoch {st.epoch}"
            )
            self._restore_continuation_state()
            self._train_data_iter = iter(self.train_loader)
            self._start_time = self._log_last_time = self._show_last_time = time.time()
            self._log_last_it = self._show_last_it = st.iteration
            gpu_loss_sum, gpu_aux_sum, gpu_metric_count = {}, {}, 0
            self._train_metric_schema = None
            for m in self.models.values():
                m.train()
            self._checkpoint_safe = True

            while st.iteration < self.iterations:
                self._checkpoint_safe = False
                self._optimizer_state_mutated = False
                torch.compiler.cudagraph_mark_step_begin()

                with self.profiler.region("data"):
                    data = self._fetch_batch()
                st.iteration += 1
                st.rows += self.batch_size * self.gradient_accumulation_steps
                try:
                    loss_dict, aux_dict = self._run_train_step(data)
                except BaseException:
                    st.iteration -= 1
                    st.rows -= self.batch_size * self.gradient_accumulation_steps
                    if self._resume_sampler is not None:
                        self._resume_sampler.rollback_uncommitted()
                        self._pending_sampler_tokens = []
                    if self._optimizer_state_mutated:
                        self._known_divergent = True
                    raise

                add_dict_to(gpu_loss_sum, loss_dict)
                add_dict_to(gpu_aux_sum, aux_dict)
                gpu_metric_count += 1
                self.profiler.mark_iteration()

                if (
                    st.iteration % self.log_interval == 0
                    or st.iteration % self.show_interval == 0
                ):
                    loss_dict, aux_dict = self._gather_train_metrics(
                        gpu_loss_sum,
                        gpu_aux_sum,
                        gpu_metric_count,
                    )
                    gpu_loss_sum, gpu_aux_sum, gpu_metric_count = {}, {}, 0

                if st.iteration % self.log_interval == 0:
                    self._run_synchronized_operation(
                        "training metric logging",
                        lambda: self._log_metrics(loss_dict, aux_dict),
                    )

                if st.iteration % self.show_interval == 0:
                    self._run_synchronized_operation(
                        "progress display",
                        lambda: self._show_progress(loss_dict),
                    )

                if st.iteration % self.val_interval == 0 and self.val_loader is not None:
                    self._validate()

                if self.profiler is NULL_PROFILER:
                    self.profiler.step(st.iteration)
                else:
                    self._run_synchronized_operation(
                        "profiler step",
                        lambda: self.profiler.step(st.iteration),
                    )
                self._checkpoint_safe = True

                # Checkpoint last: its RNG/sampler state is the exact point from
                # which the next iteration will continue.
                if self._is_save_iteration(st.iteration):
                    self._save_checkpoint()

            if self._last_state_save_iteration != st.iteration:
                self._save_checkpoint(force_state=True)
        except BaseException as exc:
            primary_exception = exc
            if (
                self._checkpoint_safe
                and not self._known_divergent
                and not self._checkpoint_attempt_failed
                and self._last_state_save_iteration != st.iteration
            ):
                try:
                    self._save_checkpoint(force_state=True)
                except BaseException as save_exc:
                    accelerator.print(
                        "Final checkpoint save failed without replacing the"
                        f" original {type(exc).__name__}:"
                        f" {type(save_exc).__name__}: {save_exc}"
                    )
            raise
        finally:
            try:
                self._shutdown_resources()
            except BaseException as shutdown_exc:
                if primary_exception is None:
                    raise
                accelerator.print(
                    "Resource shutdown failed without replacing the original"
                    f" {type(primary_exception).__name__}:"
                    f" {type(shutdown_exc).__name__}: {shutdown_exc}"
                )

    def _shutdown_resources(self):
        """Close final resources coherently across all ranks."""

        def close_local_resources():
            resources = []
            train_data_iter = getattr(self, "_train_data_iter", None)
            if train_data_iter is not None and hasattr(train_data_iter, "close"):
                resources.append(("training data iterator", train_data_iter))
            resources.append(("profiler", self.profiler))
            if self.accelerator.is_main_process:
                resources.extend(
                    [
                        ("TensorBoard logger", self.tb_logger),
                        ("training log", self.log_file),
                    ]
                )
            failures = []
            for name, resource in resources:
                try:
                    resource.close()
                except BaseException as exc:
                    failures.append(f"{name}: {type(exc).__name__}: {exc}")
            if failures:
                raise RuntimeError("; ".join(failures))

        self._run_synchronized_operation(
            "final resource shutdown",
            close_local_resources,
        )

    def profile(self, *, wait=0, warmup=10, active=30, profile_memory=False):
        """Run a short profiling loop and save trace to rundir.

        Uses torch.profiler.profile directly. Runs (wait + warmup + active) iterations
        then prints a summary table and saves the trace.
        """
        total_iters = wait + warmup + active
        accelerator = self.accelerator

        def on_trace_ready(p: torch.profiler.profile):
            if accelerator.is_main_process:
                dev_name = "cpu" if self.use_cpu else "cuda"
                print(p.key_averages().table(sort_by=f"{dev_name}_time_total", row_limit=16))
                print("Profile finished. Saving trace...")
                tracefile_path = os.path.join(self.rundir, "profile_trace")
                torch.profiler.tensorboard_trace_handler(tracefile_path, use_gzip=True)(p)
                print(f"Profile trace saved at {tracefile_path}.")

        schedule = torch.profiler.schedule(
            wait=wait, warmup=warmup, active=active, repeat=1
        )
        activities = [torch.profiler.ProfilerActivity.CPU]
        if not self.use_cpu:
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        self._train_data_iter = iter(self.train_loader)
        for m in self.models.values():
            m.train()

        with torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            on_trace_ready=on_trace_ready,
            record_shapes=True,
            profile_memory=profile_memory,
            with_stack=True,
            with_flops=True,
            with_modules=True,
        ) as profiler:
            for _ in range(total_iters):
                torch.compiler.cudagraph_mark_step_begin()
                data = self._fetch_batch()
                self._run_train_step(data)
                profiler.step()

        accelerator.print("Profiling complete.")

    def _fetch_batch(self):
        """Get the next collectively available batch, restarting at epoch boundaries."""
        data = None
        is_available = True
        fetch_error = None
        try:
            data = next(self._train_data_iter)
        except StopIteration:
            if getattr(self, "_resume_sampler", None) is not None:
                self._resume_sampler.advance_yield_epoch()
            try:
                self._train_data_iter = iter(self.train_loader)
                data = next(self._train_data_iter)
            except StopIteration:
                is_available = False
            except BaseException as exc:
                is_available = False
                fetch_error = f"{type(exc).__name__}: {exc}"
        except BaseException as exc:
            is_available = False
            fetch_error = f"{type(exc).__name__}: {exc}"

        fetch_errors = (
            [fetch_error]
            if self.accelerator.num_processes == 1
            else gather_object([fetch_error])
        )
        fetch_errors = [
            f"rank {rank}: {error}"
            for rank, error in enumerate(fetch_errors)
            if error is not None
        ]
        if fetch_errors:
            self._known_divergent = True
            raise RuntimeError(
                "training batch fetch failed collectively: " + "; ".join(fetch_errors)
            )

        if self.accelerator.num_processes == 1:
            empty_ranks = [] if is_available else [0]
        else:
            availability = torch.tensor(
                [is_available],
                dtype=torch.bool,
                device=self.accelerator.device,
            )
            availability = self.accelerator.gather(availability)
            empty_ranks = [
                rank
                for rank, available in enumerate(availability.tolist())
                if not available
            ]
        if empty_ranks:
            raise RuntimeError(
                "Training data availability check failed: no batch is available on"
                f" rank(s) {empty_ranks} of {self.accelerator.num_processes};"
                f" dataset_type={self.dataset_type!r}, train_datas={self.train_datas!r},"
                f" batch_size_per_process={self.batch_size_per_process},"
                f" num_workers={self.num_worker}. Check dataset partitioning and"
                " drop_last settings."
            )
        if getattr(self, "_resume_sampler", None) is not None:
            if not hasattr(self, "_pending_sampler_tokens"):
                self._pending_sampler_tokens = []
            self._pending_sampler_tokens.append(
                self._resume_sampler.stage_batch(self.batch_size_per_process)
            )
        from dataset.stream import BatchEnvelope
        if isinstance(data, BatchEnvelope):
            token_descriptor = getattr(
                data.token,
                "coordination_descriptor",
                (
                    data.token.epoch,
                    data.token.batch_index,
                    data.token.before_digest,
                    data.token.after_digest,
                ),
            )
            token_descriptors = (
                [token_descriptor]
                if self.accelerator.num_processes == 1
                else gather_object([token_descriptor])
            )
            if any(value != token_descriptors[0] for value in token_descriptors[1:]):
                self._known_divergent = True
                raise RuntimeError(
                    "training stream token differs across ranks: "
                    + "; ".join(
                        f"rank {rank}={value!r}"
                        for rank, value in enumerate(token_descriptors)
                    )
                )
            if not hasattr(self, "_pending_stream_tokens"):
                self._pending_stream_tokens = []
            self._pending_stream_tokens.append(data.token)
            data = data.data
        return data

    def _flush_pipeline_metrics(self, consumer_batches_s):
        dataset = getattr(getattr(self, "train_loader", None), "dataset", None)
        if getattr(dataset, "pipeline_stats", None) is None:
            return {}, None
        local_snapshot = dataset.pipeline_metrics_snapshot()
        snapshots = (
            [local_snapshot]
            if self.accelerator.num_processes == 1
            else gather_object([local_snapshot])
        )
        from dataset.telemetry import aggregate_pipeline_snapshots

        metrics = aggregate_pipeline_snapshots(
            snapshots,
            consumer_batches_s=consumer_batches_s,
            rows_per_batch=self.batch_size_per_process,
        )
        tuning_state_method = getattr(dataset, "pipeline_tuning_state_dict", None)
        current_state = None if tuning_state_method is None else tuning_state_method()
        event = None
        if current_state is not None:
            updated_state = None
            if self.accelerator.is_main_process:
                updated_state = dataset.pipeline_tuning_update(
                    metrics,
                    self.state.iteration,
                )
            if self.accelerator.num_processes > 1:
                values = [updated_state]
                torch.distributed.broadcast_object_list(values, src=0)
                updated_state = values[0]
            if updated_state is not None:
                if not self.accelerator.is_main_process:
                    dataset.load_pipeline_tuning_state_dict(updated_state)
                decisions = updated_state.get("decisions", [])
                if (
                    decisions
                    and decisions[-1].get("iteration") == self.state.iteration
                ):
                    event = decisions[-1]
            current_state = tuning_state_method()
            for name, value in current_state["settings"].items():
                metrics[f"autotune/{name}"] = float(value)
            metrics["autotune/frozen"] = float(current_state["frozen"])
        return metrics, event

    def _log_metrics(self, loss_dict, aux_dict):
        """Write training metrics to TensorBoard and the JSONL log file (main process only)."""
        # flush on every process: it clears the profiler's recording buffers
        prof_stats = self.profiler.flush_timings()
        st = self.state
        if not self.accelerator.is_main_process:
            return

        log_value_dict(self.tb_logger, "train", loss_dict, st.iteration, st.rows)
        if aux_dict:
            log_value_dict(self.tb_logger, "train_aux", aux_dict, st.iteration, st.rows)

        iters_per_second = (st.iteration - self._log_last_it) / (
            time.time() - self._log_last_time
        )
        elapsed_time = time.time() - self._start_time
        running_stat_dict = {
            "epoch": st.epoch,
            "rows": st.rows,
            "elapsed": elapsed_time,
            "it/s": iters_per_second,
            "entry/s": iters_per_second
            * self.batch_size
            * self.gradient_accumulation_steps,
        }
        for name, sched in self.schedulers.items():
            lr_key = "lr" if name == "main" else f"lr_{name}"
            running_stat_dict[lr_key] = sched.get_last_lr()[0]
        running_stat_dict.update(prof_stats)
        log_value_dict(
            self.tb_logger,
            "running_stat",
            running_stat_dict,
            st.iteration,
            st.rows,
        )
        self._log_last_it = st.iteration
        self._log_last_time = time.time()

        json_log_dict = {
            "it": st.iteration,
            "train_loss": loss_dict,
            "train_aux": aux_dict,
            "running_stat": running_stat_dict,
        }
        self.log_file.write(json.dumps(json_log_dict) + "\n")
        self.log_file.flush()

    def _log_metrics_observed(self, loss_dict, aux_dict):
        """Write training metrics to TensorBoard and the JSONL log file (main process only)."""
        # flush on every process: it clears the profiler's recording buffers
        prof_stats = self.profiler.flush_timings()
        st = self.state
        now = time.time()
        iters_per_second = (st.iteration - self._log_last_it) / max(
            1e-12, now - self._log_last_time
        )
        pipeline_stats, tuning_event = self._flush_pipeline_metrics(
            iters_per_second * self.gradient_accumulation_steps
        )
        if not self.accelerator.is_main_process:
            self._log_last_it = st.iteration
            self._log_last_time = now
            return

        log_value_dict(self.tb_logger, "train", loss_dict, st.iteration, st.rows)
        if aux_dict:
            log_value_dict(self.tb_logger, "train_aux", aux_dict, st.iteration, st.rows)

        elapsed_time = now - self._start_time
        running_stat_dict = {
            "epoch": st.epoch,
            "rows": st.rows,
            "elapsed": elapsed_time,
            "it/s": iters_per_second,
            "entry/s": iters_per_second * self.batch_size * self.gradient_accumulation_steps,
        }
        for name, sched in self.schedulers.items():
            lr_key = "lr" if name == "main" else f"lr_{name}"
            running_stat_dict[lr_key] = sched.get_last_lr()[0]
        running_stat_dict.update(prof_stats)
        log_value_dict(self.tb_logger, "running_stat", running_stat_dict, st.iteration, st.rows)
        if pipeline_stats:
            log_value_dict(
                self.tb_logger,
                "data_pipeline",
                pipeline_stats,
                st.iteration,
                st.rows,
            )
        self._log_last_it = st.iteration
        self._log_last_time = now

        json_log_dict = {
            "it": st.iteration,
            "train_loss": loss_dict,
            "train_aux": aux_dict,
            "running_stat": running_stat_dict,
        }
        if pipeline_stats:
            json_log_dict["data_pipeline"] = pipeline_stats
        if tuning_event is not None:
            json_log_dict["data_pipeline_tuning"] = tuning_event
        self.log_file.write(json.dumps(json_log_dict) + "\n")
        self.log_file.flush()

    def _loss_summary(self, loss_dict: dict) -> str:
        """One-line summary of *loss_dict* for progress prints.

        Subclasses may override to include paradigm-specific loss components.
        """
        return f"{loss_dict['total_loss']:.4f}"

    def _show_progress(self, loss_dict):
        """Print a one-line training progress summary to stdout (main process only)."""
        st = self.state
        if not self.accelerator.is_local_main_process:
            return

        iters_per_second = (st.iteration - self._show_last_it) / (time.time() - self._show_last_time)
        elapsed_time = time.time() - self._start_time
        eta_time = (self.iterations - st.iteration) / iters_per_second
        print(
            f"Iter: {st.iteration}/{self.iterations} ({st.iteration/self.iterations*100:.2f}%)"
            f" | Elapsed: {format_time(elapsed_time)}"
            f" | Speed: {iters_per_second:.2f} it/s"
            f" | ETA: {format_time(eta_time)}"
            f" | Loss: {self._loss_summary(loss_dict)}",
            flush=True,
        )
        self._show_last_it = st.iteration
        self._show_last_time = time.time()

    def _gather_averaged_metrics(
        self,
        metric_dict,
        num_batches,
        *,
        empty_context="evaluation data",
    ):
        """Sum *metric_dict* across processes and average over the global batch count.

        Values may be 0-dim tensors or Python scalars.  Must be called on all
        processes (it performs a collective gather).  Returns
        ``(averaged_dict, total_batches)`` on the main process and
        ``(None, None)`` on other processes.
        """
        accelerator = self.accelerator
        total_batches_tensor = accelerator.reduce(
            torch.tensor(
                [num_batches],
                dtype=torch.long,
                device=accelerator.device,
            ),
            reduction="sum",
        )
        total_batches = int(total_batches_tensor.item())
        if total_batches == 0:
            raise RuntimeError(
                f"No global batches were produced for {empty_context};"
                " cannot average metrics."
            )

        gathered = {}
        finalized_stats = {}
        for k, v in sorted(metric_dict.items()):
            if isinstance(v, SumCount):
                total_sum = accelerator.reduce(v.sum, reduction="sum")
                total_count = accelerator.reduce(
                    torch.tensor(v.count, dtype=torch.long, device=accelerator.device),
                    reduction="sum",
                )
                finalized_stats[k] = total_sum / total_count.to(total_sum.dtype)
                continue
            if isinstance(v, Maximum):
                candidate = (
                    v.value
                    if v.valid_count
                    else torch.full_like(v.value, -torch.inf)
                )
                maximum = accelerator.reduce(candidate, reduction="max")
                valid_count = accelerator.reduce(
                    torch.tensor(
                        v.valid_count, dtype=torch.long, device=accelerator.device
                    ),
                    reduction="sum",
                )
                if int(valid_count.item()) == 0:
                    raise RuntimeError(f"evaluation maximum {k!r} has no values")
                finalized_stats[k] = maximum
                continue
            if isinstance(v, SufficientStats):
                tensors = {
                    name: accelerator.reduce(value, reduction="sum")
                    for name, value in sorted(v.tensors.items())
                }
                counts = {
                    name: int(
                        accelerator.reduce(
                            torch.tensor(
                                count, dtype=torch.long, device=accelerator.device
                            ),
                            reduction="sum",
                        ).item()
                    )
                    for name, count in sorted(v.counts.items())
                }
                finalized_stats[k] = SufficientStats(
                    v.scope, v.finalizer_id, tensors, counts
                ).finalize()
                continue
            if not isinstance(v, torch.Tensor):
                v = torch.tensor([v], dtype=torch.float32, device=accelerator.device)
            gathered[k] = v
        gathered = accelerator.gather(gathered) if gathered else {}
        if not accelerator.is_main_process:
            return None, None
        averaged = {
            k: (
                torch.max(v).item()
                if k in {"value_abserr_max", "value_relerr_max"}
                else torch.sum(v).item() / total_batches
            )
            for k, v in gathered.items()
        }
        averaged.update(
            {
                key: value.item() if isinstance(value, torch.Tensor) else value
                for key, value in finalized_stats.items()
            }
        )
        return averaged, total_batches

    @staticmethod
    def _accumulate_evaluation_metrics(totals, values):
        for key, value in values.items():
            if isinstance(value, (SumCount, Maximum, SufficientStats)) and key in totals:
                if isinstance(totals[key], type(value)):
                    totals[key] = totals[key].combine(value)
                elif totals[key] == 0:
                    totals[key] = value
                else:
                    raise TypeError(
                        f"evaluation statistic type changed for {key!r}"
                    )
            elif key in {"value_abserr_max", "value_relerr_max"} and key in totals:
                if isinstance(value, torch.Tensor):
                    totals[key] = torch.maximum(totals[key], value)
                else:
                    totals[key] = max(totals[key], value)
            elif key in totals:
                totals[key] = totals[key] + value
            else:
                totals[key] = value

    @staticmethod
    def _zero_evaluation_stat(value):
        """Return a schema-preserving empty contribution for a replayed batch."""
        if isinstance(value, SumCount):
            return SumCount(
                value.scope, torch.zeros_like(value.sum), 0
            )
        if isinstance(value, Maximum):
            return Maximum("evaluation", value.value, 0)
        if isinstance(value, SufficientStats):
            return SufficientStats(
                value.scope,
                value.finalizer_id,
                {
                    name: torch.zeros_like(tensor)
                    for name, tensor in value.tensors.items()
                },
                {name: 0 for name in value.counts},
            )
        if isinstance(value, torch.Tensor):
            return torch.zeros_like(value)
        return 0.0

    def _preflight_evaluation_schema(
        self, *, state_attr=None, **namespaces
    ):
        """Reach schema consensus before any key-dependent numerical collective."""
        schema = []
        errors = []
        for namespace, values in sorted(namespaces.items()):
            if not isinstance(values, dict):
                errors.append(
                    f"{namespace} must be a dict, got {type(values).__name__}"
                )
                continue
            for key, value in sorted(values.items()):
                if not isinstance(key, str):
                    errors.append(f"{namespace} key {key!r} is not a string")
                    continue
                if isinstance(value, SumCount):
                    tensors = {"sum": value.sum}
                    descriptor = (
                        namespace,
                        key,
                        "SumCount",
                        value.scope,
                        None,
                        ("count",),
                    )
                elif isinstance(value, Maximum):
                    tensors = {"value": value.value}
                    descriptor = (
                        namespace,
                        key,
                        "Maximum",
                        value.scope,
                        None,
                        ("valid_count",),
                    )
                elif isinstance(value, SufficientStats):
                    tensors = value.tensors
                    descriptor = (
                        namespace,
                        key,
                        "SufficientStats",
                        value.scope,
                        value.finalizer_id,
                        tuple(sorted(value.counts)),
                    )
                else:
                    errors.append(
                        f"{namespace}[{key!r}] is an opaque "
                        f"{type(value).__name__}"
                    )
                    continue
                tensor_schema = []
                for tensor_key, tensor in sorted(tensors.items()):
                    if (
                        not isinstance(tensor_key, str)
                        or not isinstance(tensor, torch.Tensor)
                        or tensor.dtype == torch.bool
                        or tensor.is_complex()
                        or not _devices_match(
                            tensor.device, self.accelerator.device
                        )
                    ):
                        errors.append(
                            f"{namespace}[{key!r}].{tensor_key!r} is not a "
                            "real tensor on the evaluation device"
                        )
                        continue
                    tensor_schema.append(
                        (
                            tensor_key,
                            tuple(tensor.shape),
                            str(tensor.dtype),
                            tensor.device.type,
                        )
                    )
                schema.append(descriptor + (tuple(tensor_schema),))
        payload = (tuple(schema), tuple(errors))
        rank_payloads = (
            [payload]
            if self.accelerator.num_processes == 1
            else gather_object([payload])
        )
        all_errors = [
            f"rank {rank}: {error}"
            for rank, (_, rank_errors) in enumerate(rank_payloads)
            for error in rank_errors
        ]
        reference = rank_payloads[0][0]
        if any(rank_schema != reference for rank_schema, _ in rank_payloads[1:]):
            all_errors.append(
                "evaluation statistic schema differs across ranks: "
                + "; ".join(
                    f"rank {rank}={rank_schema!r}"
                    for rank, (rank_schema, _) in enumerate(rank_payloads)
                )
            )
        if not all_errors and state_attr is not None:
            expected = getattr(self, state_attr, None)
            if expected is None:
                setattr(self, state_attr, reference)
            elif reference != expected:
                all_errors.append(
                    "evaluation statistic schema changed across batches: "
                    f"expected={expected!r}; current={reference!r}"
                )
        if all_errors:
            raise RuntimeError(
                "Evaluation statistic preflight failed: "
                + " | ".join(all_errors)
            )

    def _finalize_masked_global_batch(
        self, metric_dict, data, *, include_metrics=True
    ):
        """Reduce typed rank-local statistics and close one global batch."""
        finalized = {}
        for key, value in sorted(metric_dict.items()):
            if isinstance(value, SumCount):
                if value.scope == "evaluation":
                    finalized[key] = (
                        value
                        if include_metrics
                        else SumCount(
                            "evaluation", torch.zeros_like(value.sum), 0
                        )
                    )
                    continue
                local_sum = (
                    value.sum
                    if include_metrics
                    else torch.zeros_like(value.sum)
                )
                local_count = value.count if include_metrics else 0
                global_sum = self.accelerator.reduce(
                    local_sum, reduction="sum"
                )
                global_count = int(
                    self.accelerator.reduce(
                        torch.tensor(
                            local_count,
                            dtype=torch.long,
                            device=self.accelerator.device,
                        ),
                        reduction="sum",
                    ).item()
                )
                global_value = SumCount(
                    "global_batch", global_sum, global_count
                ).finalize()
                finalized[key] = SumCount(
                    "evaluation", global_value, 1
                )
                continue
            if isinstance(value, SufficientStats):
                if value.scope == "evaluation":
                    finalized[key] = (
                        value
                        if include_metrics
                        else SufficientStats(
                            "evaluation",
                            value.finalizer_id,
                            {
                                name: torch.zeros_like(tensor)
                                for name, tensor in value.tensors.items()
                            },
                            {name: 0 for name in value.counts},
                        )
                    )
                    continue
                tensors = {
                    name: self.accelerator.reduce(
                        (
                            tensor
                            if include_metrics
                            else torch.zeros_like(tensor)
                        ),
                        reduction="sum",
                    )
                    for name, tensor in sorted(value.tensors.items())
                }
                counts = {
                    name: int(
                        self.accelerator.reduce(
                            torch.tensor(
                                count if include_metrics else 0,
                                dtype=torch.long,
                                device=self.accelerator.device,
                            ),
                            reduction="sum",
                        ).item()
                    )
                    for name, count in sorted(value.counts.items())
                }
                global_value = SufficientStats(
                    "global_batch",
                    value.finalizer_id,
                    tensors,
                    counts,
                ).finalize()
                finalized[key] = SumCount(
                    "evaluation", global_value, 1
                )
                continue
            if isinstance(value, Maximum):
                finalized[key] = (
                    value
                    if include_metrics
                    else Maximum(
                        "evaluation", value.value, 0
                    )
                )
                continue
            raise ValueError(
                f"evaluation metric {key!r} is an opaque "
                f"{type(value).__name__}; producers must return SumCount, "
                "Maximum, or registered SufficientStats"
            )
        return finalized

    def _as_global_batch_statistics(
        self,
        metric_dict,
        data,
        *,
        maximum_keys=(),
    ):
        """Reject opaque scalars instead of guessing their reduction algebra."""
        typed = {}
        for key, value in metric_dict.items():
            if isinstance(value, (SumCount, Maximum, SufficientStats)):
                typed[key] = value
            else:
                raise ValueError(
                    f"built-in evaluation output {key!r} is opaque; producers "
                    "must return SumCount, Maximum, or registered SufficientStats"
                )
        return typed

    def _vq_eval_modules(self):
        """Return trained-model modules that expose deferred eval perplexity stats."""
        modules = []
        for model in self.models.values():
            for module in self._unwrap(model).modules():
                if hasattr(module, "reset_eval_perplexity_stats"):
                    modules.append(module)
        return modules

    def _reset_vq_eval_stats(self):
        self._active_vq_eval_modules = self._vq_eval_modules()
        for module in self._active_vq_eval_modules:
            module.reset_eval_perplexity_stats()

    def _set_vq_eval_stats_enabled(self, enabled: bool):
        for module in self._active_vq_eval_modules:
            module.set_eval_perplexity_stats_enabled(enabled)

    def _global_vq_eval_metrics(self):
        """Reduce code counts after uneven forwards and compute nonlinear global metrics."""
        perplexities = []
        normalized_perplexities = []
        for module in self._active_vq_eval_modules:
            cluster_size = self.accelerator.reduce(
                module._eval_cluster_size,
                reduction="sum",
            )
            if int(cluster_size.sum().item()) == 0:
                continue
            perplexity, normalized = module.eval_perplexity_from_cluster_size(
                cluster_size
            )
            perplexities.append(perplexity)
            normalized_perplexities.append(normalized)
        if not perplexities:
            return {}
        return {
            "vq_perplexity": torch.stack(perplexities).mean().item(),
            "vq_normed_perplexity": (
                torch.stack(normalized_perplexities).mean().item()
            ),
        }

    def _iter_collective_safe_batches(self, dataloader, *, max_batches=None):
        """Yield equally many forwards per rank and identify each rank's real batches."""
        from dataset.stream import BatchEnvelope
        planned_dataset = getattr(dataloader, "dataset", None)
        if bool(
            getattr(
                planned_dataset,
                "yields_batches",
                getattr(planned_dataset, "YIELDS_BATCHES", False),
            )
        ):
            iterator = None
            iterator_errors = []
            try:
                iterator = iter(dataloader)
            except BaseException as exc:
                iterator_errors.append(f"{type(exc).__name__}: {exc}")
            self._synchronize_phase_errors(
                "planned evaluation iterator creation", iterator_errors
            )
            batch_index = 0
            while max_batches is None or batch_index < max_batches:
                envelope = None
                is_available = True
                fetch_errors = []
                try:
                    envelope = next(iterator)
                except StopIteration:
                    is_available = False
                except BaseException as exc:
                    is_available = False
                    fetch_errors.append(f"{type(exc).__name__}: {exc}")
                self._synchronize_phase_errors(
                    f"planned evaluation batch {batch_index} fetch",
                    fetch_errors,
                )
                availability = torch.tensor(
                    [is_available],
                    dtype=torch.bool,
                    device=self.accelerator.device,
                )
                if self.accelerator.num_processes > 1:
                    availability = self.accelerator.gather(availability)
                available_ranks = [
                    rank
                    for rank, available in enumerate(availability.tolist())
                    if available
                ]
                if not available_ranks:
                    return
                if len(available_ranks) != self.accelerator.num_processes:
                    raise RuntimeError(
                        "planned evaluation availability differs across ranks: "
                        f"available ranks {available_ranks}"
                    )
                if isinstance(envelope, BatchEnvelope):
                    data = dict(envelope.data)
                    data["is_real"] = envelope.is_real
                    yield data, True
                else:
                    yield envelope, True
                batch_index += 1
            return

        if self.accelerator.num_processes == 1:
            data_iter = iter(dataloader)
            num_batches = 0
            while True:
                if max_batches is not None and num_batches >= max_batches:
                    return
                try:
                    data = next(data_iter)
                except StopIteration:
                    return
                yield data, True
                num_batches += 1

        iterator_errors = []
        data_iter = None
        try:
            data_iter = iter(dataloader)
        except BaseException as exc:
            iterator_errors.append(f"{type(exc).__name__}: {exc}")
        self._synchronize_phase_errors(
            "validation iterator creation",
            iterator_errors,
        )
        last_data = None
        batch_index = 0
        num_real_batches = 0
        while True:
            is_available = True
            fetch_errors = []
            if max_batches is not None and num_real_batches >= max_batches:
                is_available = False
            else:
                try:
                    data = next(data_iter)
                except StopIteration:
                    is_available = False
                except BaseException as exc:
                    is_available = False
                    fetch_errors = [f"{type(exc).__name__}: {exc}"]
            self._synchronize_phase_errors(
                f"validation batch {batch_index} fetch",
                fetch_errors,
            )

            availability = torch.tensor(
                [is_available],
                dtype=torch.bool,
                device=self.accelerator.device,
            )
            if self.accelerator.num_processes > 1:
                availability = self.accelerator.gather(availability)
            available_ranks = [
                rank for rank, available in enumerate(availability.tolist()) if available
            ]
            if not available_ranks:
                return
            if batch_index == 0 and len(available_ranks) != self.accelerator.num_processes:
                candidate_batches = gather_object(
                    [
                        send_to_device(data, torch.device("cpu"))
                        if is_available
                        else None
                    ]
                )
                if not is_available:
                    data = send_to_device(
                        next(batch for batch in candidate_batches if batch is not None),
                        self.accelerator.device,
                    )

            if is_available:
                last_data = data
                num_real_batches += 1
            else:
                if last_data is None:
                    last_data = data
                data = last_data
            yield data, is_available
            batch_index += 1

    def _validate(self):
        """Run a full validation pass, gather metrics across processes, and log results."""
        st = self.state
        accelerator = self.accelerator

        val_start_time = time.time()
        self._validation_evaluation_schema = None
        val_loss_dict, val_aux_dict = {}, {}
        num_val_batches = 0
        local_val_entries = 0
        def print_validation_start():
            print(
                f"\nValidation at iteration {st.iteration}/{self.iterations}"
                f" ({st.iteration/self.iterations*100:.2f}%)...",
                flush=True,
            )
        self._run_synchronized_operation(
            "validation start output",
            print_validation_start,
            main_process_only=True,
        )

        setup_errors = []
        try:
            for m in self.models.values():
                m.eval()
            self._reset_vq_eval_stats()
        except BaseException as exc:
            setup_errors.append(f"{type(exc).__name__}: {exc}")
        self._synchronize_phase_errors("validation setup", setup_errors)
        with torch.no_grad():
            for val_data, include_metrics in self._iter_collective_safe_batches(
                self.val_loader
            ):
                if include_metrics:
                    local_val_entries += int(
                        val_data["is_real"].sum().item()
                        if isinstance(val_data, dict) and "is_real" in val_data
                        else self.batch_size_per_process * self.eval_bs_multipler
                    )
                step_errors = []
                val_losses, val_auxs = {}, {}
                try:
                    self._set_vq_eval_stats_enabled(include_metrics)
                    val_losses, val_auxs = self.validate_step(val_data)
                except BaseException as exc:
                    step_errors.append(f"{type(exc).__name__}: {exc}")
                self._synchronize_phase_errors(
                    "validation step",
                    step_errors,
                )
                self._preflight_evaluation_schema(
                    state_attr="_validation_evaluation_schema",
                    validation_loss=val_losses,
                    validation_aux=val_auxs,
                )
                val_losses = self._finalize_masked_global_batch(
                    val_losses,
                    val_data,
                    include_metrics=include_metrics,
                )
                val_auxs = self._finalize_masked_global_batch(
                    val_auxs,
                    val_data,
                    include_metrics=include_metrics,
                )
                if not include_metrics:
                    for totals, values in (
                        (val_loss_dict, val_losses),
                        (val_aux_dict, val_auxs),
                    ):
                        for key, value in values.items():
                            if key not in totals:
                                totals[key] = self._zero_evaluation_stat(
                                    value
                                )
                else:
                    self._accumulate_evaluation_metrics(val_loss_dict, val_losses)
                    self._accumulate_evaluation_metrics(val_aux_dict, val_auxs)
                    num_val_batches += 1
        teardown_errors = []
        try:
            self._set_vq_eval_stats_enabled(True)
            for m in self.models.values():
                m.train()
        except BaseException as exc:
            teardown_errors.append(f"{type(exc).__name__}: {exc}")
        self._synchronize_phase_errors("validation teardown", teardown_errors)
        global_vq_metrics = self._global_vq_eval_metrics()

        # gather and average metrics across processes
        val_loss_dict, loss_schema, loss_errors = self._normalize_train_metrics(
            val_loss_dict,
            "validation loss",
        )
        val_aux_dict, aux_schema, aux_errors = self._normalize_train_metrics(
            val_aux_dict,
            "validation aux",
        )
        self._synchronize_phase_errors(
            "validation metric normalization",
            loss_errors + aux_errors,
        )
        metric_schema = loss_schema + aux_schema
        if accelerator.num_processes > 1:
            rank_schemas = gather_object([metric_schema])
            if any(schema != rank_schemas[0] for schema in rank_schemas[1:]):
                raise RuntimeError(
                    "Validation metric schema differs across ranks: "
                    + "; ".join(
                        f"rank {rank}={schema!r}"
                        for rank, schema in enumerate(rank_schemas)
                    )
                )
        validation_dataset_type = getattr(self, "val_dataset_type", None) or getattr(
            self, "dataset_type", "<unknown>"
        )
        validation_context = (
            f"validation dataset_type={validation_dataset_type!r},"
            f" val_datas={getattr(self, 'val_datas', '<unknown>')!r},"
            " batch_size_per_process="
            f"{self.batch_size_per_process * self.eval_bs_multipler}"
        )
        val_loss_dict, total_val_batches = self._gather_averaged_metrics(
            val_loss_dict,
            num_val_batches,
            empty_context=validation_context,
        )
        val_aux_dict, _ = self._gather_averaged_metrics(
            val_aux_dict,
            num_val_batches,
            empty_context=validation_context,
        )
        total_val_entries = int(
            accelerator.reduce(
                torch.tensor(local_val_entries, device=accelerator.device),
                reduction="sum",
            ).item()
        )
        if accelerator.is_main_process:
            for key, value in global_vq_metrics.items():
                val_aux_dict[key] = value
        val_elapsed_time = time.time() - val_start_time

        def log_validation_results():
            elapsed_time = time.time() - self._start_time
            num_val_entries = total_val_entries
            log_value_dict(self.tb_logger, "validation", val_loss_dict, st.iteration, st.rows)
            if val_aux_dict:
                log_value_dict(
                    self.tb_logger, "validation_aux", val_aux_dict, st.iteration, st.rows
                )
            json_log_dict = {
                "it": st.iteration,
                "epoch": st.epoch,
                "elapsed": elapsed_time,
                "val_loss": val_loss_dict,
                "val_aux": val_aux_dict,
                "num_val_entries": num_val_entries,
            }
            self.log_file.write(json.dumps(json_log_dict) + "\n")
            self.log_file.flush()
            print(
                f"Validation finished with {num_val_entries} entries,"
                f" using {format_time(val_elapsed_time)}."
            )
            print(f"Validation loss: {self._loss_summary(val_loss_dict)}")
            print(flush=True)
        self._run_synchronized_operation(
            "validation logging",
            log_validation_results,
            main_process_only=True,
        )

        # subtract validation time from training time on every process:
        # progress prints run on each node's local main
        self._log_last_time += val_elapsed_time
        self._show_last_time += val_elapsed_time

    # ── Hooks for subclasses ──────────────────────────────────────

    def on_before_step(self, data):
        """Called before each training step. Return extra kwargs for train_step."""
        return {}

    def on_after_step(self, data):
        """Called after each optimizer step (e.g. for EMA model updates)."""

    def train_step(self, data, **kwargs):
        """Forward + loss. Must return (loss, loss_dict, aux_dict)."""
        raise NotImplementedError

    def validate_step(self, data, **kwargs):
        """Eval-mode forward + loss. Must return (loss_dict, aux_dict)."""
        raise NotImplementedError

    # ── Evaluation entry point ───────────────────────────────────

    def test(self, *, max_batches=None, result_file=None, result_metadata=None, **kwargs):
        """Run evaluation on the test set, gather metrics, and optionally write results.

        Returns averaged metrics dict on main process, ``None`` on others.
        """
        accelerator = self.accelerator

        for m in self.models.values():
            m.eval()
        self._reset_vq_eval_stats()

        metric_dict = {}
        self._test_evaluation_schema = None
        num_batches = 0
        local_test_entries = 0
        start_time = time.time()
        with torch.no_grad():
            batches = self._iter_collective_safe_batches(
                self.test_loader,
                max_batches=max_batches,
            )
            for data, include_metrics in tqdm(
                batches,
                disable=not accelerator.is_local_main_process,
            ):
                if include_metrics:
                    local_test_entries += int(
                        data["is_real"].sum().item()
                        if isinstance(data, dict) and "is_real" in data
                        else self.batch_size_per_process * self.eval_bs_multipler
                    )
                step_errors = []
                metrics = {}
                try:
                    self._set_vq_eval_stats_enabled(include_metrics)
                    metrics = self.test_step(data, **kwargs)
                except BaseException as exc:
                    step_errors.append(f"{type(exc).__name__}: {exc}")
                self._synchronize_phase_errors("test step", step_errors)
                self._preflight_evaluation_schema(
                    state_attr="_test_evaluation_schema",
                    test_metric=metrics,
                )
                metrics = self._finalize_masked_global_batch(
                    metrics, data, include_metrics=include_metrics
                )
                if not include_metrics:
                    for key, value in metrics.items():
                        if key not in metric_dict:
                            metric_dict[key] = self._zero_evaluation_stat(
                                value
                            )
                    continue
                self._accumulate_evaluation_metrics(metric_dict, metrics)
                num_batches += 1

        self._set_vq_eval_stats_enabled(True)
        global_vq_metrics = self._global_vq_eval_metrics()
        averaged, total_batches = self._gather_averaged_metrics(
            metric_dict,
            num_batches,
            empty_context=(
                f"test dataset_type={getattr(self, 'dataset_type', '<unknown>')!r},"
                f" test_datas={getattr(self, 'test_datas', '<unknown>')!r},"
                " batch_size_per_process="
                f"{self.batch_size_per_process * self.eval_bs_multipler}"
            ),
        )
        total_test_entries = int(
            accelerator.reduce(
                torch.tensor(local_test_entries, device=accelerator.device),
                reduction="sum",
            ).item()
        )

        if accelerator.is_main_process:
            for key, value in global_vq_metrics.items():
                averaged[key] = value
            elapsed = time.time() - start_time
            num_entries = total_test_entries
            print(f"Test finished with {num_entries} entries, in {elapsed:.2f}s.")
            print("Metrics:")
            for k, v in averaged.items():
                print(f"\t{k}: {v:.4f}")

            if result_file is not None:
                self._write_test_results(result_file, averaged, num_entries, result_metadata)
            return averaged
        return None

    def test_step(self, data, **kwargs):
        """Default test step delegates to the typed validation algebra.

        Subclasses may override for custom metric computation.
        """
        loss_dict, aux_dict = self.validate_step(data, **kwargs)
        return {**loss_dict, **aux_dict}

    def _write_test_results(self, result_file, metrics, num_entries, extra_metadata=None):
        """Write evaluation results to a JSON file."""
        log_dict = {
            "checkpoint": self.checkpoint,
            "dataset_type": self.dataset_type,
            "dataset_args": self.dataset_args,
            "num_entries": num_entries,
            "metrics": metrics,
        }
        if extra_metadata:
            log_dict.update(extra_metadata)
        with open(result_file, "w") as f:
            f.write(json.dumps(log_dict, indent=4) + "\n")
