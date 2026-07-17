"""Base trainer: accelerator setup, data, model, optimizer, checkpoint, logging, and training loop."""

import torch
from accelerate import (
    Accelerator,
    DataLoaderConfiguration,
    DistributedDataParallelKwargs,
)
from accelerate.utils import GradientAccumulationPlugin
from accelerate.utils.other import is_compiled_module
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass
import inspect
import json
import time
import os

from tqdm.auto import tqdm

from dataset import build_dataset
from model import build_model
from utils.training_utils import (
    build_lr_scheduler,
    build_optimizer,
    weights_init,
    build_data_loader,
    weight_clipping,
    state_dict_drop_size_unmatched,
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

    All parameters are keyword-only.  Unknown keys (e.g. ``config``) are silently
    absorbed by ``**_extra`` so that ``vars(args)`` can be passed directly.

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
        use_cpu: Force training on CPU even if CUDA is available.
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
        no_shuffle: Disable training data shuffling.
        eval_bs_multipler: Multiplier applied to ``batch_size`` for validation.
        # Model
        model_type: Registered model class name (passed to ``build_model``).
        model_args: Extra keyword arguments forwarded to the model constructor.
        init_cfg: Weight initialization config passed to ``weights_init``.
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
        use_cpu: bool = False,
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
        no_shuffle: bool = False,
        eval_bs_multipler: int = 1,
        # Model
        model_type: str,
        model_args: dict | None = None,
        init_cfg: dict | None = None,
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
        # Extra (absorb unknown CLI keys like 'config')
        **_extra,
    ):
        # Store all parameters
        self.rundir = rundir
        self.iterations = iterations
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.random_seed = random_seed
        self.performance_level = performance_level
        self.use_cpu = use_cpu

        self.dataset_type = dataset_type
        self.train_datas = train_datas
        self.val_datas = val_datas
        self.dataset_args = dataset_args or {}
        self.val_dataset_type = val_dataset_type
        self.val_dataset_args = val_dataset_args or {}
        self.dataloader_args = dataloader_args or {}
        self.data_pipelines = data_pipelines
        self.num_worker = num_worker
        self.no_shuffle = no_shuffle
        self.eval_bs_multipler = eval_bs_multipler

        self.model_type = model_type
        self.model_args = model_args or {}
        self.init_cfg = init_cfg or {}
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
        # Iterations of temporary (non-permanent) snapshots saved by this session;
        # cleanup only ever deletes snapshots recorded here.
        self._temp_snapshot_iters = set()
        self._setup_accelerator()
        self._setup_logging()
        self._setup_data()
        self._init_models()
        self._setup_optimizer()
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
        self.find_unused_parameters = find_unused_parameters

        self._apply_init_defaults()
        self._init_eval_attrs(**extra)
        self._setup_accelerator()
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
                    if default is None and (name.endswith("_args") or name == "init_cfg"):
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

    def _setup_logging(self):
        """Open TensorBoard writer and JSONL log file (main process only)."""
        if self.accelerator.is_main_process:
            self.tb_logger = SummaryWriter(os.path.join(self.rundir, "log"))
            self.log_file = open(os.path.join(self.rundir, "training_log.jsonl"), "a")
        else:
            self.tb_logger, self.log_file = None, None

    def _setup_data(self):
        """Build train and validation datasets and their dataloaders."""
        self.batch_size_per_process = self.batch_size // self.accelerator.num_processes
        shuffle = not self.no_shuffle

        self.train_dataset = build_dataset(
            self.dataset_type,
            self.train_datas,
            shuffle=shuffle,
            pipeline_args=self.data_pipelines,
            **self.dataset_args,
        )
        self.train_loader = build_data_loader(
            self.train_dataset,
            self.batch_size_per_process,
            num_workers=self.num_worker,
            shuffle=shuffle,
            **self.dataloader_args,
        )

        if self.val_datas or self.val_dataset_type:
            self.val_dataset = build_dataset(
                self.val_dataset_type or self.dataset_type,
                self.val_datas,
                shuffle=False,
                pipeline_args=self.data_pipelines,
                **(self.val_dataset_args if self.val_dataset_type else self.dataset_args),
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

    def _setup_test_data(self):
        """Build test dataset and dataloader for evaluation."""
        self.batch_size_per_process = self.batch_size // self.accelerator.num_processes

        self.test_dataset = build_dataset(
            self.dataset_type,
            self.test_datas,
            shuffle=False,
            **self.dataset_args,
        )
        self.test_loader = build_data_loader(
            self.test_dataset,
            self.batch_size_per_process * self.eval_bs_multipler,
            num_workers=self.num_worker,
            shuffle=False,
            **self.dataloader_args,
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
        """Create one LR scheduler per optimizer, resuming from the current iteration."""
        self.schedulers = {
            name: build_lr_scheduler(
                opt,
                self.lr_scheduler_type,
                self.iterations,
                last_it=self.state.iteration - 1,
                **self.lr_scheduler_args,
            )
            for name, opt in self.optimizers.items()
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

    def _model_ckpt_path(self, name: str, iteration: int) -> str:
        return os.path.join(self.ckpt_dir, f"ckpt_{name}_{iteration:07d}.pt")

    def _state_path(self, iteration: int) -> str:
        return os.path.join(self.ckpt_dir, f"state_{iteration:07d}.pt")

    def _load_checkpoint(self):
        """Resume from the latest training state in *ckpt_dir*, falling back to
        legacy merged checkpoints in the rundir root, then to weight init /
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

        self._init_weights()

    def _load_state_checkpoint(self, state_filename):
        """Load models, optimizers, and scaler anchored on *state_filename*."""
        iteration = get_iteration_from_ckpt_filename(state_filename)
        if iteration is None:
            raise RuntimeError(f"Cannot parse iteration from state file {state_filename}")

        for name, m in self._checkpointed_models().items():
            model_ckpt = self._model_ckpt_path(name, iteration)
            if not os.path.isfile(model_ckpt):
                raise FileNotFoundError(
                    f"Training state {state_filename} refers to iteration {iteration},"
                    f" but model checkpoint {model_ckpt} is missing"
                )
            model_state_dict, _, _ = load_torch_ckpt(model_ckpt)
            m.load_state_dict(model_state_dict)

        state = torch.load(state_filename, map_location="cpu", weights_only=True)
        for name, opt_state in state["optimizers"].items():
            self.optimizers[name].load_state_dict(opt_state)
        if self.accelerator.scaler is not None and state.get("scaler") is not None:
            self.accelerator.scaler.load_state_dict(state["scaler"])

        metadata = state["metadata"]
        self.state.iteration = int(metadata.get("iteration", 0))
        self.state.epoch = int(metadata.get("epoch", 0))
        self.state.rows = int(metadata.get("rows", 0))
        self.accelerator.print(f"Loaded from {state_filename}")

    def _load_legacy_checkpoint(self, ckpt_filename):
        """Load a legacy merged checkpoint (models + optimizer in one file)."""
        model_state_dict, training_state_dicts, metadata = load_torch_ckpt(ckpt_filename)
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

    def _init_weights(self):
        """Apply weight init to all trained models, then optional pretrained weights."""
        for m in self.models.values():
            m.apply(weights_init(self.init_cfg))
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

    def _save_checkpoint(self, force_state: bool = False):
        """Save per-model checkpoint files and (if due) the training state, then
        prune temporary snapshots and old state files.

        Args:
            force_state: Save the training state even if ``state_save_interval``
                does not match (used for the final save at end of training).
        """
        st = self.state
        if not self.accelerator.is_main_process:
            return

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

        self._cleanup_checkpoints()

    def _save_training_state(self):
        """Atomically write optimizer/scaler states and counters to a state file."""
        st = self.state
        state = {
            "optimizers": {name: opt.state_dict() for name, opt in self.optimizers.items()},
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

        state_files = []  # (iteration, path)
        model_groups = {}  # iteration -> [paths]
        for fn in os.listdir(self.ckpt_dir):
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
        for _, path in state_files[:num_delete]:
            os.remove(path)
        kept_state_iters = {it for it, _ in state_files[num_delete:]}

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
        to_prepare.append(self.test_loader)
        prepared = accelerator.prepare(*to_prepare)
        for i, name in enumerate(model_names):
            self.models[name] = prepared[i]
        self.test_loader = prepared[len(model_names)]

    # ── Training loop ─────────────────────────────────────────────

    def _unwrap(self, model):
        """Strip DDP and ``torch.compile`` wrappers from *model*."""
        m = self.accelerator.unwrap_model(model)
        if is_compiled_module(m):
            m = m._orig_mod
        return m

    @property
    def unwrapped_model(self):
        """Return the raw primary model, stripping DDP and ``torch.compile`` wrappers."""
        return self._unwrap(self.model)

    def _prepare_for_training(self):
        """Wrap trained models, optimizers and dataloaders with accelerator for distributed training."""
        accelerator = self.accelerator
        # Prepare all trained models + optimizers + train dataloader together
        model_names = list(self.models.keys())
        optimizer_names = list(self.optimizers.keys())
        to_prepare = [self.models[n] for n in model_names]
        to_prepare.extend(self.optimizers[n] for n in optimizer_names)
        to_prepare.append(self.train_loader)
        prepared = accelerator.prepare(*to_prepare)
        # Unpack: models first, then optimizers, then train_loader
        for i, name in enumerate(model_names):
            self.models[name] = prepared[i]
        for i, name in enumerate(optimizer_names):
            self.optimizers[name] = prepared[len(model_names) + i]
        self.train_loader = prepared[len(model_names) + len(optimizer_names)]
        if self.val_loader is not None:
            self.val_loader = accelerator.prepare_data_loader(self.val_loader)

    def _all_trained_parameters(self):
        """Yield all parameters from all trained models."""
        from itertools import chain
        return chain(*(m.parameters() for m in self.models.values()))

    def _clip_gradients(self, parameters=None):
        """Apply gradient norm/value clipping to *parameters* if configured.

        Defaults to all trained parameters.  Reusable by subclasses that
        override :meth:`_run_train_step`.
        """
        if parameters is None:
            parameters = self._all_trained_parameters()
        if self.clip_grad_norm is not None:
            self.accelerator.clip_grad_norm_(parameters, max_norm=self.clip_grad_norm)
        elif self.clip_grad_value is not None:
            self.accelerator.clip_grad_value_(parameters, clip_value=self.clip_grad_value)

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

        # apply weight clipping if needed (check all trained models)
        for m in self.models.values():
            unwrapped = self._unwrap(m)
            if hasattr(unwrapped, "weight_clipping"):
                weight_clipping(unwrapped.named_parameters(), unwrapped.weight_clipping)

        total_loss_dict, total_aux_dict = {}, {}
        for micro_step in range(self.gradient_accumulation_steps):
            if micro_step > 0:
                data = self._fetch_batch()
            extra_kwargs = self.on_before_step(data)
            with accelerator.accumulate(*self.models.values()), accelerator.autocast():
                loss, loss_dict, aux_dict = self.train_step(data, **extra_kwargs)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    self._clip_gradients()

                # step/zero_grad are skipped internally on non-sync micro-batches
                for opt in self.optimizers.values():
                    opt.step()
                if accelerator.sync_gradients:
                    # keep an LR unchanged when AMP overflow skipped its optimizer's
                    # step; each optimizer decides overflow independently
                    for name, sched in self.schedulers.items():
                        if not getattr(self.optimizers[name], "step_was_skipped", False):
                            sched.step()
                for opt in self.optimizers.values():
                    opt.zero_grad(set_to_none=True)
            add_dict_to(total_loss_dict, loss_dict)
            add_dict_to(total_aux_dict, aux_dict)

        if self.gradient_accumulation_steps > 1:
            for metric_dict in (total_loss_dict, total_aux_dict):
                for k in metric_dict:
                    metric_dict[k] = metric_dict[k] / self.gradient_accumulation_steps

        self.on_after_step(data)

        return total_loss_dict, total_aux_dict

    def run(self):
        """Run the full training loop from ``state.iteration`` to ``iterations``."""
        st = self.state
        accelerator = self.accelerator

        accelerator.print(f"Start training from iteration {st.iteration}, epoch {st.epoch}")
        self._train_data_iter = iter(self.train_loader)
        self._start_time = self._log_last_time = self._show_last_time = time.time()
        self._log_last_it = self._show_last_it = st.iteration
        gpu_loss_sum, gpu_aux_sum, gpu_metric_count = {}, {}, 0

        for m in self.models.values():
            m.train()
        while st.iteration < self.iterations:
            torch.compiler.cudagraph_mark_step_begin()

            data = self._fetch_batch()
            st.iteration += 1
            st.rows += self.batch_size * self.gradient_accumulation_steps

            loss_dict, aux_dict = self._run_train_step(data)

            # accumulate metrics on GPU
            add_dict_to(gpu_loss_sum, loss_dict)
            add_dict_to(gpu_aux_sum, aux_dict)
            gpu_metric_count += 1

            # gather and average at log/show intervals
            if st.iteration % self.log_interval == 0 or st.iteration % self.show_interval == 0:
                loss_dict = {k: v / gpu_metric_count for k, v in gpu_loss_sum.items()}
                aux_dict = {k: v / gpu_metric_count for k, v in gpu_aux_sum.items()}
                loss_dict = accelerator.gather(loss_dict)
                aux_dict = accelerator.gather(aux_dict)
                for metric_dict in [loss_dict, aux_dict]:
                    for key, value in metric_dict.items():
                        metric_dict[key] = torch.mean(value, dim=0).item()
                gpu_loss_sum, gpu_aux_sum, gpu_metric_count = {}, {}, 0

            if st.iteration % self.log_interval == 0:
                self._log_metrics(loss_dict, aux_dict)

            if st.iteration % self.show_interval == 0:
                self._show_progress(loss_dict)

            if self._is_save_iteration(st.iteration):
                self._save_checkpoint()

            if st.iteration % self.val_interval == 0 and self.val_loader is not None:
                self._validate()

        # Ensure the final iteration has both model checkpoints and a training
        # state, so no progress is lost and a finished run is always resumable
        # even when the intervals do not align.
        final_state_saved = self._is_save_iteration(st.iteration) and (
            self.state_save_interval is None
            or st.iteration % self.state_save_interval == 0
        )
        if not final_state_saved:
            self._save_checkpoint(force_state=True)

        if accelerator.is_main_process:
            self.tb_logger.close()
            self.log_file.close()

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
        """Get the next training batch, advancing the epoch on iterator exhaustion."""
        try:
            data = next(self._train_data_iter)
        except StopIteration:
            self.state.epoch += 1
            self._train_data_iter = iter(self.train_loader)
            data = next(self._train_data_iter)
        return data

    def _log_metrics(self, loss_dict, aux_dict):
        """Write training metrics to TensorBoard and the JSONL log file (main process only)."""
        st = self.state
        if not self.accelerator.is_main_process:
            return

        log_value_dict(self.tb_logger, "train", loss_dict, st.iteration, st.rows)
        if aux_dict:
            log_value_dict(self.tb_logger, "train_aux", aux_dict, st.iteration, st.rows)

        iters_per_second = (st.iteration - self._log_last_it) / (time.time() - self._log_last_time)
        elapsed_time = time.time() - self._start_time
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
        log_value_dict(self.tb_logger, "running_stat", running_stat_dict, st.iteration, st.rows)
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

    def _gather_averaged_metrics(self, metric_dict, num_batches):
        """Sum *metric_dict* across processes and average over the global batch count.

        Values may be 0-dim tensors or Python scalars.  Must be called on all
        processes (it performs a collective gather).  Returns
        ``(averaged_dict, total_batches)`` on the main process and
        ``(None, None)`` on other processes.
        """
        accelerator = self.accelerator
        gathered = {}
        for k, v in metric_dict.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor([v], dtype=torch.float32, device=accelerator.device)
            gathered[k] = v
        gathered["num_batches"] = torch.tensor(
            [num_batches], dtype=torch.long, device=accelerator.device
        )
        gathered = accelerator.gather(gathered)
        if not accelerator.is_main_process:
            return None, None
        total_batches = int(torch.sum(gathered.pop("num_batches")).item())
        averaged = {k: torch.sum(v).item() / total_batches for k, v in gathered.items()}
        return averaged, total_batches

    def _validate(self):
        """Run a full validation pass, gather metrics across processes, and log results."""
        st = self.state
        accelerator = self.accelerator

        val_start_time = time.time()
        val_loss_dict, val_aux_dict = {}, {}
        num_val_batches = 0
        if accelerator.is_main_process:
            print(
                f"\nValidation at iteration {st.iteration}/{self.iterations}"
                f" ({st.iteration/self.iterations*100:.2f}%)...",
                flush=True,
            )

        for m in self.models.values():
            m.eval()
        with torch.no_grad():
            for val_data in self.val_loader:
                val_losses, val_auxs = self.validate_step(val_data)
                add_dict_to(val_loss_dict, val_losses)
                add_dict_to(val_aux_dict, val_auxs)
                num_val_batches += 1
        for m in self.models.values():
            m.train()

        # gather and average metrics across processes
        val_loss_dict, total_val_batches = self._gather_averaged_metrics(
            val_loss_dict, num_val_batches
        )
        val_aux_dict, _ = self._gather_averaged_metrics(val_aux_dict, num_val_batches)
        val_elapsed_time = time.time() - val_start_time

        if accelerator.is_main_process:
            elapsed_time = time.time() - self._start_time
            num_val_entries = (
                total_val_batches * self.batch_size_per_process * self.eval_bs_multipler
            )
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

        metric_dict = {}
        num_batches = 0
        start_time = time.time()
        with torch.no_grad():
            for data in tqdm(self.test_loader, disable=not accelerator.is_local_main_process):
                metrics = self.test_step(data, **kwargs)
                add_dict_to(metric_dict, metrics)
                num_batches += 1
                if max_batches is not None and num_batches >= max_batches:
                    break

        averaged, total_batches = self._gather_averaged_metrics(metric_dict, num_batches)

        if accelerator.is_main_process:
            elapsed = time.time() - start_time
            num_entries = int(total_batches * self.batch_size_per_process * self.eval_bs_multipler)
            print(f"Test finished with {num_entries} entries, in {elapsed:.2f}s.")
            print("Metrics:")
            for k, v in averaged.items():
                print(f"\t{k}: {v:.4f}")

            if result_file is not None:
                self._write_test_results(result_file, averaged, num_entries, result_metadata)
            return averaged
        return None

    def test_step(self, data, **kwargs):
        """Default test step: delegates to validate_step and converts tensors to floats.

        Subclasses may override for custom metric computation.
        """
        loss_dict, aux_dict = self.validate_step(data, **kwargs)
        metrics = {}
        for k, v in loss_dict.items():
            metrics[k] = v.item() if isinstance(v, torch.Tensor) else v
        for k, v in aux_dict.items():
            metrics[k] = v.item() if isinstance(v, torch.Tensor) else v
        return metrics

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
