"""Base trainer: accelerator setup, data, model, optimizer, checkpoint, logging, and training loop."""

import torch
from accelerate import (
    Accelerator,
    DataLoaderConfiguration,
    DistributedDataParallelKwargs,
)
from accelerate.utils.other import is_compiled_module
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass
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
        iterations: Total number of training iterations.
        batch_size: Global batch size (split across processes).
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
        # Distributed
        find_unused_parameters: bool = False,
        # Extra (absorb unknown CLI keys like 'config')
        **_extra,
    ):
        # Store all parameters
        self.rundir = rundir
        self.iterations = iterations
        self.batch_size = batch_size
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

        self.find_unused_parameters = find_unused_parameters

        self.state = TrainingState()
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

        self._init_eval_attrs(**extra)
        self._setup_accelerator()
        self._setup_test_data()
        self._init_models()
        self._load_eval_checkpoint()
        self._prepare_for_evaluation()
        return self

    def _init_eval_attrs(self, **extra):
        """Hook for subclasses to store extra attributes before the eval setup chain."""

    # ── Setup helpers (called from __init__) ──────────────────────

    def _setup_accelerator(self):
        """Create the Accelerator with DDP config, set random seed and performance level."""
        dataloader_config = DataLoaderConfiguration(dispatch_batches=False, non_blocking=True)
        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=self.find_unused_parameters
        )
        self.accelerator = Accelerator(
            cpu=self.use_cpu,
            dataloader_config=dataloader_config,
            kwargs_handlers=[ddp_kwargs],
        )
        seed_everything(self.random_seed)
        set_performance_level(self.performance_level)

    def _setup_logging(self):
        """Open TensorBoard writer and JSONL log file (main process only)."""
        if self.accelerator.is_local_main_process:
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
        """Create the optimizer covering all trained models' parameters."""
        trained_models = list(self.models.values())
        model_or_models = trained_models[0] if len(trained_models) == 1 else trained_models
        self.optimizer = build_optimizer(
            self.optim_type,
            model_or_models,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            **self.optim_args,
        )

    def _setup_scheduler(self):
        """Create the LR scheduler, resuming from the current iteration if applicable."""
        self.scheduler = build_lr_scheduler(
            self.optimizer,
            self.lr_scheduler_type,
            self.iterations,
            last_it=self.state.iteration - 1,
            **self.lr_scheduler_args,
        )

    # ── Checkpoint ────────────────────────────────────────────────

    def _load_checkpoint(self):
        """Resume from the latest checkpoint in *rundir*, or apply weight init / pretrained weights."""
        ckpt_filename = find_latest_ckpt(self.rundir, f"ckpt_{self.model_name}")

        if ckpt_filename:
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
            self.accelerator.print(f"Loaded from {ckpt_filename}")
            self.state.iteration = int(metadata.get("iteration", 0))
            self.state.epoch = int(metadata.get("epoch", 0))
            self.state.rows = int(metadata.get("rows", 0))
        else:
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

    def _save_checkpoint(self):
        """Save all trained models, optimizer, and scaler state to disk."""
        st = self.state
        if not self.accelerator.is_local_main_process:
            return

        last_ckpt_filename = find_latest_ckpt(self.rundir, f"ckpt_{self.model_name}")

        model_state_dict = self.unwrapped_model.state_dict()
        training_state_dicts = {"optimizer": self.optimizer.state_dict()}
        # Save additional trained models under their name
        for name, m in self.models.items():
            if name != self.model_name:
                training_state_dicts[name] = self._unwrap(m).state_dict()
        if self.accelerator.scaler is not None:
            training_state_dicts["scalar"] = self.accelerator.scaler.state_dict()
        metadata_dict = {
            "iteration": str(st.iteration),
            "epoch": str(st.epoch),
            "rows": str(st.rows),
        }
        save_torch_ckpt(
            os.path.join(self.rundir, f"ckpt_{self.model_name}_{st.iteration:07d}"),
            model_state_dict,
            training_state_dicts,
            metadata_dict,
        )

        # remove last snapshot if it's a temporary snapshot
        if last_ckpt_filename is not None:
            last_snapshot_iter = get_iteration_from_ckpt_filename(last_ckpt_filename)
            if last_snapshot_iter is None or last_snapshot_iter % self.save_interval != 0:
                os.remove(last_ckpt_filename)

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
        """Wrap trained models, optimizer and dataloaders with accelerator for distributed training."""
        accelerator = self.accelerator
        # Prepare all trained models + optimizer + train dataloader together
        model_names = list(self.models.keys())
        to_prepare = [self.models[n] for n in model_names]
        to_prepare.extend([self.optimizer, self.train_loader])
        prepared = accelerator.prepare(*to_prepare)
        # Unpack: first len(model_names) are models, then optimizer, then train_loader
        for i, name in enumerate(model_names):
            self.models[name] = prepared[i]
        self.optimizer = prepared[len(model_names)]
        self.train_loader = prepared[len(model_names) + 1]
        if self.val_loader is not None:
            self.val_loader = accelerator.prepare_data_loader(self.val_loader)

    def _all_trained_parameters(self):
        """Yield all parameters from all trained models."""
        from itertools import chain
        return chain(*(m.parameters() for m in self.models.values()))

    def _run_train_step(self, data):
        """Execute one training iteration: forward, backward, optimizer step."""
        extra_kwargs = self.on_before_step(data)

        # apply weight clipping if needed (check all trained models)
        for m in self.models.values():
            unwrapped = self._unwrap(m)
            if hasattr(unwrapped, "weight_clipping"):
                weight_clipping(unwrapped.named_parameters(), unwrapped.weight_clipping)

        accelerator = self.accelerator
        with accelerator.accumulate(*self.models.values()), accelerator.autocast():
            self.optimizer.zero_grad(set_to_none=True)
            loss, loss_dict, aux_dict = self.train_step(data, **extra_kwargs)
            accelerator.backward(loss)

            if self.clip_grad_norm is not None:
                accelerator.clip_grad_norm_(
                    self._all_trained_parameters(), max_norm=self.clip_grad_norm
                )
            elif self.clip_grad_value is not None:
                accelerator.clip_grad_value_(
                    self._all_trained_parameters(), clip_value=self.clip_grad_value
                )

            self.optimizer.step()
            self.scheduler.step()

        return loss_dict, aux_dict

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
            st.rows += self.batch_size

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

            if (
                st.iteration % self.save_interval == 0
                or st.iteration % self.temp_save_interval == 0
            ):
                self._save_checkpoint()

            if st.iteration % self.val_interval == 0 and self.val_loader is not None:
                self._validate()

        if accelerator.is_local_main_process:
            self.log_file.close()

    def profile(self, *, wait=0, warmup=10, active=30, profile_memory=False):
        """Run a short profiling loop and save trace to rundir.

        Uses torch.profiler.profile directly. Runs (wait + warmup + active) iterations
        then prints a summary table and saves the trace.
        """
        total_iters = wait + warmup + active
        accelerator = self.accelerator

        def on_trace_ready(p: torch.profiler.profile):
            if accelerator.is_local_main_process:
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
        if not self.accelerator.is_local_main_process:
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
            "entry/s": iters_per_second * self.batch_size,
            "lr": self.scheduler.get_last_lr()[0],
        }
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
            f" | Loss: {loss_dict['total_loss']:.4f}"
            f", v={loss_dict['value_loss']:.4f}"
            f", p={loss_dict['policy_loss']:.4f}",
            flush=True,
        )
        self._show_last_it = st.iteration
        self._show_last_time = time.time()

    def _validate(self):
        """Run a full validation pass, gather metrics across processes, and log results."""
        st = self.state
        accelerator = self.accelerator

        val_start_time = time.time()
        val_loss_dict, val_aux_dict = {}, {}
        num_val_batches = 0
        if accelerator.is_local_main_process:
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

        # gather all loss dict across processes
        val_loss_dict["num_val_batches"] = torch.tensor(
            [num_val_batches], dtype=torch.long, device=accelerator.device
        )
        all_val_loss_dict = accelerator.gather(val_loss_dict)
        all_val_aux_dict = accelerator.gather(val_aux_dict)

        if accelerator.is_local_main_process:
            num_val_batches_tensor = all_val_loss_dict.pop("num_val_batches")
            num_val_batches = torch.sum(num_val_batches_tensor).item()
            val_loss_dict, val_aux_dict = {}, {}
            for k, loss_tensor in all_val_loss_dict.items():
                val_loss_dict[k] = torch.sum(loss_tensor).item() / num_val_batches
            for k, aux_tensor in all_val_aux_dict.items():
                val_aux_dict[k] = torch.sum(aux_tensor).item() / num_val_batches

            val_elapsed_time = time.time() - val_start_time
            elapsed_time = time.time() - self._start_time
            num_val_entries = (
                num_val_batches * self.batch_size_per_process * self.eval_bs_multipler
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
            print(
                f"Validation loss: {val_loss_dict['total_loss']:.4f},"
                f" v={val_loss_dict['value_loss']:.4f},"
                f" p={val_loss_dict['policy_loss']:.4f}"
            )
            print(flush=True)

            # subtract validation time from training time
            self._log_last_time += val_elapsed_time
            self._show_last_time += val_elapsed_time

    # ── Hooks for subclasses ──────────────────────────────────────

    def on_before_step(self, data):
        """Called before each training step. Return extra kwargs for train_step."""
        return {}

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

        # Convert scalar metrics to tensors for gathering
        for k, v in metric_dict.items():
            if not isinstance(v, torch.Tensor):
                metric_dict[k] = torch.tensor([v], dtype=torch.float32, device=accelerator.device)
        metric_dict["num_batches"] = torch.tensor(
            [num_batches], dtype=torch.long, device=accelerator.device
        )

        all_metric_dict = accelerator.gather(metric_dict)

        if accelerator.is_local_main_process:
            num_batches_tensor = all_metric_dict.pop("num_batches")
            total_batches = torch.sum(num_batches_tensor).item()
            averaged = {}
            for k, tensor in all_metric_dict.items():
                averaged[k] = torch.sum(tensor).item() / total_batches

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
