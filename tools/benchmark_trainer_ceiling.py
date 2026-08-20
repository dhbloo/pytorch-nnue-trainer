"""Measure the steady-state trainer ceiling with a device-resident batch.

The benchmark keeps the production ``SupervisedTrainer`` step intact, including
DDP, the configured supervised loss, gradient clipping, optimizer, scheduler,
finite checks, and failure coordination.  It replaces only the input pipeline:
one deterministic batch is created per rank, copied to the device once, and
then reused at a stable address.

Launch this tool with the same Accelerate precision and compiler settings as
the corresponding training run.  Logging, validation, checkpoints, planning,
decoding, and host-to-device transfer are intentionally outside the measured
steady-state interval.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path

import torch
import yaml
from torch.utils.data import IterableDataset

from tools.model_perf import (
    make_synthetic_data,
    move_data,
    torch_performance_metadata,
)
from train import collect_run_provenance
from trainer.base import BaseTrainer, _prepared_ddp
from trainer.replica import _parameter_chunks, optimizer_parameters
from trainer.supervised import SupervisedTrainer


class _PlaceholderDataset(IterableDataset):
    def __iter__(self):
        return iter(())


class _PlaceholderLoader:
    def __init__(self) -> None:
        self.dataset = _PlaceholderDataset()

    def __iter__(self):
        return iter(self.dataset)


class _FixedDeviceLoader:
    def __init__(self, data: dict[str, torch.Tensor]) -> None:
        self.dataset = _PlaceholderDataset()
        self._data = data

    def __iter__(self):
        while True:
            yield self._data


class FixedDeviceSupervisedTrainer(SupervisedTrainer):
    """Production trainer with dataset setup replaced by an inert loader."""

    def _setup_data(self) -> None:
        self._set_batch_size_per_process()
        self._adaptive_pipeline_spec = None
        self._resume_stream = None
        self._resume_sampler = None
        self._exact_resume_reason = "device-resident trainer benchmark"
        self._data_stream_signature = None
        self.train_dataset = _PlaceholderDataset()
        self.train_loader = _PlaceholderLoader()
        self.val_dataset = None
        self.val_loader = None

    def _load_checkpoint(self) -> None:
        # A benchmark run directory is disposable and must never contribute a
        # continuation cursor.  An explicitly configured pretrained model is
        # still useful and follows the production loading path.
        self._load_pretrained_weights()

    def install_device_batch(self, *, board_size: int, seed: int) -> None:
        data = make_synthetic_data(
            self.model_type,
            self.unwrapped_model,
            self.batch_size_per_process,
            board_size,
            seed + self.accelerator.process_index,
        )
        data = move_data(data, self.accelerator.device)
        for value in data.values():
            torch._dynamo.mark_static_address(value)
        self.train_loader = _FixedDeviceLoader(data)


def _trainer_keywords() -> set[str]:
    names = set()
    for trainer_type in (BaseTrainer, SupervisedTrainer):
        parameters = inspect.signature(trainer_type.__init__).parameters
        for name, parameter in parameters.items():
            if name != "self" and parameter.kind is not inspect.Parameter.VAR_KEYWORD:
                names.add(name)
    return names


def load_trainer_config(path: Path, *, rundir: str) -> dict:
    with path.open(encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if not isinstance(loaded, Mapping):
        raise ValueError("trainer config must contain a YAML mapping")

    config = dict(loaded)
    config.pop("_provenance", None)
    trainer_type = config.pop("trainer_type", "supervised")
    if trainer_type != "supervised":
        raise ValueError(
            "trainer ceiling benchmark currently supports only supervised training"
        )
    unknown = set(config) - _trainer_keywords()
    if unknown:
        raise ValueError(f"unsupported trainer config keys: {', '.join(sorted(unknown))}")

    config.update(
        rundir=rundir,
        val_datas=None,
        profiler_args=None,
        data_pipeline=None,
        data_pipelines=None,
        num_worker=0,
    )
    return config


def _initialize_step_state(trainer: FixedDeviceSupervisedTrainer) -> None:
    trainer._train_data_iter = iter(trainer.train_loader)
    trainer._train_metric_schema = None
    trainer._known_divergent = False
    trainer._optimizer_state_mutated = False
    trainer._pending_stream_tokens = []
    trainer._pending_sampler_tokens = []
    trainer._pending_batch_memory_leases = []
    for model in trainer.models.values():
        model.train()


def _run_steps(trainer: FixedDeviceSupervisedTrainer, count: int) -> None:
    for _ in range(count):
        trainer._checkpoint_safe = False
        trainer._optimizer_state_mutated = False
        torch.compiler.cudagraph_mark_step_begin()
        data = trainer._fetch_batch()
        trainer.state.iteration += 1
        trainer.state.rows += trainer.batch_size * trainer.gradient_accumulation_steps
        trainer._run_train_step(data)
        trainer._drain_pending_step_validity(include_current=False)


@torch.no_grad()
def _optimizer_parameter_digest(trainer: FixedDeviceSupervisedTrainer) -> str:
    """Return an exact, streaming digest without retaining a model copy."""

    digest = hashlib.sha256()
    for _label, parameter in optimizer_parameters(trainer.optimizers):
        for chunk in _parameter_chunks(parameter, 16 * 1024 * 1024):
            host_bytes = chunk.contiguous().view(torch.uint8).cpu().numpy()
            digest.update(memoryview(host_bytes))
    return digest.hexdigest()


def _verify_distributed_training(
    trainer: FixedDeviceSupervisedTrainer,
    initial_parameter_digest: str,
) -> dict:
    """Validate a real update, the live DDP target, and rank replicas."""

    distributed = trainer.accelerator.num_processes > 1
    parameter_update_observed = (
        _optimizer_parameter_digest(trainer) != initial_parameter_digest
    )
    missing_update = float(not parameter_update_observed)
    if distributed:
        missing_update = _distributed_max(trainer, missing_update)
    if missing_update:
        raise RuntimeError("warmup did not change optimizer-owned parameters")
    if not distributed:
        return {
            "ddp_required": False,
            "forward_target_is_ddp": None,
            "parameter_update_observed": True,
            "replica_check_passed": None,
        }

    forward_target_is_ddp = all(
        _prepared_ddp(model) is not None for model in trainer.models.values()
    )
    missing_ddp = _distributed_max(
        trainer,
        float(not forward_target_is_ddp),
    )
    if missing_ddp:
        raise RuntimeError("trainer forward target is not live DDP")
    if not trainer._check_replicated_parameters(require_update=True):
        raise RuntimeError("warmup did not complete an optimizer update")
    return {
        "ddp_required": True,
        "forward_target_is_ddp": True,
        "parameter_update_observed": True,
        "replica_check_passed": True,
    }


def _distributed_max(trainer: FixedDeviceSupervisedTrainer, value: float) -> float:
    tensor = torch.tensor(
        value,
        dtype=torch.float64,
        device=trainer.accelerator.device,
    )
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
    return float(tensor.item())


def run(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    if args.warmup_steps < 1 or args.steps < 1:
        raise ValueError("--warmup-steps and --steps must both be positive")
    if args.board_size < 1:
        raise ValueError("--board-size must be positive")

    source = collect_run_provenance()
    trainer = None
    with tempfile.TemporaryDirectory(prefix="ntr-trainer-ceiling-") as rundir:
        try:
            config = load_trainer_config(args.config, rundir=rundir)
            trainer = FixedDeviceSupervisedTrainer(**config)
            trainer.install_device_batch(board_size=args.board_size, seed=args.seed)
            _initialize_step_state(trainer)
            initial_parameter_digest = _optimizer_parameter_digest(trainer)

            warmup_started = time.perf_counter()
            _run_steps(trainer, args.warmup_steps)
            trainer._drain_pending_step_validity(include_current=True)
            torch.cuda.synchronize(trainer.accelerator.device)
            trainer.accelerator.wait_for_everyone()
            warmup_seconds = _distributed_max(
                trainer,
                time.perf_counter() - warmup_started,
            )
            validity = _verify_distributed_training(
                trainer,
                initial_parameter_digest,
            )

            torch.cuda.reset_peak_memory_stats(trainer.accelerator.device)
            trainer.accelerator.wait_for_everyone()
            measured_started = time.perf_counter()
            _run_steps(trainer, args.steps)
            torch.cuda.synchronize(trainer.accelerator.device)
            local_elapsed = time.perf_counter() - measured_started
            elapsed = _distributed_max(trainer, local_elapsed)
            trainer._drain_pending_step_validity(include_current=True)

            peak_allocated = _distributed_max(
                trainer,
                float(torch.cuda.max_memory_allocated(trainer.accelerator.device)),
            )
            peak_reserved = _distributed_max(
                trainer,
                float(torch.cuda.max_memory_reserved(trainer.accelerator.device)),
            )
            rows_per_step = trainer.batch_size * trainer.gradient_accumulation_steps
            device_properties = torch.cuda.get_device_properties(
                trainer.accelerator.device
            )
            result = {
                "schema_version": 1,
                "scope": "production trainer step with a fixed device-resident batch",
                "source": source,
                "validity": validity,
                "model": {
                    "type": trainer.model_type,
                    "args": trainer.model_args,
                    "parameters": sum(
                        parameter.numel()
                        for parameter in trainer.unwrapped_model.parameters()
                    ),
                },
                "workload": {
                    "global_batch_size": trainer.batch_size,
                    "local_batch_size": trainer.batch_size_per_process,
                    "gradient_accumulation_steps": trainer.gradient_accumulation_steps,
                    "board_size": args.board_size,
                    "warmup_steps": args.warmup_steps,
                    "measured_steps": args.steps,
                    "world_size": trainer.accelerator.num_processes,
                    "mixed_precision": trainer.accelerator.mixed_precision,
                    "loss_type": trainer.loss_type,
                    "loss_args": trainer.loss_args,
                    "optimizer_type": trainer.optim_type,
                    "optimizer_args": trainer.optim_args,
                    "clip_grad_norm": trainer.clip_grad_norm,
                    "clip_grad_value": trainer.clip_grad_value,
                    "seed": args.seed,
                },
                "timing": {
                    "warmup_seconds": warmup_seconds,
                    "max_rank_elapsed_seconds": elapsed,
                    "mean_step_ms": elapsed * 1000 / args.steps,
                    "samples_per_second": rows_per_step * args.steps / elapsed,
                },
                "memory": {
                    "allocator_limit_bytes": trainer.cuda_memory_limit_bytes,
                    "allocator_limit_fraction": trainer.max_memory_fraction,
                    "max_rank_peak_allocated_bytes": int(peak_allocated),
                    "max_rank_peak_reserved_bytes": int(peak_reserved),
                },
                "environment": {
                    "gpu": device_properties.name,
                    "gpu_total_memory_bytes": device_properties.total_memory,
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "cudnn": torch.backends.cudnn.version(),
                    "python": platform.python_version(),
                    **torch_performance_metadata(trainer.performance_level),
                },
            }
            encoded = json.dumps(result, indent=2, sort_keys=True)
            if trainer.accelerator.is_main_process:
                print(encoded, flush=True)
                if args.output is not None:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(encoded + "\n", encoding="utf-8")
        finally:
            try:
                if trainer is not None:
                    try:
                        trainer._shutdown_resources()
                    finally:
                        trainer.accelerator.end_training()
            finally:
                if torch.distributed.is_initialized():
                    torch.distributed.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--board-size", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
