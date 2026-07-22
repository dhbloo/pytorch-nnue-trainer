"""Supervised trainer: value + policy losses with optional knowledge distillation."""

import torch
from contextlib import nullcontext
from accelerate.utils.other import is_compiled_module

from model import build_model
from utils.file_utils import load_torch_ckpt
from utils.misc_utils import deep_update_dict
from trainer.base import BaseTrainer
from trainer.loss.supervised import compute_supervised_losses
from trainer.metric.supervised import compute_supervised_metrics


class SupervisedTrainer(BaseTrainer):
    """Trainer for supervised value + policy learning with optional knowledge distillation.

    Extends :class:`BaseTrainer` with supervised loss computation (value + policy)
    and optional teacher-student knowledge distillation.  The loss type is
    controlled by ``loss_type`` (e.g. ``"KL+KL"``, ``"CE+CE"``), where the
    left side selects the value loss and the right side the policy loss.

    When ``kd_model_type`` is set, a teacher model is loaded and its outputs
    are blended into the loss according to ``kd_alpha`` and ``kd_T``.

    Args:
        loss_type: Value+policy loss selector (e.g. ``"KL+KL"``, ``"CE+CE"``).
        loss_args: Extra keyword arguments forwarded to ``compute_supervised_losses``.
        kd_model_type: Registered model class name for the KD teacher, or ``None``
            to disable knowledge distillation.
        kd_model_args: Extra keyword arguments forwarded to the teacher model
            constructor.
        kd_checkpoint: Path to the teacher model checkpoint, or ``None``.
        kd_T: Temperature for knowledge distillation softmax scaling.
        kd_alpha: Interpolation weight between ground-truth loss (``1 - alpha``)
            and distillation loss (``alpha``).
        kd_use_train_mode: Keep the teacher in train mode instead of eval mode.
        kd_disable_amp: Disable automatic mixed precision for the teacher forward
            pass.
        **kwargs: Remaining keyword arguments forwarded to :class:`BaseTrainer`.
    """

    def __init__(
        self,
        *,
        loss_type: str = "KL+KL",
        loss_args: dict | None = None,
        kd_model_type: str | None = None,
        kd_model_args: dict | None = None,
        kd_checkpoint: str | None = None,
        kd_T: float = 1.0,
        kd_alpha: float = 1.0,
        kd_use_train_mode: bool = False,
        kd_disable_amp: bool = False,
        **kwargs,
    ):
        self.loss_type = loss_type
        self.loss_args = loss_args or {}
        self.kd_model_type = kd_model_type
        self.kd_model_args = kd_model_args or {}
        self.kd_checkpoint = kd_checkpoint
        self.kd_T = kd_T
        self.kd_alpha = kd_alpha
        self.kd_use_train_mode = kd_use_train_mode
        self.kd_disable_amp = kd_disable_amp
        super().__init__(**kwargs)

    def _init_eval_attrs(self, *, test_model_args=None, loss_type="CE+CE", **extra):
        """Apply eval-specific overrides (defaults come from ``_apply_init_defaults``)."""
        if test_model_args:
            deep_update_dict(self.model_args, test_model_args)
        self.loss_type = loss_type

    def _init_models(self):
        """Build the main model and KD teacher model if ``kd_model_type`` is set."""
        super()._init_models()
        self._loss_kwargs = dict(self.loss_args)
        self._kd_model_key = None

        if self.kd_model_type is not None:
            kd_model = build_model(self.kd_model_type, **self.kd_model_args)
            self._configure_model_compilation(kd_model)
            kd_state, _, _ = load_torch_ckpt(self.kd_checkpoint)
            kd_model.load_state_dict(kd_state)
            self.accelerator.print(
                f"Loaded teacher model {kd_model.name} from {self.kd_checkpoint}"
            )
            self._kd_model_key = kd_model.name
            self.aux_models[self._kd_model_key] = self.accelerator.prepare_model(
                kd_model, evaluation_mode=True
            )
            if self.kd_use_train_mode:
                self.kd_model.train()
            else:
                self.kd_model.eval()

            assert self.kd_T is not None
            assert self.kd_alpha is not None and 0 <= self.kd_alpha <= 1
            self._loss_kwargs["kd_T"] = self.kd_T
            self._loss_kwargs["kd_alpha"] = self.kd_alpha

    @property
    def kd_model(self):
        """The KD teacher model, or ``None`` if knowledge distillation is disabled."""
        if self._kd_model_key is None:
            return None
        return self.aux_models[self._kd_model_key]

    def _loss_summary(self, loss_dict: dict) -> str:
        """Show total loss with value and policy components."""
        return (
            f"{loss_dict['total_loss']:.4f}"
            f", v={loss_dict['value_loss']:.4f}"
            f", p={loss_dict['policy_loss']:.4f}"
        )

    def on_before_step(self, data):
        """Run teacher model inference and return ``kd_results`` for the training step."""
        if self.kd_model is None:
            return {"kd_results": None}
        with torch.no_grad(), (
            nullcontext() if self.kd_disable_amp else self.accelerator.autocast()
        ):
            kd_results = self.kd_model(data)
        return {"kd_results": kd_results}

    def _prepare_for_training(self):
        """Additionally compile the fused forward+loss step function."""
        super()._prepare_for_training()
        # Compile model forward + loss as ONE graph: inductor fuses the loss
        # kernels and the eager loss forward/backward launch overhead
        # disappears.  The accelerate-compiled wrapper is bypassed (its
        # `_orig_mod`, still DDP-wrapped if any) so there is a single compile
        # entry point; accelerate's own wrapper still serves eval-mode calls.
        model = self.model
        if is_compiled_module(model):
            model = model._orig_mod
        self._train_forward_model = model
        self._compiled_train_forward = self._maybe_compile(self._train_forward)

    def _train_forward(self, data, kd_results=None):
        """Model forward + supervised loss, fused into one compiled graph."""
        results = self._train_forward_model(data)
        return compute_supervised_losses(
            self.loss_type, data, results, kd_results=kd_results, **self._loss_kwargs
        )

    def train_step(self, data, kd_results=None, **kwargs):
        """Run the fused (compiled) forward+loss and return (loss, loss_dict, aux_dict)."""
        return self._compiled_train_forward(data, kd_results)

    def validate_step(self, data, **kwargs):
        """Eval-mode forward pass with optional KD teacher. Returns (loss_dict, aux_dict)."""
        kd_results = None
        if self.kd_model is not None:
            with nullcontext() if self.kd_disable_amp else self.accelerator.autocast():
                kd_results = self.kd_model(data)
        with self.accelerator.autocast():
            results = self.model(data)
            _, loss_dict, aux_dict = compute_supervised_losses(
                self.loss_type, data, results, kd_results, **self._loss_kwargs
            )
        return loss_dict, aux_dict

    def test_step(self, data, **kwargs):
        """Compute supervised metrics for one test batch."""
        with self.accelerator.autocast():
            results = self.model(data)
        return compute_supervised_metrics(data, results, **kwargs)
