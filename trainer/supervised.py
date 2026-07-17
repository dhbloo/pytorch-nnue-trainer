"""Supervised trainer: value + policy losses with optional knowledge distillation."""

import torch
from contextlib import nullcontext

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
        """Store supervised-specific attributes for evaluation before setup chain runs."""
        if test_model_args:
            deep_update_dict(self.model_args, test_model_args)
        self.loss_type = loss_type
        self.loss_args = {}
        # Disable KD branch entirely so _init_models skips teacher loading
        self.kd_model_type = None
        self.kd_model_args = {}
        self.kd_checkpoint = None
        self.kd_use_train_mode = False
        self.kd_disable_amp = False
        self.kd_T = 1.0
        self.kd_alpha = 1.0

    def _init_models(self):
        """Build the main model and KD teacher model if ``kd_model_type`` is set."""
        super()._init_models()
        self._loss_kwargs = dict(self.loss_args)
        self._kd_model_key = None

        if self.kd_model_type is not None:
            kd_model = build_model(self.kd_model_type, **self.kd_model_args)
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

    def on_before_step(self, data):
        """Run teacher model inference and return ``kd_results`` for the training step."""
        if self.kd_model is None:
            return {"kd_results": None}
        with torch.no_grad(), (
            nullcontext() if self.kd_disable_amp else self.accelerator.autocast()
        ):
            kd_results = self.kd_model(data)
        return {"kd_results": kd_results}

    def train_step(self, data, **kwargs):
        """Forward pass through model, compute supervised losses, and return (loss, loss_dict, aux_dict)."""
        results = self.model(data)
        loss, loss_dict, aux_dict = compute_supervised_losses(
            self.loss_type, data, results, **kwargs, **self._loss_kwargs
        )
        return loss, loss_dict, aux_dict

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
