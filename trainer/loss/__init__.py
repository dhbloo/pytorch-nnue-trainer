"""Loss functions for the trainer package.

Re-exports all public symbols from sub-modules for backward compatibility.
"""

from trainer.loss.utils import apply_kd_temperature, convert_value_target, prepare_policy
from trainer.loss.supervised import (
    value_loss_kl,
    value_loss_ce,
    value_loss_mse,
    policy_loss_kl,
    policy_loss_ce,
    policy_loss_mse,
    policy_reg_loss,
    value_uncertainty_loss,
    value_relative_uncertainty_loss,
    VALUE_LOSS_FN,
    POLICY_LOSS_FN,
    compute_supervised_losses,
)
