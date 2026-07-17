"""Supervised loss functions, dispatch tables, and orchestration.

Each loss function handles only its specific computation — no branching on loss type strings.
All functions are pure (no closures capturing outer scope).
"""

import torch
import torch.nn.functional as F

from utils.training_utils import cross_entropy_with_softlabel
from trainer.loss.utils import apply_kd_temperature, convert_value_target, prepare_policy


# ==================== Value losses ====================


def value_loss_kl(value, value_target):
    """KL-divergence value loss using soft-label cross entropy."""
    return cross_entropy_with_softlabel(value, value_target, use_kl_divergence=True)


def value_loss_ce(value, value_target, focal_gamma=0):
    """Cross-entropy value loss with optional focal loss."""
    return cross_entropy_with_softlabel(value, value_target, focal_gamma=focal_gamma)


def value_loss_mse(value, value_target):
    """MSE value loss computed on winrate."""
    if value.ndim == 1:
        return F.mse_loss(torch.sigmoid(value), value_target)
    else:
        value = torch.softmax(value, dim=1)  # [B, 3]
        winrate = (value[:, 0] - value[:, 1] + 1) / 2
        winrate_target = (value_target[:, 0] - value_target[:, 1] + 1) / 2
        return F.mse_loss(winrate, winrate_target)


# ==================== Policy losses ====================


def policy_loss_kl(policy, policy_target, weight=None):
    """KL-divergence policy loss with optional per-cell weighting."""
    return cross_entropy_with_softlabel(
        policy, policy_target, weight=weight, use_kl_divergence=True
    )


def policy_loss_ce(policy, policy_target, weight=None, focal_gamma=0):
    """Cross-entropy policy loss with optional focal loss and per-cell weighting."""
    return cross_entropy_with_softlabel(
        policy, policy_target, weight=weight, focal_gamma=focal_gamma
    )


def policy_loss_mse(policy, policy_target, weight=None):
    """MSE policy loss between softmax(policy) and target."""
    policy_softmaxed = torch.softmax(policy, dim=1)
    policy_loss = F.mse_loss(policy_softmaxed, policy_target, reduction="none")
    if weight is not None:
        policy_loss = policy_loss * weight
    return torch.mean(policy_loss)


# ==================== Regularization ====================


def policy_reg_loss(policy, mask=None):
    """Policy regularization: squared mean of logits."""
    policy = torch.flatten(policy, start_dim=1)
    if mask is not None:
        mask = torch.flatten(mask, start_dim=1)
        policy = policy.masked_fill(mask == 0, -1e6)
        policy_mean = torch.sum(policy * mask) / torch.sum(mask)
    else:
        policy_mean = torch.mean(policy)
    return policy_mean.square()


# ==================== Uncertainty losses ====================


def value_uncertainty_loss(uncertainty, value, value_target):
    """Huber loss for value uncertainty prediction."""
    uncertainty_gt = _value_uncertainty_gt(value, value_target)
    return F.huber_loss(uncertainty, uncertainty_gt)


def value_relative_uncertainty_loss(rel_uncert, value_small, value_large, value_target):
    """MSE loss for relative uncertainty between a small and large model."""
    ug_small = _value_uncertainty_gt(value_small, value_target)
    ug_large = _value_uncertainty_gt(value_large, value_target)
    rel_gt = ug_large / ug_small
    rel_gt = torch.where(ug_large < ug_small, rel_gt, 1.0)
    return F.mse_loss(rel_uncert, rel_gt)


def _value_uncertainty_gt(value, value_target):
    """Compute ground-truth value uncertainty (detached)."""
    value = value.detach()
    if value.ndim == 1:
        winrate = torch.sigmoid(value)
        winrate_target = value_target
    else:
        value = torch.softmax(value, dim=1)
        winrate = (value[:, 0] - value[:, 1] + 1) / 2
        winrate_target = (value_target[:, 0] - value_target[:, 1] + 1) / 2
    return torch.square(winrate - winrate_target)


# ==================== Dispatch tables ====================

VALUE_LOSS_FN = {"KL": value_loss_kl, "CE": value_loss_ce, "MSE": value_loss_mse}
POLICY_LOSS_FN = {"KL": policy_loss_kl, "CE": policy_loss_ce, "MSE": policy_loss_mse}


# ==================== Orchestration ====================


def compute_supervised_losses(
    loss_type,
    data,
    results,
    kd_results=None,
    kd_T=1.0,
    kd_alpha=1.0,
    policy_reg_lambda=0,
    value_policy_ratio=1,
    **extra_args,
):
    """Compute value + policy losses (with optional KD). Drop-in replacement for calc_loss."""
    value_loss_type, policy_loss_type = loss_type.split("+")

    # unpack model outputs
    value, policy = results["value"], results["policy"]
    aux_losses = results.get("aux_losses")
    aux_outputs = results.get("aux_outputs")
    board_mask = results.get("board_mask")

    if kd_results is not None:
        value_target, policy_target = kd_results["value"], kd_results["policy"]
    else:
        value_target, policy_target = data["value_target"], data["policy_target"]

    # ── value loss ────────────────────────────────────────────
    v, vt = value, value_target
    if kd_results is not None:
        v, vt = apply_kd_temperature(v, vt, kd_T, is_value=True)
    v, vt = convert_value_target(v, vt)

    if value_loss_type == "CE":
        v_loss = value_loss_ce(v, vt, focal_gamma=extra_args.get("value_focal_gamma", 0))
    else:
        v_loss = VALUE_LOSS_FN[value_loss_type](v, vt)

    # ── policy loss ───────────────────────────────────────────
    p, pt = prepare_policy(policy, policy_target, mask=board_mask)

    if kd_results is not None:
        # apply temperature to teacher logits
        if board_mask is not None:
            flat_mask = torch.flatten(board_mask, start_dim=1)
            pt = pt.masked_fill(flat_mask == 0, -1e6)
        pt = torch.softmax(pt / kd_T, dim=1)
        p = p / kd_T
        if board_mask is not None:
            pt = pt.masked_fill(flat_mask == 0, 0)

    # per-cell weight (e.g. ignore forbidden points)
    if extra_args.get("ignore_forbidden_point_policy", False):
        forbidden = torch.flatten(data["forbidden_point"], start_dim=1).float()
        policy_loss_weight = 1.0 - forbidden
    else:
        policy_loss_weight = None

    if policy_loss_type == "NONE":
        p_loss = torch.tensor(0.0, device=policy.device)
    elif policy_loss_type == "CE":
        p_loss = policy_loss_ce(
            p, pt, weight=policy_loss_weight,
            focal_gamma=extra_args.get("policy_focal_gamma", 0),
        )
    elif policy_loss_type == "MSE":
        p_loss = policy_loss_mse(p, pt, weight=policy_loss_weight)
    else:
        p_loss = POLICY_LOSS_FN[policy_loss_type](p, pt, weight=policy_loss_weight)

    # ── combine ───────────────────────────────────────────────
    value_lambda = 2 * value_policy_ratio / (value_policy_ratio + 1)
    policy_lambda = 2 / (value_policy_ratio + 1)
    total_loss = value_lambda * v_loss + policy_lambda * p_loss
    loss_dict = {"value_loss": v_loss.detach(), "policy_loss": p_loss.detach()}

    # policy regularization
    if float(policy_reg_lambda) > 0:
        p_reg = policy_reg_loss(policy, mask=board_mask)
        total_loss += float(policy_reg_lambda) * p_reg
        loss_dict["policy_reg"] = p_reg.detach()

    # ── aux losses ────────────────────────────────────────────
    total_loss, loss_dict = _apply_aux_losses(
        total_loss, loss_dict, aux_losses,
        value_loss_type=value_loss_type, policy_loss_type=policy_loss_type,
        value_target=vt, policy_target=pt, board_mask=board_mask,
        value_lambda=value_lambda, policy_lambda=policy_lambda,
        policy_reg_lambda=policy_reg_lambda,
        extra_args=extra_args,
    )

    # ── knowledge distillation merge ──────────────────────────
    if kd_results is not None:
        real_loss, real_loss_dict, _ = compute_supervised_losses(
            loss_type, data, results,
            policy_reg_lambda=policy_reg_lambda,
            value_policy_ratio=value_policy_ratio,
            **extra_args,
        )
        if kd_alpha == 1.0:
            # ground-truth loss is only logged; keep it out of the backward graph
            real_loss = real_loss.detach()
        total_loss = kd_alpha * total_loss * (kd_T**2) + (1 - kd_alpha) * real_loss
        kd_loss_dict = {"kd_" + k: v for k, v in loss_dict.items()}
        loss_dict = real_loss_dict
        loss_dict.update(kd_loss_dict)

    # ── aux outputs ───────────────────────────────────────────
    aux_outputs_dict = {}
    if aux_outputs:
        for name, out in aux_outputs.items():
            aux_outputs_dict[name] = out.detach()

    loss_dict["total_loss"] = total_loss.detach()
    return total_loss, loss_dict, aux_outputs_dict


def _apply_aux_losses(
    total_loss, loss_dict, aux_losses, *,
    value_loss_type, policy_loss_type,
    value_target, policy_target, board_mask,
    value_lambda, policy_lambda, policy_reg_lambda,
    extra_args,
):
    """Process model-returned aux_losses dict and add to total_loss / loss_dict."""
    if not aux_losses:
        return total_loss, loss_dict

    for aux_name, aux_loss in aux_losses.items():
        print_name = f"{aux_name}_loss"
        weight = extra_args.get(f"{aux_name}_lambda", None)
        if weight is not None:
            weight = float(weight)

        # pre-defined loss terms: (loss_type_string, inputs)
        if isinstance(aux_loss, tuple) and len(aux_loss) == 2:
            loss_type, aux_input = aux_loss
            if loss_type == "value_loss":
                aux_loss = VALUE_LOSS_FN[value_loss_type](aux_input, value_target)
                if weight is None:
                    weight = value_lambda
            elif loss_type == "policy_loss":
                p, pt = prepare_policy(aux_input, policy_target, mask=board_mask)
                aux_loss = POLICY_LOSS_FN[policy_loss_type](p, pt)
                if weight is None:
                    weight = policy_lambda
            elif loss_type == "value_uncertainty_loss":
                aux_loss = value_uncertainty_loss(*aux_input)
            elif loss_type == "value_relative_uncertainty_loss":
                aux_loss = value_relative_uncertainty_loss(*aux_input)
            elif loss_type == "policy_reg":
                if weight is None and policy_reg_lambda == 0:
                    continue
                aux_loss = policy_reg_loss(aux_input, mask=board_mask)
                print_name = aux_name
                if weight is None:
                    weight = policy_reg_lambda
            else:
                raise ValueError(f"Unsupported predefined aux loss: {loss_type}")

        if weight is None:
            weight = 1.0
        assert isinstance(aux_loss, torch.Tensor)
        total_loss += weight * aux_loss
        loss_dict[print_name] = aux_loss.detach()

    return total_loss, loss_dict
