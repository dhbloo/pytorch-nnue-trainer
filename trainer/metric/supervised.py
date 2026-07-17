"""Supervised evaluation metrics: accuracy, MSE, and optional cross-eval error."""

import torch

from trainer.loss.supervised import compute_supervised_losses
from trainer.loss.utils import convert_value_target, prepare_policy


def top_k_accuracy(policy, policy_target, k):
    """Compute top-k move overlap accuracy between policy and target."""
    _, topkmoves = torch.topk(policy, dim=1, k=k, sorted=False)
    _, topkmoves_target = torch.topk(policy_target, dim=1, k=k, sorted=False)
    move_correct_count = 0
    for moves, moves_target in zip(topkmoves, topkmoves_target):
        moveset = set(moves.tolist())
        moveset_target = set(moves_target.tolist())
        move_correct_count += len(moveset.intersection(moveset_target))
    return move_correct_count / k / len(topkmoves)


def _ce_loss_metrics(data, results):
    """Compute CE loss metrics and return as ``{lossCE_<name>: float}``."""
    _, losses_ce_ce, _ = compute_supervised_losses("CE+CE", data, results)
    return {f"lossCE_{k}": v.item() for k, v in losses_ce_ce.items()}


def _policy_accuracy_metrics(policy, policy_target):
    """Compute bestmove and top-k policy accuracy metrics."""
    bestmove_eq = torch.argmax(policy, dim=1) == torch.argmax(policy_target, dim=1)
    bestmove_acc = torch.sum(bestmove_eq) / bestmove_eq.size(0)
    return {
        "bestmove_acc": bestmove_acc.item(),
        "top2move_acc": top_k_accuracy(policy, policy_target, k=2),
        "top3move_acc": top_k_accuracy(policy, policy_target, k=3),
    }


def _value_accuracy_metrics(value, value_target):
    """Compute winrate accuracy/MSE (and drawrate if 3-head value)."""
    metrics = {}
    if value.ndim == 1:
        value = torch.sigmoid(value)
        winrate_correct = (value - 0.5) * (value_target - 0.5) >= 0
        winrate_mse = torch.mean((value - value_target) ** 2)
    else:
        value = torch.softmax(value, dim=1)

        value_iswin = value[:, 0] >= value[:, 1]
        value_iswin_target = value_target[:, 0] >= value_target[:, 1]
        winrate_correct = value_iswin == value_iswin_target

        value_norm = (value[:, 0] - value[:, 1] + 1) / 2
        value_norm_target = (value_target[:, 0] - value_target[:, 1] + 1) / 2
        winrate_mse = torch.mean((value_norm - value_norm_target) ** 2)

        value_isdraw = torch.argmax(value, dim=1) == 2
        value_isdraw_target = torch.argmax(value_target, dim=1) == 2
        drawrate_correct = value_isdraw == value_isdraw_target
        drawrate_mse = torch.mean((value[:, 2] - value_target[:, 2]) ** 2)

        metrics["drawrate_acc"] = (torch.sum(drawrate_correct) / drawrate_correct.size(0)).item()
        metrics["drawrate_mse"] = drawrate_mse.item()

    metrics["winrate_acc"] = (torch.sum(winrate_correct) / winrate_correct.size(0)).item()
    metrics["winrate_mse"] = winrate_mse.item()
    return metrics, value, value_target


def _cross_eval_metrics(value, value_target):
    """Compute absolute and relative value prediction errors."""
    abs_err = torch.abs(value - value_target)
    return {
        "value_abserr_mean": abs_err.mean().item(),
        "value_abserr_max": abs_err.max().item(),
        "value_relerr_mean": (abs_err / (torch.abs(value) + 1e-4)).mean().item(),
        "value_relerr_max": (abs_err / (torch.abs(value) + 1e-4)).max().item(),
    }


def compute_supervised_metrics(data, results, do_cross_eval=False):
    """Compute supervised metrics from model outputs.

    Returns a dict of Python floats with CE losses, bestmove/top-k accuracy,
    value accuracy/MSE, draw metrics, and optional cross-eval errors.
    """
    value, policy, *retvals = results
    value_target = data["value_target"]
    policy_target = data["policy_target"]

    policy, policy_target = prepare_policy(policy, policy_target)
    value, value_target = convert_value_target(value, value_target)

    metrics = _ce_loss_metrics(data, results)
    metrics.update(_policy_accuracy_metrics(policy, policy_target))

    value_metrics, value, value_target = _value_accuracy_metrics(value, value_target)
    metrics.update(value_metrics)

    if do_cross_eval:
        metrics.update(_cross_eval_metrics(value, value_target))

    return metrics
