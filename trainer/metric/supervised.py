"""Supervised evaluation metrics: accuracy, MSE, and optional cross-eval error.

All helpers return 0-dim GPU tensors; conversion to Python floats happens once
at the end of evaluation (in ``BaseTrainer.test``) to avoid per-batch GPU syncs.
"""

import torch

from dataset.core import Maximum, SumCount
from trainer.loss.supervised import (
    compute_supervised_losses,
    compute_supervised_loss_statistics,
)
from trainer.loss.utils import convert_value_target, prepare_policy


def top_k_accuracy(policy, policy_target, k):
    """Compute top-k move overlap accuracy between policy and target."""
    _, topkmoves = torch.topk(policy, dim=1, k=k, sorted=False)
    _, topkmoves_target = torch.topk(policy_target, dim=1, k=k, sorted=False)
    # Count per-sample intersection size; top-k indices are unique within a row,
    # so checking each predicted move against all target moves counts overlaps.
    overlap = (topkmoves.unsqueeze(2) == topkmoves_target.unsqueeze(1)).any(dim=2).sum(dim=1)
    return overlap.float().mean() / k


def _ce_loss_metrics(data, results):
    """Compute CE loss metrics and return as ``{lossCE_<name>: tensor}``."""
    _, losses_ce_ce, _ = compute_supervised_losses("CE+CE", data, results)
    return {f"lossCE_{k}": v for k, v in losses_ce_ce.items()}


def _policy_accuracy_metrics(policy, policy_target):
    """Compute bestmove and top-k policy accuracy metrics."""
    bestmove_eq = torch.argmax(policy, dim=1) == torch.argmax(policy_target, dim=1)
    return {
        "bestmove_acc": bestmove_eq.float().mean(),
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

        metrics["drawrate_acc"] = drawrate_correct.float().mean()
        metrics["drawrate_mse"] = drawrate_mse

    metrics["winrate_acc"] = winrate_correct.float().mean()
    metrics["winrate_mse"] = winrate_mse
    return metrics, value, value_target


def _cross_eval_metrics(value, value_target):
    """Compute absolute and relative value prediction errors."""
    abs_err = torch.abs(value - value_target)
    return {
        "value_abserr_mean": abs_err.mean(),
        "value_abserr_max": abs_err.max(),
        "value_relerr_mean": (abs_err / (torch.abs(value) + 1e-4)).mean(),
        "value_relerr_max": (abs_err / (torch.abs(value) + 1e-4)).max(),
    }


def compute_supervised_metrics(data, results, do_cross_eval=False):
    """Compute supervised metrics from model outputs.

    Returns a dict of 0-dim tensors with CE losses, bestmove/top-k accuracy,
    value accuracy/MSE, draw metrics, and optional cross-eval errors.
    """
    if "is_real" in data:
        mask = data["is_real"].bool()
        if mask.ndim != 1:
            raise ValueError(f"is_real must have shape (B,), got {tuple(mask.shape)}")
        if not torch.any(mask):
            if len(mask) == 0:
                raise ValueError("masked evaluation batch must contain padded rows")

            def first_row(value):
                if (
                    isinstance(value, torch.Tensor)
                    and value.ndim > 0
                    and value.shape[0] == len(mask)
                ):
                    return value[:1]
                if isinstance(value, dict):
                    return {key: first_row(item) for key, item in value.items()}
                if isinstance(value, tuple):
                    return tuple(first_row(item) for item in value)
                if isinstance(value, list):
                    return [first_row(item) for item in value]
                return value

            probe_data = {
                key: first_row(value)
                for key, value in data.items()
                if key != "is_real"
            }
            probe_metrics = compute_supervised_metrics(
                probe_data,
                first_row(results),
                do_cross_eval=do_cross_eval,
            )
            zero = results["value"].sum() * 0
            return {name: zero.to(value.dtype) for name, value in probe_metrics.items()}
        batch_size = len(mask)
        data = {
            key: (
                value[mask]
                if isinstance(value, torch.Tensor)
                and value.ndim > 0
                and value.shape[0] == batch_size
                else value
            )
            for key, value in data.items()
            if key != "is_real"
        }
        results = {
            key: (
                value[mask]
                if isinstance(value, torch.Tensor)
                and value.ndim > 0
                and value.shape[0] == batch_size
                else value
            )
            for key, value in results.items()
        }

    value, policy = results["value"], results["policy"]
    value_target = data["value_target"]
    policy_target = data["policy_target"]

    policy, policy_target = prepare_policy(policy, policy_target, mask=results.get("board_mask"))
    value, value_target = convert_value_target(value, value_target)

    metrics = _ce_loss_metrics(data, results)
    metrics.update(_policy_accuracy_metrics(policy, policy_target))

    value_metrics, value, value_target = _value_accuracy_metrics(value, value_target)
    metrics.update(value_metrics)

    if do_cross_eval:
        metrics.update(_cross_eval_metrics(value, value_target))

    return metrics


def compute_supervised_metric_statistics(data, results, do_cross_eval=False):
    """Compute exact typed test statistics without guessing scalar denominators."""
    value = results["value"]
    if "is_real" in data:
        mask = data["is_real"].to(device=value.device, dtype=torch.bool)
        if mask.ndim != 1 or len(mask) != len(value):
            raise ValueError("is_real must contain one flag per evaluation row")
    else:
        mask = torch.ones(len(value), dtype=torch.bool, device=value.device)
    batch_size = len(mask)

    def select(item):
        if (
            isinstance(item, torch.Tensor)
            and item.ndim > 0
            and item.shape[0] == batch_size
        ):
            return item[mask]
        return item

    selected_data = {
        key: select(item) for key, item in data.items() if key != "is_real"
    }
    selected_results = {
        key: select(item)
        for key, item in results.items()
        if key not in {"aux_losses", "aux_outputs"}
    }
    value = selected_results["value"]
    policy = selected_results["policy"]
    value_target = selected_data["value_target"]
    policy_target = selected_data["policy_target"]
    policy, policy_target = prepare_policy(
        policy, policy_target, mask=selected_results.get("board_mask")
    )
    value, value_target = convert_value_target(value, value_target)

    ce_stats, _ = compute_supervised_loss_statistics("CE+CE", data, results)
    metrics = {f"lossCE_{key}": stat for key, stat in ce_stats.items()}

    bestmove_eq = torch.argmax(policy, dim=1) == torch.argmax(
        policy_target, dim=1
    )
    metrics["bestmove_acc"] = SumCount(
        "global_batch", bestmove_eq.float().sum(), int(len(bestmove_eq))
    )
    for k in (2, 3):
        _, predicted = torch.topk(policy, dim=1, k=k, sorted=False)
        _, target = torch.topk(policy_target, dim=1, k=k, sorted=False)
        overlap = (
            (predicted.unsqueeze(2) == target.unsqueeze(1))
            .any(dim=2)
            .sum()
        )
        metrics[f"top{k}move_acc"] = SumCount(
            "global_batch", overlap.float(), int(k * len(policy))
        )

    if value.ndim == 1:
        probability = torch.sigmoid(value)
        winrate_correct = (
            (probability - 0.5) * (value_target - 0.5) >= 0
        )
        winrate_error = (probability - value_target).square()
    else:
        probability = torch.softmax(value, dim=1)
        winrate_correct = (probability[:, 0] >= probability[:, 1]) == (
            value_target[:, 0] >= value_target[:, 1]
        )
        winrate = (probability[:, 0] - probability[:, 1] + 1) / 2
        target_winrate = (
            value_target[:, 0] - value_target[:, 1] + 1
        ) / 2
        winrate_error = (winrate - target_winrate).square()
        drawrate_correct = torch.argmax(probability, dim=1) == 2
        drawrate_correct_target = torch.argmax(value_target, dim=1) == 2
        drawrate_error = (
            probability[:, 2] - value_target[:, 2]
        ).square()
        metrics["drawrate_acc"] = SumCount(
            "global_batch",
            (drawrate_correct == drawrate_correct_target).float().sum(),
            int(len(probability)),
        )
        metrics["drawrate_mse"] = SumCount(
            "global_batch", drawrate_error.sum(), int(len(probability))
        )
    metrics["winrate_acc"] = SumCount(
        "global_batch", winrate_correct.float().sum(), int(len(probability))
    )
    metrics["winrate_mse"] = SumCount(
        "global_batch", winrate_error.sum(), int(len(probability))
    )

    if do_cross_eval:
        abs_error = torch.abs(probability - value_target)
        relative_error = abs_error / (torch.abs(probability) + 1e-4)
        metrics.update(
            {
                "value_abserr_mean": SumCount(
                    "global_batch", abs_error.sum(), int(abs_error.numel())
                ),
                "value_abserr_max": Maximum(
                    "evaluation",
                    (
                        abs_error.max()
                        if abs_error.numel()
                        else probability.new_tensor(-torch.inf)
                    ),
                    int(abs_error.numel()),
                ),
                "value_relerr_mean": SumCount(
                    "global_batch",
                    relative_error.sum(),
                    int(relative_error.numel()),
                ),
                "value_relerr_max": Maximum(
                    "evaluation",
                    (
                        relative_error.max()
                        if relative_error.numel()
                        else probability.new_tensor(-torch.inf)
                    ),
                    int(relative_error.numel()),
                ),
            }
        )
    return metrics
