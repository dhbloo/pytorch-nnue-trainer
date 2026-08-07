"""Supervised evaluation statistics: accuracy, MSE, and optional cross-eval error.

Producers return typed ``SumCount``/``Maximum`` statistics so evaluation can
combine partial batches exactly; finalization happens once at the end of
evaluation to avoid per-batch GPU syncs.
"""

import torch

from dataset.core import Maximum, SumCount
from trainer.loss.supervised import compute_supervised_loss_statistics
from trainer.loss.utils import convert_value_target, prepare_policy


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
