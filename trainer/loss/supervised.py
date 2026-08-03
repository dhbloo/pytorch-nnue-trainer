"""Supervised loss functions, dispatch tables, and orchestration.

Each loss function handles only its specific computation — no branching on loss type strings.
All functions are pure (no closures capturing outer scope).
"""

import torch
import torch.nn.functional as F

from dataset.core import SufficientStats, SumCount
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


def _dispatch_value_loss(value_loss_type, value, value_target, extra_args):
    """Dispatch a value loss, applying type-specific extra args (focal gamma for CE)."""
    if value_loss_type == "CE":
        return value_loss_ce(value, value_target, focal_gamma=extra_args.get("value_focal_gamma", 0))
    return VALUE_LOSS_FN[value_loss_type](value, value_target)


def _dispatch_policy_loss(policy_loss_type, policy, policy_target, weight, extra_args):
    """Dispatch a policy loss with per-cell weight (and focal gamma for CE)."""
    if policy_loss_type == "CE":
        return policy_loss_ce(
            policy, policy_target, weight=weight,
            focal_gamma=extra_args.get("policy_focal_gamma", 0),
        )
    return POLICY_LOSS_FN[policy_loss_type](policy, policy_target, weight=weight)


def _evaluation_mask(data, reference):
    if "is_real" not in data:
        return torch.ones(len(reference), dtype=torch.bool, device=reference.device)
    mask = data["is_real"].to(device=reference.device, dtype=torch.bool)
    if mask.ndim != 1 or len(mask) != len(reference):
        raise ValueError("is_real must contain one flag per evaluation row")
    return mask


def _selected(value, mask):
    if (
        isinstance(value, torch.Tensor)
        and value.ndim > 0
        and value.shape[0] == len(mask)
    ):
        return value[mask]
    if isinstance(value, tuple):
        return tuple(_selected(item, mask) for item in value)
    if isinstance(value, list):
        return [_selected(item, mask) for item in value]
    if isinstance(value, dict):
        return {key: _selected(item, mask) for key, item in value.items()}
    return value


def _sum_count(values):
    return SumCount(
        "global_batch",
        values.sum(),
        int(values.numel()),
    )


def _value_loss_stat(value_loss_type, value, target, extra_args):
    if value_loss_type == "MSE":
        if value.ndim == 1:
            values = (torch.sigmoid(value) - target).square()
        else:
            probability = torch.softmax(value, dim=1)
            winrate = (probability[:, 0] - probability[:, 1] + 1) / 2
            target_winrate = (target[:, 0] - target[:, 1] + 1) / 2
            values = (winrate - target_winrate).square()
    else:
        values = cross_entropy_with_softlabel(
            value,
            target,
            reduction="none",
            focal_gamma=(
                extra_args.get("value_focal_gamma", 0)
                if value_loss_type == "CE"
                else 0
            ),
            use_kl_divergence=value_loss_type == "KL",
        )
    return _sum_count(values)


def _policy_loss_stat(
    policy_loss_type, policy, target, weight, extra_args
):
    if policy_loss_type == "NONE":
        return SumCount(
            "global_batch", policy.sum() * 0, int(policy.shape[0])
        )
    if policy_loss_type == "MSE":
        values = (torch.softmax(policy, dim=1) - target).square()
        if weight is not None:
            values = values * weight
        return _sum_count(values)
    values = cross_entropy_with_softlabel(
        policy,
        target,
        reduction="none",
        weight=weight,
        focal_gamma=(
            extra_args.get("policy_focal_gamma", 0)
            if policy_loss_type == "CE"
            else 0
        ),
        use_kl_divergence=policy_loss_type == "KL",
    )
    return SumCount("global_batch", values.sum(), int(values.shape[0]))


def _policy_regularizer_stat(policy, board_mask):
    flat_policy = torch.flatten(policy, start_dim=1)
    if board_mask is None:
        value_sum = flat_policy.sum()
        valid_cells = int(flat_policy.numel())
    else:
        valid = torch.flatten(board_mask, start_dim=1).bool()
        if valid.shape != flat_policy.shape:
            raise ValueError("board mask and policy logits have incompatible shapes")
        value_sum = (flat_policy * valid.to(flat_policy.dtype)).sum()
        valid_cells = int(valid.sum().item())
    return SufficientStats(
        "global_batch",
        "policy_mean_square_v1",
        {"valid_logit_sum": value_sum},
        {"valid_cells": valid_cells},
    )


def _vq_loss_stat(payload, outer_weight=1.0):
    if (
        not isinstance(payload, dict)
        or set(payload) != {"value", "slots"}
        or not isinstance(payload["value"], torch.Tensor)
        or not isinstance(payload["slots"], list)
        or not payload["slots"]
    ):
        raise ValueError("VQ auxiliary has an invalid loss sidecar")
    tensors = {}
    counts = {}
    for slot in payload["slots"]:
        required = {
            "slot_id",
            "slot_weight",
            "embed_sum",
            "commit_sum",
            "elements",
            "vectors",
            "commitment_weight",
            "entropy_weight",
        }
        if not isinstance(slot, dict) or not required <= set(slot):
            raise ValueError("VQ loss slot has an invalid schema")
        unexpected = set(slot) - required - {"probability_sum"}
        if unexpected:
            raise ValueError(f"VQ loss slot has unexpected fields {sorted(unexpected)!r}")
        slot_id = slot["slot_id"]
        if not isinstance(slot_id, str) or not slot_id or ":" in slot_id:
            raise ValueError(f"invalid VQ slot id {slot_id!r}")
        if f"embed_sum:{slot_id}" in tensors:
            raise ValueError(f"duplicate VQ slot id {slot_id!r}")
        embed_sum = slot["embed_sum"]
        commit_sum = slot["commit_sum"]
        if (
            not isinstance(embed_sum, torch.Tensor)
            or embed_sum.ndim != 0
            or not isinstance(commit_sum, torch.Tensor)
            or commit_sum.ndim != 0
        ):
            raise ValueError(f"VQ slot {slot_id!r} sums must be scalar tensors")
        for name in ("elements", "vectors"):
            if not isinstance(slot[name], int) or slot[name] < 0:
                raise ValueError(f"VQ slot {slot_id!r} has invalid {name}")
        slot_weight = float(slot["slot_weight"]) * float(outer_weight)
        commitment_weight = float(slot["commitment_weight"])
        entropy_weight = float(slot["entropy_weight"])
        tensors.update(
            {
                f"embed_sum:{slot_id}": embed_sum,
                f"commit_sum:{slot_id}": commit_sum,
                f"slot_weight_sum:{slot_id}": embed_sum.new_tensor(slot_weight),
                f"slot_weight_square_sum:{slot_id}": embed_sum.new_tensor(
                    slot_weight**2
                ),
                f"commit_weight_sum:{slot_id}": embed_sum.new_tensor(
                    commitment_weight
                ),
                f"commit_weight_square_sum:{slot_id}": embed_sum.new_tensor(
                    commitment_weight**2
                ),
            }
        )
        counts.update(
            {
                f"elements:{slot_id}": slot["elements"],
                f"vectors:{slot_id}": slot["vectors"],
                f"constant_replicas:{slot_id}": 1,
            }
        )
        if entropy_weight != 0:
            probability_sum = slot.get("probability_sum")
            if (
                not isinstance(probability_sum, torch.Tensor)
                or probability_sum.ndim != 1
            ):
                raise ValueError(
                    f"VQ slot {slot_id!r} requires a probability-sum vector"
                )
            tensors.update(
                {
                    f"probability_sum:{slot_id}": probability_sum,
                    f"entropy_weight_sum:{slot_id}": embed_sum.new_tensor(
                        entropy_weight
                    ),
                    f"entropy_weight_square_sum:{slot_id}": embed_sum.new_tensor(
                        entropy_weight**2
                    ),
                }
            )
        elif "probability_sum" in slot:
            raise ValueError(
                f"VQ slot {slot_id!r} provides probabilities with zero entropy weight"
            )
    return SufficientStats(
        "global_batch", "vq_composition_v1", tensors, counts
    )


def _uncertainty_stat(loss_type, inputs):
    if loss_type == "value_uncertainty_loss":
        uncertainty, value, target = inputs
        values = F.huber_loss(
            uncertainty, _value_uncertainty_gt(value, target), reduction="none"
        )
    else:
        rel_uncert, value_small, value_large, target = inputs
        ug_small = _value_uncertainty_gt(value_small, target)
        ug_large = _value_uncertainty_gt(value_large, target)
        rel_target = torch.where(
            ug_large < ug_small, ug_large / ug_small, 1.0
        )
        values = F.mse_loss(rel_uncert, rel_target, reduction="none")
    return _sum_count(values)


def _add_composition_term(
    tensors, counts, *, group, ordinal, stat, weight
):
    item_id = f"{ordinal:04d}"
    if isinstance(stat, SumCount):
        prefix = f"{group}.{item_id}"
        tensors[f"mean_sum:{prefix}"] = stat.sum
        tensors[f"mean_weight_sum:{prefix}"] = stat.sum.new_tensor(weight)
        tensors[f"mean_weight_square_sum:{prefix}"] = stat.sum.new_tensor(
            float(weight) ** 2
        )
        counts[f"mean_count:{prefix}"] = stat.count
        counts[f"mean_replicas:{prefix}"] = 1
        return
    if (
        isinstance(stat, SufficientStats)
        and stat.finalizer_id == "policy_mean_square_v1"
    ):
        prefix = f"{group}.{item_id}"
        value_sum = stat.tensors["valid_logit_sum"]
        tensors[f"reg_sum:{prefix}"] = value_sum
        tensors[f"reg_weight_sum:{prefix}"] = value_sum.new_tensor(weight)
        tensors[f"reg_weight_square_sum:{prefix}"] = value_sum.new_tensor(
            float(weight) ** 2
        )
        counts[f"reg_cells:{prefix}"] = stat.counts["valid_cells"]
        counts[f"reg_replicas:{prefix}"] = 1
        return
    if (
        isinstance(stat, SufficientStats)
        and stat.finalizer_id == "vq_composition_v1"
    ):
        scaled = _scale_vq_stat(stat, weight)
        prefix = f"vq:{group}.{item_id}:"
        tensors.update({prefix + key: value for key, value in scaled.tensors.items()})
        counts.update({prefix + key: value for key, value in scaled.counts.items()})
        return
    raise ValueError(f"unsupported total-loss statistic {type(stat).__name__}")


def _scale_vq_stat(stat, weight):
    tensors = dict(stat.tensors)
    for key in tuple(tensors):
        if not key.startswith("slot_weight_sum:"):
            continue
        slot_id = key.split(":", 1)[1]
        tensors[key] = tensors[key] * float(weight)
        square_key = f"slot_weight_square_sum:{slot_id}"
        tensors[square_key] = tensors[square_key] * float(weight) ** 2
    return SufficientStats(stat.scope, stat.finalizer_id, tensors, dict(stat.counts))


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
    if "is_real" in data:
        is_real = data["is_real"].bool()
        if is_real.ndim != 1:
            raise ValueError(f"is_real must have shape (B,), got {tuple(is_real.shape)}")
        batch_size = len(is_real)
        filtered_data = {
            key: (
                value[is_real]
                if isinstance(value, torch.Tensor)
                and value.ndim > 0
                and value.shape[0] == batch_size
                else value
            )
            for key, value in data.items()
            if key != "is_real"
        }

        def filter_tree(value):
            if (
                isinstance(value, torch.Tensor)
                and value.ndim > 0
                and value.shape[0] == batch_size
            ):
                return value[is_real]
            if isinstance(value, dict):
                return {key: filter_tree(item) for key, item in value.items()}
            if isinstance(value, tuple):
                return tuple(filter_tree(item) for item in value)
            if isinstance(value, list):
                return [filter_tree(item) for item in value]
            return value

        if not torch.any(is_real):
            if len(is_real) == 0:
                raise ValueError("masked evaluation batch must contain padded rows")
            probe_data = {
                key: (
                    value[:1]
                    if isinstance(value, torch.Tensor)
                    and value.ndim > 0
                    and value.shape[0] == batch_size
                    else value
                )
                for key, value in data.items()
                if key != "is_real"
            }
            def first_tree(value):
                if (
                    isinstance(value, torch.Tensor)
                    and value.ndim > 0
                    and value.shape[0] == batch_size
                ):
                    return value[:1]
                if isinstance(value, dict):
                    return {key: first_tree(item) for key, item in value.items()}
                if isinstance(value, tuple):
                    return tuple(first_tree(item) for item in value)
                if isinstance(value, list):
                    return [first_tree(item) for item in value]
                return value

            probe_results = first_tree(results)
            probe_kd = first_tree(kd_results) if kd_results is not None else None
            _, probe_losses, probe_aux = compute_supervised_losses(
                loss_type,
                probe_data,
                probe_results,
                kd_results=probe_kd,
                kd_T=kd_T,
                kd_alpha=kd_alpha,
                policy_reg_lambda=policy_reg_lambda,
                value_policy_ratio=value_policy_ratio,
                **extra_args,
            )
            zero = results["value"].sum() * 0
            return (
                zero,
                {key: zero.detach().to(value.dtype) for key, value in probe_losses.items()},
                {
                    key: torch.zeros_like(value)
                    for key, value in probe_aux.items()
                },
            )
        return compute_supervised_losses(
            loss_type,
            filtered_data,
            filter_tree(results),
            kd_results=filter_tree(kd_results) if kd_results is not None else None,
            kd_T=kd_T,
            kd_alpha=kd_alpha,
            policy_reg_lambda=policy_reg_lambda,
            value_policy_ratio=value_policy_ratio,
            **extra_args,
        )

    terms, loss_dict = _compute_loss_terms(
        loss_type, data, results, kd_results, kd_T,
        policy_reg_lambda, value_policy_ratio, extra_args,
    )

    # ── knowledge distillation merge ──────────────────────────
    if kd_results is not None:
        real_terms, real_loss_dict = _compute_loss_terms(
            loss_type, data, results, None, kd_T,
            policy_reg_lambda, value_policy_ratio, extra_args,
        )
        kd_loss = _sum_terms(terms, target_dependent=True)
        real_loss = _sum_terms(real_terms, target_dependent=True)
        if kd_alpha == 1.0:
            # ground-truth loss is only logged; keep it out of the backward graph
            real_loss = real_loss.detach()
        total_loss = kd_alpha * kd_loss * (kd_T**2) + (1 - kd_alpha) * real_loss
        # target-independent terms (regularizers, plain-tensor aux losses) are
        # identical in both branches: add them once, unscaled by T^2 or alpha
        indep_loss = _sum_terms(real_terms, target_dependent=False)
        if indep_loss is not None:
            total_loss = total_loss + indep_loss
        kd_loss_dict = {"kd_" + k: v for k, v in loss_dict.items()}
        loss_dict = real_loss_dict
        loss_dict.update(kd_loss_dict)
    else:
        total_loss = _sum_terms(terms)

    # ── aux outputs ───────────────────────────────────────────
    aux_outputs = results.get("aux_outputs")
    aux_outputs_dict = {}
    if aux_outputs:
        for name, out in aux_outputs.items():
            aux_outputs_dict[name] = out.detach()

    loss_dict["total_loss"] = total_loss.detach()
    return total_loss, loss_dict, aux_outputs_dict


def _compute_evaluation_terms(
    loss_type,
    data,
    results,
    kd_results,
    kd_T,
    policy_reg_lambda,
    value_policy_ratio,
    extra_args,
):
    value_loss_type, policy_loss_type = loss_type.split("+")
    value = results["value"]
    policy = results["policy"]
    mask = _evaluation_mask(data, value)
    value = value[mask]
    policy = policy[mask]
    board_mask = _selected(results.get("board_mask"), mask)

    if kd_results is None:
        value_target = _selected(data["value_target"], mask)
        policy_target = _selected(data["policy_target"], mask)
    else:
        value_target = _selected(kd_results["value"], mask)
        policy_target = _selected(kd_results["policy"], mask)

    v, vt = value, value_target
    if kd_results is not None:
        v, vt = apply_kd_temperature(v, vt, kd_T, is_value=True)
    aux_value_target = vt
    v, vt = convert_value_target(v, vt)
    value_stat = _value_loss_stat(value_loss_type, v, vt, extra_args)

    p, pt = prepare_policy(policy, policy_target, mask=board_mask)
    if kd_results is not None:
        if board_mask is not None:
            flat_mask = torch.flatten(board_mask, start_dim=1)
            pt = pt.masked_fill(flat_mask == 0, -1e6)
        pt = torch.softmax(pt / kd_T, dim=1)
        p = p / kd_T
        if board_mask is not None:
            pt = pt.masked_fill(flat_mask == 0, 0)

    if extra_args.get("ignore_forbidden_point_policy", False):
        forbidden = torch.flatten(
            _selected(data["forbidden_point"], mask), start_dim=1
        ).float()
        policy_loss_weight = 1.0 - forbidden
    else:
        policy_loss_weight = None
    policy_stat = _policy_loss_stat(
        policy_loss_type, p, pt, policy_loss_weight, extra_args
    )

    value_lambda = 2 * value_policy_ratio / (value_policy_ratio + 1)
    policy_lambda = 2 / (value_policy_ratio + 1)
    terms = [
        (value_stat, True, value_lambda),
        (policy_stat, True, policy_lambda),
    ]
    loss_dict = {
        "value_loss": value_stat,
        "policy_loss": policy_stat,
    }

    if float(policy_reg_lambda) > 0:
        reg_stat = _policy_regularizer_stat(policy, board_mask)
        terms.append((reg_stat, False, float(policy_reg_lambda)))
        loss_dict["policy_reg"] = reg_stat

    aux_losses = results.get("aux_losses")
    if aux_losses:
        for aux_name, aux_loss in aux_losses.items():
            if not (
                isinstance(aux_loss, tuple)
                and len(aux_loss) == 2
                and isinstance(aux_loss[0], str)
            ):
                raise ValueError(
                    f"evaluation auxiliary loss {aux_name!r} is opaque; "
                    "use a registered predefined loss tuple"
                )
            loss_kind, aux_input = aux_loss
            print_name = f"{aux_name}_loss"
            weight = extra_args.get(f"{aux_name}_lambda")
            target_dependent = False
            if loss_kind == "value_loss":
                aux_value = _selected(aux_input, mask)
                if kd_results is not None:
                    aux_value = aux_value / kd_T
                aux_value, aux_target = convert_value_target(
                    aux_value, aux_value_target
                )
                stat = _value_loss_stat(
                    value_loss_type, aux_value, aux_target, extra_args
                )
                target_dependent = True
                if weight is None:
                    weight = value_lambda
            elif loss_kind == "policy_loss":
                if policy_loss_type == "NONE":
                    continue
                aux_policy = _selected(aux_input, mask)
                aux_policy, aux_target = prepare_policy(
                    aux_policy, pt, mask=board_mask
                )
                if kd_results is not None:
                    aux_policy = aux_policy / kd_T
                stat = _policy_loss_stat(
                    policy_loss_type,
                    aux_policy,
                    aux_target,
                    policy_loss_weight,
                    extra_args,
                )
                target_dependent = True
                if weight is None:
                    weight = policy_lambda
            elif loss_kind in {
                "value_uncertainty_loss",
                "value_relative_uncertainty_loss",
            }:
                stat = _uncertainty_stat(
                    loss_kind, _selected(aux_input, mask)
                )
            elif loss_kind == "policy_reg":
                if weight is None and policy_reg_lambda == 0:
                    continue
                stat = _policy_regularizer_stat(
                    _selected(aux_input, mask), board_mask
                )
                print_name = aux_name
                if weight is None:
                    weight = policy_reg_lambda
            elif loss_kind == "vq_loss":
                stat = _vq_loss_stat(aux_input)
            else:
                raise ValueError(f"Unsupported predefined aux loss: {loss_kind}")
            if weight is None:
                weight = 1.0
            weight = float(weight)
            terms.append((stat, target_dependent, weight))
            loss_dict[print_name] = stat
    return terms, loss_dict


def _composition_stat(terms, *, finalizer_id, kd_alpha=None, kd_T=None):
    tensors = {}
    counts = {}
    for ordinal, (group, stat, weight) in enumerate(terms):
        _add_composition_term(
            tensors,
            counts,
            group=group,
            ordinal=ordinal,
            stat=stat,
            weight=weight,
        )
    if finalizer_id == "kd_total_v1":
        reference = next(iter(tensors.values()))
        alpha = float(kd_alpha)
        temperature = float(kd_T)
        tensors.update(
            {
                "kd_alpha_sum": reference.new_tensor(alpha),
                "kd_alpha_square_sum": reference.new_tensor(alpha**2),
                "kd_temperature_sum": reference.new_tensor(temperature),
                "kd_temperature_square_sum": reference.new_tensor(
                    temperature**2
                ),
            }
        )
        counts["kd_config_replicas"] = 1
    return SufficientStats(
        "global_batch", finalizer_id, tensors, counts
    )


def _evaluation_aux_output_statistics(results):
    typed = {}
    for name, value in (results.get("aux_outputs") or {}).items():
        if name in {"vq_perplexity", "vq_normed_perplexity"}:
            continue
        if name.startswith("vq_cluster_size_q") or name == "vq_num_expired_codes":
            if not isinstance(value, torch.Tensor) or value.ndim != 0:
                raise ValueError(f"{name} must be a scalar replicated constant")
            typed[name] = SufficientStats(
                "evaluation",
                "replicated_constant_v1",
                {"sum": value.detach(), "square_sum": value.detach().square()},
                {"replicas": 1},
            )
            continue
        raise ValueError(
            f"evaluation auxiliary output {name!r} is opaque; "
            "use a registered sufficient statistic"
        )
    return typed


def compute_supervised_loss_statistics(
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
    """Return exact typed statistics for validation loss and auxiliary output."""
    kd_terms, kd_loss_dict = _compute_evaluation_terms(
        loss_type,
        data,
        results,
        kd_results,
        kd_T,
        policy_reg_lambda,
        value_policy_ratio,
        extra_args,
    )
    if kd_results is None:
        total_terms = [
            ("all", stat, weight) for stat, _, weight in kd_terms
        ]
        loss_dict = kd_loss_dict
        loss_dict["total_loss"] = _composition_stat(
            total_terms, finalizer_id="loss_composition_v1"
        )
    else:
        real_terms, real_loss_dict = _compute_evaluation_terms(
            loss_type,
            data,
            results,
            None,
            kd_T,
            policy_reg_lambda,
            value_policy_ratio,
            extra_args,
        )
        total_terms = [
            ("kd", stat, weight)
            for stat, dependent, weight in kd_terms
            if dependent
        ]
        total_terms.extend(
            ("real", stat, weight)
            for stat, dependent, weight in real_terms
            if dependent
        )
        total_terms.extend(
            ("independent", stat, weight)
            for stat, dependent, weight in real_terms
            if not dependent
        )
        loss_dict = real_loss_dict
        loss_dict.update(
            {f"kd_{name}": stat for name, stat in kd_loss_dict.items()}
        )
        loss_dict["total_loss"] = _composition_stat(
            total_terms,
            finalizer_id="kd_total_v1",
            kd_alpha=kd_alpha,
            kd_T=kd_T,
        )
    return loss_dict, _evaluation_aux_output_statistics(results)


def _sum_terms(terms, target_dependent=None):
    """Sum loss terms left-to-right, optionally filtering by target dependence."""
    total = None
    for term, is_dep in terms:
        if target_dependent is not None and is_dep != target_dependent:
            continue
        total = term if total is None else total + term
    return total


def _compute_loss_terms(
    loss_type, data, results, kd_results, kd_T,
    policy_reg_lambda, value_policy_ratio, extra_args,
):
    """Compute all loss terms against one target source (teacher if kd_results, else data).

    Returns (terms, loss_dict) where terms is an ordered list of
    (loss_tensor, is_target_dependent) contributions to the total loss.
    """
    value_loss_type, policy_loss_type = loss_type.split("+")

    # unpack model outputs
    value, policy = results["value"], results["policy"]
    aux_losses = results.get("aux_losses")
    board_mask = results.get("board_mask")

    if kd_results is not None:
        value_target, policy_target = kd_results["value"], kd_results["policy"]
    else:
        value_target, policy_target = data["value_target"], data["policy_target"]

    # ── value loss ────────────────────────────────────────────
    v, vt = value, value_target
    if kd_results is not None:
        v, vt = apply_kd_temperature(v, vt, kd_T, is_value=True)
    aux_value_target = vt  # pre-conversion target; aux heads convert per their own shape
    v, vt = convert_value_target(v, vt)
    v_loss = _dispatch_value_loss(value_loss_type, v, vt, extra_args)

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
    else:
        p_loss = _dispatch_policy_loss(policy_loss_type, p, pt, policy_loss_weight, extra_args)

    # ── combine ───────────────────────────────────────────────
    value_lambda = 2 * value_policy_ratio / (value_policy_ratio + 1)
    policy_lambda = 2 / (value_policy_ratio + 1)
    terms = [(value_lambda * v_loss + policy_lambda * p_loss, True)]
    loss_dict = {"value_loss": v_loss.detach(), "policy_loss": p_loss.detach()}

    # policy regularization (target-independent)
    if float(policy_reg_lambda) > 0:
        p_reg = policy_reg_loss(policy, mask=board_mask)
        terms.append((float(policy_reg_lambda) * p_reg, False))
        loss_dict["policy_reg"] = p_reg.detach()

    # ── aux losses ────────────────────────────────────────────
    _apply_aux_losses(
        terms, loss_dict, aux_losses,
        value_loss_type=value_loss_type, policy_loss_type=policy_loss_type,
        value_target=aux_value_target, policy_target=pt, board_mask=board_mask,
        policy_loss_weight=policy_loss_weight,
        kd_T=kd_T if kd_results is not None else None,
        value_lambda=value_lambda, policy_lambda=policy_lambda,
        policy_reg_lambda=policy_reg_lambda,
        extra_args=extra_args,
    )

    return terms, loss_dict


def _apply_aux_losses(
    terms, loss_dict, aux_losses, *,
    value_loss_type, policy_loss_type,
    value_target, policy_target, board_mask, policy_loss_weight, kd_T,
    value_lambda, policy_lambda, policy_reg_lambda,
    extra_args,
):
    """Append model-returned aux loss terms to terms / loss_dict.

    ``kd_T`` is the KD temperature when the targets are teacher outputs, else
    None. Terms are tagged target-dependent (True) when they compare model
    outputs against the (possibly temperature-softened) targets; regularizers
    and plain tensor losses are tagged False so the KD merge adds them once.
    """
    if not aux_losses:
        return

    for aux_name, aux_loss in aux_losses.items():
        print_name = f"{aux_name}_loss"
        weight = extra_args.get(f"{aux_name}_lambda", None)
        if weight is not None:
            weight = float(weight)
        target_dependent = False

        # pre-defined loss terms: (loss_type_string, inputs)
        if isinstance(aux_loss, tuple) and len(aux_loss) == 2:
            loss_type, aux_input = aux_loss
            if loss_type == "value_loss":
                aux_v = aux_input if kd_T is None else aux_input / kd_T
                aux_v, aux_vt = convert_value_target(aux_v, value_target)
                aux_loss = _dispatch_value_loss(value_loss_type, aux_v, aux_vt, extra_args)
                target_dependent = True
                if weight is None:
                    weight = value_lambda
            elif loss_type == "policy_loss":
                if policy_loss_type == "NONE":
                    continue
                p, pt = prepare_policy(aux_input, policy_target, mask=board_mask)
                if kd_T is not None:
                    p = p / kd_T
                aux_loss = _dispatch_policy_loss(
                    policy_loss_type, p, pt, policy_loss_weight, extra_args
                )
                target_dependent = True
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
            elif loss_type == "vq_loss":
                if (
                    not isinstance(aux_input, dict)
                    or set(aux_input) != {"value", "slots"}
                    or not isinstance(aux_input["value"], torch.Tensor)
                ):
                    raise ValueError(
                        f"VQ auxiliary {aux_name!r} has invalid loss sidecar"
                    )
                aux_loss = aux_input["value"]
            else:
                raise ValueError(f"Unsupported predefined aux loss: {loss_type}")

        if weight is None:
            weight = 1.0
        assert isinstance(aux_loss, torch.Tensor)
        terms.append((weight * aux_loss, target_dependent))
        loss_dict[print_name] = aux_loss.detach()
