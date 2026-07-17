"""Common helpers shared across loss modules."""

import torch


def apply_kd_temperature(value_or_policy, target, temperature, is_value):
    """Apply temperature scaling for knowledge distillation.

    Scales the prediction by 1/T and softens the target with temperature T.

    Args:
        value_or_policy: prediction logits.
        target: teacher logits (will be softened).
        temperature: KD temperature T.
        is_value: True for value head, False for policy head.
    Returns:
        (scaled_prediction, softened_target) tuple.
    """
    if is_value:
        if target.size(1) > 1:
            target = torch.softmax(target / temperature, dim=1)
        else:
            target = torch.sigmoid(target / temperature)
        value_or_policy = value_or_policy / temperature
    else:
        # policy: target is logits, soften with temperature
        target = torch.softmax(target / temperature, dim=1)
        value_or_policy = value_or_policy / temperature
    return value_or_policy, target


def convert_value_target(value, value_target):
    """Convert value target when model has no draw info (3-tuple -> 1D).

    If value is (B, 1), squeezes channel dim. If value_target is (B, 3),
    converts to scalar winrate in [0, 1].

    Args:
        value: (B, C) or (B,) model value output.
        value_target: (B, C) or (B,) target.
    Returns:
        (value, value_target) — possibly squeezed/converted.
    """
    if value.size(1) == 1:
        value = value[:, 0]
        if value_target.size(1) == 1:
            value_target = value_target[:, 0]
        else:
            value_target = (value_target[:, 0] - value_target[:, 1] + 1) / 2
    return value, value_target


def prepare_policy(policy, policy_target, mask=None):
    """Flatten and mask policy tensors for loss computation.

    Args:
        policy: (B, *) model policy logits.
        policy_target: (B, *) target distribution.
        mask: optional (B, *) binary mask.
    Returns:
        (policy, policy_target) — flattened and masked.
    """
    policy_target = torch.flatten(policy_target, start_dim=1)
    policy = torch.flatten(policy, start_dim=1)
    assert policy_target.shape[1] == policy.shape[1]

    if mask is not None:
        mask = torch.flatten(mask, start_dim=1)
        policy = policy.masked_fill(mask == 0, -1e6)
        policy_target = policy_target.masked_fill(mask == 0, 0)

    return policy, policy_target
