"""Canonical rule/side conditioning, independent of the spatial input layout."""

import torch
from torch import nn

from utils.data_utils import Rule


class RuleConditionEncoder(nn.Module):
    """Look up fixed one-hot features without materializing intermediate IDs.

    Rule indices use ``Rule.index``. STM is -1 for black and +1 for white;
    zero is accepted only for rules which are not split by side.
    """

    def __init__(self, split_by_side=("renju",)):
        super().__init__()
        if not isinstance(split_by_side, (list, tuple)):
            raise TypeError("split_by_side must be a list of canonical rule names")
        split_rules = tuple(Rule.from_string(name) for name in split_by_side)
        if len(set(split_rules)) != len(split_rules):
            raise ValueError("split_by_side contains duplicate rules")

        names, mapping, split_mask = [], [], []
        for rule in sorted(Rule, key=lambda rule: rule.index):
            first = len(names)
            split = rule in split_rules
            names.extend((f"{rule}_black", f"{rule}_white") if split else (str(rule),))
            mapping.append((first, first + int(split)))
            split_mask.append(split)
        self.condition_names = tuple(names)
        self.dim_feature = len(names)
        # Tensor metadata preserves compatibility with EMA and weight loaders.
        self.register_buffer("condition_map", torch.tensor(mapping, dtype=torch.int64))
        self.register_buffer(
            "_lookup", torch.eye(self.dim_feature)[self.condition_map.flatten()], persistent=False
        )
        self.register_buffer("_split_mask", torch.tensor(split_mask), persistent=False)

    def validate_checkpoint_mapping(self, state_dict, prefix=""):
        """Validate semantics for model loads and parameter-only EMA restores."""
        key = prefix + "condition_map"
        incoming = state_dict.get(key)
        expected = self.condition_map
        if incoming is None or incoming.dtype != expected.dtype or not torch.equal(
            incoming.cpu(), expected.cpu()
        ):
            raise RuntimeError(f"{key}: checkpoint rule mapping differs from the configured mapping")

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        try:
            self.validate_checkpoint_mapping(state_dict, prefix)
        except RuntimeError as error:
            error_msgs.append(str(error))
            # Do not change the constructor's semantic mapping on a failed load.
            state_dict = dict(state_dict)
            state_dict[prefix + "condition_map"] = self.condition_map
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, rule_index, stm_input, inv_side=False):
        if rule_index.dtype not in (torch.int32, torch.int64):
            raise TypeError("rule_index must have dtype int32 or int64")
        if stm_input.dtype != torch.float32:
            raise TypeError("stm_input must have dtype float32")
        for name, value in (("rule_index", rule_index), ("stm_input", stm_input)):
            if value.ndim != 1 and not (value.ndim == 2 and value.shape[1] == 1):
                raise ValueError(f"{name} must have shape (B,) or (B, 1)")
        if rule_index.shape[0] != stm_input.shape[0]:
            raise ValueError("rule_index and stm_input must have the same batch size")
        if rule_index.device != stm_input.device or rule_index.device != self._lookup.device:
            raise ValueError("rule_index, stm_input, and the encoder must be on the same device")

        rules = rule_index.reshape(-1)
        stm = stm_input.reshape(-1)
        if inv_side:
            stm = -stm
        # Clamp before any gather so even invalid negative IDs cannot wrap around.
        safe_rules = rules.clamp(0, len(Rule) - 1)
        valid_rule = (rules >= 0) & (rules < len(Rule))
        valid_stm = (stm == -1) | (stm == 1) | ((stm == 0) & ~self._split_mask[safe_rules])
        valid = (valid_rule & valid_stm).all()
        message = "Invalid rule_index or stm_input: rules must be 0..2; split rules require STM -1/+1"
        if rules.device.type == "cpu" and not torch.compiler.is_compiling():
            if not valid:
                raise ValueError(message)
        else:
            # A device-side assertion avoids a host sync and remains in compiled CUDA graphs.
            torch._assert_async(valid, message)
        return self._lookup[safe_rules * 2 + (stm > 0)]
