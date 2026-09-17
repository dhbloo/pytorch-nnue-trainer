"""Shared configuration metadata contracts."""

import ast
from collections.abc import Mapping

import yaml


def parse_optimizer_args(value):
    """Preserve numeric types in configargparse's stringified YAML mappings."""
    if isinstance(value, str):
        # YAMLConfigFileParser converts nested mappings with str(), producing
        # Python literals such as 1e-08 that YAML 1.1 treats as strings.
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            value = yaml.safe_load(value)
    if not isinstance(value, Mapping):
        raise ValueError("optim_args must be a mapping")
    return dict(value)


RUN_PROVENANCE_KEY = "_provenance"
RUN_PROVENANCE_FIELDS = frozenset(
    {
        "git_commit",
        "torch_version",
        "cuda_version",
        "accelerate_version",
    }
)


def parse_run_provenance(value):
    if isinstance(value, str):
        value = yaml.safe_load(value)
    if not isinstance(value, dict) or set(value) != RUN_PROVENANCE_FIELDS:
        raise ValueError(
            f"{RUN_PROVENANCE_KEY} must contain exactly "
            f"{sorted(RUN_PROVENANCE_FIELDS)}"
        )
    for name, field_value in value.items():
        if field_value is not None and not isinstance(field_value, str):
            raise ValueError(f"{RUN_PROVENANCE_KEY}.{name} must be a string or null")
    return value
