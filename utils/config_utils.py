"""Shared configuration metadata contracts."""

import yaml


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
