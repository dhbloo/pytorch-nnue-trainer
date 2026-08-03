#!/bin/bash

CONFIG="$1"
if [[ -z "$CONFIG" ]]; then
    echo "Please provide a config file"
    exit 1
fi
shift 1

install_packages() {
    echo "First make sure you have installed a Pytorch environment and CUDA toolkit!"
    echo "Installing required packages..."
    # Installs the dependencies from pyproject.toml and builds the dataset
    # pipeline extensions via the root setup.py.
    pip install . || exit 1
}

prompt_confirm() {
    local prompt_message="$1"
    local response
    read -p "$prompt_message (y/n): " response
    if [[ "$response" == "y" || "$response" == "Y" ]]; then
        return 0 # Success (yes)
    else
        return 1 # Failure (no)
    fi
}

NNUE_PYTHON="${NNUE_PYTHON:-python}"
NNUE_MIXED_PRECISION="${NNUE_MIXED_PRECISION:-bf16}"
NNUE_DYNAMO_BACKEND="${NNUE_DYNAMO_BACKEND:-inductor}"
NNUE_DYNAMO_MODE="${NNUE_DYNAMO_MODE:-max-autotune}"

get_accelerate_config_file() {
    "$NNUE_PYTHON" - <<'PY'
from accelerate.commands.config.config_args import default_config_file

print(default_config_file)
PY
}

set_accelerate_performance_defaults() {
    local accelerate_config_file="$1"

    "$NNUE_PYTHON" - \
        "$accelerate_config_file" \
        "$NNUE_MIXED_PRECISION" \
        "$NNUE_DYNAMO_BACKEND" \
        "$NNUE_DYNAMO_MODE" <<'PY'
import json
import sys
from pathlib import Path

import yaml
from accelerate.commands.config.config_utils import DYNAMO_BACKENDS
from accelerate.utils.constants import TORCH_DYNAMO_MODES


config_path = Path(sys.argv[1])
mixed_precision = sys.argv[2].lower()
dynamo_backend = sys.argv[3].upper()
dynamo_mode = sys.argv[4]

if mixed_precision not in {"no", "fp16", "bf16", "fp8"}:
    raise ValueError(f"Unsupported mixed precision mode: {mixed_precision}")
if dynamo_backend not in {"NO", *DYNAMO_BACKENDS}:
    raise ValueError(f"Unsupported Dynamo backend: {dynamo_backend}")
if dynamo_mode not in TORCH_DYNAMO_MODES:
    raise ValueError(f"Unsupported Dynamo mode: {dynamo_mode}")

with config_path.open(encoding="utf-8") as config_stream:
    if config_path.suffix == ".json":
        config = json.load(config_stream)
    else:
        config = yaml.safe_load(config_stream)

config["mixed_precision"] = mixed_precision
dynamo_config = config.get("dynamo_config") or {}
dynamo_config["dynamo_backend"] = dynamo_backend
dynamo_config["dynamo_mode"] = dynamo_mode
config["dynamo_config"] = dynamo_config

with config_path.open("w", encoding="utf-8") as config_stream:
    if config_path.suffix == ".json":
        json.dump(config, config_stream, indent=2)
        config_stream.write("\n")
    else:
        yaml.safe_dump(config, config_stream, sort_keys=False)

print(
    "Configured Accelerate defaults: "
    f"mixed_precision={mixed_precision}, "
    f"dynamo_backend={dynamo_backend.lower()}, "
    f"dynamo_mode={dynamo_mode}"
)
PY
}

if prompt_confirm "Is this your first time running?"; then
    install_packages
fi

ACCELERATE_CONFIG_FILE="$(get_accelerate_config_file)" || exit 1
if [[ ! -f "$ACCELERATE_CONFIG_FILE" ]]; then
    echo "No Accelerate config found. Configuring training..."
    accelerate config || exit 1
    set_accelerate_performance_defaults "$ACCELERATE_CONFIG_FILE" || exit 1
fi

accelerate launch train.py -c "$CONFIG" "$@"
