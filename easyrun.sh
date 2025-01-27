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
    pip install accelerate configargparse tqdm tensorboard matplotlib pybind11 lz4 pykeops
}

install_dataset_pipeline_modules() {
    echo "Installing dataset pipeline modules..."

    modules=(
        dataset/pipeline/line_encoding_cpp
        dataset/pipeline/forbidden_point_cpp
    )

    for module in "${modules[@]}"; do
        pip install "$module"
    done
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

if prompt_confirm "Is this your first time running?"; then
    install_packages
    if prompt_confirm "Do you want to install the dataset pipeline modules?"; then
        install_dataset_pipeline_modules
    fi
    echo "Configuring training..."
    accelerate config
fi

accelerate launch train.py -c "$CONFIG" "$@"
