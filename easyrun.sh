CONFIG="$1"
if [[ -z "$CONFIG" ]]; then
    echo "Please provide a config file"
    exit 1
fi
shift 1

read -p "Is this your first time running? (y/n): " answer

if [[ "$answer" == "y" || "$answer" == "Y" ]]; then
    echo "First make sure you have install a Pytorch environment and CUDA toolkit!"
    echo "Installing required packages..."
    pip install accelerate configargparse tqdm tensorboard matplotlib pybind11 lz4 pykeops

    echo "Configuring training..."
    accelerate config
fi

accelerate launch train.py -c $CONFIG "$@"