CONFIG=$1

pip install accelerate configargparse tqdm tensorboard matplotlib pybind11 lz4 pykeops
accelerate config
accelerate launch train.py -c $CONFIG