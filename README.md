# NNUE Trainer

## Usage

### Requirements

+ 64-bit Python 3.8 and Pytorch 1.10 or later.
+ Accelerate 0.6.2 or later.
+ Other python libraries: `configargparse tqdm tensorboardX matplotlib `.

### Train a network

We recommend to launch from the accelerate-cli, which automatically handles settings for distributed training. But first of all, you need to configurate the training environment:

```
accelerate config
```

All configs about training can be directly specified from command line, for example:

```bash
accelerate launch train.py -r runs/01 --dataset_type katago_numpy --model_type mix6v2 --model_args "{dim_middle: 128, dim_policy: 32, dim_value: 32, input_type: basic}" -d ./data --batch_size 256 --iterations 1000000
```

Config can also be specified in a yaml file, which can be passed into command line:

```bash
accelerate launch train.py -c configs/train_config.yaml
```

#### Resuming from a checkpoint

Training config is saved to `run_config.yaml` under the running directory. When resuming from an interrupted run, just pass the config to the trainer. The trainer will automatically find the last checkpoint and resume from it.

```bash
accelerate launch train.py -c rundir/run_config.yaml
```

#### Train on the CPU

By defaults, accelerate uses GPU for training. However if you want to use CPU, pass `--use_cpu` to the trainer.

```bash
accelerate launch train.py --use_cpu [other options...]
```

### Test a trained network

You can evaluate a trained network on a test dataset:

```bash
accelerate launch test.py -c <path to run config> -p <path to checkpoint> -d <path to test data>
```

Or you can evaluate a trained network by running a real-time game, which does not acquire a dataset input. However, not all model supports running the real-time game, if they require extra inputs other than plain board.

```bash
accelerate launch test_play.py -c <path to run config> -p <path to checkpoint>
```

### Export a trained network

Checkpoint file can be exported to pytorch JIT, ONNX format, or serialized file for engine to read. When exporting to Pytorch JIT, ONNX, a sample dataset should be provided.

Export to Pytorch JIT:

```bash
python export.py -c <path to run config> -p <path to checkpoint> -o <output file> --export_type jit -d <path to dataset>
```

Export to ONNX:

```bash
python export.py -c <path to run config> -p <path to checkpoint> -o <output file> --export_type onnx -d <path to dataset>
```

Export to (lz4 compressed) serialized binary:

```bash
python export.py -c <path to run config> -p <path to checkpoint> -o <output file> --export_type serialization-lz4 --export_args "{[extra export options]...}"
```

### Visualize a dataset

A simple tool is provided to inspect a dataset. For example, to view a packed binary file:

```bash
python visualize_dataset.py --dataset_type packed_binary <path to data file>
```

## Serialized Weight Format

In order to improve the ease of managing weights for different rules and board size configuration, binary serialized network weights are serialized with a standardized header, which contains information about the weights, including architecture id, applicable rules and board sizes, name, description and checksum. 
