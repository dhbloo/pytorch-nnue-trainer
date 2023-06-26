# NNUE Trainer

## Usage

### Requirements

+ 64-bit Python 3.9 and Pytorch 1.10 or later.
+ Huggingface Accelerate 0.6.2 or later.
+ Other python libraries: `configargparse tqdm tensorboard matplotlib pybind11 lz4`.

### Setup dataset pipeline 

After install all required packages in specified in requirements, it is necessary to build some extra c++ sources for the trainer to transform some data into features. First of all, you need to setup the C++ compiling environment. Then install the dataset pipelines by doing the following commands.

+ Line Encoding: Fast line encoding for transforming board features.

  ```bash
  cd dataset/pipeline/line_encoding_cpp
  python setup.py build
  python setup.py install
  ```

+ Forbidden Point (optional): Finding forbidden points for Renju rule. Can be skipped if you don't need to train a network for Renju rule.

  ```bash
  cd dataset/pipeline/forbidden_point_cpp
  python setup.py build
  python setup.py install
  ```

### Train a network

You are recommended to launch from the accelerate-cli, which automatically handles settings for distributed training. But first of all, you need to configurate the training environment:

```
accelerate config
```

All configs about training can be directly specified from command line. For example, suppose you have a unprocessed katago selfplay dataset in folder `./data`, and you want to train a `mix6` network with specific model arguments, you can run the following command:

```bash
accelerate launch train.py -r run_dirs/run01 -d ./data --dataset_type katago_numpy --model_type mix6 --model_args "{dim_middle: 128, dim_policy: 32, dim_value: 32, input_type: basic}" --batch_size 256 --iterations 1000000
```

Configs can also be specified in a yaml file to be reused for multiple runs, which can be passed in command line using `-c` flag. For example, if you have a config file `configs/train_config.yaml`, you can run the following command:

```bash
accelerate launch train.py -c configs/train_config.yaml
```

Examples of config file can be found in `configs/example`.

#### Resuming from a checkpoint

Training config is saved to `run_config.yaml` under the running directory. When resuming from an interrupted run, just pass the config to the trainer. The trainer will automatically find the last checkpoint and resume from it. For example, if you have a run directory at `run_dirs/run01` and want to resume from it, you can run the following command:

```bash
accelerate launch train.py -c run_dirs/run01/run_config.yaml
```

#### Train on the CPU

By defaults, accelerate uses GPU for training. However if you want to use CPU only, pass `--use_cpu` to the trainer.

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

Export to (lz4 compressed) serialized binary to be used in Rapfi:

```bash
python export.py -c <path to run config> -p <path to checkpoint> -o <output file> --export_type serialization-lz4 --export_args "{[extra export options]...}"
```

Note that some NNUE serializer requires specifying exporting arguments, such as rule (options are `freestyle`, `standard`, `renju`), board size (usually a number between 5 and 22), etc. For example, to export a binary network file of `mix7` NNUE for Renju rule and 15x15 board size, you can run the following command:

```bash
python export.py -c <path to run config> -p <path to checkpoint> -o <output file> --export_type serialization-lz4 --export_args "{rule: renju, board_size: 15}"
```

### Visualize a dataset

A simple tool is provided to inspect a dataset. For example, to view a processed katago dataset file (`.npz`):

```bash
python visualize_dataset.py --dataset_type processed_katago_numpy <path to data file>
```

Or to view a packed binary dataset file (`.binpack`) produced by [c-gomoku-cli](https://github.com/dhbloo/c-gomoku-cli):

```bash
python visualize_dataset.py --dataset_type packed_binary <path to data file>
```

## Serialized Weight Format

In order to improve the ease of managing weights for different rules and board size configuration, binary serialized network weights are serialized with a standardized header, which contains information about the weights, including architecture id, applicable rules, board sizes, default settings and description. Binary weights data of the network is appended after the header.

```C
struct Header {
    uint32_t magic;				// 0xacd8cc6a = crc32("gomoku network weight version 1")
    uint32_t arch_hash;			// architecture hash, which is hash of the network architecture (network type, num of channels, layers, ...)
    uint32_t rule_mask;			// applicable rule bitmask (1=gomoku, 2=standard, 4=renju)
    uint32_t boardsize_mask;	// applicable board size bitmask (lsb set at index i means board size i+1 is usable for this weight)
    uint32_t desc_len;			// length of desc string (=0 for no description)
    char description[desc_len];	// weight description (encoded in utf-8)
}
```

