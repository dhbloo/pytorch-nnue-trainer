from collections import OrderedDict

from accelerate import PartialState
import configargparse
import yaml
import os

from trainer.supervised import SupervisedTrainer
from utils.config_utils import RUN_PROVENANCE_KEY, parse_run_provenance


_TRAIN_CONFIG_KEYS = frozenset(
    {
        "_provenance",
        "config",
        "train_datas",
        "val_datas",
        "rundir",
        "trainer_type",
        "load_from",
        "use_cpu",
        "dataset_type",
        "dataset_args",
        "val_dataset_type",
        "val_dataset_args",
        "dataloader_args",
        "data_pipelines",
        "num_worker",
        "model_type",
        "model_args",
        "optim_type",
        "optim_args",
        "lr_scheduler_type",
        "lr_scheduler_args",
        "loss_type",
        "loss_args",
        "iterations",
        "batch_size",
        "gradient_accumulation_steps",
        "eval_bs_multipler",
        "learning_rate",
        "weight_decay",
        "clip_grad_norm",
        "clip_grad_value",
        "no_shuffle",
        "log_interval",
        "show_interval",
        "save_interval",
        "val_interval",
        "temp_save_interval",
        "state_save_interval",
        "num_keep_states",
        "kd_model_type",
        "kd_model_args",
        "kd_checkpoint",
        "kd_T",
        "kd_alpha",
        "kd_use_train_mode",
        "kd_disable_amp",
        "find_unused_parameters",
        "random_seed",
        "performance_level",
        "max_memory_fraction",
        "profiler_args",
        "profile",
        "profile_active_iters",
        "profile_warmup_iters",
        "profile_memory",
    }
)

_EVAL_CONFIG_KEYS = frozenset(
    {
        "config",
        "checkpoint",
        "test_datas",
        "train_datas",
        "do_cross_eval",
        "use_cpu",
        "test_dataset_type",
        "test_dataset_args",
        "dataset_type",
        "dataset_args",
        "dataloader_args",
        "model_type",
        "model_args",
        "test_model_args",
        "batch_size",
        "eval_bs_multipler",
        "num_worker",
        "max_batches",
        "random_seed",
        "max_memory_fraction",
    }
)

_ALLOWED_CONFIG_KEYS = _TRAIN_CONFIG_KEYS | _EVAL_CONFIG_KEYS


def _canonicalize_config_key(key):
    if not isinstance(key, str):
        raise configargparse.ConfigFileParserException(f"Unknown config key: {key}")
    if key in _ALLOWED_CONFIG_KEYS:
        return key
    if key.startswith("--") and key[2:] in _ALLOWED_CONFIG_KEYS:
        return key[2:]
    raise configargparse.ConfigFileParserException(f"Unknown config key: {key}")


class _StrictYAMLConfigFileParser(configargparse.YAMLConfigFileParser):
    def parse(self, stream):
        yaml_module, SafeLoader, _ = self._load_yaml()
        try:
            parsed_obj = yaml_module.load(stream, Loader=SafeLoader)
        except Exception as e:
            raise configargparse.ConfigFileParserException(
                f"Couldn't parse config file: {e}"
            )

        if not isinstance(parsed_obj, dict):
            raise configargparse.ConfigFileParserException(
                "The config file doesn't appear to contain 'key: value' pairs "
                "(aka. a YAML mapping). "
                "yaml.load('%s') returned type '%s' instead of 'dict'."
                % (getattr(stream, "name", "stream"), type(parsed_obj).__name__)
            )

        config_items = OrderedDict()
        for key, value in parsed_obj.items():
            if key in {RUN_PROVENANCE_KEY, f"--{RUN_PROVENANCE_KEY}"}:
                try:
                    parse_run_provenance(value)
                except ValueError as e:
                    raise configargparse.ConfigFileParserException(str(e))
                continue
            key = _canonicalize_config_key(key)
            if isinstance(value, list):
                config_items[key] = value
            elif value is not None:
                config_items[key] = str(value)
        return config_items


def parse_args_and_init():
    parser = configargparse.ArgParser(
        description="Test",
        config_file_parser_class=_StrictYAMLConfigFileParser,
        ignore_unknown_config_file_keys=True,
        allow_abbrev=False,
    )
    parser.add("-c", "--config", is_config_file=True, help="Config file path")
    parser.add("-p", "--checkpoint", required=True, help="Model checkpoint file to test")
    parser.add(
        "-d",
        "--test_datas",
        nargs="+",
        help="Test dataset file or directory paths"
        " (If not set, then the training set paths will be used)",
    )
    parser.add("--train_datas", nargs="+", help="Training dataset file or directory paths")
    parser.add("--do_cross_eval", action="store_true", help="Use the dataset to do cross eval check")
    parser.add("--use_cpu", action="store_true", help="Use cpu only")
    parser.add("--test_dataset_type", help="Test dataset type (same as train set if not set)")
    parser.add("--test_dataset_args", type=yaml.safe_load, default={}, help="Extra test dataset arguments")
    parser.add("--dataset_type", help="Train dataset type")
    parser.add("--dataset_args", type=yaml.safe_load, default={}, help="Extra train dataset arguments")
    parser.add("--dataloader_args", type=yaml.safe_load, default={}, help="Extra dataloader arguments")
    parser.add("--model_type", required=True, help="Model type")
    parser.add("--model_args", type=yaml.safe_load, default={}, help="Extra model arguments")
    parser.add("--test_model_args", type=yaml.safe_load, default={}, help="Override model args for testing")
    parser.add("--batch_size", type=int, default=128, help="Batch size")
    parser.add("--eval_bs_multipler", type=int, default=4, help="Multiply batch size by this number")
    parser.add("--num_worker", type=int, default=0, help="Num of dataloader workers")
    parser.add("--max_batches", type=int, help="Test the amount of batches only")
    parser.add("--random_seed", type=int, default=42, help="Random seed")
    parser.add(
        "--max_memory_fraction",
        type=float,
        help="Optional per-process CUDA allocator limit in (0, 1]",
    )

    args = parser.parse_args()

    if PartialState(cpu=args.use_cpu).is_local_main_process:
        parser.print_values()
        print("-" * 60)

    return args


def test(
    checkpoint,
    do_cross_eval,
    use_cpu,
    test_datas,
    train_datas,
    test_dataset_type,
    test_dataset_args,
    dataset_type,
    dataset_args,
    dataloader_args,
    model_type,
    model_args,
    test_model_args,
    batch_size,
    eval_bs_multipler,
    num_worker,
    max_batches,
    random_seed,
    max_memory_fraction,
):
    if not os.path.exists(checkpoint) or not os.path.isfile(checkpoint):
        raise RuntimeError(f"Checkpoint {checkpoint} must be a valid file")

    # Resolve test data fallbacks
    if test_datas is None:
        if train_datas is None:
            raise RuntimeError("Test dataset must be set if train dataset is not set.")
        test_datas = train_datas
    if test_dataset_type is None:
        if dataset_type is None:
            raise RuntimeError("Test dataset type must be set if train dataset type is not set.")
        test_dataset_type = dataset_type
        test_dataset_args = dataset_args

    # Compute result file path
    rundir = os.path.dirname(checkpoint)
    ckpt_filename_noext = os.path.splitext(os.path.basename(checkpoint))[0]
    result_file = os.path.join(rundir, f"{ckpt_filename_noext}_test_result.json")

    trainer = SupervisedTrainer.init_for_evaluation(
        checkpoint=checkpoint,
        model_type=model_type,
        model_args=model_args,
        test_model_args=test_model_args,
        test_datas=test_datas,
        dataset_type=test_dataset_type,
        dataset_args=test_dataset_args,
        dataloader_args=dataloader_args,
        batch_size=batch_size,
        eval_bs_multipler=eval_bs_multipler,
        num_worker=num_worker,
        use_cpu=use_cpu,
        random_seed=random_seed,
        max_memory_fraction=max_memory_fraction,
    )

    trainer.test(
        max_batches=max_batches,
        result_file=result_file,
        result_metadata={
            "test_datas": test_datas,
            "dataloader_args": dataloader_args,
            "use_cpu": use_cpu,
            "do_cross_eval": do_cross_eval,
        },
        do_cross_eval=do_cross_eval,
    )


def main():
    args = vars(parse_args_and_init())
    args.pop("config", None)
    test(**args)


if __name__ == "__main__":
    main()
