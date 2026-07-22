from accelerate import PartialState
import configargparse
import yaml
import os

from trainer.supervised import SupervisedTrainer


def parse_args_and_init():
    parser = configargparse.ArgParser(
        description="Test", config_file_parser_class=configargparse.YAMLConfigFileParser
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
    parser.add("--num_worker", type=int, default=8, help="Num of dataloader workers")
    parser.add("--max_batches", type=int, help="Test the amount of batches only")
    parser.add("--random_seed", type=int, default=42, help="Random seed")
    parser.add(
        "--max_memory_fraction",
        type=float,
        help="Optional per-process CUDA allocator limit in (0, 1]",
    )

    args, _ = parser.parse_known_args()  # parse args

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
    **kwargs,
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


if __name__ == "__main__":
    args = parse_args_and_init()
    test(**vars(args))
