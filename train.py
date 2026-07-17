from accelerate import PartialState
import configargparse
import yaml
import os

from utils.file_utils import make_dir
from trainer import build_trainer


def parse_args_and_init():
    parser = configargparse.ArgParser(
        description="Trainer", config_file_parser_class=configargparse.YAMLConfigFileParser
    )
    parser.add("-c", "--config", is_config_file=True, help="Config file path")
    parser.add("-d", "--train_datas", nargs="+", help="Training dataset file or directory paths")
    parser.add("-v", "--val_datas", nargs="+", help="Validation dataset file or directory paths")
    parser.add("-r", "--rundir", required=True, help="Run directory")
    parser.add("--trainer_type", default="supervised", help="Trainer type")
    parser.add("--load_from", help="Load pretrained weights from file")
    parser.add("--use_cpu", action="store_true", help="Use cpu only")
    parser.add("--dataset_type", required=True, help="Dataset type")
    parser.add("--dataset_args", type=yaml.safe_load, default={}, help="Extra dataset arguments")
    parser.add("--val_dataset_type", help="Validate dataset type (same as train set if not set)")
    parser.add(
        "--val_dataset_args", type=yaml.safe_load, default={}, help="Extra validate dataset arguments"
    )
    parser.add("--dataloader_args", type=yaml.safe_load, default={}, help="Extra dataloader arguments")
    parser.add("--data_pipelines", type=yaml.safe_load, default=None, help="Data-pipeline arguments")
    parser.add("--num_worker", type=int, default=4, help="Num of dataloader workers")
    parser.add("--model_type", required=True, help="Model type")
    parser.add("--model_args", type=yaml.safe_load, default={}, help="Extra model arguments")
    parser.add("--optim_type", default="adamw", help="Optimizer type")
    parser.add("--optim_args", type=yaml.safe_load, default={}, help="Extra optimizer arguments")
    parser.add("--lr_scheduler_type", default="constant", help="LR scheduler type")
    parser.add("--lr_scheduler_args", type=yaml.safe_load, default={}, help="Extra LR scheduler arguments")
    parser.add("--init_cfg", type=yaml.safe_load, default={}, help="Init configuration")
    parser.add("--loss_type", default="KL+KL", help="Loss type")
    parser.add("--loss_args", type=yaml.safe_load, default={}, help="Extra loss arguments")
    parser.add("--iterations", type=int, default=1000000, help="Num iterations")
    parser.add("--batch_size", type=int, default=128, help="Total batch size of all GPUs")
    parser.add(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Num of micro-batches accumulated per optimizer step"
        " (effective batch size = batch_size * this)",
    )
    parser.add("--eval_bs_multipler", type=int, default=1, help="Eval batch size multipler")
    parser.add("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add("--weight_decay", type=float, default=1e-2, help="Weight decay")
    parser.add("--clip_grad_norm", type=float, help="Gradient clipping max norm")
    parser.add("--clip_grad_value", type=float, help="Gradient clipping max value")
    parser.add("--no_shuffle", action="store_true", help="Do not shuffle dataset")
    parser.add("--log_interval", type=int, default=500, help="Num iterations to log")
    parser.add("--show_interval", type=int, default=1000, help="Num iterations to display")
    parser.add("--save_interval", type=int, default=100000, help="Num iterations to save snapshot")
    parser.add("--val_interval", type=int, default=50000, help="Num iterations to do validation")
    parser.add(
        "--temp_save_interval",
        type=int,
        default=5000,
        help="Num iterations to save a temporary snapshot (removed when there is newer one)",
    )
    parser.add(
        "--state_save_interval",
        type=int,
        help="Num iterations to save training state (defaults to every snapshot save)",
    )
    parser.add(
        "--num_keep_states",
        type=int,
        default=1,
        help="Num of most recent training state files to keep (-1 to keep all)",
    )
    parser.add("--kd_model_type", help="Knowledge distillation model type")
    parser.add(
        "--kd_model_args",
        type=yaml.safe_load,
        default={},
        help="Knowledge distillation extra model arguments",
    )
    parser.add("--kd_checkpoint", help="Knowledge distillation model checkpoint")
    parser.add("--kd_T", type=float, default=1.0, help="Knowledge distillation temperature")
    parser.add(
        "--kd_alpha",
        type=float,
        default=1.0,
        help="Distillation loss ratio in [0,1] (1 for distillation loss only)",
    )
    parser.add("--kd_use_train_mode", action="store_true", help="Set teacher to train mode")
    parser.add("--kd_disable_amp", action="store_true", help="Disable mixed precision for teacher")
    parser.add("--find_unused_parameters", action="store_true", help="Enable find_unused_parameters in DDP")
    parser.add("--random_seed", type=int, default=42, help="Random seed")
    parser.add(
        "--performance_level",
        type=int,
        default=2,
        help="Performance level to use. A higher value will trade higher performance with less precision and reproducibility",
    )
    parser.add(
        "--profiler_args",
        type=yaml.safe_load,
        default=None,
        help='Profiler configuration, e.g. "{timing: true, trace_at: [100000], trace_iters: 30}"',
    )
    parser.add("--profile", action="store_true", help="Enable profiling")
    parser.add("--profile_active_iters", type=int, default=30, help="Num iterations to profile")
    parser.add("--profile_warmup_iters", type=int, default=10, help="Warmup iterations before profiling")
    parser.add("--profile_memory", action="store_true", help="Enable memory profiling")
    parser.add("--profile_finish_exit", action="store_true", help="Exit after profiling is finished")

    args, _ = parser.parse_known_args()  # parse args

    if PartialState(cpu=args.use_cpu).is_main_process:
        parser.print_values()  # print argument values
        make_dir(args.rundir)  # make run directory
        # write run config
        run_cfg_filename = os.path.join(args.rundir, "run_config.yaml")
        if args.config is None or os.path.abspath(args.config) != os.path.abspath(run_cfg_filename):
            parser.write_config_file(args, [run_cfg_filename])
        print("-" * 60)

    return args


def train(**kwargs):
    # Pop profile-related keys before constructing trainer
    do_profile = kwargs.pop("profile", False)
    profile_active_iters = kwargs.pop("profile_active_iters", 30)
    profile_warmup_iters = kwargs.pop("profile_warmup_iters", 10)
    profile_memory = kwargs.pop("profile_memory", False)
    kwargs.pop("profile_finish_exit", None)

    trainer_type = kwargs.pop("trainer_type", "supervised")
    trainer = build_trainer(trainer_type, **kwargs)

    if do_profile:
        trainer.profile(
            warmup=profile_warmup_iters,
            active=profile_active_iters,
            profile_memory=profile_memory,
        )
    else:
        trainer.run()


if __name__ == "__main__":
    args = parse_args_and_init()
    train(**vars(args))

