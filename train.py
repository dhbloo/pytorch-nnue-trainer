import torch
import torch.nn.functional as F
from accelerate import (
    Accelerator,
    DataLoaderConfiguration,
    DistributedDataParallelKwargs,
    ProfileKwargs,
    PartialState,
)
from accelerate.utils.other import is_compiled_module
from torch.utils.tensorboard import SummaryWriter
from contextlib import nullcontext
import configargparse
import yaml
import json
import time
import os

from dataset import build_dataset
from model import build_model
from utils.training_utils import (
    build_lr_scheduler,
    cross_entropy_with_softlabel,
    build_optimizer,
    weights_init,
    build_data_loader,
    weight_clipping,
    state_dict_drop_size_unmatched,
)
from utils.misc_utils import (
    seed_everything,
    set_performance_level,
    add_dict_to,
    log_value_dict,
    format_time,
)
from utils.file_utils import (
    make_dir,
    save_torch_ckpt,
    load_torch_ckpt,
    find_latest_ckpt,
    get_iteration_from_ckpt_filename,
)


def parse_args_and_init():
    parser = configargparse.ArgParser(
        description="Trainer", config_file_parser_class=configargparse.YAMLConfigFileParser
    )
    parser.add("-c", "--config", is_config_file=True, help="Config file path")
    parser.add("-d", "--train_datas", nargs="+", help="Training dataset file or directory paths")
    parser.add("-v", "--val_datas", nargs="+", help="Validation dataset file or directory paths")
    parser.add("-r", "--rundir", required=True, help="Run directory")
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
    parser.add("--profile", action="store_true", help="Enable profiling")
    parser.add("--profile_active_iters", type=int, default=30, help="Num iterations to profile")
    parser.add("--profile_warmup_iters", type=int, default=10, help="Warmup iterations before profiling")
    parser.add("--profile_memory", action="store_true", help="Enable memory profiling")
    parser.add("--profile_finish_exit", action="store_true", help="Exit after profiling is finished")

    args, _ = parser.parse_known_args()  # parse args

    if PartialState(cpu=args.use_cpu).is_local_main_process:
        parser.print_values()  # print argument values
        make_dir(args.rundir)  # make run directory
        # write run config
        run_cfg_filename = os.path.join(args.rundir, "run_config.yaml")
        if args.config is None or os.path.abspath(args.config) != os.path.abspath(run_cfg_filename):
            parser.write_config_file(args, [run_cfg_filename])
        print("-" * 60)

    return args


def calc_loss(
    loss_type,
    data,
    results,
    kd_results=None,
    kd_T=1.0,
    kd_alpha=1.0,
    policy_reg_lambda=0,
    value_policy_ratio=1,
    **extra_args,
):
    value_loss_type, policy_loss_type = loss_type.split("+")

    # get predicted value and policy from model results
    value, policy, *retvals = results
    aux_losses = retvals[0] if len(retvals) >= 1 else None
    aux_outputs = retvals[1] if len(retvals) >= 2 else None
    board_mask = retvals[2] if len(retvals) >= 3 else None

    if kd_results is not None:
        # get value and policy target from teacher model
        value_target, policy_target, *kd_retvals = kd_results
    else:
        # get value and poliay target from data
        value_target, policy_target = data["value_target"], data["policy_target"]

    # ===============================================================
    # value loss
    def get_value_loss(value: torch.Tensor, value_target: torch.Tensor) -> torch.Tensor:
        if kd_results is not None:
            # apply softmax to value target according to temparature
            if value_target.size(1) > 1:
                value_target = torch.softmax(value_target / kd_T, dim=1)
            else:  # for value without draw info
                value_target = torch.sigmoid(value_target / kd_T)
            # scale prediction value according to temparature
            value = value / kd_T

        # convert value_target if value has no draw info
        if value.size(1) == 1:
            value = value[:, 0]  # squeeze value channel dim
            if value_target.size(1) == 1:
                value_target = value_target[:, 0]  # squeeze value channel dim
            else:
                value_target = (value_target[:, 0] - value_target[:, 1] + 1) / 2  # in [0, 1]

        if value_loss_type == "KL":
            return cross_entropy_with_softlabel(value, value_target, use_kl_divergence=True)
        elif value_loss_type == "CE":
            return cross_entropy_with_softlabel(
                value,
                value_target,
                focal_gamma=extra_args.get("value_focal_gamma", 0),
            )
        elif value_loss_type == "MSE":
            if value.ndim == 1:
                return F.mse_loss(torch.sigmoid(value), value_target)
            else:
                value = torch.softmax(value, dim=1)  # [B, 3]
                winrate = (value[:, 0] - value[:, 1] + 1) / 2
                winrate_target = (value_target[:, 0] - value_target[:, 1] + 1) / 2
                return F.mse_loss(winrate, winrate_target)
        else:
            raise ValueError(f"Unsupported value loss: {value_loss_type}")

    # ===============================================================
    # policy loss
    def get_policy_loss(policy: torch.Tensor, policy_target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # reshape policy
        policy_target = torch.flatten(policy_target, start_dim=1)
        policy = torch.flatten(policy, start_dim=1)
        assert policy_target.shape[1] == policy.shape[1]

        # apply board region mask to policy, so after exp() they are close to zero
        if mask is not None:
            mask = torch.flatten(mask, start_dim=1)
            policy = policy.masked_fill(mask == 0, -1e6)
            policy_target = policy_target.masked_fill(mask == 0, 0)

        if kd_results is not None:
            if mask is not None:
                policy_target = policy_target.masked_fill(mask == 0, -1e6)
            # apply softmax to policy target according to temparature
            policy_target = torch.softmax(policy_target / kd_T, dim=1)
            # scale prediction policy according to temparature
            policy = policy / kd_T
            if mask is not None:
                policy_target = policy_target.masked_fill(mask == 0, 0)

        # check if there is loss weight for policy at each empty cells
        if extra_args.get("ignore_forbidden_point_policy", False):
            forbidden_point = torch.flatten(data["forbidden_point"], start_dim=1)  # [B, H*W]
            policy_loss_weight = 1.0 - forbidden_point.float()  # [B, H*W]
        else:
            policy_loss_weight = None

        if policy_loss_type == "KL":
            return cross_entropy_with_softlabel(
                policy,
                policy_target,
                weight=policy_loss_weight,
                use_kl_divergence=True,
            )
        elif policy_loss_type == "CE":
            return cross_entropy_with_softlabel(
                policy,
                policy_target,
                weight=policy_loss_weight,
                focal_gamma=extra_args.get("policy_focal_gamma", 0),
            )
        elif policy_loss_type == "MSE":
            policy_softmaxed = torch.softmax(policy, dim=1)
            policy_loss = F.mse_loss(policy_softmaxed, policy_target, reduction="none")
            if policy_loss_weight is not None:
                policy_loss = policy_loss * policy_loss_weight
            return torch.mean(policy_loss)
        elif policy_loss_type == "NONE":
            return torch.tensor(0.0, device=policy.device)
        else:
            raise ValueError(f"Unsupported policy loss: {policy_loss_type}")

    # ===============================================================
    # policy reg loss
    def get_policy_reg_loss(policy: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        policy = torch.flatten(policy, start_dim=1)  # reshape policy
        if mask is not None:
            mask = torch.flatten(mask, start_dim=1)  # reshape mask
            policy = policy.masked_fill(mask == 0, -1e6)
            policy_mean = torch.sum(policy * mask) / torch.sum(mask)
        else:
            policy_mean = torch.mean(policy)
        return policy_mean.square()

    # ===============================================================
    # value uncertainty loss
    def get_value_uncertainty_gt(value: torch.Tensor) -> torch.Tensor:
        # stop gradient for value
        value = value.detach()

        if value.ndim == 1:
            winrate = torch.sigmoid(value)  # [B]
            winrate_target = value_target  # [B]
        else:
            value = torch.softmax(value, dim=1)  # [B, 3]
            # compute winrate in [0, 1] from 3-tuple value
            winrate = (value[:, 0] - value[:, 1] + 1) / 2  # [B]
            winrate_target = (value_target[:, 0] - value_target[:, 1] + 1) / 2  # [B]

        # compute value uncertainty (uncertainty = |winrate - winrate_target|^2)
        uncertainty_gt = torch.square(winrate - winrate_target)  # [B]
        return uncertainty_gt

    def get_value_uncertainty_loss(uncertainty: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        uncertainty_gt = get_value_uncertainty_gt(value)
        uncertainty_loss = F.huber_loss(uncertainty, uncertainty_gt)  # [B]
        return uncertainty_loss

    def get_value_relative_uncertainty_loss(
        relative_uncertainty: torch.Tensor, value_smallnet: torch.Tensor, value_largenet: torch.Tensor
    ) -> torch.Tensor:
        uncertainty_gt_smallnet = get_value_uncertainty_gt(value_smallnet)
        uncertainty_gt_largenet = get_value_uncertainty_gt(value_largenet)
        relative_uncertainty_gt = uncertainty_gt_largenet / uncertainty_gt_smallnet
        relative_uncertainty_gt = torch.where(
            uncertainty_gt_largenet < uncertainty_gt_smallnet, relative_uncertainty_gt, 1.0
        )
        relative_uncertainty_loss = F.mse_loss(relative_uncertainty, relative_uncertainty_gt)
        return relative_uncertainty_loss

    # ===============================================================

    value_loss = get_value_loss(value, value_target)
    policy_loss = get_policy_loss(policy, policy_target, board_mask)
    value_lambda = 2 * value_policy_ratio / (value_policy_ratio + 1)
    policy_lambda = 2 / (value_policy_ratio + 1)
    total_loss = value_lambda * value_loss + policy_lambda * policy_loss
    loss_dict = {
        "value_loss": value_loss.detach(),
        "policy_loss": policy_loss.detach(),
    }

    if float(policy_reg_lambda) > 0:
        policy_reg = get_policy_reg_loss(policy, board_mask)
        total_loss += float(policy_reg_lambda) * policy_reg
        loss_dict["policy_reg"] = policy_reg.detach()

    if aux_losses:
        for aux_loss_name, aux_loss in aux_losses.items():
            aux_loss_print_name = f"{aux_loss_name}_loss"
            aux_loss_weight = None
            if f"{aux_loss_name}_lambda" in extra_args:
                aux_loss_weight = float(extra_args[f"{aux_loss_name}_lambda"])

            # use pre-defined loss terms if aux_loss is a tuple of (string, inputs)
            if isinstance(aux_loss, tuple) and len(aux_loss) == 2:
                aux_loss_type, aux_input = aux_loss
                if aux_loss_type == "value_loss":
                    aux_loss = get_value_loss(aux_input, value_target)
                    if aux_loss_weight is None:
                        aux_loss_weight = value_lambda
                elif aux_loss_type == "policy_loss":
                    aux_loss = get_policy_loss(aux_input, policy_target, board_mask)
                    if aux_loss_weight is None:
                        aux_loss_weight = policy_lambda
                elif aux_loss_type == "value_uncertainty_loss":
                    aux_loss = get_value_uncertainty_loss(*aux_input)
                elif aux_loss_type == "value_relative_uncertainty_loss":
                    aux_loss = get_value_relative_uncertainty_loss(*aux_input)
                elif aux_loss_type == "policy_reg":
                    if aux_loss_weight is None and policy_reg_lambda == 0:
                        continue
                    aux_loss = get_policy_reg_loss(aux_input, board_mask)
                    aux_loss_print_name = aux_loss_name
                    if aux_loss_weight is None:
                        aux_loss_weight = policy_reg_lambda
                else:
                    raise ValueError(f"Unsupported predefined aux loss: {aux_loss}")

            if aux_loss_weight is None:
                aux_loss_weight = 1.0
            assert isinstance(aux_loss, torch.Tensor)
            total_loss += float(aux_loss_weight) * aux_loss
            loss_dict[aux_loss_print_name] = aux_loss.detach()

    if kd_results is not None:
        # also get a true loss from real data
        real_loss, real_loss_dict, _ = calc_loss(
            loss_type,
            data,
            results,
            policy_reg_lambda=policy_reg_lambda,
            value_policy_ratio=value_policy_ratio,
            **extra_args,
        )

        # merge real loss and knowledge distillation loss
        total_loss = kd_alpha * total_loss * (kd_T**2) + (1 - kd_alpha) * real_loss
        kd_loss_dict = {"kd_" + k: v for k, v in loss_dict.items()}
        loss_dict = real_loss_dict
        loss_dict.update(kd_loss_dict)

    aux_outputs_dict = {}
    if aux_outputs:
        for aux_name, aux_output in aux_outputs.items():
            assert isinstance(aux_output, torch.Tensor)
            aux_outputs_dict[aux_name] = aux_output.detach()

    loss_dict["total_loss"] = total_loss.detach()
    return total_loss, loss_dict, aux_outputs_dict


def train(
    rundir,
    load_from,
    use_cpu,
    train_datas,
    val_datas,
    dataset_type,
    dataset_args,
    val_dataset_type,
    val_dataset_args,
    dataloader_args,
    data_pipelines,
    model_type,
    model_args,
    optim_type,
    optim_args,
    lr_scheduler_type,
    lr_scheduler_args,
    init_cfg,
    loss_type,
    loss_args,
    iterations,
    batch_size,
    eval_bs_multipler,
    num_worker,
    learning_rate,
    weight_decay,
    clip_grad_norm,
    clip_grad_value,
    no_shuffle,
    log_interval,
    show_interval,
    save_interval,
    val_interval,
    temp_save_interval,
    kd_model_type,
    kd_model_args,
    kd_checkpoint,
    kd_T,
    kd_alpha,
    kd_use_train_mode,
    kd_disable_amp,
    find_unused_parameters,
    random_seed,
    performance_level,
    profile,
    profile_active_iters,
    profile_warmup_iters,
    profile_memory,
    profile_finish_exit,
    **kwargs,
):
    # initialize accelerator
    dataloader_config = DataLoaderConfiguration(dispatch_batches=False, non_blocking=True)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)
    accelerator = Accelerator(cpu=use_cpu, dataloader_config=dataloader_config, kwargs_handlers=[ddp_kwargs])
    seed_everything(random_seed)
    set_performance_level(performance_level)

    if accelerator.is_local_main_process:
        tb_logger = SummaryWriter(os.path.join(rundir, "log"))
        log_file = open(os.path.join(rundir, "training_log.jsonl"), "a")
    else:
        tb_logger, log_file = None, None

    profile_ctx = nullcontext()
    if profile:

        def profile_handler(p: torch.profiler.profile):
            if accelerator.is_local_main_process:
                dev_name = "cpu" if use_cpu else "cuda"
                print(p.key_averages().table(sort_by=f"{dev_name}_time_total", row_limit=16))
                print("Profile finished. Saving trace...")
                tracefile_path = os.path.join(rundir, "profile_trace")
                torch.profiler.tensorboard_trace_handler(tracefile_path, use_gzip=True)(p)
                print(f"Profile trace saved at {tracefile_path}.")
            if profile_finish_exit:
                accelerator.print("Exiting...")
                exit(0)

        profile_ctx = accelerator.profile(
            ProfileKwargs(
                activities=["cpu"] if use_cpu else ["cpu", "cuda"],
                schedule_option={
                    "active": profile_active_iters,
                    "warmup": profile_warmup_iters,
                    "wait": 0,
                    "repeat": 1,
                },
                on_trace_ready=profile_handler,
                record_shapes=True,
                profile_memory=profile_memory,
                with_stack=True,
                with_flops=True,
                with_modules=True,
            )
        )

    # load train and validation dataset
    batch_size_per_process = batch_size // accelerator.num_processes
    train_dataset = build_dataset(
        dataset_type, train_datas, shuffle=not no_shuffle, pipeline_args=data_pipelines, **dataset_args
    )
    train_loader = build_data_loader(
        train_dataset,
        batch_size_per_process,
        num_workers=num_worker,
        shuffle=not no_shuffle,
        **dataloader_args,
    )
    if val_datas or val_dataset_type:
        val_dataset = build_dataset(
            val_dataset_type or dataset_type,
            val_datas,
            shuffle=False,
            pipeline_args=data_pipelines,
            **(val_dataset_args if val_dataset_type else dataset_args),
        )
        val_loader = build_data_loader(
            val_dataset,
            batch_size_per_process * eval_bs_multipler,
            num_workers=num_worker,
            shuffle=False,
            **dataloader_args,
        )
    else:
        val_dataset, val_loader = None, None

    # build model, optimizer
    model = build_model(model_type, **model_args)
    optimizer = build_optimizer(optim_type, model, lr=learning_rate, weight_decay=weight_decay, **optim_args)

    # load checkpoint if exists
    model_name = model.name
    ckpt_filename = find_latest_ckpt(rundir, f"ckpt_{model_name}")
    if ckpt_filename:
        model_state_dict, training_state_dicts, metadata = load_torch_ckpt(ckpt_filename)
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(training_state_dicts["optimizer"])
        if accelerator.scaler is not None and "scalar" in training_state_dicts:
            accelerator.scaler.load_state_dict(training_state_dicts["scalar"])
        accelerator.print(f"Loaded from {ckpt_filename}")
        it = int(metadata.get("iteration", 0))
        epoch = int(metadata.get("epoch", 0))
        rows = int(metadata.get("rows", 0))
    else:
        model.apply(weights_init(init_cfg))
        it, epoch, rows = 0, 0, 0
        if load_from is not None:
            model_state_dict, _, _ = load_torch_ckpt(load_from)
            model_state_dict = state_dict_drop_size_unmatched(model, model_state_dict)
            missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
            if len(unexpected_keys) > 0:
                accelerator.print(f"unexpected keys in state_dict: {', '.join(unexpected_keys)}")
            if len(missing_keys) > 0:
                accelerator.print(f"missing keys in state_dict: {', '.join(missing_keys)}")
            accelerator.print(f"Loaded from pretrained: {load_from}")

    # build lr scheduler
    scheduler = build_lr_scheduler(optimizer, lr_scheduler_type, iterations, last_it=it - 1, **lr_scheduler_args)

    # accelerate model training
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    if val_loader is not None:
        val_loader = accelerator.prepare_data_loader(val_loader)

    # build teacher model if knowledge distillation model is on
    if kd_model_type is not None:
        kd_model = build_model(kd_model_type, **kd_model_args)
        kd_model_state_dict, _, _ = load_torch_ckpt(kd_checkpoint)
        kd_model.load_state_dict(kd_model_state_dict)
        accelerator.print(f"Loaded teacher model {kd_model.name} from {kd_checkpoint}")

        kd_model = accelerator.prepare_model(kd_model, evaluation_mode=True)
        if kd_use_train_mode:
            kd_model.train()
        else:
            kd_model.eval()

        # add kd_T into loss_args dict
        assert kd_T is not None, f"kd_T must be set when knowledge distillation is enabled"
        assert kd_alpha is not None and (0 <= kd_alpha <= 1), f"kd_alpha must be in [0,1]"
        loss_args["kd_T"] = kd_T
        loss_args["kd_alpha"] = kd_alpha
    else:
        kd_model = None

    accelerator.print(f"Start training from iteration {it}, epoch {epoch}")
    train_data_iter = iter(train_loader)
    start_time = log_last_time = show_last_time = time.time()
    log_last_it = show_last_it = it
    gpu_loss_sum, gpu_aux_sum, gpu_metric_count = {}, {}, 0
    model.train()
    with profile_ctx as profiler:
        while it < iterations:
            torch.compiler.cudagraph_mark_step_begin()

            # fetch the next batch of data from train loader
            try:
                data = next(train_data_iter)
            except StopIteration:
                epoch += 1
                train_data_iter = iter(train_loader)
                data = next(train_data_iter)

            it += 1
            rows += batch_size
            data.update(
                {
                    "train_iter": it,
                    "train_total_iter": iterations,
                    "train_progress": it / iterations,
                }
            )

            # evaulate teacher model for knowledge distillation
            with torch.no_grad(), nullcontext() if kd_disable_amp else accelerator.autocast():
                kd_results = kd_model(data) if kd_model is not None else None

            # get unwarpped model and apply weight clipping if needed
            unwarpped_model = accelerator.unwrap_model(model)
            if is_compiled_module(unwarpped_model):
                unwarpped_model = unwarpped_model._orig_mod

            # apply weight clipping if needed
            if hasattr(unwarpped_model, "weight_clipping"):
                # A workaround to get the original model from the dynamo optimized model
                weight_clipping(unwarpped_model.named_parameters(), unwarpped_model.weight_clipping)

            with accelerator.accumulate(model), accelerator.autocast():
                # model update
                optimizer.zero_grad(set_to_none=True)
                results = model(data)
                loss, loss_dict, aux_dict = calc_loss(loss_type, data, results, kd_results, **loss_args)
                accelerator.backward(loss)

                # apply gradient clipping if needed
                if clip_grad_norm is not None:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
                elif clip_grad_value is not None:
                    accelerator.clip_grad_value_(model.parameters(), clip_value=clip_grad_value)

                optimizer.step()
                scheduler.step()

            # accumulate metrics on GPU (no cross-GPU communication)
            add_dict_to(gpu_loss_sum, loss_dict)
            add_dict_to(gpu_aux_sum, aux_dict)
            gpu_metric_count += 1

            # gather and average metrics at log/show intervals
            if it % log_interval == 0 or it % show_interval == 0:
                loss_dict = {k: v / gpu_metric_count for k, v in gpu_loss_sum.items()}
                aux_dict = {k: v / gpu_metric_count for k, v in gpu_aux_sum.items()}

                loss_dict = accelerator.gather(loss_dict)
                aux_dict = accelerator.gather(aux_dict)
                for metric_dict in [loss_dict, aux_dict]:
                    for key, value in metric_dict.items():
                        metric_dict[key] = torch.mean(value, dim=0).item()

                gpu_loss_sum = {}
                gpu_aux_sum = {}
                gpu_metric_count = 0

            # logging
            if it % log_interval == 0 and accelerator.is_local_main_process:
                log_value_dict(tb_logger, "train", loss_dict, it, rows)
                if aux_dict:
                    log_value_dict(tb_logger, "train_aux", aux_dict, it, rows)

                # Log time-related performance metrics
                iters_per_second = (it - log_last_it) / (time.time() - log_last_time)
                elapsed_time = time.time() - start_time
                running_stat_dict = {
                    "epoch": epoch,
                    "rows": rows,
                    "elapsed": elapsed_time,
                    "it/s": iters_per_second,
                    "entry/s": iters_per_second * batch_size,
                    "lr": scheduler.get_last_lr()[0],
                }
                log_value_dict(tb_logger, "running_stat", running_stat_dict, it, rows)
                log_last_it = it
                log_last_time = time.time()

                json_log_dict = {
                    "it": it,
                    "train_loss": loss_dict,
                    "train_aux": aux_dict,
                    "running_stat": running_stat_dict,
                }
                log_file.write(json.dumps(json_log_dict) + "\n")
                log_file.flush()

            # display current progress
            if it % show_interval == 0 and accelerator.is_local_main_process:
                iters_per_second = (it - show_last_it) / (time.time() - show_last_time)
                elapsed_time = time.time() - start_time
                eta_time = (iterations - it) / iters_per_second
                print(f"Iter: {it}/{iterations} ({it/iterations*100:.2f}%)"
                   f" | Elapsed: {format_time(elapsed_time)}"
                   f" | Speed: {iters_per_second:.2f} it/s"
                   f" | ETA: {format_time(eta_time)}"
                   f" | Loss: {loss_dict['total_loss']:.4f}, v={loss_dict['value_loss']:.4f}, p={loss_dict['policy_loss']:.4f}",
                   flush=True
                )
                show_last_it = it
                show_last_time = time.time()

            # checkpoint saving
            if (
                it % save_interval == 0 or it % temp_save_interval == 0
            ) and accelerator.is_local_main_process:
                # get latest checkpoint filename
                last_ckpt_filename = find_latest_ckpt(rundir, f"ckpt_{model_name}")

                # save a new checkpoint at current iteration
                model_state_dict = unwarpped_model.state_dict()
                training_state_dicts = {"optimizer": optimizer.state_dict()}
                if accelerator.scaler is not None:
                    training_state_dicts["scalar"] = accelerator.scaler.state_dict()
                metadata_dict = {"iteration": str(it), "epoch": str(epoch), "rows": str(rows)}
                save_torch_ckpt(
                    os.path.join(rundir, f"ckpt_{model_name}_{it:07d}"),
                    model_state_dict,
                    training_state_dicts,
                    metadata_dict,
                )

                # remove last snapshot if it's a temporary snapshot
                if last_ckpt_filename is not None:
                    last_snapshot_iter = get_iteration_from_ckpt_filename(last_ckpt_filename)
                    if last_snapshot_iter is None or last_snapshot_iter % save_interval != 0:
                        os.remove(last_ckpt_filename)

            # validation
            if it % val_interval == 0 and val_loader is not None:
                val_start_time = time.time()
                val_loss_dict, val_aux_dict = {}, {}
                num_val_batches = 0
                if accelerator.is_local_main_process:
                    print(f"\nValidation at iteration {it}/{iterations} ({it/iterations*100:.2f}%)...", flush=True)

                model.eval()
                with torch.no_grad():
                    for val_data in val_loader:
                        # evaulate teacher model for knowledge distillation
                        with nullcontext() if kd_disable_amp else accelerator.autocast():
                            kd_results = kd_model(val_data) if kd_model is not None else None
                        # model evaluation
                        with accelerator.autocast():
                            results = model(val_data)
                            _, val_losses, val_auxs = calc_loss(
                                loss_type, val_data, results, kd_results, **loss_args
                            )
                        add_dict_to(val_loss_dict, val_losses)
                        add_dict_to(val_aux_dict, val_auxs)
                        num_val_batches += 1
                model.train()

                # gather all loss dict across processes
                val_loss_dict["num_val_batches"] = torch.tensor(
                    [num_val_batches], dtype=torch.long, device=accelerator.device
                )
                all_val_loss_dict = accelerator.gather(val_loss_dict)
                all_val_aux_dict = accelerator.gather(val_aux_dict)

                if accelerator.is_local_main_process:
                    # average all losses
                    num_val_batches_tensor = all_val_loss_dict.pop("num_val_batches")
                    num_val_batches = torch.sum(num_val_batches_tensor).item()
                    val_loss_dict, val_aux_dict = {}, {}
                    for k, loss_tensor in all_val_loss_dict.items():
                        val_loss_dict[k] = torch.sum(loss_tensor).item() / num_val_batches
                    for k, aux_tensor in all_val_aux_dict.items():
                        val_aux_dict[k] = torch.sum(aux_tensor).item() / num_val_batches

                    val_elapsed_time = time.time() - val_start_time
                    elapsed_time = time.time() - start_time
                    num_val_entries = num_val_batches * batch_size_per_process * eval_bs_multipler
                    # log validation results
                    log_value_dict(tb_logger, "validation", val_loss_dict, it, rows)
                    if val_aux_dict:
                        log_value_dict(tb_logger, "validation_aux", val_aux_dict, it, rows)
                    json_log_dict = {
                        "it": it,
                        "epoch": epoch,
                        "elapsed": elapsed_time,
                        "val_loss": val_loss_dict,
                        "val_aux": val_aux_dict,
                        "num_val_entries": num_val_entries,
                    }
                    log_file.write(json.dumps(json_log_dict) + "\n")
                    log_file.flush()
                    print(f"Validation finished with {num_val_entries} entries, using {format_time(val_elapsed_time)}.")
                    print(f"Validation loss: {val_loss_dict['total_loss']:.4f}, v={val_loss_dict['value_loss']:.4f}, p={val_loss_dict['policy_loss']:.4f}")
                    print(flush=True)

                    # subtract validation time from training time
                    log_last_time += val_elapsed_time
                    show_last_time += val_elapsed_time

            if profile:
                profiler.step()

    if accelerator.is_local_main_process:
        log_file.close()


if __name__ == "__main__":
    args = parse_args_and_init()
    train(**vars(args))
