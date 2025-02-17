import torch
import numpy as np
import torch.nn.functional as F
from accelerate import Accelerator, DataLoaderConfiguration, DistributedDataParallelKwargs, ProfileKwargs
from accelerate.utils.other import is_compiled_module
from torch.utils.tensorboard import SummaryWriter
from collections import deque
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
from utils.misc_utils import seed_everything, set_performance_level, add_dict_to, log_value_dict
from utils.file_utils import find_latest_model_file, get_iteration_from_model_filename


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
    parser.add("--val_dataset_type", help="Validate Dataset type (If not set, then it's same as train set)")
    parser.add(
        "--val_dataset_args", type=yaml.safe_load, default={}, help="Extra validate dataset arguments"
    )
    parser.add("--dataloader_args", type=yaml.safe_load, default={}, help="Extra dataloader arguments")
    parser.add(
        "--data_pipelines", type=yaml.safe_load, default=None, help="Data-pipeline type and arguments"
    )
    parser.add("--num_worker", type=int, default=min(8, os.cpu_count()), help="Num of dataloader workers")
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
    parser.add("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add("--weight_decay", type=float, default=1e-7, help="Weight decay")
    parser.add("--clip_grad_norm", type=float, help="Gradient clipping max norm")
    parser.add("--clip_grad_value", type=float, help="Gradient clipping max value")
    parser.add("--no_shuffle", action="store_true", help="Do not shuffle dataset")
    parser.add("--seed", type=int, default=42, help="Random seed")
    parser.add("--log_interval", type=int, default=100, help="Num iterations to log")
    parser.add("--show_interval", type=int, default=1000, help="Num iterations to display")
    parser.add("--save_interval", type=int, default=100000, help="Num iterations to save snapshot")
    parser.add("--val_interval", type=int, default=50000, help="Num iterations to do validation")
    parser.add("--avg_loss_interval", type=int, default=2500, help="Num iterations to average loss")
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
    parser.add(
        "--performance_level",
        type=int,
        default=2,
        help="Performance level to use. A higher value will trade higher performance with less precision and reproducibility",
    )
    parser.add("--profile", action="store_true", help="Enable profiling")
    parser.add("--profile_active_iters", type=int, default=100, help="Num iterations to profile")
    parser.add("--profile_warmup_iters", type=int, default=100, help="Warmup iterations before profiling")
    parser.add("--profile_memory", action="store_true", help="Enable memory profiling")

    args, _ = parser.parse_known_args()  # parse args
    parser.print_values()  # print out values
    os.makedirs(args.rundir, exist_ok=True)  # make run directory
    # write run config
    run_cfg_filename = os.path.join(args.rundir, "run_config.yaml")
    if args.config is None or os.path.abspath(args.config) != os.path.abspath(run_cfg_filename):
        parser.write_config_file(args, [run_cfg_filename])
    seed_everything(args.seed)  # set seed
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
    ignore_forbidden_point_policy=False,
    value_focal_gamma=0,
    policy_focal_gamma=0,
    **extra_args,
):
    value_loss_type, policy_loss_type = loss_type.split("+")

    # get predicted value and policy from model results
    value, policy, *retvals = results
    aux_losses = retvals[0] if len(retvals) >= 1 else None
    aux_outputs = retvals[1] if len(retvals) >= 2 else None

    if kd_results is not None:
        # get value and policy target from teacher model
        value_target, policy_target, *kd_retvals = kd_results
    else:
        # get value and poliay target from data
        value_target = data["value_target"]
        policy_target = data["policy_target"]

    # reshape policy
    policy_target = torch.flatten(policy_target, start_dim=1)
    policy = torch.flatten(policy, start_dim=1)
    assert policy_target.shape[1] == policy.shape[1]

    if kd_results is not None:
        # apply softmax to value and policy target according to temparature
        if value_target.size(1) > 1:
            value_target = torch.softmax(value_target / kd_T, dim=1)
        else:  # for value without draw info
            value_target = torch.sigmoid(value_target / kd_T)
        policy_target = torch.softmax(policy_target / kd_T, dim=1)

        # scale prediction value and policy according to temparature
        value = value / kd_T
        policy = policy / kd_T

    # convert value_target if value has no draw info
    if value.size(1) == 1:
        value = value[:, 0]  # squeeze value channel dim
        if value_target.size(1) == 1:
            value_target = value_target[:, 0]  # squeeze value channel dim
        else:
            value_target = (value_target[:, 0] - value_target[:, 1] + 1) / 2  # in [0, 1]

    # ===============================================================
    # value loss
    def get_value_loss(value: torch.Tensor):
        if value_loss_type == "KL":
            return cross_entropy_with_softlabel(value, value_target, use_kl_divergence=True)
        elif value_loss_type == "CE":
            return cross_entropy_with_softlabel(value, value_target, focal_gamma=value_focal_gamma)
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
    def get_policy_loss(policy: torch.Tensor):
        # check if there is loss weight for policy at each empty cells
        if ignore_forbidden_point_policy:
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
                focal_gamma=policy_focal_gamma,
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
    def get_policy_reg_loss(policy: torch.Tensor):
        return torch.mean(policy).square()

    # ===============================================================
    # value uncertainty loss
    def get_value_uncertainty_gt(value: torch.Tensor):
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

    def get_value_uncertainty_loss(uncertainty: torch.Tensor, value: torch.Tensor):
        uncertainty_gt = get_value_uncertainty_gt(value)
        uncertainty_loss = F.huber_loss(uncertainty, uncertainty_gt)  # [B]
        return uncertainty_loss

    def get_value_relative_uncertainty_loss(
        relative_uncertainty: torch.Tensor, value_smallnet: torch.Tensor, value_largenet: torch.Tensor
    ):
        uncertainty_gt_smallnet = get_value_uncertainty_gt(value_smallnet)
        uncertainty_gt_largenet = get_value_uncertainty_gt(value_largenet)
        relative_uncertainty_gt = uncertainty_gt_largenet / uncertainty_gt_smallnet
        relative_uncertainty_gt = torch.where(
            uncertainty_gt_largenet < uncertainty_gt_smallnet, relative_uncertainty_gt, 1.0
        )
        relative_uncertainty_loss = F.mse_loss(relative_uncertainty, relative_uncertainty_gt)
        return relative_uncertainty_loss

    # ===============================================================

    value_loss = get_value_loss(value)
    policy_loss = get_policy_loss(policy)
    value_lambda = 2 * value_policy_ratio / (value_policy_ratio + 1)
    policy_lambda = 2 / (value_policy_ratio + 1)
    total_loss = value_lambda * value_loss + policy_lambda * policy_loss
    loss_dict = {
        "value_loss": value_loss.detach(),
        "policy_loss": policy_loss.detach(),
    }

    if policy_reg_lambda > 0:
        policy_reg = get_policy_reg_loss(policy)
        total_loss += float(policy_reg_lambda) * policy_reg
        loss_dict["policy_reg"] = policy_reg.detach()

    if aux_losses is not None:
        for aux_loss_name, aux_loss in aux_losses.items():
            aux_loss_print_name = f"{aux_loss_name}_loss"
            aux_loss_weight = None
            if f"{aux_loss_name}_lambda" in extra_args:
                aux_loss_weight = float(extra_args[f"{aux_loss_name}_lambda"])

            # use pre-defined loss terms if aux_loss is a tuple of (string, inputs)
            if isinstance(aux_loss, tuple) and len(aux_loss) == 2:
                aux_loss_type, aux_loss_input = aux_loss
                if aux_loss_type == "value_loss":
                    aux_loss = get_value_loss(aux_loss_input)
                    if aux_loss_weight is None:
                        aux_loss_weight = value_lambda
                elif aux_loss_type == "policy_loss":
                    aux_loss = get_policy_loss(aux_loss_input)
                    if aux_loss_weight is None:
                        aux_loss_weight = policy_lambda
                elif aux_loss_type == "value_uncertainty_loss":
                    aux_loss = get_value_uncertainty_loss(*aux_loss_input)
                elif aux_loss_type == "value_relative_uncertainty_loss":
                    aux_loss = get_value_relative_uncertainty_loss(*aux_loss_input)
                elif aux_loss_type == "policy_reg":
                    if aux_loss_weight is None and policy_reg_lambda == 0:
                        continue
                    aux_loss = get_policy_reg_loss(aux_loss_input)
                    aux_loss_print_name = aux_loss_name
                    if aux_loss_weight is None:
                        aux_loss_weight = policy_reg_lambda
                else:
                    raise ValueError(f"Unsupported predefined aux loss: {aux_loss}")

            if aux_loss_weight is None:
                aux_loss_weight = 1.0
            total_loss += aux_loss * aux_loss_weight
            loss_dict[aux_loss_print_name] = aux_loss.detach()

    if kd_results is not None:
        # also get a true loss from real data
        real_loss, real_loss_dict, _ = calc_loss(
            loss_type,
            data,
            results,
            policy_reg_lambda=policy_reg_lambda,
            value_policy_ratio=value_policy_ratio,
            ignore_forbidden_point_policy=ignore_forbidden_point_policy,
            value_focal_gamma=value_focal_gamma,
            policy_focal_gamma=policy_focal_gamma,
        )

        # merge real loss and knowledge distillation loss
        total_loss = kd_alpha * total_loss * (kd_T**2) + (1 - kd_alpha) * real_loss
        kd_loss_dict = {"kd_" + k: v for k, v in loss_dict.items()}
        loss_dict = real_loss_dict
        loss_dict.update(kd_loss_dict)

    aux_outputs_dict = {}
    if aux_outputs is not None:
        for aux_name, aux_output in aux_outputs.items():
            aux_outputs_dict[aux_name] = aux_output.detach()

    loss_dict["total_loss"] = total_loss.detach()
    return total_loss, loss_dict, aux_outputs_dict


def training_loop(
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
    avg_loss_interval,
    temp_save_interval,
    kd_model_type,
    kd_model_args,
    kd_checkpoint,
    kd_T,
    kd_alpha,
    kd_use_train_mode,
    kd_disable_amp,
    find_unused_parameters,
    performance_level,
    profile,
    profile_active_iters,
    profile_warmup_iters,
    profile_memory,
    **kwargs,
):
    # use accelerator
    dataloader_config = DataLoaderConfiguration(dispatch_batches=False)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)
    accelerator = Accelerator(cpu=use_cpu, dataloader_config=dataloader_config, kwargs_handlers=[ddp_kwargs])
    set_performance_level(performance_level)

    if accelerator.is_local_main_process:
        tb_logger = SummaryWriter(os.path.join(rundir, "log"))
        log_filename = os.path.join(rundir, "training_log.jsonl")

    profile_ctx = nullcontext()
    if profile:

        def profile_handler(p: torch.profiler.profile):
            if accelerator.is_local_main_process:
                dev_name = "cpu" if use_cpu else "cuda"
                print(p.key_averages().table(sort_by=f"{dev_name}_time_total", row_limit=16))
                print("Profile finished. Saving trace...")
            torch.profiler.tensorboard_trace_handler(os.path.join(rundir, "profile_trace"), use_gzip=True)(p)

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
            batch_size_per_process,
            num_workers=num_worker,
            shuffle=False,
            **dataloader_args,
        )
    else:
        val_dataset, val_loader = None, None

    # build model, optimizer
    model = build_model(model_type, **model_args)
    optimizer = build_optimizer(
        optim_type, model.parameters(), lr=learning_rate, weight_decay=weight_decay, **optim_args
    )

    # load checkpoint if exists
    model_name = model.name
    ckpt_filename = find_latest_model_file(rundir, f"ckpt_{model_name}")
    if ckpt_filename:
        state_dicts = torch.load(ckpt_filename, map_location=accelerator.device)
        model.load_state_dict(state_dicts["model"])
        optimizer.load_state_dict(state_dicts["optimizer"])
        if accelerator.scaler is not None and "scalar" in state_dicts:
            accelerator.scaler.load_state_dict(state_dicts["scalar"])
        accelerator.print(f"Loaded from {ckpt_filename}")
        it = state_dicts.get("iteration", 0)
        epoch = state_dicts.get("epoch", 0)
        rows = state_dicts.get("rows", 0)
    else:
        model.apply(weights_init(init_cfg))
        it, epoch, rows = 0, 0, 0
        if load_from is not None:
            state_dicts = torch.load(load_from, map_location=accelerator.device)
            missing_keys, unexpected_keys = model.load_state_dict(
                state_dict_drop_size_unmatched(model, state_dicts["model"]), strict=False
            )
            if len(unexpected_keys) > 0:
                accelerator.print(f"unexpected keys in state_dict: {', '.join(unexpected_keys)}")
            if len(missing_keys) > 0:
                accelerator.print(f"missing keys in state_dict: {', '.join(missing_keys)}")
            accelerator.print(f"Loaded from pretrained: {load_from}")

    # build lr scheduler
    scheduler = build_lr_scheduler(optimizer, lr_scheduler_type, last_it=it - 1, **lr_scheduler_args)

    # accelerate model training
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    if val_loader is not None:
        val_loader = accelerator.prepare_data_loader(val_loader)

    # build teacher model if knowledge distillation model is on
    if kd_model_type is not None:
        kd_model = build_model(kd_model_type, **kd_model_args)
        kd_state_dicts = torch.load(kd_checkpoint, map_location=accelerator.device)
        kd_model.load_state_dict(kd_state_dicts["model"])
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
    last_it = it
    last_time = time.time()
    avg_metric_dict = {}
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
            data["train_progress"] = it / iterations

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

            # update running average loss
            loss_dict = accelerator.gather(loss_dict)
            aux_dict = accelerator.gather(aux_dict)
            for metric_dict in [loss_dict, aux_dict]:
                for key, value in metric_dict.items():
                    value = torch.mean(value, dim=0).item()
                    if not key in avg_metric_dict:
                        avg_metric_dict[key] = deque(maxlen=avg_loss_interval)
                    avg_metric_dict[key].append(value)
                    metric_dict[key] = np.mean(avg_metric_dict[key])

            # logging
            if it % log_interval == 0 and accelerator.is_local_main_process:
                log_value_dict(tb_logger, "train", loss_dict, it, rows)
                if aux_dict:
                    log_value_dict(tb_logger, "train_aux", aux_dict, it, rows)
                with open(log_filename, "a") as log:
                    json_text = json.dumps(
                        {
                            "it": it,
                            "epoch": epoch,
                            "rows": rows,
                            "train_loss": loss_dict,
                            "train_aux": aux_dict,
                            "lr": scheduler.get_last_lr()[0],
                        }
                    )
                    log.writelines([json_text, "\n"])

            # display current progress
            if it % show_interval == 0 and accelerator.is_local_main_process:
                elasped = time.time() - last_time
                num_it = it - last_it
                speed = num_it / elasped
                log_value_dict(
                    tb_logger,
                    "running_stat",
                    {
                        "epoch": epoch,
                        "rows": rows,
                        "elasped_seconds": elasped,
                        "it/s": speed,
                        "entry/s": speed * batch_size,
                        "lr": scheduler.get_last_lr()[0],
                    },
                    it,
                    rows,
                )
                print(
                    f"[{it:08d}][{epoch}][{elasped:.2f}s][{speed:.2f}it/s]"
                    + f" total: {loss_dict['total_loss']:.4f},"
                    + f" value: {loss_dict['value_loss']:.4f},"
                    + f" policy: {loss_dict['policy_loss']:.4f}"
                )
                last_it = it
                last_time = time.time()

            # snapshot saving
            if (
                it % save_interval == 0 or it % temp_save_interval == 0
            ) and accelerator.is_local_main_process:
                # get latest snapshot filename
                last_snapshot_filename = find_latest_model_file(rundir, f"ckpt_{model_name}")

                # save snapshot at current iteration
                state_dict = {
                    "iteration": it,
                    "epoch": epoch,
                    "rows": rows,
                    "model": unwarpped_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                if accelerator.scaler is not None:
                    state_dict["scalar"] = accelerator.scaler.state_dict()
                torch.save(state_dict, os.path.join(rundir, f"ckpt_{model_name}_{it:08d}.pth"))

                # remove last temporary snapshot if it's not a snapshot iteration
                if last_snapshot_filename is not None:
                    last_snapshot_iter = get_iteration_from_model_filename(last_snapshot_filename)
                    if last_snapshot_iter is None or last_snapshot_iter % save_interval != 0:
                        os.remove(last_snapshot_filename)

            # validation
            if it % val_interval == 0 and val_loader is not None:
                val_start_time = time.time()
                val_loss_dict, val_aux_dict = {}, {}
                num_val_batches = 0

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

                    val_elasped = time.time() - val_start_time
                    num_val_entries = num_val_batches * batch_size_per_process
                    # log validation results
                    log_value_dict(tb_logger, "validation", val_loss_dict, it, rows)
                    if val_aux_dict:
                        log_value_dict(tb_logger, "validation_aux", val_aux_dict, it, rows)
                    with open(log_filename, "a") as log:
                        json_text = json.dumps(
                            {
                                "it": it,
                                "epoch": epoch,
                                "val_loss": val_loss_dict,
                                "val_aux": val_aux_dict,
                                "num_val_entries": num_val_entries,
                                "elasped_seconds": val_elasped,
                            }
                        )
                        log.writelines([json_text, "\n"])
                    print(
                        f"[validation][{num_val_entries} entries][{val_elasped:.2f}s]"
                        + f" total: {val_loss_dict['total_loss']:.4f},"
                        + f" value: {val_loss_dict['value_loss']:.4f},"
                        + f" policy: {val_loss_dict['policy_loss']:.4f}"
                    )

                    # substract validation time from training time
                    last_time += val_elasped

            if profile:
                profiler.step()


if __name__ == "__main__":
    args = parse_args_and_init()
    training_loop(**vars(args))
