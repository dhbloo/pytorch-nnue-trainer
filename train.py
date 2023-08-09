import torch
import numpy as np
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import configargparse
import yaml
import json
import time
import os

from dataset import build_dataset
from model import build_model
from utils.training_utils import build_lr_scheduler, cross_entropy_with_softlabel, \
    build_optimizer, weights_init, build_data_loader, weight_clipping
from utils.misc_utils import add_dict_to, seed_everything, log_value_dict
from utils.file_utils import find_latest_model_file


def parse_args_and_init():
    parser = configargparse.ArgParser(description="Trainer",
                                      config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add('-c', '--config', is_config_file=True, help='Config file path')
    parser.add('-d', '--train_datas', nargs='+', help="Training dataset file or directory paths")
    parser.add('-v', '--val_datas', nargs='+', help="Validation dataset file or directory paths")
    parser.add('-r', '--rundir', required=True, help="Run directory")
    parser.add('--load_from', help="Load pretrained weights from file")
    parser.add('--use_cpu', action='store_true', help="Use cpu only")
    parser.add('--dataset_type', required=True, help="Dataset type")
    parser.add('--dataset_args', type=yaml.safe_load, default={}, help="Extra dataset arguments")
    parser.add('--dataloader_args',
               type=yaml.safe_load,
               default={},
               help="Extra dataloader arguments")
    parser.add('--data_pipelines',
               type=yaml.safe_load,
               default=None,
               help="Data-pipeline type and arguments")
    parser.add('--num_worker',
               type=int,
               default=min(8, os.cpu_count()),
               help="Num of dataloader workers")
    parser.add('--model_type', required=True, help="Model type")
    parser.add('--model_args', type=yaml.safe_load, default={}, help="Extra model arguments")
    parser.add('--optim_type', default='adamw', help='Optimizer type')
    parser.add('--optim_args', type=yaml.safe_load, default={}, help="Extra optimizer arguments")
    parser.add('--lr_scheduler_type', default='constant', help='LR scheduler type')
    parser.add('--lr_scheduler_args',
               type=yaml.safe_load,
               default={},
               help="Extra LR scheduler arguments")
    parser.add('--init_type', default='kaiming', help='Initialization type')
    parser.add('--loss_type', default='CE+CE', help='Loss type')
    parser.add('--loss_args', type=yaml.safe_load, default={}, help="Extra loss arguments")
    parser.add('--iterations', type=int, default=1000000, help="Num iterations")
    parser.add('--batch_size', type=int, default=128, help="Batch size")
    parser.add('--learning_rate', type=float, default=1e-3, help="Learning rate")
    parser.add('--weight_decay', type=float, default=1e-7, help="Weight decay")
    parser.add('--clip_grad_norm', type=float, help="Gradient clipping max norm")
    parser.add('--no_shuffle', action='store_true', help="Do not shuffle dataset")
    parser.add('--seed', type=int, default=42, help="Random seed")
    parser.add('--log_interval', type=int, default=100, help="Num iterations to log")
    parser.add('--show_interval', type=int, default=1000, help="Num iterations to display")
    parser.add('--save_interval', type=int, default=10000, help="Num iterations to save snapshot")
    parser.add('--val_interval', type=int, default=25000, help="Num iterations to do validation")
    parser.add('--avg_loss_interval', type=int, default=2500, help="Num iterations to average loss")
    parser.add('--kd_model_type', help="Knowledge distillation model type")
    parser.add('--kd_model_args',
               type=yaml.safe_load,
               default={},
               help="Knowledge distillation extra model arguments")
    parser.add('--kd_checkpoint', help="Knowledge distillation model checkpoint")
    parser.add('--kd_T', type=float, default=1.0, help="Knowledge distillation temperature")
    parser.add('--kd_alpha',
               type=float,
               default=1.0,
               help="Distillation loss ratio in [0,1] (1 for distillation loss only)")

    args = parser.parse_args()  # parse args
    parser.print_values()  # print out values
    os.makedirs(args.rundir, exist_ok=True)  # make run directory
    # write run config
    run_cfg_filename = os.path.join(args.rundir, "run_config.yaml")
    if args.config is None or os.path.abspath(args.config) != os.path.abspath(run_cfg_filename):
        parser.write_config_file(args, [run_cfg_filename])
    seed_everything(args.seed)  # set seed
    print('-' * 60)

    return args


def calc_loss(loss_type,
              data,
              results,
              kd_results=None,
              kd_T=None,
              kd_alpha=None,
              policy_reg_lambda=None,
              ignore_forbidden_point_policy=False):
    value_loss_type, policy_loss_type = loss_type.split('+')

    # get predicted value and policy from model results
    value, policy, *retvals = results

    if kd_results is not None:
        # get value and policy target from teacher model
        value_target, policy_target, *kd_retvals = kd_results
    else:
        # get value and poliay target from data
        value_target = data['value_target']
        policy_target = data['policy_target']

    # reshape policy
    policy_target = torch.flatten(policy_target, start_dim=1)
    policy = torch.flatten(policy, start_dim=1)
    assert policy_target.shape[1] == policy.shape[1]

    if kd_results is not None:
        # apply softmax to value and policy target according to temparature
        value_target = torch.softmax(value_target / kd_T, dim=1)
        policy_target = torch.softmax(policy_target / kd_T, dim=1)

        # scale prediction value and policy according to temparature
        value = value / kd_T
        policy = policy / kd_T

    # convert value_target if value has no draw info
    if value.size(1) == 1:
        value = value[:, 0]
        value_target = value_target[:, 0] - value_target[:, 1]
        value_target = (value_target + 1) / 2  # normalize [-1, 1] to [0, 1]

    # ===============================================================
    # value loss
    if value_loss_type == 'CE':
        if value.ndim == 1:
            value_loss = (F.binary_cross_entropy_with_logits(value, value_target) -
                          F.binary_cross_entropy_with_logits(value_target + 1e-8, value_target))
        else:
            value_loss = cross_entropy_with_softlabel(value, value_target, adjust=True)
    elif value_loss_type == 'MSE':
        if value.ndim == 1:
            value_loss = F.mse_loss(torch.sigmoid(value), value_target)
        else:
            value = torch.softmax(value, dim=1)  # [B, 3]
            winrate = (value[:, 0] - value[:, 1] + 1) / 2
            winrate_target = (value_target[:, 0] - value_target[:, 1] + 1) / 2
            value_loss = F.mse_loss(winrate, winrate_target)
    else:
        assert 0, f"Unsupported value loss: {value_loss_type}"

    # ===============================================================
    # policy loss

    # check if there is loss weight for policy at each empty cells
    if ignore_forbidden_point_policy:
        forbidden_point = torch.flatten(data['forbidden_point'], start_dim=1)  # [B, H*W]
        policy_loss_weight = 1.0 - forbidden_point.float()  # [B, H*W]
    else:
        policy_loss_weight = None

    if policy_loss_type == 'CE':
        policy_loss = cross_entropy_with_softlabel(policy,
                                                   policy_target,
                                                   adjust=True,
                                                   weight=policy_loss_weight)
    elif policy_loss_type == 'MSE':
        policy_softmaxed = torch.softmax(policy, dim=1)
        policy_loss = F.mse_loss(policy_softmaxed, policy_target, size_average='none')
        if policy_loss_weight is not None:
            policy_loss = policy_loss * policy_loss_weight
        policy_loss = torch.mean(policy_loss)
    else:
        assert 0, f"Unsupported policy loss: {policy_loss_type}"

    # ===============================================================

    total_loss = value_loss + policy_loss
    loss_dict = {
        'total_loss': total_loss.detach(),
        'value_loss': value_loss.detach(),
        'policy_loss': policy_loss.detach(),
    }

    if policy_reg_lambda is not None:
        policy_reg = torch.mean(policy).square()
        total_loss += float(policy_reg_lambda) * policy_reg
        loss_dict['policy_reg'] = policy_reg.detach()

    if kd_results is not None:
        # also get a true loss from real data
        real_loss, real_loss_dict = calc_loss(
            loss_type,
            data,
            results,
            policy_reg_lambda=policy_reg_lambda,
            ignore_forbidden_point_policy=ignore_forbidden_point_policy,
        )

        # merge real loss and knowledge distillation loss
        total_loss = kd_alpha * total_loss + (1 - kd_alpha) * real_loss
        kd_loss_dict = {'kd_' + k: v for k, v in loss_dict.items()}
        loss_dict = real_loss_dict
        loss_dict.update(kd_loss_dict)

    return total_loss, loss_dict


def training_loop(rundir, load_from, use_cpu, train_datas, val_datas, dataset_type, dataset_args,
                  dataloader_args, data_pipelines, model_type, model_args, optim_type, optim_args,
                  lr_scheduler_type, lr_scheduler_args, init_type, loss_type, loss_args, iterations,
                  batch_size, num_worker, learning_rate, weight_decay, clip_grad_norm, no_shuffle,
                  log_interval, show_interval, save_interval, val_interval, avg_loss_interval,
                  kd_model_type, kd_model_args, kd_checkpoint, kd_T, kd_alpha, **kwargs):
    # use accelerator
    accelerator = Accelerator(cpu=use_cpu, dispatch_batches=False)

    if accelerator.is_local_main_process:
        tb_logger = SummaryWriter(os.path.join(rundir, "log"))
        log_filename = os.path.join(rundir, 'training_log.json')

    # load train and validation dataset
    train_dataset = build_dataset(dataset_type,
                                  train_datas,
                                  shuffle=not no_shuffle,
                                  pipeline_args=data_pipelines,
                                  **dataset_args)
    train_loader = build_data_loader(train_dataset,
                                     batch_size,
                                     num_workers=num_worker,
                                     shuffle=not no_shuffle,
                                     **dataloader_args)
    if val_datas:
        val_dataset = build_dataset(dataset_type,
                                    val_datas,
                                    shuffle=False,
                                    pipeline_args=data_pipelines,
                                    **dataset_args)
        val_loader = build_data_loader(val_dataset,
                                       batch_size,
                                       num_workers=num_worker,
                                       shuffle=False,
                                       **dataloader_args)
    else:
        val_dataset, val_loader = None, None

    # build model, optimizer
    model = build_model(model_type, **model_args)
    optimizer = build_optimizer(optim_type,
                                model.parameters(),
                                lr=learning_rate,
                                weight_decay=weight_decay,
                                **optim_args)

    # load checkpoint if exists
    model_name = model.name
    ckpt_filename = find_latest_model_file(rundir, f"ckpt_{model_name}")
    if ckpt_filename:
        state_dicts = torch.load(ckpt_filename, map_location=accelerator.device)
        model.load_state_dict(state_dicts['model'])
        optimizer.load_state_dict(state_dicts['optimizer'])
        accelerator.print(f'Loaded from checkpoint: {ckpt_filename}')
        it = state_dicts.get('iteration', 0)
        epoch = state_dicts.get('epoch', 0)
        rows = state_dicts.get('rows', 0)
    else:
        model.apply(weights_init(init_type))
        it, epoch, rows = 0, 0, 0
        if load_from is not None:
            state_dicts = torch.load(load_from, map_location=accelerator.device)
            missing_keys, unexpected_keys = model.load_state_dict(state_dicts['model'],
                                                                  strict=False)
            if len(unexpected_keys) > 0:
                accelerator.print(f"unexpected keys in state_dict: {', '.join(unexpected_keys)}")
            if len(missing_keys) > 0:
                accelerator.print(f"missing keys in state_dict: {', '.join(missing_keys)}")
            accelerator.print(f'Loaded from pretrained: {load_from}')

    # accelerate model training
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    if val_loader is not None:
        val_loader = accelerator.prepare_data_loader(val_loader)

    # build teacher model if knowledge distillation model is on
    if kd_model_type is not None:
        kd_model = build_model(kd_model_type, **kd_model_args)
        kd_state_dicts = torch.load(kd_checkpoint, map_location=accelerator.device)
        kd_model.load_state_dict(kd_state_dicts['model'])
        accelerator.print(f'Loaded teacher model {kd_model.name} from: {kd_checkpoint}')

        kd_model = accelerator.prepare_model(kd_model)
        kd_model.eval()

        # add kd_T into loss_args dict
        assert kd_T is not None, f"kd_T must be set when knowledge distillation is enabled"
        assert kd_alpha is not None and (0 <= kd_alpha <= 1), f"kd_alpha must be in [0,1]"
        loss_args['kd_T'] = kd_T
        loss_args['kd_alpha'] = kd_alpha
    else:
        kd_model = None

    # build lr scheduler
    lr_scheduler = build_lr_scheduler(optimizer,
                                      lr_scheduler_type,
                                      last_it=it - 1,
                                      **lr_scheduler_args)

    accelerator.print(f'Start training from iteration {it}, epoch {epoch}')
    last_it = it
    last_time = time.time()
    stop_training = False
    avg_loss_dict = {}
    model.train()
    while not stop_training:
        epoch += 1
        for data in train_loader:
            it += 1
            rows += batch_size
            if it > iterations:
                stop_training = True
                break

            # evaulate teacher model for knowledge distillation
            with torch.no_grad():
                kd_results = kd_model(data) if kd_model is not None else None

            # apply weight clipping if needed
            if hasattr(accelerator.unwrap_model(model), 'weight_clipping'):
                unwarpped_model = accelerator.unwrap_model(model)
                clip_parameters = unwarpped_model.weight_clipping
                weight_clipping(unwarpped_model.named_parameters(), clip_parameters)

            # model update
            optimizer.zero_grad()
            results = model(data)
            loss, loss_dict = calc_loss(loss_type, data, results, kd_results, **loss_args)
            accelerator.backward(loss)

            # apply gradient clipping if needed
            if clip_grad_norm is not None:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)

            optimizer.step()
            lr_scheduler.step()

            # update running average loss
            loss_dict = accelerator.gather(loss_dict)
            for key, value in loss_dict.items():
                value = torch.mean(value, dim=0).item()
                if not key in avg_loss_dict:
                    avg_loss_dict[key] = deque(maxlen=avg_loss_interval)
                avg_loss_dict[key].append(value)
                loss_dict[key] = np.mean(avg_loss_dict[key])

            # logging
            if it % log_interval == 0 and accelerator.is_local_main_process:
                log_value_dict(tb_logger, 'train', loss_dict, it)
                with open(log_filename, 'a') as log:
                    json_text = json.dumps({
                        'it': it,
                        'epoch': epoch,
                        'rows': rows,
                        'train_loss': loss_dict,
                        'lr': lr_scheduler.get_last_lr()[0],
                    })
                    log.writelines([json_text, '\n'])

            # display current progress
            if it % show_interval == 0 and accelerator.is_local_main_process:
                elasped = time.time() - last_time
                num_it = it - last_it
                speed = num_it / elasped
                log_value_dict(
                    tb_logger, 'running_stat', {
                        'epoch': epoch,
                        'rows': rows,
                        'elasped_seconds': elasped,
                        'it/s': speed,
                        'entry/s': speed * batch_size,
                        'lr': lr_scheduler.get_last_lr()[0],
                    }, it)
                print(f"[{it:08d}][{epoch}][{elasped:.2f}s][{speed:.2f}it/s]" +
                      f" total: {loss_dict['total_loss']:.4f}," +
                      f" value: {loss_dict['value_loss']:.4f}," +
                      f" policy: {loss_dict['policy_loss']:.4f}")
                last_it = it
                last_time = time.time()

            # snapshot saving
            if it % save_interval == 0 and accelerator.is_local_main_process:
                snapshot_filename = os.path.join(rundir, f"ckpt_{model_name}_{it:08d}.pth")
                torch.save(
                    {
                        "iteration": it,
                        "epoch": epoch,
                        "rows": rows,
                        "model": accelerator.get_state_dict(model),
                        "optimizer": optimizer.state_dict(),
                    }, snapshot_filename)

            # validation
            if it % val_interval == 0 and val_loader is not None:
                val_start_time = time.time()
                val_loss_dict = {}
                num_val_batches = 0

                with torch.no_grad():
                    for val_data in val_loader:
                        # evaulate teacher model for knowledge distillation
                        kd_results = kd_model(val_data) if kd_model is not None else None
                        # model evaluation
                        results = model(val_data)
                        _, val_losses = calc_loss(loss_type, val_data, results, kd_results,
                                                  **loss_args)
                        add_dict_to(val_loss_dict, val_losses)
                        num_val_batches += 1

                # gather all loss dict across processes
                val_loss_dict['num_val_batches'] = torch.LongTensor([num_val_batches
                                                                     ]).to(accelerator.device)
                all_val_loss_dict = accelerator.gather(val_loss_dict)

                if accelerator.is_local_main_process:
                    # average all losses
                    num_val_batches_tensor = all_val_loss_dict.pop('num_val_batches')
                    num_val_batches = torch.sum(num_val_batches_tensor).item()
                    val_loss_dict = {}
                    for k, loss_tensor in all_val_loss_dict.items():
                        val_loss_dict[k] = torch.sum(loss_tensor).item() / num_val_batches

                    val_elasped = time.time() - val_start_time
                    num_val_entries = num_val_batches * batch_size
                    # log validation results
                    log_value_dict(tb_logger, 'validation', val_loss_dict, it)
                    with open(log_filename, 'a') as log:
                        json_text = json.dumps({
                            'it': it,
                            'epoch': epoch,
                            'val_loss': val_loss_dict,
                            'num_val_entries': num_val_entries,
                            'elasped_seconds': val_elasped,
                        })
                        log.writelines([json_text, '\n'])
                    print(f"[validation][{num_val_entries} entries][{val_elasped:.2f}s]" +
                          f" total: {val_loss_dict['total_loss']:.4f}," +
                          f" value: {val_loss_dict['value_loss']:.4f}," +
                          f" policy: {val_loss_dict['policy_loss']:.4f}")

                    # substract validation time from training time
                    last_time += val_elasped


if __name__ == "__main__":
    args = parse_args_and_init()
    training_loop(**vars(args))
