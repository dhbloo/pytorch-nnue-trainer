import torch
import numpy as np
import torch.nn.functional as F
from accelerate import Accelerator
from tensorboardX import SummaryWriter
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
    parser.add('--avg_loss_interval',
               type=int,
               default=2500,
               help="Num iterations to average loss")

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


def calc_loss(loss_type, data, value, policy):
    value_loss_type, policy_loss_type = loss_type.split('+')
    value_target = data['value_target']
    policy_target = data['policy_target']

    # reshape policy
    policy_target = torch.flatten(policy_target, start_dim=1)
    policy = torch.flatten(policy, start_dim=1)

    # convert value_target if value has no draw info
    if value.size(1) == 1:
        value = value[:, 0]
        value_target = value_target[:, 0] - value_target[:, 1]
        value_target = (value_target + 1) / 2  # normalize [-1, 1] to [0, 1]

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

    # policy loss
    if policy_loss_type == 'CE':
        policy_loss = cross_entropy_with_softlabel(policy, policy_target, adjust=True)
    elif policy_loss_type == 'MSE':
        policy_loss = F.mse_loss(torch.softmax(policy, dim=1), policy_target)
    else:
        assert 0, f"Unsupported policy loss: {policy_loss_type}"

    total_loss = value_loss + policy_loss
    return total_loss, {
        'total_loss': total_loss.detach(),
        'value_loss': value_loss.detach(),
        'policy_loss': policy_loss.detach(),
    }


def training_loop(rundir, load_from, use_cpu, train_datas, val_datas, dataset_type, dataset_args,
                  dataloader_args, model_type, model_args, optim_type, optim_args,
                  lr_scheduler_type, lr_scheduler_args, init_type, loss_type, iterations,
                  batch_size, num_worker, learning_rate, weight_decay, clip_grad_norm, no_shuffle,
                  log_interval, show_interval, save_interval, val_interval, avg_loss_interval,
                  **kwargs):
    # use accelerator
    accelerator = Accelerator(cpu=use_cpu, dispatch_batches=False)

    if accelerator.is_local_main_process:
        tb_logger = SummaryWriter(os.path.join(rundir, "log"))
        log_filename = os.path.join(rundir, 'training_log.json')

    # load train and validation dataset
    train_dataset = build_dataset(dataset_type,
                                  train_datas,
                                  shuffle=not no_shuffle,
                                  **dataset_args)
    train_loader = build_data_loader(train_dataset,
                                     batch_size,
                                     num_workers=num_worker,
                                     shuffle=not no_shuffle,
                                     **dataloader_args)
    if val_datas:
        val_dataset = build_dataset(dataset_type, val_datas, shuffle=False, **dataset_args)
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
        epoch, it = state_dicts.get('epoch', 0), state_dicts.get('iteration', 0)
    else:
        model.apply(weights_init(init_type))
        epoch, it = 0, 0
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
            if it > iterations:
                stop_training = True
                break

            # model update
            optimizer.zero_grad()
            value, policy, *retvals = model(data)
            loss, loss_dict = calc_loss(loss_type, data, value, policy)
            accelerator.backward(loss)

            # apply gradient clipping if needed
            if clip_grad_norm is not None:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)

            optimizer.step()
            lr_scheduler.step()

            # apply weight clipping if needed
            if hasattr(accelerator.unwrap_model(model), 'weight_clipping'):
                unwarpped_model = accelerator.unwrap_model(model)
                clip_parameters = unwarpped_model.weight_clipping
                weight_clipping(unwarpped_model.named_parameters(), clip_parameters)

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
                        value, policy, *retvals = model(val_data)
                        _, val_losses = calc_loss(loss_type, val_data, value, policy)
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
