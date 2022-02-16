import torch
import numpy as np
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
from utils.training_utils import cross_entropy_with_softlabel, \
    build_optimizer, weights_init, build_data_loader
from utils.misc_utils import add_dict_to, seed_everything, write_loss_dict
from utils.file_utils import find_latest_model_file


def parse_args_and_init():
    parser = configargparse.ArgParser(description="Trainer",
                                      config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add('-c', '--config', is_config_file=True, help='Config file path')
    parser.add('-d', '--train_datas', nargs='+', help="Training dataset file or directory paths")
    parser.add('-v', '--val_datas', nargs='+', help="Validation dataset file or directory paths")
    parser.add('-r', '--rundir', required=True, help="Run directory")
    parser.add('--use_cpu', action='store_true', help="Use cpu only")
    parser.add('--dataset_type', required=True, help="Dataset type")
    parser.add('--dataset_args', type=yaml.safe_load, default={}, help="Extra dataset arguments")
    parser.add('--model_type', required=True, help="Model type")
    parser.add('--model_args', type=yaml.safe_load, default={}, help="Extra model arguments")
    parser.add('--optim_type', default='adamw', help='Optimizer type')
    parser.add('--optim_args', type=yaml.safe_load, default={}, help="Extra optimizer arguments")
    parser.add('--init_type', default='kaiming', help='Initialization type')
    parser.add('--loss_type', default='CE+CE', help='Loss type')
    parser.add('--iterations', type=int, default=1000000, help="Num iterations")
    parser.add('--batch_size', type=int, default=128, help="Batch size")
    parser.add('--num_worker', type=int, default=8, help="Num of dataloader workers")
    parser.add('--learning_rate', type=float, default=1e-3, help="Learning rate")
    parser.add('--weight_decay', type=float, default=1e-6, help="Weight decay")
    parser.add('--shuffle', action='store_true', default=True, help="Shuffle dataset")
    parser.add('--seed', type=int, default=42, help="Random seed")
    parser.add('--log_interval', type=int, default=100, help="Num iterations to log")
    parser.add('--show_interval', type=int, default=1000, help="Num iterations to display")
    parser.add('--save_interval', type=int, default=10000, help="Num iterations to save snapshot")
    parser.add('--val_interval', type=int, default=10000, help="Num iterations to do validation")
    parser.add('--avg_loss_interval',
               type=int,
               default=1000,
               help="Num iterations to average loss")

    args = parser.parse_args()  # parse args
    parser.print_values()  # print out values
    os.makedirs(args.rundir, exist_ok=True)  # make run directory
    # write run config
    run_cfg_filename = os.path.join(args.rundir, "run_config.yaml")
    if (args.config is not None
            and os.path.abspath(args.config) != os.path.abspath(run_cfg_filename)):
        parser.write_config_file(args, [run_cfg_filename])
    seed_everything(args.seed)  # set seed
    print('-' * 60)

    return args


def calc_loss(loss_type, value, policy, data):
    value_target = data['value_target']
    policy_target = data['policy_target']

    # reshape policy
    policy_target = torch.flatten(policy_target, start_dim=1)
    policy = torch.flatten(policy, start_dim=1)

    if loss_type == 'CE+CE':
        value_loss = cross_entropy_with_softlabel(value, value_target, adjust=True)
        policy_loss = cross_entropy_with_softlabel(policy, policy_target, adjust=True)
    elif loss_type == 'CE+MSE':
        value_loss = cross_entropy_with_softlabel(value, value_target, adjust=True)
        policy_loss = ((policy - policy_target)**2).mean()
    else:
        assert 0, f"Unsupported loss: {loss_type}"

    total_loss = value_loss + policy_loss
    return total_loss, {
        'total_loss': total_loss.item(),
        'value_loss': value_loss.item(),
        'policy_loss': policy_loss.item()
    }


def training_loop(rundir, use_cpu, train_datas, val_datas, dataset_type, dataset_args, model_type,
                  model_args, optim_type, optim_args, init_type, loss_type, iterations, batch_size,
                  num_worker, learning_rate, weight_decay, shuffle, log_interval, show_interval,
                  save_interval, val_interval, avg_loss_interval, **kwargs):
    tb_logger = SummaryWriter(os.path.join(rundir, "log"))
    log_filename = os.path.join(rundir, 'training_log.json')

    # use accelerator
    accelerator = Accelerator(cpu=use_cpu)

    # load train and validation dataset
    train_dataset = build_dataset(dataset_type, train_datas, shuffle=shuffle, **dataset_args)
    train_loader = build_data_loader(train_dataset,
                                     batch_size,
                                     num_workers=num_worker,
                                     shuffle=shuffle)
    if val_datas:
        val_dataset = build_dataset(dataset_type, val_datas, shuffle=shuffle, **dataset_args)
        val_loader = build_data_loader(val_dataset,
                                       batch_size,
                                       num_workers=num_worker,
                                       shuffle=shuffle)
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
    ckpt_filename = find_latest_model_file(rundir, f"ckpt_{model.name}")
    if ckpt_filename:
        state_dicts = torch.load(ckpt_filename)
        model.load_state_dict(state_dicts['model'])
        optimizer.load_state_dict(state_dicts['optimizer'])
        accelerator.print(f'Loaded from checkpoint: {ckpt_filename}')
    else:
        model.apply(weights_init(init_type))

    # accelerate model training
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    if val_loader:
        val_loader = accelerator.prepare_data_loader(val_loader)

    epoch, it = 0, 0
    last_it = 0
    last_time = time.time()
    stop_training = False
    avg_loss_dict = {}
    while not stop_training:
        epoch += 1
        for data in train_loader:
            it += 1
            if it >= iterations:
                stop_training = True
                break

            # model update
            optimizer.zero_grad()
            value, policy = model(data)
            loss, loss_dict = calc_loss(loss_type, value, policy, data)
            accelerator.backward(loss)
            optimizer.step()

            # update running average loss
            for key, value in loss_dict.items():
                if not key in avg_loss_dict:
                    avg_loss_dict[key] = deque(maxlen=avg_loss_interval)
                avg_loss_dict[key].append(value)
                loss_dict[key] = np.mean(avg_loss_dict[key])

            # logging
            if it % log_interval == 0 and accelerator.is_local_main_process:
                write_loss_dict(tb_logger, 'train', {'epoch': epoch, **loss_dict}, it)
                with open(log_filename, 'a') as log:
                    json_text = json.dumps({'it': it, 'train_loss': loss_dict})
                    log.writelines([json_text, '\n'])

            # display current progress
            if it % show_interval == 0 and accelerator.is_local_main_process:
                elasped = time.time() - last_time
                num_it = it - last_it
                print(f"[{it:08d}][{epoch}][{elasped:.2f}s][{num_it/elasped:.2f}it/s]" +
                      f" total: {loss_dict['total_loss']:.4f}," +
                      f" value: {loss_dict['value_loss']:.4f}," +
                      f" policy: {loss_dict['policy_loss']:.4f}")
                last_it = it
                last_time = time.time()

            # snapshot saving
            if it % save_interval == 0 and accelerator.is_local_main_process:
                snapshot_filename = os.path.join(rundir, f"ckpt_{model.name}_{it:08d}.pth")
                torch.save(
                    {
                        "model": accelerator.get_state_dict(model),
                        "optimizer": optimizer.state_dict(),
                    }, snapshot_filename)

            # validation
            if it % val_interval == 0 and val_loader and accelerator.is_local_main_process:
                val_loss_dict = {}
                num_val_entries = 0

                with torch.no_grad():
                    for val_data in val_loader:
                        value, policy = model(val_data)
                        _, val_losses = calc_loss(loss_type, value, policy, val_data)
                        add_dict_to(val_loss_dict, val_losses)
                        num_val_entries += 1

                # average all losses
                for k in val_loss_dict:
                    val_loss_dict[k] /= num_val_entries

                write_loss_dict(tb_logger, 'validation', val_loss_dict, it)
                with open(log_filename, 'a') as log:
                    json_text = json.dumps({'it': it, 'val_loss': val_loss_dict})
                    log.writelines([json_text, '\n'])


if __name__ == "__main__":
    args = parse_args_and_init()
    training_loop(**vars(args))
