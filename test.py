import torch
import numpy as np
from accelerate import Accelerator, PartialState, DataLoaderConfiguration
from tqdm.auto import tqdm
import configargparse
import yaml
import json
import time
import os

from dataset import build_dataset
from model import build_model
from train import calc_loss
from utils.file_utils import load_torch_ckpt
from utils.training_utils import build_data_loader
from utils.misc_utils import add_dict_to, seed_everything, set_performance_level, deep_update_dict


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

    args, _ = parser.parse_known_args()  # parse args

    if PartialState(cpu=args.use_cpu).is_local_main_process:
        parser.print_values()
        print("-" * 60)

    return args


def top_k_accuracy(policy, policy_target, k):
    _, topkmoves = torch.topk(policy, dim=1, k=k, sorted=False)
    _, topkmoves_target = torch.topk(policy_target, dim=1, k=k, sorted=False)
    move_correct_count = 0
    for moves, moves_target in zip(topkmoves, topkmoves_target):
        moveset = set(moves.tolist())
        moveset_target = set(moves_target.tolist())
        move_correct_count += len(moveset.intersection(moveset_target))
    return move_correct_count / k / len(topkmoves)


def calc_metric(data, results, do_cross_eval):
    value, policy, *retvals = results
    value_target = data["value_target"]
    policy_target = data["policy_target"]

    # reshape policy
    policy_target = torch.flatten(policy_target, start_dim=1)
    policy = torch.flatten(policy, start_dim=1)

    # convert value_target if value has no draw info
    if value.size(1) == 1:
        value = value[:, 0]
        value_target = value_target[:, 0] - value_target[:, 1]
        value_target = (value_target + 1) / 2  # normalize [-1, 1] to [0, 1]

    # calc metrics
    metrics = {}

    # 0. losses
    _, losses_ce_ce, _ = calc_loss("CE+CE", data, results)
    for k, v in losses_ce_ce.items():
        metrics[f"lossCE_{k}"] = v.item()

    # 1. bestmove accuracy
    bestmove_eq = torch.argmax(policy, dim=1) == torch.argmax(policy_target, dim=1)
    bestmove_acc = torch.sum(bestmove_eq) / bestmove_eq.size(0)
    metrics["bestmove_acc"] = bestmove_acc.item()
    metrics["top2move_acc"] = top_k_accuracy(policy, policy_target, k=2)
    metrics["top3move_acc"] = top_k_accuracy(policy, policy_target, k=3)

    # 2. value accuracy, mse
    if value.ndim == 1:
        # sigmoid activation
        value = torch.sigmoid(value)

        winrate_correct = (value - 0.5) * (value_target - 0.5) >= 0
        winrate_mse = torch.mean((value - value_target) ** 2)
    else:
        # softmax activation
        value = torch.softmax(value, dim=1)

        value_iswin = value[:, 0] >= value[:, 1]
        value_iswin_target = value_target[:, 0] >= value_target[:, 1]
        winrate_correct = value_iswin == value_iswin_target

        value_norm = (value[:, 0] - value[:, 1] + 1) / 2
        value_norm_target = (value_target[:, 0] - value_target[:, 1] + 1) / 2
        winrate_mse = torch.mean((value_norm - value_norm_target) ** 2)

        value_isdraw = torch.argmax(value, dim=1) == 2
        value_isdraw_target = torch.argmax(value_target, dim=1) == 2
        drawrate_correct = value_isdraw == value_isdraw_target
        drawrate_mse = torch.mean((value[:, 2] - value_target[:, 2]) ** 2)

        drawrate_acc = torch.sum(drawrate_correct) / drawrate_correct.size(0)
        metrics["drawrate_acc"] = drawrate_acc.item()
        metrics["drawrate_mse"] = drawrate_mse.item()
    winrate_acc = torch.sum(winrate_correct) / winrate_correct.size(0)
    metrics["winrate_acc"] = winrate_acc.item()
    metrics["winrate_mse"] = winrate_mse.item()

    # cross eval. compute absolute and relative error
    if do_cross_eval:
        metrics["value_abserr_mean"] = torch.abs(value - value_target).mean().item()
        metrics["value_abserr_max"] = torch.abs(value - value_target).max().item()
        metrics["value_relerr_mean"] = (
            (torch.abs(value - value_target) / (torch.abs(value) + 1e-4)).mean().item()
        )
        metrics["value_relerr_max"] = (
            (torch.abs(value - value_target) / (torch.abs(value) + 1e-4)).max().item()
        )

    return metrics


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
    **kwargs,
):
    if not os.path.exists(checkpoint) or not os.path.isfile(checkpoint):
        raise RuntimeError(f"Checkpoint {checkpoint} must be a valid file")
    rundir = os.path.dirname(checkpoint)
    ckpt_filename_noext = os.path.splitext(os.path.basename(checkpoint))[0]
    log_filename = os.path.join(rundir, f"{ckpt_filename_noext}_test_result.json")

    # use accelerator
    dataloader_config = DataLoaderConfiguration(dispatch_batches=False)
    accelerator = Accelerator(cpu=use_cpu, dataloader_config=dataloader_config)
    seed_everything(random_seed)
    set_performance_level(0)

    # load test dataset
    if test_datas is None:
        if train_datas is None:
            raise RuntimeError("Test dataset must be set if train dataset is not set.")
        test_datas = train_datas
    if test_dataset_type is None:
        if dataset_type is None:
            raise RuntimeError("Test dataset type must be set if train dataset type is not set.")
        test_dataset_type = dataset_type
        test_dataset_args = dataset_args
    batch_size_per_process = batch_size // accelerator.num_processes
    test_dataset = build_dataset(test_dataset_type, test_datas, shuffle=False, **test_dataset_args)
    test_loader = build_data_loader(
        test_dataset,
        batch_size_per_process * eval_bs_multipler,
        num_workers=num_worker,
        shuffle=False,
        **dataloader_args,
    )

    # build model
    if test_model_args:
        model_args = deep_update_dict(model_args, test_model_args)
    model = build_model(model_type, **model_args)

    # load checkpoint
    model_state_dict, _, metadata = load_torch_ckpt(checkpoint)
    model.load_state_dict(model_state_dict)
    epoch, it = metadata.get("epoch", "?"), metadata.get("iteration", "?")
    accelerator.print(f"Loaded from {checkpoint}, epoch: {epoch}, it: {it}")

    # accelerate model testing
    model, test_loader = accelerator.prepare(model, test_loader)
    model.eval()

    # testing
    metric_dict = {}
    num_batches = 0
    start_time = time.time()
    with torch.no_grad(), accelerator.autocast():
        for data in tqdm(test_loader, disable=not accelerator.is_local_main_process):
            results = model(data)
            metrics = calc_metric(data, results, do_cross_eval)
            add_dict_to(metric_dict, metrics)
            num_batches += 1
            if max_batches is not None and num_batches >= max_batches:
                break

    # convert metrics floats to tensor
    for k, loss in metric_dict.items():
        metric_dict[k] = torch.FloatTensor([loss]).to(accelerator.device)
    metric_dict["num_batches"] = torch.LongTensor([num_batches]).to(accelerator.device)
    # gather all metric dict across processes
    all_metric_dict = accelerator.gather(metric_dict)

    if accelerator.is_local_main_process:
        # average all metrics
        num_batches_tensor = all_metric_dict.pop("num_batches")
        num_batches = torch.sum(num_batches_tensor).item()
        metric_dict = {}
        for k, metric_tensor in all_metric_dict.items():
            metric_dict[k] = torch.sum(metric_tensor).item() / num_batches

        elasped = time.time() - start_time
        num_entries = num_batches * batch_size_per_process * eval_bs_multipler
        # log test results
        with open(log_filename, "w") as log:
            log_dict = {
                "test_datas": test_datas,
                "dataset_type": test_dataset_type,
                "dataset_args": test_dataset_args,
                "dataloader_args": dataloader_args,
                "use_cpu": use_cpu,
                "do_cross_eval": do_cross_eval,
                "num_entries": num_entries,
                "metrics": metric_dict,
            }
            json_text = json.dumps(log_dict, indent=4)
            log.writelines([json_text, "\n"])
        print(f"Test finished with {num_entries} entries, in {elasped:.2f}s.")
        print("Metrics:")
        for k, metric in metric_dict.items():
            print(f"\t{k}: {metric:.4f}")


if __name__ == "__main__":
    args = parse_args_and_init()
    test(**vars(args))
