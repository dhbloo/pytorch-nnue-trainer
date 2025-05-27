import math
import random
import torch
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader


def weights_init(init_cfg: dict):
    """Generate a init function given a init type"""
    weight_init_type = init_cfg.get("weight_init_type", "kaiming")
    if weight_init_type == "kaiming":
        weight_init_method = init.kaiming_normal_
        weight_init_args = {"a": 0, "mode": "fan_in"}
    elif weight_init_type == "xavier":
        weight_init_method = init.xavier_normal_
        weight_init_args = {"gain": math.sqrt(2)}
    elif weight_init_type == "orthogonal":
        weight_init_method = init.orthogonal_
        weight_init_args = {"gain": math.sqrt(2)}
    elif weight_init_type == "normal":
        weight_init_method = init.normal_
        weight_init_args = {"mean": 0.0, "std": 0.02}
    elif weight_init_type == "truncated_normal":
        weight_init_method = init.trunc_normal_
        weight_init_args = {"mean": 0.0, "std": 0.02}
    elif weight_init_type == "constant":
        weight_init_method = init.constant_
        weight_init_args = {"val": 0.0}
    elif weight_init_type == "default":
        weight_init_method = lambda *args, **kwargs: None
        weight_init_args = {}
    else:
        raise ValueError(f"Unsupported initialization: {weight_init_type}")
    weight_init_args.update(init_cfg.get("weight_init_args", {}))

    bias_init_type = init_cfg.get("bias_init_type", "constant")
    if bias_init_type == "constant":
        bias_init_method = init.constant_
        bias_init_args = {"val": 0.0}
    elif bias_init_type == "default":
        bias_init_method = lambda *args, **kwargs: None
        bias_init_args = {}
    else:
        raise ValueError(f"Unsupported initialization: {bias_init_type}")
    bias_init_args.update(init_cfg.get("bias_init_args", {}))

    def init_fun(m):
        """Note that the init function is called in the post order traversal fashion"""
        classname = m.__class__.__name__
        # First we check if the layer has custom initialization method.
        # If so, we just call it without our initialization.
        if hasattr(m, "initialize"):
            m.initialize()
        # Call our unifrom initialization methods for all Conv and Linear layers
        elif classname.startswith("Conv") or classname.startswith("Linear"):
            if hasattr(m, "weight") and m.weight is not None:
                weight_init_method(m.weight.data, **weight_init_args)

            if hasattr(m, "bias") and m.bias is not None:
                bias_init_method(m.bias.data, **bias_init_args)

    return init_fun


def build_optimizer(
    optim_type: str,
    model: torch.nn.Module,
    lr: float,
    weight_decay: float = 0.0,
    only_track_requires_grad=True,
    **kwargs,
):
    parameters = model.parameters()
    if only_track_requires_grad:
        # only track parameters with requires_grad=True
        parameters = [p for p in parameters if p.requires_grad]

    if optim_type == "adamw":
        args = {"lr": lr, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": weight_decay}
        args.update(kwargs)
        opt = optim.AdamW(parameters, **args)
    elif optim_type == "adamw-ams":
        args = {"lr": lr, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": weight_decay, "amsgrad": True}
        args.update(kwargs)
        opt = optim.AdamW(parameters, **args)
    elif optim_type == "sgd":
        args = {"lr": lr, "momentum": 0, "dampening": 0, "weight_decay": weight_decay}
        args.update(kwargs)
        opt = optim.SGD(parameters, **args)
    elif optim_type == "sgd-momentum":
        args = {"lr": lr, "momentum": 0.9, "dampening": 0.1, "nesterov": False, "weight_decay": weight_decay}
        args.update(kwargs)
        opt = optim.SGD(parameters, **args)
    elif optim_type == "sgd-nesterov":
        args = {"lr": lr, "momentum": 0.9, "dampening": 0.1, "nesterov": True, "weight_decay": weight_decay}
        args.update(kwargs)
        opt = optim.SGD(parameters, **args)
    elif optim_type == "muon-adamw":
        from utils.muon import Muon, get_params_for_muon
        from utils.chained_optimizer import ChainedOptimizer, OptimizerSpec

        params_id_to_name = {id(p): name for name, p in model.named_parameters()}
        muon_params_id_set = set(id(p) for p in get_params_for_muon(model))
        muon_args = {"weight_decay": max(1e-2, 0.0 if weight_decay is None else weight_decay)}
        muon_args.update(kwargs.pop("muon_args", {}))
        adamw_args = {"betas": (0.9, 0.999), "eps": 1e-8}
        adamw_args.update(kwargs.pop("adamw_args", {}))
        spec_muon = OptimizerSpec(Muon, muon_args, lambda param: id(param) in muon_params_id_set)
        spec_adamw = OptimizerSpec(optim.AdamW, adamw_args, None)
        specs = [spec_muon, spec_adamw]
        callback = None
        if kwargs.pop("verbose", False):
            callback = lambda p, spec_idx: print(
                f"Adding param {params_id_to_name[id(p)]} ({p.shape}) to "
                f"optimizer{spec_idx} {str(specs[spec_idx].class_type)}"
            )
        kwargs.update({"lr": lr, "weight_decay": weight_decay, "optimizer_selection_callback": callback})
        opt = ChainedOptimizer(parameters, specs, **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer: {optim_type}")

    return opt


def build_lr_scheduler(optimizer, lr_schedule_type="constant", last_it=-1, **kwargs):
    if lr_schedule_type == "constant":
        scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=0, last_epoch=last_it)
    elif lr_schedule_type == "step":
        step_size = kwargs.get("step_size", 50000)
        step_gamma = kwargs.get("step_gamma", 0.9)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=step_gamma, last_epoch=last_it
        )
    else:
        raise ValueError(f"Unsupported lr scheduler: {lr_schedule_type}")

    return scheduler


def build_data_loader(
    dataset,
    batch_size=1,
    shuffle=False,
    shuffle_buffer_size=32768,
    num_workers=0,
    drop_last=True,
    batch_by_boardsize=False,
    **kwargs,
):
    if shuffle and isinstance(dataset, IterableDataset):
        # Warp non shuffleable dataset with ShuffleDataset
        if not dataset.is_internal_shuffleable:
            dataset = ShuffleDataset(dataset, shuffle_buffer_size)
        shuffle = False

    if batch_by_boardsize:
        assert isinstance(dataset, IterableDataset), "batch_by_boardsize must be used with IterableDataset"
        dataset = BatchByBoardSizeDataset(dataset, batch_size)

    # Default to pin_memory=True for better performance
    if "pin_memory" not in kwargs:
        kwargs["pin_memory"] = True

    dataloader = DataLoader(
        dataset,
        batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        persistent_workers=(num_workers > 0),
        **kwargs,
    )
    return dataloader


def weight_clipping(named_parameters, clip_parameters):
    named_parameters = dict(named_parameters)
    for group in clip_parameters:
        min_weight = group["min_weight"]
        max_weight = group["max_weight"]
        for i, param_name in enumerate(group["params"]):
            p = named_parameters[param_name]
            p_data_fp32 = p.data
            if "virtual_params" in group:
                virtual_param_name = group["virtual_params"][i]
                virtual_param = named_parameters[virtual_param_name]
                virtual_param = virtual_param.repeat(
                    *[p_data_fp32.shape[i] // virtual_param.shape[i] for i in range(virtual_param.ndim)]
                )
                min_weight_t = p_data_fp32.new_full(p_data_fp32.shape, min_weight) - virtual_param
                p_data_fp32 = torch.max(p_data_fp32, min_weight_t)
                max_weight_t = p_data_fp32.new_full(p_data_fp32.shape, max_weight) - virtual_param
                p_data_fp32 = torch.min(p_data_fp32, max_weight_t)
            else:
                p_data_fp32.clamp_(min_weight, max_weight)
            p.data.copy_(p_data_fp32)


def cross_entropy_with_softlabel(
    input, target, reduction="mean", weight=None, focal_gamma=0.0, use_kl_divergence=False, eps=1e-8
):
    """
    :param input: (batch, *) logits before sigmoid/softmax activation.
    :param target: (batch, *) same shape as input, must be a valid distribution
        (sum(target[i, ...]) == 1) for multi-class classification or in [0, 1] for binary classification.
    :param weight: (batch, *) same shape as input. If specified, a weight is applied for each category.
        If used in binary classification, this is treated as positive weight.
    :param focal_gamma: focal loss gamma parameter. Default to 0 as disabled.
    :param use_kl_divergence: subtract soft-label bias from the loss
    :param eps: small value to prevent log(0) when target is 0. Default to 1e-8.
    """
    assert (
        focal_gamma == 0.0 or use_kl_divergence is False
    ), "Focal loss and KL divergence cannot be used together."

    if input.ndim > 1:
        # Cross-entropy Loss
        input = input.view(input.shape[0], -1)
        target = target.view(target.shape[0], -1)
        if weight is not None:
            target = target * weight.view(weight.shape[0], -1)

        logprobs = F.log_softmax(input, dim=1)
        if focal_gamma > 0.0:
            focal_weight = (1 - torch.exp(logprobs)) ** focal_gamma
            logprobs = logprobs * focal_weight
        batchloss = -torch.sum(target * logprobs, dim=1)

        if use_kl_divergence:
            logprobs_target = torch.log(torch.clamp(target, min=eps))
            batchloss += torch.sum(target * logprobs_target, dim=1)
    else:
        # Binary Cross-entropy Loss
        batchloss = F.binary_cross_entropy_with_logits(input, target, reduction="none", pos_weight=weight)

        if focal_gamma > 0.0:
            probs = torch.sigmoid(input)
            pt = target * probs + (1 - target) * (1 - probs)
            focal_weight = (1 - pt) ** focal_gamma
            batchloss = batchloss * focal_weight

        if use_kl_divergence:
            logprobs_target = torch.log(torch.clamp(target, min=eps))
            loginvprobs_target = torch.log(torch.clamp(1 - target, min=eps))
            batchloss += target * logprobs_target + (1 - target) * loginvprobs_target

    if reduction == "none":
        return batchloss
    elif reduction == "mean":
        return torch.mean(batchloss)
    elif reduction == "sum":
        return torch.sum(batchloss)
    else:
        raise ValueError(f"Unsupported reduction mode {reduction}.")


class ShuffleDataset(IterableDataset):
    def __init__(self, dataset: IterableDataset, buffer_size: int):
        super().__init__()
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        shufbuf = []
        try:
            dataset_iter = iter(self.dataset)
            for i in range(self.buffer_size):
                shufbuf.append(next(dataset_iter))
        except StopIteration:
            self.buffer_size = len(shufbuf)

        try:
            while True:
                try:
                    item = next(dataset_iter)
                    evict_idx = random.randint(0, self.buffer_size - 1)
                    yield shufbuf[evict_idx]
                    shufbuf[evict_idx] = item
                except StopIteration:
                    break
            while len(shufbuf) > 0:
                yield shufbuf.pop()
        except GeneratorExit:
            pass


class BatchByBoardSizeDataset(IterableDataset):
    def __init__(self, dataset, batch_size):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        boardsize_to_databuf = {}
        try:
            dataset_iter = iter(self.dataset)
            while True:
                try:
                    data = next(dataset_iter)
                    board_size = tuple(data["board_size"])
                    if board_size not in boardsize_to_databuf:
                        boardsize_to_databuf[board_size] = []
                    databuf = boardsize_to_databuf[board_size]
                    databuf.append(data)

                    assert len(databuf) <= self.batch_size
                    if len(databuf) == self.batch_size:
                        while len(databuf) > 0:
                            yield databuf.pop()
                except StopIteration:
                    break  # discard last incomplete batch for all board size
        except GeneratorExit:
            pass


def state_dict_drop_size_unmatched(model: torch.nn.Module, loaded_state_dict: dict) -> dict:
    """
    Drop key and values from loaded_state_dict that have shape
    unmatched with the current model's parameters.
    This will not drop other unmatched keys.
    """
    current_model_dict = model.state_dict()
    new_state_dict = {}
    for k, v in loaded_state_dict.items():
        if k not in current_model_dict or current_model_dict[k].size() == v.size():
            new_state_dict[k] = v
    return new_state_dict
