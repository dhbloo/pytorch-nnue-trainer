import math
import random
import torch
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader
from itertools import chain

from utils.misc_utils import Registry


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


OPTIMIZERS = Registry("optimizer")
"""Registry of optimizer factories.

Each entry is a callable ``(parameters, model_or_models, lr, weight_decay, **kwargs)
-> Optimizer``, where *parameters* is the (already filtered) parameter list and
*model_or_models* is the original model or list of models for factories that
need module structure (e.g. muon's per-parameter routing).
"""


@OPTIMIZERS.register("adamw")
def _make_adamw(parameters, model, lr, weight_decay, **kwargs):
    args = {"lr": lr, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": weight_decay}
    args.update(kwargs)
    return optim.AdamW(parameters, **args)


@OPTIMIZERS.register("adamw-ams")
def _make_adamw_ams(parameters, model, lr, weight_decay, **kwargs):
    args = {"lr": lr, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": weight_decay, "amsgrad": True}
    args.update(kwargs)
    return optim.AdamW(parameters, **args)


@OPTIMIZERS.register("sgd")
def _make_sgd(parameters, model, lr, weight_decay, **kwargs):
    args = {"lr": lr, "momentum": 0, "dampening": 0, "weight_decay": weight_decay}
    args.update(kwargs)
    return optim.SGD(parameters, **args)


@OPTIMIZERS.register("sgd-momentum")
def _make_sgd_momentum(parameters, model, lr, weight_decay, **kwargs):
    args = {"lr": lr, "momentum": 0.9, "dampening": 0.1, "nesterov": False, "weight_decay": weight_decay}
    args.update(kwargs)
    return optim.SGD(parameters, **args)


@OPTIMIZERS.register("sgd-nesterov")
def _make_sgd_nesterov(parameters, model, lr, weight_decay, **kwargs):
    # Nesterov momentum requires zero dampening
    args = {"lr": lr, "momentum": 0.9, "dampening": 0, "nesterov": True, "weight_decay": weight_decay}
    args.update(kwargs)
    return optim.SGD(parameters, **args)


@OPTIMIZERS.register("muon-adamw")
def _make_muon_adamw(parameters, model, lr, weight_decay, **kwargs):
    from utils.muon import Muon, get_params_for_muon
    from utils.chained_optimizer import ChainedOptimizer, OptimizerSpec

    models = model if isinstance(model, (list, tuple)) else [model]
    params_id_to_name = {}
    muon_params_id_set = set()
    for m in models:
        params_id_to_name.update({id(p): name for name, p in m.named_parameters()})
        muon_params_id_set.update(id(p) for p in get_params_for_muon(m))
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
    return ChainedOptimizer(parameters, specs, **kwargs)


def build_optimizer(
    optim_type: str,
    model: torch.nn.Module | list[torch.nn.Module],
    lr: float,
    weight_decay: float = 0.0,
    only_track_requires_grad=True,
    **kwargs,
):
    if optim_type not in OPTIMIZERS:
        raise ValueError(f"Unsupported optimizer: {optim_type}")

    if isinstance(model, (list, tuple)):
        parameters = chain(*(m.parameters() for m in model))
    else:
        parameters = model.parameters()
    if only_track_requires_grad:
        # only track parameters with requires_grad=True
        parameters = [p for p in parameters if p.requires_grad]

    return OPTIMIZERS[optim_type](parameters, model, lr, weight_decay, **kwargs)


def build_lr_scheduler(optimizer, lr_schedule_type, iterations, last_it=-1, **kwargs):
    if lr_schedule_type == "constant":
        scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=iterations, last_epoch=last_it)
    elif lr_schedule_type == "step":
        step_size = kwargs.get("step_size", 50000)
        step_gamma = kwargs.get("step_gamma", 0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=step_gamma, last_epoch=last_it)
    elif lr_schedule_type == "cosine":
        eta_min = kwargs.get("eta_min", 1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations, eta_min=eta_min, last_epoch=last_it)
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

    # Default to persistent workers to avoid worker restart cost between epochs
    if "persistent_workers" not in kwargs:
        kwargs["persistent_workers"] = num_workers > 0

    dataloader = DataLoader(
        dataset,
        batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        **kwargs,
    )
    return dataloader


def resolve_weight_clipping(named_parameters, clip_parameters):
    """Resolve parameter names in *clip_parameters* to parameter tensors.

    Done once at setup so the per-iteration apply does not rebuild the name
    lookup.  Returns a list of ``(min_weight, max_weight, params,
    virtual_params)`` tuples; ``virtual_params`` is ``None`` for plain clamping.
    """
    named_parameters = dict(named_parameters)
    resolved = []
    for group in clip_parameters:
        params = [named_parameters[name] for name in group["params"]]
        virtual_params = None
        if "virtual_params" in group:
            virtual_params = [named_parameters[name] for name in group["virtual_params"]]
            if len(virtual_params) != len(params):
                raise ValueError(
                    f"weight clipping group has {len(params)} params"
                    f" but {len(virtual_params)} virtual_params"
                )
        resolved.append((group["min_weight"], group["max_weight"], params, virtual_params))
    return resolved


def apply_weight_clipping(resolved_groups):
    """Clamp parameters per the groups produced by :func:`resolve_weight_clipping`."""
    for min_weight, max_weight, params, virtual_params in resolved_groups:
        if virtual_params is None:
            for p in params:
                p.data.clamp_(min_weight, max_weight)
        else:
            for p, virtual_param in zip(params, virtual_params):
                p_data = p.data
                virtual = virtual_param.repeat(
                    *[p_data.shape[i] // virtual_param.shape[i] for i in range(virtual_param.ndim)]
                )
                min_weight_t = p_data.new_full(p_data.shape, min_weight) - virtual
                p_data = torch.max(p_data, min_weight_t)
                max_weight_t = p_data.new_full(p_data.shape, max_weight) - virtual
                p_data = torch.min(p_data, max_weight_t)
                p.data.copy_(p_data)


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
