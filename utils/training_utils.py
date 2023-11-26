import math
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader


def weights_init(init_type):
    '''Generate a init function given a init type'''
    def init_fun(m):
        classname = m.__class__.__name__
        # First we check if the layer has custom init method.
        # If so, we just call it without our uniform initialization.
        if hasattr(m, 'custom_init'):
            m.custom_init()
        # Call our unifrom initialization methods for all Conv and Linear layers
        elif (classname.startswith('Conv') or classname.startswith('Linear')) \
            and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'normal':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'default':
                pass
            else:
                assert 0, f"Unsupported initialization: {init_type}"

            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

        if hasattr(m, 'post_init'):
            m.post_init()

    return init_fun


def build_optimizer(optim_type, parameters, lr, weight_decay=0.0, **kwargs):
    if optim_type == 'adamw':
        opt = optim.AdamW(parameters,
                          lr=lr,
                          betas=(kwargs.get('beta1', 0.9), kwargs.get('beta2', 0.999)),
                          eps=1e-8,
                          weight_decay=weight_decay)
    elif optim_type == 'sgd':
        opt = optim.SGD(parameters, lr=lr, momentum=0, dampening=0, weight_decay=weight_decay)
    elif optim_type == 'sgd_momentum':
        opt = optim.SGD(parameters,
                        lr=lr,
                        momentum=kwargs.get('momentum', 0.9),
                        dampening=kwargs.get('dampening', 0.1),
                        weight_decay=weight_decay)
    else:
        assert 0, f"Unsupported optimizer: {optim_type}"

    return opt


def build_lr_scheduler(optimizer, lr_schedule_type='constant', last_it=-1, **kwargs):
    if lr_schedule_type == 'constant':
        scheduler = optim.lr_scheduler.ConstantLR(optimizer,
                                                  factor=1.0,
                                                  total_iters=0,
                                                  last_epoch=last_it)
    elif lr_schedule_type == 'step':
        step_size = kwargs.get('step_size', 50000)
        step_gamma = kwargs.get('step_gamma', 0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=step_size,
                                              gamma=step_gamma,
                                              last_epoch=last_it)
    else:
        assert 0, f"Unsupported lr scheduler: {lr_schedule_type}"

    return scheduler


def build_data_loader(dataset,
                      batch_size=1,
                      shuffle=False,
                      shuffle_buffer_size=10000,
                      drop_last=True,
                      batch_by_boardsize=False,
                      **kwargs):
    if shuffle and isinstance(dataset, IterableDataset):
        # Warp non shuffleable dataset with ShuffleDataset
        if not dataset.is_internal_shuffleable:
            dataset = ShuffleDataset(dataset, shuffle_buffer_size)
        shuffle = False

    if batch_by_boardsize:
        assert isinstance(dataset, IterableDataset), \
            "batch_by_boardsize must be used with IterableDataset"
        dataset = BatchByBoardSizeDataset(dataset, batch_size)

    dataloader = DataLoader(dataset,
                            batch_size,
                            shuffle=shuffle,
                            drop_last=drop_last,
                            persistent_workers=True,
                            **kwargs)
    return dataloader


def weight_clipping(named_parameters, clip_parameters):
    named_parameters = dict(named_parameters)
    for group in clip_parameters:
        min_weight = group['min_weight']
        max_weight = group['max_weight']
        for i, param_name in enumerate(group['params']):
            p = named_parameters[param_name]
            p_data_fp32 = p.data
            if 'virtual_params' in group:
                virtual_param_name = group['virtual_params'][i]
                virtual_param = named_parameters[virtual_param_name]
                virtual_param = virtual_param.repeat(*[
                    p_data_fp32.shape[i] // virtual_param.shape[i]
                    for i in range(virtual_param.ndim)
                ])
                min_weight_t = p_data_fp32.new_full(p_data_fp32.shape, min_weight) - virtual_param
                p_data_fp32 = torch.max(p_data_fp32, min_weight_t)
                max_weight_t = p_data_fp32.new_full(p_data_fp32.shape, max_weight) - virtual_param
                p_data_fp32 = torch.min(p_data_fp32, max_weight_t)
            else:
                p_data_fp32.clamp_(min_weight, max_weight)
            p.data.copy_(p_data_fp32)


def cross_entropy_with_softlabel(input, target, reduction='mean', adjust=False, weight=None):
    """
    :param input: (batch, *)
    :param target: (batch, *) same shape as input,
        each item must be a valid distribution: target[i, :].sum() == 1.
    :param adjust: subtract soft-label bias from the loss
    :param weight: (batch, *) same shape as input, 
        if not none, a weight is specified for each loss item
    """
    input = input.view(input.shape[0], -1)
    target = target.view(target.shape[0], -1)
    if weight is not None:
        weight = weight.view(weight.shape[0], -1)

    logprobs = F.log_softmax(input, dim=1)
    if weight is not None:
        logprobs = logprobs * weight
    batchloss = -torch.sum(target * logprobs, dim=1)

    if adjust:
        eps = 1e-8
        bias = target * torch.log(target + eps)
        if weight is not None:
            bias = bias * weight
        bias = torch.sum(bias, dim=1)
        batchloss += bias

    if reduction == 'none':
        return batchloss
    elif reduction == 'mean':
        return torch.mean(batchloss)
    elif reduction == 'sum':
        return torch.sum(batchloss)
    else:
        assert 0, f'Unsupported reduction mode {reduction}.'


class ShuffleDataset(IterableDataset):
    def __init__(self, dataset, buffer_size):
        super().__init__()
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        shufbuf = []
        try:
            dataset_iter = iter(self.dataset)
            for i in range(self.buffer_size):
                shufbuf.append(next(dataset_iter))
        except:
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
                    board_size = tuple(data['board_size'])
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