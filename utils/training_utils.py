import math
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader


def weights_init(init_type):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
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
                      **kwargs):
    if shuffle and isinstance(dataset, IterableDataset):
        dataset = ShuffleDataset(dataset, shuffle_buffer_size)
        dataloader = DataLoader(dataset, batch_size, drop_last=drop_last, **kwargs)
    else:
        dataloader = DataLoader(dataset,
                                batch_size,
                                shuffle=shuffle,
                                drop_last=drop_last,
                                **kwargs)
    return dataloader


def weight_clipping(parameters):
    for group in parameters:
        for p in group['params']:
            p_data_fp32 = p.data
            min_weight = group['min_weight']
            max_weight = group['max_weight']
            p_data_fp32.clamp_(min_weight, max_weight)
            p.data.copy_(p_data_fp32)


def cross_entropy_with_softlabel(input, target, reduction='mean', adjust=False):
    """
    :param input: (batch, *)
    :param target: (batch, *) same shape as input,
        each item must be a valid distribution: target[i, :].sum() == 1.
    :param adjust: subtract soft-label bias from the loss
    """
    input = input.view(input.shape[0], -1)
    target = target.view(target.shape[0], -1)

    logprobs = F.log_softmax(input, dim=1)
    batchloss = -torch.sum(target * logprobs, dim=1)

    if adjust:
        eps = 1e-8
        bias = torch.sum(target * torch.log(target + eps), dim=1)
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
