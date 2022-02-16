import random
import os
import numpy as np
import torch


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def add_dict_to(total_dict, dict_to_add):
    for k, v in dict_to_add.items():
        if k in total_dict:
            total_dict[k] += v
        else:
            total_dict[k] = v


def write_loss_dict(tb_logger, tag, loss_dict, it):
    for name, loss in loss_dict.items():
        tb_logger.add_scalar(f'{tag}/{name}', loss, it)
