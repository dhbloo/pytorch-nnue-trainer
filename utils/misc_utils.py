import random
import numpy as np
import torch
import logging
import importlib
import pkgutil


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def set_performance_level(level: int):
    if not (0 <= level <= 3):
        raise ValueError(f"Performance level must in [0,3], got {level}.")
    torch.backends.cudnn.deterministic = level == 0
    torch.backends.cudnn.benchmark = level >= 1
    torch.backends.cudnn.allow_tf32 = level >= 1
    torch.set_float32_matmul_precision("medium" if level >= 3 else "high" if level >= 2 else "highest")


def add_dict_to(total_dict, dict_to_add):
    for k, v in dict_to_add.items():
        if k in total_dict:
            total_dict[k] += v
        else:
            total_dict[k] = v


def log_value_dict(tb_logger, tag, value_dict, it, rows):
    for name, value in value_dict.items():
        tb_logger.add_scalar(f"{tag}/{name}", value, it)
        tb_logger.add_scalar(f"{tag}_rows/{name}", value, rows)


def deep_update_dict(base_dict: dict, new_dict: dict):
    """Recursively update base_dict with new_dict (inplace).

    :param base_dict: dict to be updated
    :param new_dict: dict to update from
    :return: updated base_dict
    """
    def recursive_update(dict1: dict, dict2: dict, key_prefix: str = ""):
        for key, value in dict2.items():
            if isinstance(value, dict) and key in base_dict:
                full_key = key_prefix + "." + key if key_prefix else key
                if not isinstance(base_dict[key], dict):
                    raise TypeError(f"Key {full_key} in base_dict is not a dict.")
                dict1[key] = recursive_update(base_dict[key], value, full_key)
            else:
                dict1[key] = value

    recursive_update(base_dict, new_dict)
    return base_dict


class Registry:
    def __init__(self, registry_name):
        self._dict = {}
        self._name = registry_name

    def __setitem__(self, key, value):
        if not callable(value):
            raise Exception(f"Value of a Registry must be a callable!\nValue: {value}")
        if key is None:
            key = value.__name__
        if key in self._dict:
            logging.warning(f"Key {key} already in registry {self._name}.")
        self._dict[key] = value

    def register(self, target):
        """Decorator to register a function or class."""

        def add(key, value):
            self[key] = value
            return value

        if callable(target):
            # @reg.register
            return add(None, target)
        # @reg.register('alias')
        return lambda x: add(target, x)

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def keys(self):
        return self._dict.keys()


def import_submodules(package, recursive=True):
    """Import all submodules of a module, recursively, including subpackages

    :param package: package (name or actual module)
    :type package: str | module
    :rtype: dict[str, types.ModuleType]
    """
    if isinstance(package, str):
        package = importlib.import_module(package)
    results = {}
    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = package.__name__ + "." + name
        results[full_name] = importlib.import_module(full_name)
        if recursive and is_pkg:
            results.update(import_submodules(full_name))
    return results


def ascii_hist(name, x, bins=10, width=60):
    N, X = np.histogram(x, bins=bins)
    nmax = N.max()

    print(name + f", min={x.min():.4f}, max={x.max():.4f}")
    for xi, n in zip(X, N):
        bar = "#" * int(n * 1.0 * width / nmax)
        xi = "{0: <8.4g}".format(xi).ljust(10)
        print("{0}| {1}".format(xi, bar))


def aligned_write(output, buffer, alignment=1):
    output.write(buffer)
    padding = (alignment - len(buffer) % alignment) % alignment
    output.write(b"\0" * padding)
