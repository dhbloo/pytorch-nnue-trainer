import os
import torch
from torch import Tensor


def make_dir(dirpath: str | list[str]):
    """Create dirs after first checking their existence."""

    def _mkdir(path: str):
        if os.path.exists(path):
            return
        print(f"Create folder {path}")
        os.makedirs(path, exist_ok=True)

    if isinstance(dirpath, list):
        for path in dirpath:
            _mkdir(path)
    else:
        _mkdir(dirpath)


def make_file_list(paths: list[str], exts: list[str] | None = None) -> list[str]:
    """
    Make a file list from a list of data paths.
    Args:
        paths (list[str]): A list of file/directory paths to find the files.
        exts (list[str]): A list of file extensions. If none, no filter will be applied.
    Returns:
        A list of file paths.
    """
    file_lists = []
    for path in paths:
        if os.path.isfile(path):
            file_lists.append(path)
        elif os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    ext = os.path.splitext(file)[1]
                    if exts is None or ext in exts:
                        file_lists.append(os.path.join(root, file))
        else:
            raise IOError(f'"{path}" is not a file or directory')
    return file_lists


def load_torch_ckpt(
    filepath: str, map_location="cpu"
) -> tuple[dict[str, Tensor], dict[str, dict[str, Tensor]], dict[str, str]]:
    """
    Load model state (as well as the optional optimizer state and the metadata) from the file path.
    Args:
        filepath: The file path to load the model state.
        map_location: The map location to load the model state. Default is "cpu".
    Returns:
        A tuple of (model_state_dict, extra_state_dicts, metadata).
        model_state_dict: The model state dict.
        extra_state_dicts: The extra state dict. This may contain optimizer/scheduler/scalar states.
        metadata: The metadata dict.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file {filepath} does not exist")

    try:
        ckpt_state_dict = torch.load(filepath, map_location=map_location, weights_only=True)
    except:
        ckpt_state_dict = torch.load(filepath, map_location=map_location, weights_only=False)

    model_state_dict = ckpt_state_dict.pop("model")
    extra_state_dicts = {k: v for k, v in ckpt_state_dict.items() if isinstance(v, dict)}

    metadata = {}
    for k, v in ckpt_state_dict.items():
        if k in extra_state_dicts:
            continue
        if isinstance(v, str):
            metadata[k] = v
        else:
            # To load older checkpoints, we allow the metadata to be any other types,
            # as long as it can be converted into a str type.
            try:
                metadata[k] = str(v)
            except:
                pass

    return model_state_dict, extra_state_dicts, metadata


def save_torch_ckpt(
    filepath: str,
    model_state_dict: dict[str, Tensor],
    extra_state_dicts: dict[str, dict[str, Tensor]] = {},
    metadata: dict[str, str] | None = None,
):
    """
    Save model state (as well as the optional optimizer state and the metadata) to the file path.
    Args:
        filepath: The file path to save the model state.
        model_state_dict: The model state dict to save.
        extra_state_dicts: The extra state dict to save. Prefix will be added to each key of the
            state dict. This can be used to save optimizer/scheduler/scalar states.
        metadata: The metadata to save. If None, no metadata will be saved.
    """
    make_dir(os.path.dirname(filepath))

    state_dict = {}
    state_dict["model"] = model_state_dict
    for prefix, sd in extra_state_dicts.items():
        if not isinstance(sd, dict):
            raise TypeError(f"Extra state dict {prefix} must be a dict, got {type(sd)}")
        state_dict[prefix] = sd

    if metadata is not None:
        for key, value in metadata.items():
            if not isinstance(value, str):
                raise TypeError(f"Metadata value {key} must be a string, got {type(value)}")
            state_dict[key] = value

    # Add file name extension if not given
    if not os.path.splitext(filepath)[1]:
        filepath += ".pt"
    torch.save(state_dict, filepath)


def find_latest_ckpt(
    dirname: str, prefix: str, exts: list[str] = [".pt", ".pth", ".sft", ".safetensors"]
) -> str | None:
    """
    Find latest checkpoint (with largest postfix) in a directory with prefix and exts.
    Returns:
        Filepath of the latest checkpoint. None if not exists.
    """
    if not os.path.exists(dirname):
        return None

    ckpt_list = []
    extensions = tuple(exts)
    for fn in os.listdir(dirname):
        filepath = os.path.join(dirname, fn)
        if os.path.isfile(filepath) and fn.startswith(prefix) and fn.endswith(extensions):
            ckpt_list.append(filepath)

    if not ckpt_list:
        return None

    ckpt_list.sort()
    return ckpt_list[-1]


def get_iteration_from_ckpt_filename(model_name: str) -> int | None:
    """
    Get iteration number from model name.
    Returns:
        The iteration number. None if failed to parse.
    """
    model_name_with_ext = os.path.basename(model_name)
    model_name = os.path.splitext(model_name_with_ext)[0]

    parts = model_name.split("_")
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    return None
