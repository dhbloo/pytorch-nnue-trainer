import os


def make_file_list(data_paths: list[str], file_exts: list[str] | None = None) -> list[str]:
    """
    Make a file list from a list of data paths.
    Args:
        data_path (list[str]): A list of data paths.
        file_exts (list[str]): A list of file extensions. If none, no filter will be apply.
    Returns:
        A list of file paths.
    """
    file_lists = []
    for path in data_paths:
        if os.path.isfile(path):
            file_lists.append(path)
        elif os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    ext = os.path.splitext(file)[1]
                    if file_exts is None or ext in file_exts:
                        file_lists.append(os.path.join(root, file))
        else:
            raise IOError(f'"{path}" is not a file or directory')
    return file_lists


def ensure_dir(path: str):
    """Create path by first checking its existence."""
    if not os.path.exists(path):
        print("Create folder ", path)
        os.makedirs(path)
    else:
        print(path, " already exists.")


def ensure_dirs(paths: list[str] | str):
    """Create paths by first checking their existence."""
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            ensure_dir(path)
    else:
        ensure_dir(paths)


def find_latest_model_file(dirname: str, prefix: str, ext: str = ".pth") -> str | None:
    """
    Find latest file (with largest postfix) in a directory with prefix and ext.
    Returns:
        The latest file name. None if not exists.
    """
    if not os.path.exists(dirname):
        return None

    model_list = [
        os.path.join(dirname, f)
        for f in os.listdir(dirname)
        if os.path.isfile(os.path.join(dirname, f)) and f.startswith(prefix) and f.endswith(ext)
    ]
    if len(model_list) == 0:
        return None

    model_list.sort()
    return model_list[-1]


def get_iteration_from_model_filename(model_name: str | None) -> int | None:
    """
    Get iteration number from model name.
    Returns:
        The iteration number. None if not exists.
    """
    if model_name is None:
        return None

    model_name = os.path.basename(model_name)
    model_name = os.path.splitext(model_name)[0]

    parts = model_name.split("_")
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    return None
