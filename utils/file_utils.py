import os


def make_file_list(data_paths, file_exts=None):
    """
    Make a file list from a list of data paths.

    Args:
        data_path (list[str]): A list of data paths.
        file_exts (list[str]): A list of file extensions. If none, no filter will be apply.
    """
    file_lists = []
    for path in data_paths:
        if os.path.isfile(path):
            file_lists.append(path)
        elif os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    ext = os.path.splitext(file)[1][1:]
                    if file_exts is None or ext in file_exts:
                        file_lists.append(os.path.join(root, file))
        else:
            raise IOError(f'"{path}" is not a file or directory')
    return file_lists


def ensure_dir(path):
    """Create path by first checking its existence."""
    if not os.path.exists(path):
        print("Create folder ", path)
        os.makedirs(path)
    else:
        print(path, " already exists.")


def ensure_dirs(paths):
    """Create paths by first checking their existence."""
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            ensure_dir(path)
    else:
        ensure_dir(paths)


def find_latest_model_file(dirname, prefix, ext='.pth'):
    """
    Find latest file (with largest postfix) in a directory with prefix and ext.

    Returns:
        The latest file name. None if not exists.
    """
    if not os.path.exists(dirname):
        return None

    model_list = [os.path.join(dirname, f) for f in os.listdir(dirname)
                  if os.path.isfile(os.path.join(dirname, f))
                  and f.startswith(prefix)
                  and f.endswith(ext)]
    if len(model_list) == 0:
        return None

    model_list.sort()
    return model_list[-1]
