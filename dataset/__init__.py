from .packed_binary import PackedBinaryDataset
from .katago import KatagoNumpyDataset, ProcessedKatagoNumpyDataset
import os

DATASETS = {
    'packed_binary': PackedBinaryDataset,
    'katago_numpy': KatagoNumpyDataset,
    'processed_katago_numpy': ProcessedKatagoNumpyDataset,
}
DATASET_FILE_EXTS = {
    'packed_binary': ['lz4', 'bin'],
    'katago_numpy': 'npz',
    'processed_katago_numpy': 'npz',
}


def make_file_list(data_paths, file_exts=None):
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


def build_dataset(dataset_type, data_paths, rules=None, boardsizes=None, boardsize=None, **kwargs):
    assert dataset_type in DATASETS

    file_list = make_file_list(data_paths, DATASET_FILE_EXTS[dataset_type])
    rules = rules or ['freestyle', 'standard', 'renju']
    if boardsizes is None:
        if isinstance(boardsize, int):
            boardsizes = [(boardsizes, boardsizes)]
        else:
            boardsizes = [(s, s) for s in range(5, 21)]

    return DATASETS[dataset_type](file_list=file_list,
                                  rules=rules,
                                  boardsizes=boardsizes,
                                  **kwargs)
