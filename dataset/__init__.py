from .packed_binary import PackedBinaryDataset
from .katago import KatagoNumpyDataset, ProcessedKatagoNumpyDataset
from .sparse_numpy import SparseNumpyDataset
from utils.file_utils import make_file_list

DATASETS = {
    'packed_binary': PackedBinaryDataset,
    'katago_numpy': KatagoNumpyDataset,
    'processed_katago_numpy': ProcessedKatagoNumpyDataset,
    'sparse_numpy': SparseNumpyDataset,
}
DATASET_FILE_EXTS = {
    'packed_binary': ['lz4', 'bin'],
    'katago_numpy': 'npz',
    'processed_katago_numpy': 'npz',
    'sparse_numpy': 'npz',
}


def build_dataset(dataset_type,
                  data_paths,
                  rules=None,
                  boardsizes=None,
                  boardsize=None,
                  fixed_side_input=False,
                  shuffle=False,
                  **kwargs):
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
                                  fixed_side_input=fixed_side_input,
                                  shuffle=shuffle,
                                  **kwargs)
