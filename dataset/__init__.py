from utils.file_utils import make_file_list
from utils.misc_utils import Register, import_submodules

DATASETS = Register('dataset')
import_submodules(__name__)


def build_dataset(dataset_type,
                  data_paths,
                  rules=None,
                  boardsizes=None,
                  boardsize=None,
                  fixed_side_input=False,
                  shuffle=False,
                  **kwargs):
    assert dataset_type in DATASETS
    dataset_cls = DATASETS[dataset_type]

    file_list = make_file_list(data_paths, dataset_cls.FILE_EXTS)
    rules = rules or ['freestyle', 'standard', 'renju']
    if boardsizes is None:
        if isinstance(boardsize, int):
            boardsizes = [(boardsizes, boardsizes)]
        else:
            boardsizes = [(s, s) for s in range(5, 21)]

    return dataset_cls(file_list=file_list,
                       rules=rules,
                       boardsizes=boardsizes,
                       fixed_side_input=fixed_side_input,
                       shuffle=shuffle,
                       **kwargs)
