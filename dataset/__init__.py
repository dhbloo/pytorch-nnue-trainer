import random
from torch.utils.data.dataset import Dataset, IterableDataset
from utils.file_utils import make_file_list
from utils.misc_utils import Registry, import_submodules
from .pipeline import warp_dataset_with_pipeline

DATASETS = Registry("dataset")
import_submodules(__name__, recursive=False)


def _read_multi_dataset(file_list: list[str], dataset_dict: dict[str, dict], **kwargs):
    datasets = []
    for dataset_name, dataset_args in dataset_dict.items():
        dataset_type = dataset_args.pop("dataset_type")
        assert dataset_type in DATASETS, f"Invalid dataset type in {dataset_name}: {dataset_type}"
        dataset_cls = DATASETS[dataset_type]

        data_paths = dataset_args.pop("data_paths", None)
        if data_paths is None:
            data_paths = file_list
        elif isinstance(data_paths, str):
            data_paths = [data_paths]
        else:
            assert isinstance(data_paths, list), f"data_paths in {dataset_name} must be list of str"
        flist = make_file_list(data_paths, dataset_cls.FILE_EXTS)
        dataset = dataset_cls(file_list=flist, **kwargs, **dataset_args)
        datasets.append(dataset)
    return datasets


@DATASETS.register("multi")
class MultiDataset(Dataset):
    """MultiDataset combines all (random accessable) datasets into one dataset."""

    def __init__(self, file_list: list[str], dataset_dict: dict[str, dict], **kwargs) -> None:
        super().__init__()
        self.fixed_side_input = kwargs.get("fixed_side_input", False)
        self.datasets = _read_multi_dataset(file_list, dataset_dict, **kwargs)
        # check if all datasets have __len__ and __getitem__ method
        assert all(
            hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__") for dataset in self.datasets
        ), "All datasets must be random accessable datasets"

    @property
    def is_fixed_side_input(self):
        return self.fixed_side_input

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, index):
        for dataset in self.datasets:
            if index < len(dataset):
                return dataset[index]
            index -= len(dataset)
        raise IndexError("Index out of range")


@DATASETS.register("iterative_multi")
class MultiIterativeDataset(IterableDataset):
    """MultiDataset combines all iterative datasets into one dataset."""

    def __init__(
        self, file_list: list[str], dataset_dict: dict[str, dict], sync_length=True, **kwargs
    ) -> None:
        super().__init__()
        self.fixed_side_input = kwargs.get("fixed_side_input", False)
        self.sync_length = sync_length
        self.datasets = _read_multi_dataset(file_list, dataset_dict, **kwargs)
        # check if all datasets have __iter__ method
        assert all(
            hasattr(dataset, "__iter__") for dataset in self.datasets
        ), "All datasets must be iterative datasets"

    @property
    def is_fixed_side_input(self):
        return self.fixed_side_input

    @property
    def is_internal_shuffleable(self):
        return all(map(lambda d: d.is_internal_shuffleable, self.datasets))

    def __iter__(self):
        dataset_iters = [iter(ds) for ds in self.datasets]
        sync_length_flag = False
        while len(dataset_iters) > 0:
            # randomly select a dataset to get one data entry
            idx = random.randint(0, len(dataset_iters) - 1)
            dataset_iter = dataset_iters[idx]

            try:
                data = next(dataset_iter)
                yield data
                if sync_length_flag:
                    del dataset_iters[idx]  # remove the unfinished dataset if sync_length_flag is True
            except StopIteration:
                del dataset_iters[idx]  # remove the exhausted dataset
                if self.sync_length:  # only take common length part
                    sync_length_flag = True


def build_dataset(
    dataset_type,
    data_paths,
    rules=None,
    boardsizes=None,
    boardsize=None,
    fixed_side_input=False,
    shuffle=False,
    pipeline_args=None,
    **kwargs,
) -> Dataset | IterableDataset:
    assert dataset_type in DATASETS, f"Unknown dataset type: {dataset_type}"
    dataset_cls = DATASETS[dataset_type]

    if dataset_cls == MultiDataset or dataset_cls == MultiIterativeDataset:
        file_list = data_paths
    else:
        file_list = make_file_list(data_paths, dataset_cls.FILE_EXTS)

    rules = rules or ["freestyle", "standard", "renju"]
    if boardsizes is None:
        if isinstance(boardsize, int):
            boardsizes = [(boardsize, boardsize)]
        else:
            boardsizes = [(s, s) for s in range(9, 22)]

    dataset = dataset_cls(
        file_list=file_list,
        rules=set(rules),
        boardsizes=set(boardsizes),
        fixed_side_input=fixed_side_input,
        shuffle=shuffle,
        **kwargs,
    )

    if pipeline_args is not None:
        dataset = warp_dataset_with_pipeline(dataset, pipeline_args)

    return dataset
