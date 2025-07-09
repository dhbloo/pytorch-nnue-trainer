import numpy as np
from torch.utils.data.dataset import Dataset, IterableDataset
from utils.file_utils import make_file_list
from utils.misc_utils import Registry, import_submodules
from .pipeline import warp_dataset_with_pipeline

DATASETS = Registry("dataset")
import_submodules(__name__, recursive=False)


def _read_multi_dataset(file_list: list[str], dataset_dict: dict[str, dict], **kwargs):
    datasets = []
    blend_ratios = []
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

        blend_ratio = float(dataset_args.pop("blend_ratio", 1.0))
        blend_ratios.append(blend_ratio)

        flist = make_file_list(data_paths, dataset_cls.FILE_EXTS)
        dataset = dataset_cls(file_list=flist, **kwargs, **dataset_args)
        datasets.append(dataset)
    
    return datasets, blend_ratios


@DATASETS.register("multi")
class MultiDataset(Dataset):
    """MultiDataset combines all (random accessable) datasets into one dataset."""

    def __init__(self, file_list: list[str], dataset_dict: dict[str, dict], **kwargs) -> None:
        super().__init__()
        self.fixed_side_input = kwargs.get("fixed_side_input", False)
        self.datasets, _ = _read_multi_dataset(file_list, dataset_dict, **kwargs)
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
        self.datasets, self.blend_ratios = _read_multi_dataset(file_list, dataset_dict, **kwargs)
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
        def normalize_array(arr):
            arr_total = sum(arr)
            assert arr_total > 0.0, f"Sum of the array should be positive!"
            return [x / arr_total for x in arr]

        dataset_iters = [iter(ds) for ds in self.datasets]
        blend_ratios = normalize_array(self.blend_ratios)
        sync_length_flag = False

        def remove_at(idx):
            nonlocal dataset_iters, blend_ratios
            del dataset_iters[idx], blend_ratios[idx]
            if len(blend_ratios) > 0:
                blend_ratios = normalize_array(blend_ratios)  # re-normalize blend ratios

        while len(dataset_iters) > 0:
            # randomly select a dataset to get one data entry
            idx = np.random.choice(len(dataset_iters), p=blend_ratios)
            dataset_iter = dataset_iters[idx]

            try:
                data = next(dataset_iter)
                yield data
                if sync_length_flag:
                    remove_at(idx)  # remove the unfinished dataset if sync_length_flag is True
            except StopIteration:
                remove_at(idx)  # remove the exhausted dataset
                if self.sync_length:  # only take common length part
                    sync_length_flag = True


def build_dataset(
    dataset_type: str,
    data_paths: list[str],
    rules: None | list[str] = None,
    boardsizes: None | int | tuple[int, int] | list[tuple[int, int]] = None,
    fixed_side_input: bool = False,
    fixed_board_size: None | int | tuple[int, int] = None,
    shuffle: bool=False,
    pipeline_args: None | dict = None,
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
        boardsizes = [(s, s) for s in range(9, 22)]
    elif isinstance(boardsizes, int):
        boardsizes = [(boardsizes, boardsizes)]
    elif isinstance(boardsizes, tuple):
        boardsizes = [boardsizes]
    assert isinstance(boardsizes, list), f"boardsizes must be list of tuples, but got {boardsizes}"

    if isinstance(fixed_board_size, int):
        fixed_board_size = (fixed_board_size, fixed_board_size)
    assert fixed_board_size is None or isinstance(fixed_board_size, tuple), \
        f"fixed_board_size must be None or tuple, but got {fixed_board_size}"

    dataset = dataset_cls(
        file_list=file_list,
        rules=set(rules),
        boardsizes=set(boardsizes),
        fixed_side_input=fixed_side_input,
        fixed_board_size=fixed_board_size,
        shuffle=shuffle,
        **kwargs,
    )

    if pipeline_args is not None:
        dataset = warp_dataset_with_pipeline(dataset, pipeline_args)

    return dataset
