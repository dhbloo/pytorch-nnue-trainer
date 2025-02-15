from torch.utils.data.dataset import Dataset, IterableDataset
from utils.file_utils import make_file_list
from utils.misc_utils import Registry, import_submodules
from .pipeline import warp_dataset_with_pipeline

DATASETS = Registry("dataset")
import_submodules(__name__, recursive=False)


@DATASETS.register("multi")
class MultiIterativeDataset(IterableDataset):
    """MultiDataset combines all iterative datasets into one dataset."""

    def __init__(self, file_list, dataset_dict, sync_length=False, **kwargs) -> None:
        super().__init__()

        self.fixed_side_input = kwargs["fixed_side_input"]
        self.sync_length = sync_length
        self.datasets = []
        for dataset_name, dataset_args in dataset_dict.items():
            dataset_type = dataset_args.pop("dataset_type")
            assert dataset_type in DATASETS, f"Invalid dataset type in {dataset_name}: {dataset_type}"
            dataset_cls = DATASETS[dataset_type]
            assert issubclass(dataset_cls, IterableDataset)

            data_paths = dataset_args.pop("data_paths", None)
            if isinstance(data_paths, str):
                data_paths = [data_paths]
            else:
                assert data_paths is None or isinstance(
                    data_paths, (list, tuple)
                ), f"Invalid data_paths in {dataset_name}, must be None or list of str"
            flist = make_file_list(data_paths or file_list, dataset_cls.FILE_EXTS)
            dataset = dataset_cls(file_list=flist, **kwargs, **dataset_args)
            self.datasets.append(dataset)

    @property
    def is_fixed_side_input(self):
        return self.fixed_side_input

    @property
    def is_internal_shuffleable(self):
        return all(map(lambda d: d.is_internal_shuffleable, self.datasets))

    def __iter__(self):
        dataset_iters = [iter(ds) for ds in self.datasets]
        while len(dataset_iters) > 0:
            finished_dataset_indices = []
            # yield one data entry from each dataset
            for idx, dataset_iter in enumerate(dataset_iters):
                try:
                    data = next(dataset_iter)
                    yield data
                except StopIteration:
                    if self.sync_length:  # only take common length part
                        return
                    finished_dataset_indices.append(idx)

            # remove all finished datasets
            for index in sorted(finished_dataset_indices, reverse=True):
                del dataset_iters[index]


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

    if dataset_cls == MultiIterativeDataset:
        file_list = data_paths
    else:
        file_list = make_file_list(data_paths, dataset_cls.FILE_EXTS)

    rules = rules or ["freestyle", "standard", "renju"]
    if boardsizes is None:
        if isinstance(boardsize, int):
            boardsizes = [(boardsize, boardsize)]
        else:
            boardsizes = [(s, s) for s in range(9, 21)]

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
