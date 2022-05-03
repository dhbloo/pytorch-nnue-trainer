from typing import List
from torch.utils.data.dataset import Dataset, IterableDataset
from utils.misc_utils import Register, import_submodules
from abc import ABC, abstractmethod


class BasePipeline(ABC):
    """The base class for all dataset pipeline."""
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, *args, **kwargs):
        return self.process(*args, **kwargs)

    @abstractmethod
    def process(self, data: dict) -> dict:
        """Process a data entry."""
        raise NotImplementedError()


PIPELINES = Register('pipeline')
import_submodules(__name__, recursive=False)


class DatasetPipelineWrapper(Dataset):
    def __init__(self, dataset, pipelines) -> None:
        super().__init__()
        self.dataset = dataset
        self.pipelines = pipelines

    @property
    def is_fixed_side_input(self):
        return self.dataset.fixed_side_input

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        data = self.dataset.__getitem__(index)
        for pipeline in self.pipelines:
            data = pipeline(data)
        return data


class IterativePipelineWarpper(IterableDataset):
    def __init__(self, dataset, pipelines) -> None:
        super().__init__()
        self.dataset = dataset
        self.pipelines = pipelines

    @property
    def is_fixed_side_input(self):
        return self.dataset.fixed_side_input

    @property
    def is_internal_shuffleable(self):
        return self.dataset.is_internal_shuffleable

    def __iter__(self):
        dataset_iter = iter(self.dataset)
        try:
            while True:
                try:
                    data = next(dataset_iter)
                    for pipeline in self.pipelines:
                        data = pipeline(data)
                    yield data
                except StopIteration:
                    break
        except GeneratorExit:
            pass


def build_data_pipeline(pipeline_args) -> List[BasePipeline]:
    pipelines = []
    for pipeline_type, pipeline_kwargs in pipeline_args.items():
        assert pipeline_type in PIPELINES
        pipelines.append(PIPELINES[pipeline_type](**pipeline_kwargs))
    return pipelines


def warp_dataset_with_pipeline(dataset, pipeline_args):
    if isinstance(dataset, IterableDataset):
        return IterativePipelineWarpper(dataset, build_data_pipeline(pipeline_args))
    elif isinstance(dataset, Dataset):
        return DatasetPipelineWrapper(dataset, build_data_pipeline(pipeline_args))
    else:
        assert 0, "Unsupported dataset type"