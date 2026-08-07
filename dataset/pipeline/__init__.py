from torch.utils.data.dataset import Dataset, IterableDataset
from utils.misc_utils import Registry, import_submodules
from abc import ABC, abstractmethod


class BasePipeline(ABC):
    """The base class for all dataset pipeline."""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, *args, **kwargs):
        return self.process(*args, **kwargs)

    pipeline_id = ""
    schema_version = 1
    input_fields = ()
    output_fields = ()

    def signature_state(self) -> dict:
        raise TypeError(f"opaque pipeline {type(self).__name__} has no signature_state")

    @abstractmethod
    def process(self, data: dict) -> dict:
        """Process a data entry."""
        raise NotImplementedError()


PIPELINES = Registry("pipeline")
import_submodules(__name__, recursive=False)


class DatasetPipelineWrapper(Dataset):
    def __init__(self, dataset, pipelines) -> None:
        super().__init__()
        self.dataset = dataset
        self.pipelines = pipelines

    @property
    def is_fixed_side_input(self):
        return self.dataset.is_fixed_side_input

    @property
    def capabilities(self):
        return getattr(self.dataset, "capabilities", None)

    @property
    def runtime_context(self):
        return getattr(self.dataset, "runtime_context", None)

    @property
    def yields_batches(self):
        return bool(
            getattr(self.dataset, "yields_batches", getattr(self.dataset, "YIELDS_BATCHES", False))
        )

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        data = self.dataset.__getitem__(index)
        for pipeline in self.pipelines:
            data = pipeline(data)
        return data

    def map_record_ref(self, index):
        return self.dataset.map_record_ref(index)


class IterativePipelineWrapper(IterableDataset):
    def __init__(self, dataset, pipelines) -> None:
        super().__init__()
        self.dataset = dataset
        self.pipelines = pipelines

    @property
    def is_fixed_side_input(self):
        return self.dataset.is_fixed_side_input

    @property
    def is_internal_shuffleable(self):
        return self.dataset.is_internal_shuffleable

    @property
    def capabilities(self):
        return getattr(self.dataset, "capabilities", None)

    @property
    def runtime_context(self):
        return getattr(self.dataset, "runtime_context", None)

    @property
    def yields_batches(self):
        return bool(
            getattr(self.dataset, "yields_batches", getattr(self.dataset, "YIELDS_BATCHES", False))
        )

    @property
    def YIELDS_BATCHES(self):
        return self.yields_batches

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


def build_data_pipeline(pipeline_args) -> list[BasePipeline]:
    pipelines = []
    for pipeline_type, pipeline_kwargs in pipeline_args.items():
        if pipeline_type not in PIPELINES:
            raise ValueError(f"unknown dataset pipeline {pipeline_type!r}")
        pipelines.append(PIPELINES[pipeline_type](**pipeline_kwargs))
    return pipelines


def warp_dataset_with_pipeline(dataset, pipeline_args):
    if isinstance(dataset, IterableDataset):
        return IterativePipelineWrapper(dataset, build_data_pipeline(pipeline_args))
    elif isinstance(dataset, Dataset):
        return DatasetPipelineWrapper(dataset, build_data_pipeline(pipeline_args))
    else:
        raise ValueError(f"Unsupported dataset type, {type(dataset)}")
