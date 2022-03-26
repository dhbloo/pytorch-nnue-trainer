from utils.misc_utils import Register, import_submodules
from abc import ABC, abstractmethod
from io import IOBase


class BaseSerializer(ABC):
    @property
    def is_binary(self):
        """Whether this serializer outputs binary data."""
        return True

    @abstractmethod
    def serialize(self, out: IOBase, model, device):
        """Serialize a model to an output IO stream."""
        raise NotImplementedError()


SERIALIZERS = Register('serialization')
import_submodules(__name__, recursive=False)


def build_serializer(model_type, **kwargs) -> BaseSerializer:
    assert model_type in SERIALIZERS
    return SERIALIZERS[model_type](**kwargs)
