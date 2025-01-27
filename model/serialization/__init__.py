from utils.misc_utils import Registry, import_submodules
from abc import ABC, abstractmethod
from io import IOBase
from datetime import datetime


class BaseSerializer(ABC):
    """The base class for all model serializer."""

    def __init__(self, rules=["freestyle"], boardsizes=[15], description=None) -> None:
        super().__init__()
        self._rules = rules
        self._boardsizes = boardsizes
        self._description = description
        if self._description is None:
            timestamp_str = datetime.now().strftime("%c")
            self._description = f"Weight exported by pytorch-nnue-trainer at {timestamp_str}."

    @property
    def is_binary(self) -> bool:
        """Whether this serializer outputs binary data."""
        return True

    @property
    def needs_header(self) -> bool:
        """Whether binary header is needed before the serializer output."""
        return self.is_binary

    def rule_mask(self, model) -> int:
        """Applicable rule for serialized weight file."""
        rule_mask = 0
        if "freestyle" in self._rules:
            rule_mask |= 1
        if "standard" in self._rules:
            rule_mask |= 2
        if "renju" in self._rules:
            rule_mask |= 4
        return rule_mask

    def boardsize_mask(self, model) -> int:
        """Applicable board size for serialized weight file."""
        boardsize_mask = 0
        for board_size in self._boardsizes:
            assert 1 <= board_size <= 32
            boardsize_mask |= 1 << (board_size - 1)
        return boardsize_mask

    def description(self, model) -> str:
        """Description of serialized weight file."""
        return f"model={model.name}; {self._description}"

    @abstractmethod
    def arch_hash(self, model) -> int:
        """A hash value for the network architecture."""
        raise NotImplementedError()

    @abstractmethod
    def serialize(self, out: IOBase, model, device):
        """Serializes a model to an output IO stream."""
        raise NotImplementedError()


SERIALIZERS = Registry("serialization")
import_submodules(__name__, recursive=False)


def build_serializer(model_type, **kwargs) -> BaseSerializer:
    assert model_type in SERIALIZERS
    return SERIALIZERS[model_type](**kwargs)
