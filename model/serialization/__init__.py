from utils.misc_utils import Registry, import_submodules
from abc import ABC, abstractmethod
from io import IOBase


class BaseSerializer(ABC):
    """The base class for all model serializer."""

    def __init__(self, rules=["freestyle"], boardsizes=[15], description=None) -> None:
        super().__init__()
        self._rules = rules
        self._boardsizes = boardsizes
        self._description = description or ""

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


def get_rules_from_args(export_args: dict):
    if "rule" in export_args:
        rules = [export_args.pop("rule")]
    elif "rule_list" in export_args:
        rules = export_args.pop("rule_list")
        assert isinstance(rules, list), f"rule_list must be a list of str, got {rules}"
    else:
        rules = export_args.pop("rules", ["freestyle", "standard", "renju"])
    if len(rules) == 0:
        raise ValueError("No supported rules specified")
    for rule in rules:
        if rule not in ["freestyle", "standard", "renju"]:
            raise ValueError(f"Invalid rule {rule}, must be in [freestyle, standard, renju]")
    return rules


def get_boardsizes_from_args(export_args: dict):
    if "board_size" in export_args:
        boardsizes = [export_args.pop("board_size")]
    elif "min_board_size" in export_args and "max_board_size" in export_args:
        min_board_size = export_args.pop("min_board_size")
        max_board_size = export_args.pop("max_board_size")
        boardsizes = list(range(min_board_size, max_board_size + 1))
    elif "board_size_list" in export_args:
        boardsizes = export_args.pop("board_size_list")
        assert isinstance(boardsizes, list), f"boardsizes={boardsizes}"
    else:
        boardsizes = export_args.pop("boardsizes", list(range(1, 32 + 1)))
    for boardsize in boardsizes:
        if not isinstance(boardsize, int):
            raise ValueError(f"Invalid board size {boardsize}, must be int")
        if not (1 <= boardsize <= 32):
            raise ValueError(f"Invalid board size {boardsize}, must be in [1, 32]")
    return boardsizes


def build_serializer(model_type, **export_args) -> BaseSerializer:
    assert model_type in SERIALIZERS
    rules = get_rules_from_args(export_args)
    boardsizes = get_boardsizes_from_args(export_args)
    return SERIALIZERS[model_type](rules=rules, boardsizes=boardsizes, **export_args)
