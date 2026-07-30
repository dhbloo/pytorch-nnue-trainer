from utils.misc_utils import Registry, import_submodules
from abc import ABC, abstractmethod
from io import IOBase


def rules_to_mask(rules) -> int:
    """Compose the engine-facing rule bitmask (freestyle=1, standard=2, renju=4)."""
    rule_mask = 0
    if "freestyle" in rules:
        rule_mask |= 1
    if "standard" in rules:
        rule_mask |= 2
    if "renju" in rules:
        rule_mask |= 4
    return rule_mask


def boardsizes_to_mask(boardsizes) -> int:
    """Compose the engine-facing board size bitmask (bit i-1 for board size i)."""
    boardsize_mask = 0
    for board_size in boardsizes:
        assert 1 <= board_size <= 32
        boardsize_mask |= 1 << (board_size - 1)
    return boardsize_mask


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
        return rules_to_mask(self._rules)

    def boardsize_mask(self, model) -> int:
        """Applicable board size for serialized weight file."""
        return boardsizes_to_mask(self._boardsizes)

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


def has_rules_in_args(export_args: dict) -> bool:
    return any(key in export_args for key in ("rule", "rule_list", "rules"))


def has_boardsizes_in_args(export_args: dict) -> bool:
    if "min_board_size" in export_args and "max_board_size" in export_args:
        return True
    return any(key in export_args for key in ("board_size", "board_size_list", "boardsizes"))


def get_rules_from_args(export_args: dict):
    if "rule" in export_args:
        rules = [export_args.pop("rule")]
    elif "rule_list" in export_args:
        rules = export_args.pop("rule_list")
        assert isinstance(rules, list), f"rule_list must be a list of str, got {rules}"
    elif "rules" in export_args:
        rules = export_args.pop("rules")
        assert isinstance(rules, list), f"rules must be a list of str, got {rules}"
    else:
        # No permissive default: the rule mask is embedded in the exported
        # weight metadata, and silently advertising all rules would make the
        # engine accept the weight for rules it was never trained on.
        raise ValueError(
            "No rules specified for export. Set the applicable rules explicitly "
            'in export_args, e.g. --export_args "{rule: freestyle}" '
            "(or rule_list/rules for multiple rules)."
        )
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
    elif "boardsizes" in export_args:
        boardsizes = export_args.pop("boardsizes")
        assert isinstance(boardsizes, list), f"boardsizes={boardsizes}"
    else:
        print(
            "Warning: no board sizes specified for export, the exported weight "
            "will advertise support for all board sizes 1-32."
        )
        boardsizes = list(range(1, 32 + 1))
    for boardsize in boardsizes:
        if not isinstance(boardsize, int):
            raise ValueError(f"Invalid board size {boardsize}, must be int")
        if not (1 <= boardsize <= 32):
            raise ValueError(f"Invalid board size {boardsize}, must be in [1, 32]")
    return boardsizes


def build_serializer(model_type, **export_args) -> BaseSerializer:
    if model_type not in SERIALIZERS:
        raise ValueError(
            f"No serializer registered for model type '{model_type}'. "
            f"Supported types: {sorted(SERIALIZERS.keys())}"
        )
    # The export entry points in export.py require rules explicitly (the rule
    # mask is stamped into the weight metadata). Bare construction without any
    # rule/board-size keys (e.g. registry smoke tests) falls back to the
    # restrictive BaseSerializer defaults instead of a permissive mask.
    rules = get_rules_from_args(export_args) if has_rules_in_args(export_args) else ["freestyle"]
    boardsizes = (
        get_boardsizes_from_args(export_args) if has_boardsizes_in_args(export_args) else [15]
    )
    return SERIALIZERS[model_type](rules=rules, boardsizes=boardsizes, **export_args)
