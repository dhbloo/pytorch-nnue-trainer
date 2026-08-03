import numpy as np
import torch
import hashlib
from enum import Enum


class Result(Enum):
    LOSS = 0
    DRAW = 1
    WIN = 2

    def opposite(self) -> "Result":
        return Result(2 - self.value)


class Rule(Enum):
    FREESTYLE = 0
    STANDARD = 1
    RENJU = 4

    @property
    def index(self) -> int:
        Indices = [0, 1, None, None, 2]
        return Indices[self.value]

    def __str__(self) -> str:
        RuleName = ["freestyle", "standard", None, None, "renju"]
        return RuleName[self.value]

    @staticmethod
    def from_str(rule_str: str) -> "Rule":
        RuleStr = {"freestyle": Rule.FREESTYLE, "standard": Rule.STANDARD, "renju": Rule.RENJU}
        return RuleStr[rule_str]

    @staticmethod
    def from_index(rule_idx: int) -> "Rule":
        RuleIndex = [Rule.FREESTYLE, Rule.STANDARD, Rule.RENJU]
        return RuleIndex[rule_idx]


class Move:
    PASS: "Move"

    def __init__(self, x: int, y: int):
        self.x, self.y = x, y

    @property
    def is_pass(self):
        return self.x < 0 and self.y < 0

    @property
    def pos(self):
        return np.array([self.x, self.y])

    @property
    def value(self):
        return (self.x << 5) | self.y if not self.is_pass else -1

    def __repr__(self):
        return f"({self.x},{self.y})" if not self.is_pass else "(pass)"

    def __str__(self):
        return chr(self.x + ord("a")) + str(self.y + 1) if not self.is_pass else "pass"


Move.PASS = Move(-1, -1)


class Symmetry(Enum):
    IDENTITY = 0
    ROTATE_90 = 1  # (x, y) -> (y, s - x)
    ROTATE_180 = 2  # (x, y) -> (s - x, s - y)
    ROTATE_270 = 3  # (x, y) -> (s - y, x)
    FLIP_X = 4  # (x, y) -> (x, s - y)
    FLIP_Y = 5  # (x, y) -> (s - x, y)
    FLIP_XY = 6  # (x, y) -> (y, x)
    FLIP_YX = 7  # (x, y) -> (s - y, s - x)

    @staticmethod
    def available_symmetries(boardsize: tuple[int, int], symmetry_type="default") -> list["Symmetry"]:
        height, width = (int(v) for v in boardsize)
        if symmetry_type == "default":
            if height == width:
                return [
                    Symmetry.IDENTITY,
                    Symmetry.ROTATE_90,
                    Symmetry.ROTATE_180,
                    Symmetry.ROTATE_270,
                    Symmetry.FLIP_X,
                    Symmetry.FLIP_Y,
                    Symmetry.FLIP_XY,
                    Symmetry.FLIP_YX,
                ]
            else:
                return [Symmetry.IDENTITY, Symmetry.ROTATE_180, Symmetry.FLIP_X, Symmetry.FLIP_Y]
        elif symmetry_type == "rotate":
            if height != width:
                raise ValueError(
                    f"rotate symmetry changes shape for non-square board {(height, width)}"
                )
            return [Symmetry.IDENTITY, Symmetry.ROTATE_90, Symmetry.ROTATE_180, Symmetry.ROTATE_270]
        elif symmetry_type == "flip":
            if height != width:
                raise ValueError(
                    f"flip symmetry includes diagonal transforms for non-square board "
                    f"{(height, width)}"
                )
            return [Symmetry.IDENTITY, Symmetry.FLIP_X, Symmetry.FLIP_Y, Symmetry.FLIP_XY, Symmetry.FLIP_YX]
        elif symmetry_type == "flip_diag_rotate180":
            if height != width:
                raise ValueError(
                    f"diagonal symmetry changes shape for non-square board {(height, width)}"
                )
            return [Symmetry.IDENTITY, Symmetry.ROTATE_180, Symmetry.FLIP_XY, Symmetry.FLIP_YX]
        else:
            raise ValueError(f"unsupported symmetry_type: {symmetry_type}")

    def apply_to_move(self, move: Move, boardsize: tuple[int, int]) -> Move:
        """Apply symmetry transformation to a move (x, y)"""
        if move.is_pass:
            return move
        allowed = Symmetry.available_symmetries(boardsize)
        if self not in allowed:
            raise ValueError(
                f"symmetry {self.name} changes shape for non-square board {tuple(boardsize)}"
            )
        height, width = (int(v) for v in boardsize)
        if not (0 <= move.x < width and 0 <= move.y < height):
            raise ValueError(f"move {move} is outside board {(height, width)}")
        mapping_list = [
            lambda x, y: (x, y),
            lambda x, y: (y, width - 1 - x),
            lambda x, y: (width - 1 - x, height - 1 - y),
            lambda x, y: (height - 1 - y, x),
            lambda x, y: (x, height - 1 - y),
            lambda x, y: (width - 1 - x, y),
            lambda x, y: (y, x),
            lambda x, y: (height - 1 - y, width - 1 - x),
        ]
        new_x, new_y = mapping_list[self.value](move.x, move.y)
        return Move(x=new_x, y=new_y)

    def apply_to_array(self, array: np.ndarray, y_dim=-2, x_dim=-1) -> np.ndarray:
        """Apply a copy of symmetry transformation to an array of shape (..., y, x)"""
        op_list = [
            (False, False, False),
            (True, False, True),
            (True, True, False),
            (False, True, True),
            (False, True, False),
            (True, False, False),
            (False, False, True),
            (True, True, True),
        ]
        flip_x, flip_y, swap = op_list[self.value]
        if flip_x:
            array = np.flip(array, axis=x_dim)
        if flip_y:
            array = np.flip(array, axis=y_dim)
        if swap:
            array = np.swapaxes(array, x_dim, y_dim)
        return array.copy()



class SamplePostProcessor:
    """Own all field-aware sample and batch transformations."""

    def process_sample(
        self,
        data: dict,
        fixed_side_input=False,
        fixed_board_size=None,
        symmetry_type=None,
        symmetry_index=None,
        drop_extra=False,
    ) -> dict:
        """
        Apply post processing to the data dict that contains some numpy arrays.
        Keys to be processed:
            board_input: int8 ndarray of shape (C, H, W).
            value_target: float ndarray of shape (3), win-loss-draw probability.
            policy_target: int8 ndarray of shape (H, W) or (H*W+1) with pass move.
            position: a list of Move objects. (optional)
        Other keys are kept as they are, and some may be used to help processing:
            board_size: int8 ndarray of shape (2), height and width.
            stm_input: float ndarray of shape (1), -1.0 for black and 1.0 for white.

        Args:
            data: A dict containing numpy arrays.
            fixed_side_input: Whether to fix the side of the input, so that the
                first channel is always black and the second channel is always white.
            fixed_board_size: The fixed board size to use. If None, the size of input plane
                will be the same as the board size. Otherwise, the input plane will be padded
                to the fixed board size.
            symmetry_type: The type of symmetry to apply to the data. False for no symmetry.
            symmetry_index: The index of the symmetry to apply to the data. None for random.
            drop_extra: Drop extra data except for the core ndarray.
        """
        from dataset.core import validate_field_dict

        data = dict(data)
        validate_field_dict(data, batched=False)

        if fixed_side_input and data["stm_input"] > 0:
            # Flip side when stm is white. Both arrays may be views into a dataset's
            # storage, so take copies: never mutate the stored win/loss values, and
            # keep board_input contiguous for default_collate (np.flip returns a
            # negative-stride view that torch.as_tensor rejects).
            data["board_input"] = np.ascontiguousarray(np.flip(data["board_input"], axis=0))
            value_target = data["value_target"]
            perm = np.arange(len(value_target))
            perm[[0, 1]] = [1, 0]  # swap win/loss, keep any further channels
            data["value_target"] = value_target[perm]
            if "sparse_feature_input" in data:
                data["sparse_feature_input"] = np.take(
                    data["sparse_feature_input"],
                    indices=[4, 5, 6, 7, 0, 1, 2, 3, 9, 8, 11, 10],
                    axis=0,
                )

        if symmetry_type:
            if symmetry_type == True:
                symmetry_type = "default"  # Default symmetry type
            symmetries = Symmetry.available_symmetries(data["board_size"], symmetry_type)
            if symmetry_index is None:
                from dataset.core import uniform_below

                digest = hashlib.sha256()
                digest.update(b"NNUE-sample-transform-fallback-v1\0")
                for key in (
                    "board_size",
                    "board_input",
                    "stm_input",
                    "value_target",
                    "policy_target",
                ):
                    value = np.ascontiguousarray(np.asarray(data[key]))
                    digest.update(key.encode("ascii") + b"\0")
                    digest.update(value.dtype.str.encode("ascii") + b"\0")
                    digest.update(repr(value.shape).encode("ascii") + b"\0")
                    digest.update(value.tobytes())
                symmetry_index, _ = uniform_below(
                    len(symmetries),
                    0,
                    "sample_symmetry_fallback",
                    (digest.digest(),),
                )
            picked_symmetry = symmetries[symmetry_index]

            # Apply symmetry to the board_input, policy_target
            data["board_input"] = picked_symmetry.apply_to_array(data["board_input"])
            if "sparse_feature_input" in data:
                data["sparse_feature_input"] = picked_symmetry.apply_to_array(
                    data["sparse_feature_input"]
                )
            if data["policy_target"].ndim == 1:
                if (
                    data["policy_target"].shape[0]
                    != np.prod(data["board_size"]) + 1
                ):
                    raise ValueError(
                        "flattened policy target does not match board_size"
                    )
                policy_target_sym = data["policy_target"][:-1].reshape(tuple(data["board_size"]))
                policy_target_sym = picked_symmetry.apply_to_array(policy_target_sym)
                data["policy_target"] = np.concatenate(
                    [policy_target_sym.reshape(-1), data["policy_target"][-1:]]
                )
            else:
                data["policy_target"] = picked_symmetry.apply_to_array(data["policy_target"])

            # Apply symmetry to the optional position
            if "position" in data:
                data["position"] = [
                    picked_symmetry.apply_to_move(m, data["board_size"]) for m in data["position"]
                ]

        if fixed_board_size is not None:
            padded_h, padded_w = fixed_board_size
            board_channels, board_h, board_w = data["board_input"].shape
            data["board_input"] = np.pad(
                data["board_input"],
                ((0, 0), (0, padded_h - board_h), (0, padded_w - board_w)),
                mode="constant",
                constant_values=0,
            )
            if "sparse_feature_input" in data:
                sparse = data["sparse_feature_input"]
                data["sparse_feature_input"] = np.pad(
                    sparse,
                    ((0, 0), (0, padded_h - sparse.shape[-2]), (0, padded_w - sparse.shape[-1])),
                    mode="constant",
                    constant_values=0,
                )

            if data["policy_target"].ndim == 1:
                if (
                    data["policy_target"].shape[0]
                    != np.prod(data["board_size"]) + 1
                ):
                    raise ValueError(
                        "flattened policy target does not match board_size"
                    )
                policy_target_board = data["policy_target"][:-1].reshape((board_h, board_w))
                policy_target_board = np.pad(
                    policy_target_board,
                    ((0, padded_h - board_h), (0, padded_w - board_w)),
                    mode="constant",
                    constant_values=0,
                )
                data["policy_target"] = np.concatenate(
                    [policy_target_board.reshape(-1), data["policy_target"][-1:]]
                )
            else:
                data["policy_target"] = np.pad(
                    data["policy_target"],
                    ((0, padded_h - board_h), (0, padded_w - board_w)),
                    mode="constant",
                    constant_values=0,
                )

            if data["board_input"].shape != (
                board_channels,
                padded_h,
                padded_w,
            ):
                raise RuntimeError("board padding produced an invalid shape")
            if data["policy_target"].shape not in {
                (padded_h, padded_w),
                (padded_h * padded_w + 1,),
            }:
                raise RuntimeError("policy padding produced an invalid shape")

        if drop_extra:
            keys_to_preserve = ["board_size", "board_input", "stm_input", "value_target", "policy_target"]
            data = {k: data[k] for k in keys_to_preserve}

        if "position" in data:
            transformed_position = data.pop("position")
            data["position_string"] = "".join([str(m) for m in transformed_position])
            data["last_move"] = (
                transformed_position[-1].pos
                if transformed_position
                else np.array([-1, -1], dtype=np.int64)
            )

        return data
    def process_batch(
        self,
        data,
        fixed_side_input=False,
        symmetry_type=None,
        symmetry_indices=None,
    ):
        from dataset.core import FIELD_SPECS, validate_field_dict

        validate_field_dict(data, batched=True)
        if not fixed_side_input and not symmetry_type:
            return data

        batch_size = len(data["board_input"])
        if symmetry_indices is not None and len(symmetry_indices) != batch_size:
            raise ValueError(
                "symmetry_indices must contain one entry per batch row"
            )
        samples = []
        for index in range(batch_size):
            sample = {}
            for key, value in data.items():
                spec = FIELD_SPECS.get(key)
                sample[key] = (
                    value
                    if spec.scope == "batch_shared"
                    else value[index]
                )
            samples.append(
                self.process_sample(
                    sample,
                    fixed_side_input=fixed_side_input,
                    symmetry_type=symmetry_type,
                    symmetry_index=(
                        None
                        if symmetry_indices is None
                        else int(symmetry_indices[index])
                    ),
                )
            )
        result = {}
        for key in samples[0]:
            spec = FIELD_SPECS[key]
            values = [sample[key] for sample in samples]
            if spec.scope == "batch_shared":
                result[key] = values[0]
            elif all(isinstance(value, torch.Tensor) for value in values):
                result[key] = torch.stack(values, dim=0)
            elif all(
                isinstance(
                    value,
                    (np.ndarray, np.number, int, float, bool),
                )
                for value in values
            ):
                result[key] = np.stack(
                    [
                        np.ascontiguousarray(np.asarray(value))
                        for value in values
                    ],
                    axis=0,
                )
            else:
                result[key] = values
        return result


SAMPLE_POST_PROCESSOR = SamplePostProcessor()


def post_process_data(
    data: dict,
    fixed_side_input=False,
    fixed_board_size=None,
    symmetry_type=None,
    symmetry_index=None,
    drop_extra=False,
) -> dict:
    return SAMPLE_POST_PROCESSOR.process_sample(
        data,
        fixed_side_input=fixed_side_input,
        fixed_board_size=fixed_board_size,
        symmetry_type=symmetry_type,
        symmetry_index=symmetry_index,
        drop_extra=drop_extra,
    )


def post_process_batch(
    data: dict,
    fixed_side_input=False,
    symmetry_type=None,
    symmetry_indices=None,
) -> dict:
    return SAMPLE_POST_PROCESSOR.process_batch(
        data,
        fixed_side_input=fixed_side_input,
        symmetry_type=symmetry_type,
        symmetry_indices=symmetry_indices,
    )


def filter_data_by_condition(condition: str, data_dict: dict) -> int:
    """
    Filter the data dict (inplace) by the condition expression.
    Assume the data dict contains numpy arrays with the same length at the first dim.
    Returns: The filtered length of the data dict.
    """
    from dataset.filter import evaluate_filter_condition

    condition = evaluate_filter_condition(condition, data_dict)

    selected_indices = np.nonzero(condition)[0]
    for k in data_dict.keys():
        data_dict[k] = data_dict[k][selected_indices, ...]

    return len(next(iter(data_dict.values())))
