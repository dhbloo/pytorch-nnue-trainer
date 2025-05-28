import numpy as np
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
        sx, sy = boardsize
        if symmetry_type == "default":
            if sx == sy:
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
            return [Symmetry.IDENTITY, Symmetry.ROTATE_90, Symmetry.ROTATE_180, Symmetry.ROTATE_270]
        elif symmetry_type == "flip":
            return [Symmetry.IDENTITY, Symmetry.FLIP_X, Symmetry.FLIP_Y, Symmetry.FLIP_XY, Symmetry.FLIP_YX]
        elif symmetry_type == "flip_diag_rotate180":
            return [Symmetry.IDENTITY, Symmetry.ROTATE_180, Symmetry.FLIP_XY, Symmetry.FLIP_YX]
        else:
            raise ValueError(f"unsupported symmetry_type: {symmetry_type}")

    def apply_to_move(self, move: Move, boardsize: tuple[int, int]) -> Move:
        """Apply symmetry transformation to a move (x, y)"""
        if move.is_pass:
            return move
        assert self in Symmetry.available_symmetries(boardsize)
        sx, sy = boardsize
        sx, sy = sx - 1, sy - 1
        mapping_list = [
            lambda x, y: (x, y),
            lambda x, y: (y, sy - x),
            lambda x, y: (sx - x, sy - y),
            lambda x, y: (sx - y, x),
            lambda x, y: (x, sy - y),
            lambda x, y: (sx - x, y),
            lambda x, y: (y, x),
            lambda x, y: (sx - y, sy - x),
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


def even_select_range(N, M):
    """
    Selects M elements from N elements evenly.

    Returns:
        A range of N booleans indicating whether to select one element.
    """
    assert 0 < M < N

    if M > N / 2:
        q, r = divmod(N, N - M)
        j = 0
        for i in range(N):
            if q * j + min(j, r) == i:
                j = min(j + 1, N - M - 1)
                yield False
            else:
                yield True
    else:
        q, r = divmod(N, M)
        j = 0
        for i in range(N):
            if q * j + min(j, r) == i:
                j = min(j + 1, M - 1)
                yield True
            else:
                yield False


def make_subset_range(length, partition_num, partition_idx, shuffle=False, sample_rate=1.0):
    """Make a subset range from a partition of randomly (shuffled) sampled range(length)."""
    # divide indices according to worker id and worker num
    assert 0 <= partition_idx < partition_num
    indices = np.arange(partition_idx, length, partition_num)
    length = len(indices)

    # randomly shuffle indices
    if shuffle:
        np.random.shuffle(indices)

    # select a subset of indices according to sample rate
    assert 0.0 <= sample_rate <= 1.0
    if sample_rate == 1.0:
        for index in indices:
            yield index
    elif sample_rate > 0:
        for index in indices:
            if np.random.random() < sample_rate:
                yield index


def post_process_data(
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
    if fixed_side_input and data["stm_input"] > 0:
        # Flip side when stm is white
        data["board_input"] = np.flip(data["board_input"], axis=0)
        value_target = data["value_target"]
        value_target[0], value_target[1] = value_target[1], value_target[0]

    if symmetry_type:
        if symmetry_type == True:
            symmetry_type = "default"  # Default symmetry type
        symmetries = Symmetry.available_symmetries(data["board_size"], symmetry_type)
        if symmetry_index is None:
            symmetry_index = np.random.randint(len(symmetries))
        picked_symmetry = symmetries[symmetry_index]

        # Apply symmetry to the board_input, policy_target
        data["board_input"] = picked_symmetry.apply_to_array(data["board_input"])
        if data["policy_target"].ndim == 1:
            assert (
                data["policy_target"].shape[0] == np.prod(data["board_size"]) + 1
            ), f"Invalid policy target shape ({data['policy_target'].shape}) for symmetry, must be (H*W+1)"
            policy_target_sym = data["policy_target"][:-1].reshape((-1, *data["board_size"]))
            policy_target_sym = picked_symmetry.apply_to_array(policy_target_sym)
            policy_target_sym = policy_target_sym.reshape((policy_target_sym.shape[0], -1))
            data["policy_target"] = np.concatenate([policy_target_sym, data["policy_target"][-1:]])
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

        if data["policy_target"].ndim == 1:
            assert (
                data["policy_target"].shape[0] == np.prod(data["board_size"]) + 1
            ), f"Invalid policy target shape ({data['policy_target'].shape}) for symmetry, must be (H*W+1)"
            policy_target_board = data["policy_target"][:-1].reshape((-1, *data["board_size"]))
            policy_target_board = np.pad(
                policy_target_board,
                ((0, padded_h - board_h), (0, padded_w - board_w)),
                mode="constant",
                constant_values=0,
            )
            policy_target_board = policy_target_board.reshape((policy_target_board.shape[0], -1))
            data["policy_target"] = np.concatenate([policy_target_board, data["policy_target"][-1:]])
        else:
            data["policy_target"] = np.pad(
                data["policy_target"],
                ((0, padded_h - board_h), (0, padded_w - board_w)),
                mode="constant",
                constant_values=0,
            )

        assert data["board_input"].shape == (board_channels, padded_h, padded_w)
        assert data["policy_target"].shape == (padded_h, padded_w) or data["policy_target"].shape == (
            padded_h * padded_w + 1,
        )

    if drop_extra:
        keys_to_preserve = ["board_size", "board_input", "stm_input", "value_target", "policy_target"]
        data = {k: data[k] for k in keys_to_preserve}

    if "position" in data:
        transformed_position = data.pop("position")
        data["position_string"] = "".join([str(m) for m in transformed_position])
        data["last_move"] = transformed_position[-1].pos

    return data


def filter_data_by_condition(condition: str, data_dict: dict) -> int:
    """
    Filter the data dict (inplace) by the condition expression.
    Assume the data dict contains numpy arrays with the same length at the first dim.
    Returns: The filtered length of the data dict.
    """
    try:
        condition = eval(condition, {"np": np, **data_dict})
    except Exception as e:
        raise ValueError(f"Invalid custom condition: {condition}, error: {e}")

    selected_indices = np.nonzero(condition)[0]
    for k in data_dict.keys():
        data_dict[k] = data_dict[k][selected_indices, ...]

    return len(next(iter(data_dict.values())))
