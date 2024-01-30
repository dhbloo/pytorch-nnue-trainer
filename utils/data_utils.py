import numpy as np
from enum import Enum


class Result(Enum):
    LOSS = 0
    DRAW = 1
    WIN = 2

    def opposite(r):
        return Result(2 - r.value)


class Rule(Enum):
    FREESTYLE = 0
    STANDARD = 1
    RENJU = 4

    def __str__(self) -> str:
        RuleName = ['freestyle', 'standard', None, None, 'renju']
        return RuleName[self.value]


class Move():
    def __init__(self, x=None, y=None, pos=None):
        assert pos is not None or (x is not None and y is not None)
        if pos:
            self.x, self.y = pos
        else:
            self.x = x
            self.y = y

    @property
    def pos(self):
        return (self.x, self.y)

    @property
    def value(self):
        return (self.x << 5) | self.y

    def __repr__(self):
        return f'({self.x},{self.y})'

    def __str__(self):
        return chr(self.x + ord("a")) + str(self.y + 1)


class Symmetry(Enum):
    IDENTITY = 0
    ROTATE_90 = 1  # (x, y) -> (y, s - x)
    ROTATE_180 = 2  # (x, y) -> (s - x, s - y)
    ROTATE_270 = 3  # (x, y) -> (s - y, x)
    FLIP_X = 4  # (x, y) -> (x, s - y)
    FLIP_Y = 5  # (x, y) -> (s - x, y)
    FLIP_XY = 6  # (x, y) -> (y, x)
    FLIP_YX = 7  # (x, y) -> (s - y, s - x)

    def available_symmetries(boardsize, symmetry_type="default"):
        sx, sy = boardsize
        if symmetry_type == "default":
            if sx == sy:
                return [
                    Symmetry.IDENTITY, Symmetry.ROTATE_90, Symmetry.ROTATE_180, Symmetry.ROTATE_270,
                    Symmetry.FLIP_X, Symmetry.FLIP_Y, Symmetry.FLIP_XY, Symmetry.FLIP_YX
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
            assert 0, f"unsupported symmetry_type: {symmetry_type}"

    def apply(self, pos, boardsize):
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
        return mapping_list[self.value](*pos)

    def apply_to_array(self, array: np.ndarray, y_dim=-2, x_dim=-1):
        op_list = [(False, False, False), (True, False, True), (True, True, False),
                   (False, True, True), (False, True, False), (True, False, False),
                   (False, False, True), (True, True, True)]
        flip_x, flip_y, swap = op_list[self.value]
        if flip_x: array = np.flip(array, axis=x_dim)
        if flip_y: array = np.flip(array, axis=y_dim)
        if swap: array = np.swapaxes(array, x_dim, y_dim)
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