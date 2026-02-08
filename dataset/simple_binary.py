import numpy as np
import lz4.frame
import ctypes
import random
import torch.utils.data
import io
from torch.utils.data.dataset import IterableDataset
from utils.data_utils import *
from . import DATASETS


class Entry(ctypes.Structure):
    _fields_ = [
        ("result", ctypes.c_uint16, 2),
        ("ply", ctypes.c_uint16, 9),
        ("boardsize", ctypes.c_uint16, 5),
        ("rule", ctypes.c_uint16, 3),
        ("move", ctypes.c_uint16, 13),
        ("position", ctypes.c_uint16 * 1024),
    ]


class EntryHead(ctypes.Structure):
    _fields_ = [
        ("result", ctypes.c_uint16, 2),
        ("ply", ctypes.c_uint16, 9),
        ("boardsize", ctypes.c_uint16, 5),
        ("rule", ctypes.c_uint16, 3),
        ("move", ctypes.c_uint16, 13),
    ]


def write_entry(
    f: io.RawIOBase,
    result: Result,
    boardsize: int,
    rule: Rule,
    move: Move,
    position: list[Move],
):
    entry = Entry()
    entry.result = result.value
    entry.ply = len(position)
    entry.boardsize = boardsize
    entry.rule = rule.value
    entry.move = move.value
    for i, m in enumerate(position):
        entry.position[i] = m.value

    f.write(bytearray(entry)[: 4 + 2 * len(position)])


def read_entry(f: io.RawIOBase) -> tuple[Result, int, int, Rule, Move, list[Move]]:
    ehead = EntryHead()
    f.readinto(ehead)

    result = Result(ehead.result)
    ply = int(ehead.ply)
    boardsize = int(ehead.boardsize)
    rule = Rule(ehead.rule)
    move = Move((ehead.move >> 5) & 31, ehead.move & 31)

    pos_array = (ctypes.c_uint16 * ehead.ply)()
    f.readinto(pos_array)
    position = [Move((m >> 5) & 31, m & 31) for m in pos_array]

    return result, ply, boardsize, rule, move, position


@DATASETS.register("simple_binary")
class SimpleBinaryDataset(IterableDataset):
    FILE_EXTS = [".lz4", ".bin"]

    def __init__(
        self,
        file_list: list[str],
        rules: set[str],
        boardsizes: set[tuple[int, int]],
        fixed_side_input: bool = False,
        fixed_board_size: None | tuple[int, int] = None,
        apply_symmetry: bool = False,
        drop_extra: bool = False,
        shuffle: bool = False,
        sample_rate: float = 1.0,
        max_partition_per_file: int = 1,
        **kwargs
    ):
        super().__init__()
        self.file_list = file_list
        self.rules = rules
        self.boardsizes = boardsizes
        self.fixed_side_input = fixed_side_input
        self.fixed_board_size = fixed_board_size
        self.apply_symmetry = apply_symmetry
        self.drop_extra = drop_extra
        self.shuffle = shuffle
        self.sample_rate = sample_rate
        self.max_partition_per_file = max_partition_per_file

    @property
    def is_fixed_side_input(self):
        return self.fixed_side_input

    @property
    def is_internal_shuffleable(self):
        return False

    def _open_binary_file(self, filename: str):
        if filename.endswith("lz4"):
            return lz4.frame.open(filename, "rb")
        else:
            return open(filename, "rb")

    def _prepare_data_from_entry(
        self,
        result: Result,
        ply: int,
        bsize: int,
        rule: Rule,
        move: Move,
        position: list[Move],
    ):
        # Skip other rules and board sizes
        boardsize = (bsize, bsize)
        if str(rule) not in self.rules:
            return None
        if boardsize not in self.boardsizes:
            return None

        stm_input = -1 if ply % 2 == 0 else 1  # (Black = -1, White = 1)
        board_input = np.zeros((2, bsize, bsize), dtype=np.int8)
        for i, m in enumerate(position):
            current_stm_input = -1 if i % 2 == 0 else 1
            side_idx = 0 if current_stm_input == stm_input else 1
            board_input[side_idx, m.y, m.x] = 1

        value_target = np.array(
            [result == Result.WIN, result == Result.LOSS, result == Result.DRAW], dtype=np.float32
        )
        policy_target = np.zeros(boardsize, dtype=np.float32)
        policy_target[move.y, move.x] = 1

        data = {
            # global info
            "board_size": np.array(boardsize, dtype=np.int8),  # H, W
            "rule_index": rule.index,
            # inputs
            "board_input": board_input,  # [C, H, W], C=(Black,White)
            "stm_input": np.array([stm_input], dtype=np.float32),  # [1] Black = -1.0, White = 1.0
            # targets
            "value_target": value_target,  # [3] (Black Win, White Win, Draw)
            "policy_target": policy_target,  # [H, W]
            # other infos
            "position": position,
            "ply": ply,
        }
        data = post_process_data(
            data,
            fixed_side_input=self.fixed_side_input,
            fixed_board_size=self.fixed_board_size,
            symmetry_type=self.apply_symmetry,
            drop_extra=self.drop_extra,
        )
        return data

    def __iter__(self):
        partition_num, partition_idx = get_partition_num_and_idx()
        partition_per_file = min(partition_num, self.max_partition_per_file)
        assert partition_num % partition_per_file == 0, \
            f"partition_num {partition_num} should be divisible by partition_per_file {partition_per_file}"

        for file_index in make_subset_range(
            len(self.file_list),
            partition_num=partition_num // partition_per_file,
            partition_idx=partition_idx // partition_per_file,
            shuffle=self.shuffle,
        ):
            filename = self.file_list[file_index]
            with self._open_binary_file(filename) as f:
                while f.peek() != b"":
                    data = self._prepare_data_from_entry(*read_entry(f))

                    # random skip data according to sample rate
                    if random.random() >= self.sample_rate:
                        continue

                    if data is not None:
                        yield data
