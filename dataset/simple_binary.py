import numpy as np
import lz4.frame
import ctypes
import random
import torch.utils.data
import io
from torch.utils.data.dataset import IterableDataset
from utils.data_utils import Result, Move, Rule, Symmetry, make_subset_range
from . import DATASETS


class Entry(ctypes.Structure):
    _fields_ = [('result', ctypes.c_uint16, 2), ('ply', ctypes.c_uint16, 9),
                ('boardsize', ctypes.c_uint16, 5), ('rule', ctypes.c_uint16, 3),
                ('move', ctypes.c_uint16, 13), ('position', ctypes.c_uint16 * 1024)]


class EntryHead(ctypes.Structure):
    _fields_ = [('result', ctypes.c_uint16, 2), ('ply', ctypes.c_uint16, 9),
                ('boardsize', ctypes.c_uint16, 5), ('rule', ctypes.c_uint16, 3),
                ('move', ctypes.c_uint16, 13)]


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

    f.write(bytearray(entry)[:4 + 2 * len(position)])


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


@DATASETS.register('simple_binary')
class SimpleBinaryDataset(IterableDataset):
    FILE_EXTS = ['.lz4', '.bin']

    def __init__(self,
                 file_list,
                 rules,
                 boardsizes,
                 fixed_side_input,
                 apply_symmetry=False,
                 shuffle=False,
                 sample_rate=1.0,
                 max_worker_per_file=2,
                 **kwargs):
        super().__init__()
        self.file_list = file_list
        self.rules = rules
        self.boardsizes = boardsizes
        self.fixed_side_input = fixed_side_input
        self.apply_symmetry = apply_symmetry
        self.shuffle = shuffle
        self.sample_rate = sample_rate
        self.max_worker_per_file = max_worker_per_file

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

        stm_is_black = ply % 2 == 0

        board_input = np.zeros((2, bsize, bsize), dtype=np.int8)
        for i, m in enumerate(position):
            if self.fixed_side_input:
                side_idx = i % 2
            else:
                side_idx = 0 if i % 2 == ply % 2 else 1
            board_input[side_idx, m.y, m.x] = 1

        if self.fixed_side_input and not stm_is_black:
            result = Result.opposite(result)
        value_target = np.array(
            [result == Result.WIN, result == Result.LOSS, result == Result.DRAW], dtype=np.int8)

        policy_target = np.zeros(boardsize, dtype=np.int8)
        policy_target[move.y, move.x] = 1

        if self.apply_symmetry:
            symmetries = Symmetry.available_symmetries(boardsize)
            picked_symmetry = np.random.choice(symmetries)
            board_input = picked_symmetry.apply_to_array(board_input)
            policy_target = picked_symmetry.apply_to_array(policy_target)
            position = [Move(*picked_symmetry.apply(m.pos, boardsize)) for m in position]

        return {
            # global info
            'board_size': np.array(boardsize, dtype=np.int8),  # H, W
            'rule': str(rule),

            # inputs
            'board_input': board_input,  # [C, H, W], C=(Black,White)
            'stm_input': -1.0 if stm_is_black else 1.0,  # [1] Black = -1.0, White = 1.0

            # targets
            'value_target': value_target,  # [3] (Black Win, White Win, Draw)
            'policy_target': policy_target,  # [H, W]

            # other infos
            'position_string': "".join([str(m) for m in position]),
            'last_move': position[-1].pos,
            'ply': ply,
        }

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_num = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        worker_per_file = min(worker_num, self.max_worker_per_file)
        assert worker_num % worker_per_file == 0

        for file_index in make_subset_range(len(self.file_list),
                                            partition_num=worker_num // worker_per_file,
                                            partition_idx=worker_id // worker_per_file,
                                            shuffle=self.shuffle):
            filename = self.file_list[file_index]
            with self._open_binary_file(filename) as f:
                while f.peek() != b'':
                    data = self._prepare_data_from_entry(*read_entry(f))

                    # random skip data according to sample rate
                    if random.random() >= self.sample_rate:
                        continue

                    if data is not None:
                        yield data
