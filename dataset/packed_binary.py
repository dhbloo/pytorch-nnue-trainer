import numpy as np
from torch.utils.data.dataset import IterableDataset
from enum import Enum
import lz4.frame
import ctypes
import random
from utils.data_utils import Move, Rule, Symmetry


class Result(Enum):
    LOSS = 0
    DRAW = 1
    WIN = 2

    def opposite(r):
        return Result(2 - r.value)


class Entry(ctypes.Structure):
    _fields_ = [('result', ctypes.c_uint16, 2), ('ply', ctypes.c_uint16, 9),
                ('boardsize', ctypes.c_uint16, 5), ('rule', ctypes.c_uint16, 3),
                ('move', ctypes.c_uint16, 13), ('position', ctypes.c_uint16 * 1024)]


class EntryHead(ctypes.Structure):
    _fields_ = [('result', ctypes.c_uint16, 2), ('ply', ctypes.c_uint16, 9),
                ('boardsize', ctypes.c_uint16, 5), ('rule', ctypes.c_uint16, 3),
                ('move', ctypes.c_uint16, 13)]


def write_entry(f, result: Result, boardsize: int, rule: Rule, move: Move, position: list):
    entry = Entry()
    entry.result = result.value
    entry.ply = len(position)
    entry.boardsize = boardsize
    entry.rule = rule.value
    entry.move = move.value
    for i, m in enumerate(position):
        entry.position[i] = m.value

    f.write(bytearray(entry)[:4 + 2 * len(position)])


def read_entry(f):
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


class PackedBinaryDataset(IterableDataset):
    def __init__(self,
                 file_list,
                 rules,
                 boardsizes,
                 fixed_side_input,
                 apply_symmetry=False,
                 shuffle=False,
                 sample_rate=1.0,
                 **kwargs):
        super().__init__()
        self.file_list = file_list
        self.rules = rules
        self.boardsizes = boardsizes
        self.fixed_side_input = fixed_side_input
        self.apply_symmetry = apply_symmetry
        self.shuffle = shuffle
        self.sample_rate = sample_rate

    @property
    def is_fixed_side_input(self):
        return self.fixed_side_input

    def _open_binary_file(self, filename: str):
        if filename.endswith("lz4"):
            return lz4.frame.open(filename, "rb")
        else:
            return open(filename, "rb")

    def _prepare_data_from_entry(self, result, ply, bsize, rule, move, position):
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
            'stm_input': -1 if stm_is_black else 1,  # [1] Black = -1, White = 1

            # targets
            'value_target': value_target,  # [3] (Black Win, White Win, Draw)
            'policy_target': policy_target,  # [H, W]

            # other infos
            'position_string': "".join([str(m) for m in position]),
            'last_move': position[-1].pos,
            'ply': ply,
        }

    def __iter__(self):
        file_list = [fn for fn in self.file_list]
        if self.shuffle:
            random.shuffle(file_list)
        for filename in file_list:
            with self._open_binary_file(filename) as f:
                while f.peek() != b'':
                    data = self._prepare_data_from_entry(*read_entry(f))
                    if data is not None and random.random() < self.sample_rate:
                        yield data
