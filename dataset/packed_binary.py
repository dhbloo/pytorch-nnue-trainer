import numpy as np
import lz4.frame
import ctypes
import random
import torch.utils.data
import io
from typing import Optional
from torch.utils.data.dataset import IterableDataset
from utils.data_utils import Result, Move, Rule, Symmetry, make_subset_range
from utils.winrate_model import WinrateModel
from . import DATASETS


class EntryHead(ctypes.Structure):
    _fields_ = [
        ('boardSize', ctypes.c_uint32, 5),
        ('rule', ctypes.c_uint32, 3),
        ('result', ctypes.c_uint32, 4),
        ('totalPly', ctypes.c_uint32, 10),
        ('initPly', ctypes.c_uint32, 10),
        ('gameTag', ctypes.c_uint32, 14),
        ('moveCount', ctypes.c_uint32, 18),
    ]


class EntryMove(ctypes.Structure):
    _fields_ = [
        ('isFirst', ctypes.c_uint16, 1),
        ('isLast', ctypes.c_uint16, 1),
        ('isNoEval', ctypes.c_uint16, 1),
        ('isPass', ctypes.c_uint16, 1),
        ('reserved', ctypes.c_uint16, 2),
        ('move', ctypes.c_uint16, 10),
        ('eval', ctypes.c_int16),
    ]


class MoveData():
    def __init__(self):
        self.moves = []
        self.evals = []
        self.is_ended = False

    def __getitem__(self, index) -> tuple[Optional[Move], Optional[int]]:
        """Returns none move for pass, and none eval for no eval."""
        return (self.moves[index], self.evals[index])

    def __len__(self):
        return len(self.moves)

    def _append_entry_move(self, entry_move: EntryMove):
        """
        Returns true if the move is the last move in the movelist.
        No further entry move should be appended after this.
        """
        assert len(self) > 0 or entry_move.isFirst, "Must be first move"
        assert not self.is_ended, "Cannot append after the last move"

        if entry_move.isLast:
            self.is_ended = True
        move = None
        if not entry_move.isPass:
            move = Move((entry_move.move >> 5) & 31, entry_move.move & 31)
        eval = None if entry_move.isNoEval else entry_move.eval
        self.moves.append(move)
        self.evals.append(eval)


class EntryData():
    def __init__(
        self,
        boardsize: int,
        rule: Rule,
        result: Result,
        totalply: int,
        gametag: int,
        init_position: list[Move],
    ):
        self.boardsize = boardsize
        self.rule = rule
        self.result = result
        self.totalply = totalply
        self.gametag = gametag
        self.init_position = init_position
        self.moves: list[MoveData] = []

    def _append_entry_move(self, entry_move: EntryMove):
        if len(self.moves) == 0 or self.moves[-1].is_ended:
            self.moves.append(MoveData())
        self.moves[-1]._append_entry_move(entry_move)


def read_entry(f: io.RawIOBase) -> EntryData:
    ehead = EntryHead()
    f.readinto(ehead)

    pos_array = (ctypes.c_uint16 * int(ehead.initPly))()
    f.readinto(pos_array)
    position = [Move((m >> 5) & 31, m & 31) for m in pos_array]

    entry = EntryData(int(ehead.boardSize), Rule(ehead.rule), Result(ehead.result),
                      int(ehead.totalPly), int(ehead.gameTag), position)

    for _ in range(int(ehead.moveCount)):
        emove = EntryMove()
        f.readinto(emove)
        entry._append_entry_move(emove)

    return entry


@DATASETS.register('packed_binary')
class PackedBinaryDataset(IterableDataset):
    FILE_EXTS = ['.lz4', '.binpack']

    def __init__(self,
                 file_list,
                 rules,
                 boardsizes,
                 fixed_side_input,
                 has_pass_move=False,
                 value_lambda=0.0,
                 dynamic_value_lambda=True,
                 multipv_temperature=0.03,
                 use_mate_multipv=False,
                 winrate_model=None,
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
        self.has_pass_move = has_pass_move
        self.value_lambda = value_lambda
        self.dynamic_value_lambda = dynamic_value_lambda
        self.multipv_temperature = multipv_temperature
        self.use_mate_multipv = use_mate_multipv
        self.apply_symmetry = apply_symmetry
        self.shuffle = shuffle
        self.sample_rate = sample_rate
        self.max_worker_per_file = max_worker_per_file
        self.winrate_model = winrate_model or WinrateModel()

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

    def _setup_policy_target(self, boardsize: tuple[int, int], movedata: MoveData):
        H, W = boardsize
        policy = np.zeros(H * W + (1 if self.has_pass_move else 0), dtype=np.float32)

        # single bestmove
        move, besteval = movedata[0]
        if len(movedata) == 1 or self.multipv_temperature == 0.0 \
            or any(ev is None for ev in movedata.evals) \
            or (not self.use_mate_multipv and besteval >= self.winrate_model.eval_mate_threshold):
            if move is None and self.has_pass_move:
                policy[-1] = 1.0
            else:
                policy[move.y * W + move.x] = 1.0
        # multipv
        else:
            winrates = np.array([self.winrate_model.eval_to_winrate(ev) for ev in movedata.evals],
                                dtype=np.float32)
            winrates_exp = np.exp(winrates / self.multipv_temperature)
            winrates_softmax = winrates_exp / np.sum(winrates_exp)
            for i, (move, _) in enumerate(movedata):
                if move is None:
                    policy[-1] = winrates_softmax[i]
                else:
                    policy[move.y * W + move.x] = winrates_softmax[i]

        return policy if self.has_pass_move else policy.reshape(boardsize)

    def _prepare_data(
        self,
        result: Result,
        boardsize: tuple[int, int],
        rule: Rule,
        game_stage: float,
        position: list[Optional[Move]],
        movedata: MoveData,
    ) -> dict:
        ply = len(position)
        stm_is_black = ply % 2 == 0
        bestmove, eval = movedata[0]

        # setup board input
        board_input = np.zeros((2, boardsize[0], boardsize[1]), dtype=np.int8)
        for i, m in enumerate(position):
            if m is None:  # skip pass move
                continue
            if self.fixed_side_input:
                side_idx = i % 2
            else:
                side_idx = 0 if i % 2 == ply % 2 else 1
            board_input[side_idx, m.y, m.x] = 1

        # make value target
        if self.fixed_side_input and not stm_is_black:
            result = Result.opposite(result)
        wld_result = np.array([result == Result.WIN, result == Result.LOSS, result == Result.DRAW],
                              dtype=np.float32)
        if self.value_lambda == 0:
            value_target = wld_result
        else:
            value_lambda = self.value_lambda
            if self.dynamic_value_lambda:
                value_lambda = value_lambda * (1.0 - game_stage)
            wld_eval = wld_result if eval is None else self.winrate_model.eval_to_wld(eval)
            value_target = wld_result * (1 - value_lambda) + wld_eval * value_lambda

        # make policy target
        policy_target = self._setup_policy_target(boardsize, movedata)

        # apply symmetry transformation
        if self.apply_symmetry:
            symmetries = Symmetry.available_symmetries(boardsize)
            picked_symmetry = np.random.choice(symmetries)
            board_input = picked_symmetry.apply_to_array(board_input)
            if self.has_pass_move:
                policy_board = policy_target[:-1].reshape(boardsize)
                policy_board = picked_symmetry.apply_to_array(policy_board)
                policy_pass = policy_target[-1:]
                policy_target = np.concatenate([policy_board.flatten(), policy_pass])
            else:
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
            'policy_target': policy_target,  # [H, W] or [H*W+1] (append pass at last channel)

            # other infos
            'position_string': "".join([str(m) for m in position]),
            'last_move': position[-1].pos,
            'ply': ply,
            'raw_eval': float('nan') if eval is None else float(eval),
        }

    def _process_entry(self, entry: EntryData) -> list[dict]:
        # Skip other rules and board sizes
        boardsize = (entry.boardsize, entry.boardsize)
        if str(entry.rule) not in self.rules:
            return []
        if boardsize not in self.boardsizes:
            return []

        data_list = []
        current_result = entry.result
        current_position = entry.init_position.copy()
        for moveidx, movedata in enumerate(entry.moves):
            current_move, current_eval = movedata[0]
            game_stage = min(max(moveidx / (len(entry.moves) - 1), 0.0), 1.0)

            # random skip data according to sample rate
            if random.random() < self.sample_rate:
                data = self._prepare_data(current_result, boardsize, entry.rule,
                                          game_stage, current_position, movedata)
                data_list.append(data)

            current_result = Result.opposite(current_result)
            current_position.append(current_move)

        return data_list

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
                    data_list = self._process_entry(read_entry(f))
                    for data in data_list:
                        yield data
