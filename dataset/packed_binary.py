import numpy as np
import lz4.frame
import ctypes
import random
import torch.utils.data
import io
from torch.utils.data.dataset import IterableDataset
from utils.data_utils import Result, Move, Rule, Symmetry, make_subset_range, post_process_data
from utils.winrate_model import WinrateModel
from . import DATASETS


class EntryHead(ctypes.Structure):
    _fields_ = [
        ("boardSize", ctypes.c_uint32, 5),
        ("rule", ctypes.c_uint32, 3),
        ("result", ctypes.c_uint32, 4),
        ("totalPly", ctypes.c_uint32, 10),
        ("initPly", ctypes.c_uint32, 10),
        ("gameTag", ctypes.c_uint32, 14),
        ("moveCount", ctypes.c_uint32, 18),
    ]


class EntryMove(ctypes.Structure):
    _fields_ = [
        ("isFirst", ctypes.c_uint16, 1),
        ("isLast", ctypes.c_uint16, 1),
        ("isNoEval", ctypes.c_uint16, 1),
        ("isPass", ctypes.c_uint16, 1),
        ("reserved", ctypes.c_uint16, 2),
        ("move", ctypes.c_uint16, 10),
        ("eval", ctypes.c_int16),
    ]


class MoveData:
    def __init__(self):
        self.moves = []
        self.evals = []
        self.is_ended = False

    def __getitem__(self, index) -> tuple[Move, int | None]:
        """Returns the pair of (move, eval). Eval is None if not available."""
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
        if entry_move.isPass:
            move = Move.PASS
        else:
            move = Move((entry_move.move >> 5) & 31, entry_move.move & 31)
        eval = None if entry_move.isNoEval else entry_move.eval
        self.moves.append(move)
        self.evals.append(eval)


class EntryData:
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

    entry = EntryData(
        int(ehead.boardSize),
        Rule(ehead.rule),
        Result(ehead.result),
        int(ehead.totalPly),
        int(ehead.gameTag),
        position,
    )

    for _ in range(int(ehead.moveCount)):
        emove = EntryMove()
        f.readinto(emove)
        entry._append_entry_move(emove)

    return entry


@DATASETS.register("packed_binary")
class PackedBinaryDataset(IterableDataset):
    FILE_EXTS = [".lz4", ".binpack"]

    def __init__(
        self,
        file_list: list[str],
        rules: set[str],
        boardsizes: set[tuple[int, int]],
        fixed_side_input: bool = False,
        has_pass_move: bool = False,
        apply_symmetry: bool = False,
        shuffle: bool = False,
        sample_rate: float = 1.0,
        max_worker_per_file: int = 2,
        value_td_lambda: float = 0.0,
        dynamic_value_lambda: bool = True,
        multipv_temperature: float = 0.05,
        use_mate_multipv: bool = False,
        winrate_model_args: dict = {},
        **kwargs
    ):
        """
        Args:
            value_td_lambda: The weight of the soft target in the value target.
                0.0 for pure hard target, 1.0 for pure soft target
            dynamic_value_lambda: Decay value_td_lambda to zero as game stage increases
            multipv_temperature: The temperature for the multipv softmax
            use_mate_multipv: Whether to use mate score for multipv softmax
        """
        super().__init__()
        self.file_list = file_list
        self.rules = rules
        self.boardsizes = boardsizes
        self.fixed_side_input = fixed_side_input
        self.has_pass_move = has_pass_move
        self.value_td_lambda = value_td_lambda
        self.dynamic_value_lambda = dynamic_value_lambda
        self.multipv_temperature = multipv_temperature
        self.use_mate_multipv = use_mate_multipv
        self.apply_symmetry = apply_symmetry
        self.shuffle = shuffle
        self.sample_rate = sample_rate
        self.max_worker_per_file = max_worker_per_file
        self.winrate_model = WinrateModel(**winrate_model_args)

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

    def _setup_value_target(
        self, result: Result, bestmove_eval: int | None, game_stage: float
    ) -> np.ndarray:
        wld_result = np.array(
            [result == Result.WIN, result == Result.LOSS, result == Result.DRAW], dtype=np.float32
        )
        if self.value_td_lambda == 0 or bestmove_eval is None:
            return wld_result
        else:
            wld_eval = self.winrate_model.eval_to_wld(bestmove_eval)
            td_lambda = self.value_td_lambda
            if self.dynamic_value_lambda:
                td_lambda *= 1.0 - game_stage
            return wld_result * (1 - td_lambda) + wld_eval * td_lambda

    def _setup_policy_target(self, boardsize: tuple[int, int], movedata: MoveData):
        H, W = boardsize
        policy = np.zeros(H * W + (1 if self.has_pass_move else 0), dtype=np.float32)

        # single bestmove
        move, besteval = movedata[0]
        if (
            len(movedata) == 1
            or self.multipv_temperature == 0.0
            or any(ev is None for ev in movedata.evals)
            or (
                not self.use_mate_multipv
                and besteval is not None
                and besteval >= self.winrate_model.eval_mate_threshold
            )
        ):
            if move.is_pass:
                if self.has_pass_move:
                    policy[-1] = 1.0
            else:
                policy[move.y * W + move.x] = 1.0
        # multipv
        else:
            winrates = np.array(
                [self.winrate_model.eval_to_winrate(ev) for ev in movedata.evals], dtype=np.float32
            )
            winrates_shifted = winrates - np.max(winrates)
            winrates_exp = np.exp(winrates_shifted / self.multipv_temperature)
            winrates_softmax = winrates_exp / np.sum(winrates_exp)
            for i, move in enumerate(movedata.moves):
                if move.is_pass:
                    if self.has_pass_move:
                        policy[-1] = winrates_softmax[i]
                else:
                    policy[move.y * W + move.x] = winrates_softmax[i]

        return policy if self.has_pass_move else policy.reshape(boardsize)

    def _process_entry(self, entry: EntryData) -> list[dict]:
        # Skip other rules and board sizes
        boardsize = (entry.boardsize, entry.boardsize)
        if str(entry.rule) not in self.rules:
            return []
        if boardsize not in self.boardsizes:
            return []

        current_result = entry.result
        current_position = entry.init_position.copy()
        current_ply = len(current_position)
        current_stm_input = -1 if current_ply % 2 == 0 else 1  # (Black = -1, White = 1)

        # setup inital board input
        current_board_input = np.zeros((2, boardsize[0], boardsize[1]), dtype=np.int8)
        for move in current_position:
            if not move.is_pass:
                current_board_input[max(current_stm_input, 0), move.y, move.x] = 1
            current_stm_input = -current_stm_input

        data_list = []
        for moveidx, movedata in enumerate(entry.moves):
            bestmove, bestmove_eval = movedata[0]

            # random skip data according to sample rate
            if random.random() < self.sample_rate:
                game_stage = min(max(moveidx / (len(entry.moves) - 1), 0.0), 1.0)
                value_target = self._setup_value_target(current_result, bestmove_eval, game_stage)
                policy_target = self._setup_policy_target(boardsize, movedata)
                if current_stm_input > 0:
                    board_input = current_board_input[[1, 0]].copy()  # flip stm
                else:
                    board_input = current_board_input.copy()
                data = {
                    # global info
                    "board_size": np.array(boardsize, dtype=np.int8),  # (2,) for H, W
                    "rule_index": entry.rule.index,
                    # inputs
                    "board_input": board_input,  # [C, H, W], C=2 (Black,White)
                    "stm_input": float(current_stm_input),  # [1] Black = -1, White = 1
                    # targets
                    "value_target": value_target,  # [3] (Black Win, White Win, Draw)
                    "policy_target": policy_target,  # [H, W] or [H*W+1] (append pass at last channel)
                    # other infos
                    "position": current_position,
                    "ply": current_ply,
                    "raw_eval": float("nan") if bestmove_eval is None else float(bestmove_eval),
                }
                data = post_process_data(data, self.fixed_side_input, self.apply_symmetry)
                transformed_position = data.pop("position")
                data["position_string"] = "".join([str(m) for m in transformed_position])
                data["last_move"] = transformed_position[-1].pos
                data_list.append(data)

            current_result = Result.opposite(current_result)
            current_position.append(bestmove)
            current_ply += 1
            if not bestmove.is_pass:
                current_board_input[max(current_stm_input, 0), bestmove.y, bestmove.x] = 1
            current_stm_input = -current_stm_input

        return data_list

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_num = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        worker_per_file = min(worker_num, self.max_worker_per_file)
        assert worker_num % worker_per_file == 0

        for file_index in make_subset_range(
            len(self.file_list),
            partition_num=worker_num // worker_per_file,
            partition_idx=worker_id // worker_per_file,
            shuffle=self.shuffle,
        ):
            filename = self.file_list[file_index]
            with self._open_binary_file(filename) as f:
                while f.peek() != b"":
                    data_list = self._process_entry(read_entry(f))
                    for data in data_list:
                        yield data
