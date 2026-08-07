import numpy as np
import lz4.frame
import ctypes
import copy
import io
from utils.data_utils import *
from utils.winrate_model import WinrateModel
from . import DATASETS
from .core import PipelineStateComposer, uniform_below
from .planner import DatasetPlanner, PlannerConfig
from .sequential_source import InterleavedSequentialSource
from .source_dataset import PlannedBatchDataset, SourceBatchDataset


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
        if len(self) == 0 and not entry_move.isFirst:
            raise RuntimeError("packed move group must begin with isFirst")
        if self.is_ended:
            raise RuntimeError("cannot append after the last packed move")

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


def _readinto_exact(f: io.RawIOBase, buf) -> None:
    """Fill *buf* completely from *f*, raising EOFError on a short read."""
    size = ctypes.sizeof(buf)
    read = f.readinto(buf)
    if read != size:
        raise EOFError(f"expected {size} bytes, got {read}")


def read_entry(f: io.RawIOBase) -> EntryData:
    ehead = EntryHead()
    _readinto_exact(f, ehead)

    pos_array = (ctypes.c_uint16 * int(ehead.initPly))()
    _readinto_exact(f, pos_array)
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
        _readinto_exact(f, emove)
        entry._append_entry_move(emove)

    return entry


def _read_exact_bytes(f: io.RawIOBase, size: int) -> bytes:
    parts = []
    remaining = size
    while remaining:
        part = f.read(remaining)
        if not part:
            break
        parts.append(part)
        remaining -= len(part)
    if remaining:
        raise EOFError(f"expected {size} bytes, got {size - remaining}")
    return b"".join(parts)


def read_raw_entry(f: io.RawIOBase) -> bytes:
    """Read one packed game without constructing its move graph."""
    header_size = ctypes.sizeof(EntryHead)
    header = _read_exact_bytes(f, header_size)
    entry_head = EntryHead.from_buffer_copy(header)
    payload_size = (
        2 * int(entry_head.initPly)
        + ctypes.sizeof(EntryMove) * int(entry_head.moveCount)
    )
    return header + _read_exact_bytes(f, payload_size)


def decode_raw_entry(raw_entry: bytes) -> EntryData:
    stream = io.BytesIO(raw_entry)
    entry = read_entry(stream)
    if stream.tell() != len(raw_entry):
        raise RuntimeError("packed-binary payload contains trailing entry bytes")
    return entry


def raw_entry_metadata(raw_entry: bytes) -> tuple[int, str, int]:
    header_size = ctypes.sizeof(EntryHead)
    if len(raw_entry) < header_size:
        raise RuntimeError("packed-binary payload is missing its header")
    entry_head = EntryHead.from_buffer_copy(raw_entry)
    expected_size = (
        header_size
        + 2 * int(entry_head.initPly)
        + ctypes.sizeof(EntryMove) * int(entry_head.moveCount)
    )
    if len(raw_entry) != expected_size:
        raise RuntimeError("packed-binary payload size disagrees with its header")
    move_offset = header_size + 2 * int(entry_head.initPly)
    group_count = 0
    group_ended = True
    for move_index in range(int(entry_head.moveCount)):
        offset = move_offset + move_index * ctypes.sizeof(EntryMove)
        entry_move = EntryMove.from_buffer_copy(raw_entry, offset)
        if group_ended:
            if not entry_move.isFirst:
                raise RuntimeError("packed move group must begin with isFirst")
            group_count += 1
            group_ended = False
        if entry_move.isLast:
            group_ended = True
    return int(entry_head.boardSize), str(Rule(entry_head.rule)), group_count


@DATASETS.register("packed_binary")
class PackedBinaryDataset(PlannedBatchDataset):
    FILE_EXTS = [".lz4", ".binpack"]

    def __init__(
        self,
        file_list: list[str],
        rules: set[str],
        boardsizes: set[tuple[int, int]],
        fixed_side_input: bool = False,
        fixed_board_size: None | tuple[int, int] = None,
        has_pass_move: bool = False,
        apply_symmetry: bool = False,
        drop_extra: bool = False,
        shuffle: bool = False,
        sample_rate: float = 1.0,
        value_td_lambda: float = 0.0,
        dynamic_value_lambda: bool = True,
        multipv_temperature: float = 0.05,
        use_mate_multipv: bool = False,
        winrate_model_args: dict | None = None,
        batch_size: int | None = None,
        batch_pipelines=(),
        shuffle_window_size: int = 32768,
        shuffle_buffer_bytes: int | None = 256 * 1024 * 1024,
        sequential_active_streams: int = 2,
        sequential_read_quantum: int = 256,
        steps_per_epoch: int | None = None,
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
        self.batch_pipelines = tuple(batch_pipelines)
        self.file_list = file_list
        self.rules = rules
        self.boardsizes = boardsizes
        self.fixed_side_input = fixed_side_input
        self.fixed_board_size = fixed_board_size
        self.has_pass_move = has_pass_move
        self.value_td_lambda = value_td_lambda
        self.dynamic_value_lambda = dynamic_value_lambda
        self.multipv_temperature = multipv_temperature
        self.use_mate_multipv = use_mate_multipv
        self.apply_symmetry = apply_symmetry
        self.drop_extra = drop_extra
        self.shuffle = shuffle
        self.sample_rate = sample_rate
        self.shuffle_window_size = shuffle_window_size
        self.shuffle_buffer_bytes = shuffle_buffer_bytes
        self.sequential_active_streams = sequential_active_streams
        self.sequential_read_quantum = sequential_read_quantum
        self.steps_per_epoch = steps_per_epoch
        self.winrate_model_args = dict(winrate_model_args or {})
        self.winrate_model = WinrateModel(**self.winrate_model_args)
    @property
    def capabilities(self):
        from .core import DatasetCapabilities

        return DatasetCapabilities(
            True,
            not any(path.lower().endswith(".lz4") for path in self.file_list),
            True,
            True,
        )

    def _build_partitioned_stream(self):
        runtime_context = getattr(self, "runtime_context", None)
        if runtime_context is None:
            raise RuntimeError("packed_binary requires a DatasetRuntimeContext")

        reader = PackedBinaryDataset(
            file_list=[],
            rules=self.rules,
            boardsizes=self.boardsizes,
            fixed_side_input=self.fixed_side_input,
            fixed_board_size=self.fixed_board_size,
            has_pass_move=self.has_pass_move,
            apply_symmetry=False,
            drop_extra=self.drop_extra,
            shuffle=False,
            sample_rate=1.0,
            value_td_lambda=self.value_td_lambda,
            dynamic_value_lambda=self.dynamic_value_lambda,
            multipv_temperature=self.multipv_temperature,
            use_mate_multipv=self.use_mate_multipv,
            winrate_model_args=self.winrate_model_args,
        )
        semantic_state = {
            "rules": sorted(self.rules),
            "boardsizes": sorted(self.boardsizes),
            "fixed_side_input": self.fixed_side_input,
            "fixed_board_size": self.fixed_board_size,
            "drop_extra": self.drop_extra,
            "has_pass_move": self.has_pass_move,
            "value_td_lambda": self.value_td_lambda,
            "dynamic_value_lambda": self.dynamic_value_lambda,
            "multipv_temperature": self.multipv_temperature,
            "use_mate_multipv": self.use_mate_multipv,
            "winrate_model_args": self.winrate_model_args,
            "apply_symmetry": self.apply_symmetry,
        }

        def shape_of(raw_entry):
            boardsize_value, rule, _ = raw_entry_metadata(raw_entry)
            boardsize = (boardsize_value, boardsize_value)
            if rule not in self.rules or boardsize not in self.boardsizes:
                return None
            return self.fixed_board_size or boardsize

        def count_subrecords(raw_entry):
            return raw_entry_metadata(raw_entry)[2]

        def materialize_entry(entry, record_key, epoch, subrecord):
            data = reader._process_subrecord(entry, subrecord)
            if data is None or not self.apply_symmetry:
                return data
            from utils.data_utils import Symmetry

            board_size = tuple(int(value) for value in np.asarray(data["board_size"]))
            kind = "default" if self.apply_symmetry is True else self.apply_symmetry
            choices = Symmetry.available_symmetries(board_size, kind)
            symmetry_index, _ = uniform_below(
                len(choices),
                runtime_context.seed,
                "symmetry",
                (epoch, record_key, 0),
            )
            return post_process_data(
                copy.deepcopy(data),
                symmetry_type=self.apply_symmetry,
                symmetry_index=symmetry_index,
            )

        self._record_source = InterleavedSequentialSource(
            self.file_list,
            format_id="packed-binary",
            schema_version=2,
            seed=runtime_context.seed,
            shuffle=self.shuffle,
            sample_rate=self.sample_rate,
            active_streams=self.sequential_active_streams,
            read_quantum=self.sequential_read_quantum,
            output_shapes={self.fixed_board_size} if self.fixed_board_size else self.boardsizes,
            open_file=reader._open_binary_file,
            read_entry=read_entry,
            read_raw_entry=read_raw_entry,
            decode_raw_entry=decode_raw_entry,
            shape_of=shape_of,
            subrecord_count=count_subrecords,
            materialize_entry=materialize_entry,
            semantic_state=semantic_state,
        )
        self._partitioned_stream = DatasetPlanner(
            self._record_source,
            runtime_context,
            PlannerConfig(
                shuffle=self.shuffle,
                shuffle_buffer_size=self.shuffle_window_size,
                shuffle_buffer_bytes=self.shuffle_buffer_bytes,
                steps_per_epoch=self.steps_per_epoch,
            ),
            pipeline_composer=(
                PipelineStateComposer(self.batch_pipelines)
                if self.batch_pipelines
                else None
            ),
        )
        self._planned_decoder = SourceBatchDataset(
            self._partitioned_stream,
            self._record_source,
        )
        return self._partitioned_stream

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

    def _process_entry(self, entry: EntryData, selected_subrecord: int) -> dict | None:
        # Skip other rules and board sizes
        boardsize = (entry.boardsize, entry.boardsize)
        if str(entry.rule) not in self.rules:
            return None
        if boardsize not in self.boardsizes:
            return None

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

        for moveidx, movedata in enumerate(entry.moves):
            bestmove, bestmove_eval = movedata[0]

            # Planned readers address one already-admitted move directly and
            # never materialize siblings.
            if moveidx == selected_subrecord:
                game_stage = min(max(moveidx / max(len(entry.moves) - 1, 1), 0.0), 1.0)
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
                    "stm_input": np.array(
                        [current_stm_input], dtype=np.float32
                    ),  # [1] Black = -1, White = 1
                    # targets
                    "value_target": value_target,  # [3] (Black Win, White Win, Draw)
                    "policy_target": policy_target,  # [H, W] or [H*W+1] (append pass at last channel)
                    # other infos
                    "position": current_position,
                    "ply": current_ply,
                    "raw_eval": np.array(
                        np.nan if bestmove_eval is None else bestmove_eval, dtype=np.float32
                    ),
                }
                return post_process_data(
                    data,
                    fixed_side_input=self.fixed_side_input,
                    fixed_board_size=self.fixed_board_size,
                    symmetry_type=self.apply_symmetry,
                    drop_extra=self.drop_extra,
                )

            current_result = Result.opposite(current_result)
            current_position.append(bestmove)
            current_ply += 1
            if not bestmove.is_pass:
                current_board_input[max(current_stm_input, 0), bestmove.y, bestmove.x] = 1
            current_stm_input = -current_stm_input

        return None

    def _process_subrecord(self, entry: EntryData, subrecord: int) -> dict | None:
        """Replay only far enough to materialize the selected move sample."""
        return self._process_entry(entry, int(subrecord))
