import numpy as np
import lz4.frame
import ctypes
import copy
import io
from utils.data_utils import *
from . import DATASETS
from .core import PipelineStateComposer, uniform_below
from .planner import DatasetPlanner, PlannerConfig
from .sequential_source import InterleavedSequentialSource
from .source_dataset import PlannedBatchDataset, SourceBatchDataset


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


def _readinto_exact(f: io.RawIOBase, buf) -> None:
    """Fill *buf* completely from *f*, raising EOFError on a short read."""
    size = ctypes.sizeof(buf)
    read = f.readinto(buf)
    if read != size:
        raise EOFError(f"expected {size} bytes, got {read}")


def read_entry(f: io.RawIOBase) -> tuple[Result, int, int, Rule, Move, list[Move]]:
    ehead = EntryHead()
    _readinto_exact(f, ehead)

    result = Result(ehead.result)
    ply = int(ehead.ply)
    boardsize = int(ehead.boardsize)
    rule = Rule(ehead.rule)
    move = Move((ehead.move >> 5) & 31, ehead.move & 31)

    pos_array = (ctypes.c_uint16 * ehead.ply)()
    _readinto_exact(f, pos_array)
    position = [Move((m >> 5) & 31, m & 31) for m in pos_array]

    return result, ply, boardsize, rule, move, position


def read_raw_entry(f: io.RawIOBase) -> bytes:
    """Read one entry without constructing its move list during admission."""
    header_size = ctypes.sizeof(EntryHead)
    header = f.read(header_size)
    if len(header) != header_size:
        raise EOFError(f"expected {header_size} bytes, got {len(header)}")
    entry_head = EntryHead.from_buffer_copy(header)
    payload_size = 2 * int(entry_head.ply)
    parts = []
    remaining = payload_size
    while remaining:
        part = f.read(remaining)
        if not part:
            break
        parts.append(part)
        remaining -= len(part)
    if remaining:
        raise EOFError(f"expected {payload_size} bytes, got {payload_size - remaining}")
    return header + b"".join(parts)


def decode_raw_entry(raw_entry: bytes):
    stream = io.BytesIO(raw_entry)
    entry = read_entry(stream)
    if stream.tell() != len(raw_entry):
        raise RuntimeError("simple-binary payload contains trailing entry bytes")
    return entry


def raw_entry_shape(raw_entry: bytes):
    if len(raw_entry) < ctypes.sizeof(EntryHead):
        raise RuntimeError("simple-binary payload is missing its header")
    entry_head = EntryHead.from_buffer_copy(raw_entry)
    return int(entry_head.boardsize), str(Rule(entry_head.rule))


@DATASETS.register("simple_binary")
class SimpleBinaryDataset(PlannedBatchDataset):
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
        batch_size: int | None = None,
        batch_pipelines=(),
        shuffle_window_size: int = 32768,
        shuffle_buffer_bytes: int | None = 256 * 1024 * 1024,
        sequential_active_streams: int = 2,
        sequential_read_quantum: int = 256,
        steps_per_epoch: int | None = None,
    ):
        super().__init__()
        self.batch_pipelines = tuple(batch_pipelines)
        self.file_list = file_list
        self.rules = rules
        self.boardsizes = boardsizes
        self.fixed_side_input = fixed_side_input
        self.fixed_board_size = fixed_board_size
        self.apply_symmetry = apply_symmetry
        self.drop_extra = drop_extra
        self.shuffle = shuffle
        self.sample_rate = sample_rate
        self.shuffle_window_size = shuffle_window_size
        self.shuffle_buffer_bytes = shuffle_buffer_bytes
        self.sequential_active_streams = sequential_active_streams
        self.sequential_read_quantum = sequential_read_quantum
        self.steps_per_epoch = steps_per_epoch
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
            raise RuntimeError("simple_binary requires a DatasetRuntimeContext")

        reader = SimpleBinaryDataset(
            file_list=[],
            rules=self.rules,
            boardsizes=self.boardsizes,
            fixed_side_input=self.fixed_side_input,
            fixed_board_size=self.fixed_board_size,
            apply_symmetry=False,
            drop_extra=self.drop_extra,
            shuffle=False,
            sample_rate=1.0,
        )
        semantic_state = {
            "rules": sorted(self.rules),
            "boardsizes": sorted(self.boardsizes),
            "fixed_side_input": self.fixed_side_input,
            "fixed_board_size": self.fixed_board_size,
            "drop_extra": self.drop_extra,
            "apply_symmetry": self.apply_symmetry,
        }

        def shape_of(raw_entry):
            boardsize_value, rule = raw_entry_shape(raw_entry)
            boardsize = (boardsize_value, boardsize_value)
            if rule not in self.rules or boardsize not in self.boardsizes:
                return None
            return self.fixed_board_size or boardsize

        def materialize_entry(entry, record_key, epoch):
            data = reader._prepare_data_from_entry(*entry)
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
            format_id="simple-binary",
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
            self._partitioned_stream, self._record_source
        )
        return self._partitioned_stream

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
