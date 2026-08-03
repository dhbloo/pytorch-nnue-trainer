import numpy as np
import torch
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data.dataset import Dataset
from torch.utils.data import get_worker_info
from utils.data_utils import *
from . import DATASETS
from .decoder import NpzRowRecordDecoder, ProcessedNpzDecoder
from .core import PipelineStateComposer, uniform_below
from .npz_source import DenseNpzSource, IndexedNpzSource
from .planner import DatasetPlanner, PlannerConfig
from .source_dataset import PlannedBatchDataset, SourceBatchDataset
from .stream import (
    MapRecordRef,
    reject_duplicate_physical_files,
)


def _dataset_content_digest(file_list):
    digest = hashlib.sha256()
    digest.update(b"NNUE-map-dataset-v1\0")
    for filename in file_list:
        file_digest = hashlib.sha256()
        with open(filename, "rb") as stream:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                file_digest.update(chunk)
        digest.update(file_digest.digest())
    return digest.digest()


def _map_symmetry_index(dataset, index, board_size):
    if not dataset.apply_symmetry:
        return None
    symmetry_type = (
        "default" if dataset.apply_symmetry is True else dataset.apply_symmetry
    )
    symmetries = Symmetry.available_symmetries(board_size, symmetry_type)
    context = getattr(dataset, "runtime_context", None)
    worker = get_worker_info()
    rank = 0 if context is None else context.rank_local_identity.rank
    worker_id = 0 if worker is None else worker.id
    root_seed = 0 if context is None else context.seed
    sample_key = (
        "map-row",
        dataset._sample_root_digest,
        type(dataset).__name__,
        (int(index),),
    )
    picked, _ = uniform_below(
        len(symmetries),
        root_seed,
        "map_symmetry",
        (
            int(getattr(dataset, "_active_epoch", 0)),
            sample_key,
            rank,
            worker_id,
            0,
        ),
    )
    return picked


@DATASETS.register("katago_numpy")
class KatagoNumpyDataset(Dataset):
    FILE_EXTS = [".npz"]

    def __init__(
        self,
        file_list: list[str],
        boardsizes: set[tuple[int, int]],
        rules: set[str] | None = None,
        fixed_side_input: bool = False,
        fixed_board_size: None | tuple[int, int] = None,
        has_pass_move: bool = False,
        apply_symmetry: bool = False,
        filter_stm: int | None = None,
        filter_condition: str | None = None,
        shuffle: bool = False,
        value_td_level: int = 0,
    ):
        super().__init__()
        self.file_list = file_list
        self.boardsizes = boardsizes
        self.fixed_side_input = fixed_side_input
        self.fixed_board_size = fixed_board_size
        self.has_pass_move = has_pass_move
        self.apply_symmetry = apply_symmetry
        self.shuffle = shuffle
        self.filter_stm = filter_stm
        self.filter_condition = filter_condition
        self.value_td_level = value_td_level
        self._sample_root_digest = _dataset_content_digest(self.file_list)
        self._active_epoch = 0
        if filter_stm is not None and not isinstance(filter_stm, int):
            raise TypeError("filter_stm must be an integer")

        self.data_dict = {}

        # Load data tensors from npz files
        for filename in self.file_list:
            data = np.load(filename)
            data_dict, length = self._unpack_data(data)

            selected_indices = []
            for i in range(length):
                if tuple(data_dict["board_size"][i]) in self.boardsizes:
                    selected_indices.append(i)

            for k in data_dict:
                data_dict[k] = data_dict[k][selected_indices, ...]

            for k in data_dict:
                if k not in self.data_dict:
                    self.data_dict[k] = [data_dict[k]]
                else:
                    self.data_dict[k].append(data_dict[k])

        # Concatenate tensors across files
        for k in self.data_dict:
            if len(self.data_dict[k]) > 1:
                self.data_dict[k] = np.concatenate(self.data_dict[k], axis=0)
            else:
                self.data_dict[k] = self.data_dict[k][0]

        # Validate row counts after board-size filtering.
        length_list = [len(array) for array in self.data_dict.values()]
        self.length = length_list[0]
        if self.length <= 0:
            raise ValueError(f"no valid data entry in {self.file_list}")
        if length_list.count(self.length) != len(length_list):
            raise ValueError("NPZ fields have unequal row counts")

    @property
    def is_fixed_side_input(self):
        return self.fixed_side_input

    def _unpack_global_feature(self, packed_data):
        if packed_data.shape[1] == 1:
            # Channel 0: side to move (black = -1.0, white = 1.0)
            stm_input = packed_data[:, [0]].astype(np.float32)
        else:
            # Original katago feature format:
            # Channel 5: komi (black negative, white positive)
            stm_input = np.where(packed_data[:, [5]] > 0, 1, -1).astype(np.float32)
        return stm_input

    def _unpack_board_feature(self, packed_data, dims=[1, 2]):
        length, n_features, n_bytes = packed_data.shape
        bsize = int(np.sqrt(n_bytes * 8))

        # Channel 1: next player stones
        # Channel 2: oppo stones
        packed_data = packed_data[:, dims]

        board_input = np.unpackbits(packed_data, axis=2, count=bsize * bsize, bitorder="big")
        board_input = board_input.reshape(length, len(dims), bsize, bsize).astype(np.int8)
        return board_input

    def _unpack_global_target(self, packed_data):
        # Channel 0: stm win probability
        # Channel 1: stm loss probability
        # Channel 2: draw probability
        base = self.value_td_level * 4
        return packed_data[:, [base + 0, base + 1, base + 2]]

    def _unpack_policy_target(self, packed_data):
        length, n_features, n_cells = packed_data.shape
        bsize = int(np.sqrt(n_cells - 1))
        if bsize * bsize + 1 != n_cells:
            raise ValueError("packed policy target has an invalid cell count")

        # Channel 0: policy target this turn
        policy_target_stm = packed_data[:, 0, : bsize * bsize + (1 if self.has_pass_move else 0)]
        policy_sum = np.sum(policy_target_stm.astype(np.float32), axis=1, keepdims=True)
        policy_target_stm = policy_target_stm / (policy_sum + 1e-9)
        if not self.has_pass_move:
            policy_target_stm = policy_target_stm.reshape(-1, bsize, bsize)
        return policy_target_stm  # [H, W] or [H*W+1] (append pass at last channel)

    def _unpack_data(self, raw_npz_data):
        raw_data_dict = {
            "binaryInputNCHWPacked": raw_npz_data["binaryInputNCHWPacked"],
            "globalInputNC": raw_npz_data["globalInputNC"],
            "globalTargetsNC": raw_npz_data["globalTargetsNC"],
            "policyTargetsNCMove": raw_npz_data["policyTargetsNCMove"],
        }
        if self.filter_stm is not None:
            if raw_data_dict["globalInputNC"].shape[1] == 1:
                condition = raw_data_dict["globalInputNC"][:, 0] == self.filter_stm
            else:
                condition = (
                    raw_data_dict["globalInputNC"][:, 5] > 0
                    if self.filter_stm == 1
                    else raw_data_dict["globalInputNC"][:, 5] < 0
                )
            selected_indices = np.nonzero(condition)[0]
            raw_data_dict = {
                key: value[selected_indices, ...] for key, value in raw_data_dict.items()
            }
        if self.filter_condition is not None:
            filter_data_by_condition(self.filter_condition, raw_data_dict)

        stm_input = self._unpack_global_feature(raw_data_dict["globalInputNC"])
        board_input_stm = self._unpack_board_feature(raw_data_dict["binaryInputNCHWPacked"])
        value_target = self._unpack_global_target(raw_data_dict["globalTargetsNC"])
        policy_target = self._unpack_policy_target(raw_data_dict["policyTargetsNCMove"])

        # Get board size from the 0 channel of packed board input
        board_mask = self._unpack_board_feature(raw_data_dict["binaryInputNCHWPacked"], dims=[0])
        board_width = np.sum(board_mask[:, 0, 0, :], axis=1)
        board_height = np.sum(board_mask[:, 0, :, 0], axis=1)
        board_size = np.stack([board_height, board_width], axis=1)  # (N, 2)

        return {
            "board_size": board_size,
            "board_input": board_input_stm,
            "stm_input": stm_input,
            "value_target": value_target,
            "policy_target": policy_target,
        }, len(board_size)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = {k: self.data_dict[k][index] for k in self.data_dict}
        return post_process_data(
            data,
            self.fixed_side_input,
            self.fixed_board_size,
            self.apply_symmetry,
            symmetry_index=_map_symmetry_index(
                self, index, tuple(int(value) for value in data["board_size"])
            ),
        )

    def map_record_ref(self, index):
        board_size = tuple(
            int(value) for value in self.data_dict["board_size"][index]
        )
        output_shape = self.fixed_board_size or board_size
        return MapRecordRef(
            type(self).__name__,
            int(index),
            (
                "map-row",
                self._sample_root_digest,
                type(self).__name__,
                (int(index),),
            ),
            tuple(output_shape),
        )


@DATASETS.register("iterative_katago_numpy")
class IterativeKatagoNumpyDataset(PlannedBatchDataset):
    """
    Similar to KatagoNumpyDataset but with iterative loading.
    This is useful when the dataset is too large to fit into memory.
    """

    FILE_EXTS = [".npz"]
    def __init__(
        self,
        file_list: list[str],
        boardsizes: set[tuple[int, int]],
        rules: set[str] | None = None,
        fixed_side_input: bool = False,
        fixed_board_size: None | tuple[int, int] = None,
        has_pass_move: bool = False,
        apply_symmetry: bool | str = False,
        filter_stm: int | None = None,
        filter_condition: str | None = None,
        value_td_level: int = 0,
        shuffle: bool = False,
        sample_rate: float = 1.0,
        batch_size: int | None = None,
        batch_pipelines=(),
        shuffle_window_size: int = 32768,
        shuffle_buffer_bytes: int | None = None,
        steps_per_epoch: int | None = None,
    ):
        super().__init__()
        self.file_list = file_list
        self.boardsizes = boardsizes
        self.fixed_side_input = fixed_side_input
        self.shuffle = shuffle
        self.sample_rate = sample_rate
        self.batch_pipelines = tuple(batch_pipelines)
        self.extra_kwargs = {
            "fixed_board_size": fixed_board_size,
            "has_pass_move": has_pass_move,
            "apply_symmetry": apply_symmetry,
            "filter_stm": filter_stm,
            "filter_condition": filter_condition,
            "value_td_level": value_td_level,
            "batch_size": batch_size,
            "batch_pipelines": self.batch_pipelines,
            "shuffle_window_size": shuffle_window_size,
            "shuffle_buffer_bytes": shuffle_buffer_bytes,
            "steps_per_epoch": steps_per_epoch,
        }

    def _build_partitioned_stream(self):
        runtime_context = getattr(self, "runtime_context", None)
        if runtime_context is None:
            raise RuntimeError("iterative_katago_numpy requires a DatasetRuntimeContext")
        options = dict(self.extra_kwargs)
        options.pop("batch_size", None)
        options.pop("batch_pipelines", None)
        options.pop("rules", None)
        options.pop("shuffle_window_size", None)
        shuffle_buffer_bytes = options.pop("shuffle_buffer_bytes", None)
        steps_per_epoch = options.pop("steps_per_epoch", None)
        symmetry = options.pop("apply_symmetry", False)

        def load(path):
            dataset = KatagoNumpyDataset(
                file_list=[path],
                boardsizes=self.boardsizes,
                fixed_side_input=self.fixed_side_input,
                apply_symmetry=False,
                **options,
            )
            return dataset, len(dataset)

        decoder = NpzRowRecordDecoder(
            "raw-katago-npz",
            runtime_context,
            load,
            lambda dataset, index: dataset[index],
            apply_symmetry=symmetry,
            catalog_rows=lambda dataset, length: (
                None,
                (
                    options["fixed_board_size"]
                    if options.get("fixed_board_size") is not None
                    else dataset.data_dict["board_size"]
                ),
            ),
            semantic_state={
                "boardsizes": sorted(self.boardsizes),
                "fixed_side_input": self.fixed_side_input,
                **options,
            },
        )
        paths = reject_duplicate_physical_files(self.file_list)
        catalogs = [
            decoder.inspect_compact(path, ordinal)
            for ordinal, path in enumerate(paths)
        ]
        self._record_source = IndexedNpzSource(
            catalogs,
            decoder,
            seed=runtime_context.seed,
            shuffle=self.shuffle,
            sample_rate=self.sample_rate,
        )
        composer = (
            PipelineStateComposer(self.batch_pipelines)
            if self.batch_pipelines
            else None
        )
        self._partitioned_stream = DatasetPlanner(
            self._record_source,
            runtime_context,
            PlannerConfig(
                shuffle=self.shuffle,
                shuffle_buffer_size=self.extra_kwargs.get(
                    "shuffle_window_size", 32768
                ),
                shuffle_buffer_bytes=shuffle_buffer_bytes,
                steps_per_epoch=steps_per_epoch,
            ),
            pipeline_composer=composer,
        )
        self._planned_decoder = SourceBatchDataset(
            self._partitioned_stream,
            self._record_source,
        )
        return self._partitioned_stream


@DATASETS.register("processed_katago_numpy")
class ProcessedKatagoNumpyDataset(Dataset):
    """
    Dataset with processed npz files from katago.
    Each npz file should contain the following keys:
    - bf: board feature (N, C, H, W)
    - gf: global feature (N, 1) for side to move (black = -1.0, white = 1.0)
    - vt: value target (N, 3) for win, loss, draw
    - pt: policy target (N, H*W) or (N, H*W+1) if has_pass_move
    """

    FILE_EXTS = [".npz"]

    def __init__(
        self,
        file_list: list[str],
        boardsizes: set[tuple[int, int]],
        rules: set[str] | None = None,
        fixed_side_input: bool = False,
        fixed_board_size: None | tuple[int, int] = None,
        has_pass_move: bool = False,
        apply_symmetry: bool = False,
        filter_stm: int | None = None,
        filter_condition: str | None = None,
        board_input_channels: list[int] | None = None,
        stm_input_channel: int | None = None,
        value_target_channels: list[int] | None = None,
        shuffle: bool = False,
    ):
        super().__init__()
        self.file_list = file_list
        self.boardsizes = boardsizes
        self.fixed_side_input = fixed_side_input
        self.fixed_board_size = fixed_board_size
        self.has_pass_move = has_pass_move
        self.apply_symmetry = apply_symmetry
        self._sample_root_digest = _dataset_content_digest(self.file_list)
        self._active_epoch = 0

        self.data_dict = {
            "bf": [],
            "gf": [],
            "vt": [],
            "pt": [],
        }

        # Read all npz files to data dict
        for filename in self.file_list:
            data = np.load(filename)

            # Skip other board size file
            if tuple(data["bf"].shape[2:]) not in self.boardsizes:
                continue

            for k in self.data_dict:
                if k in data:
                    if len(data[k]) <= 0:
                        raise ValueError(f"empty tensor {k} in file {filename}")
                    self.data_dict[k].append(data[k])

        # Concatenate tensors across files
        length_list = []
        concated_data_dict = {}
        for k, tensor_list in self.data_dict.items():
            if len(tensor_list) > 1:
                concated_data_dict[k] = np.concatenate(tensor_list, axis=0)
            elif len(tensor_list) > 0:
                concated_data_dict[k] = tensor_list[0]
            elif k == "gf" or k == "pt":
                continue  # allow tensor gf/pt to be empty
            length_list.append(len(concated_data_dict[k]))
        self.data_dict = concated_data_dict

        # Validate processed NPZ row counts.
        self.length = length_list[0]
        if self.length <= 0:
            raise ValueError(f"no valid data entry in {self.file_list}")
        if length_list.count(self.length) != len(length_list):
            raise ValueError("processed NPZ fields have unequal row counts")

        # Get board size
        self.boardsize = self.data_dict["bf"].shape[2:]
        if len(self.boardsize) != 2:
            raise ValueError("processed board feature must have two spatial axes")

        if filter_stm is not None:
            if not isinstance(filter_stm, int):
                raise TypeError("filter_stm must be an integer")
            if "gf" not in self.data_dict:
                raise ValueError("gf tensor is required for filtering stm")
            selected_indices = np.nonzero(self.data_dict["gf"][:, 0] == filter_stm)[0]
            self.data_dict = {
                key: value[selected_indices, ...] for key, value in self.data_dict.items()
            }
            self.length = len(selected_indices)
        if filter_condition is not None:
            self.length = filter_data_by_condition(filter_condition, self.data_dict)

        # Select a subset of channels if specified
        if board_input_channels is not None:
            self.data_dict["bf"] = self.data_dict["bf"][:, board_input_channels]
        if stm_input_channel is not None:
            if "gf" not in self.data_dict:
                raise ValueError(
                    "stm_input_channel requires a gf tensor"
                )
            self.data_dict["gf"] = self.data_dict["gf"][:, [stm_input_channel]]
        if value_target_channels is not None:
            self.data_dict["vt"] = self.data_dict["vt"][:, value_target_channels]

    @property
    def is_fixed_side_input(self):
        return self.fixed_side_input

    def _prepare_data(self, index):
        board_size = np.array(self.boardsize, dtype=np.int8)
        board_input = self.data_dict["bf"][index].astype(np.int8)
        if "gf" in self.data_dict:
            stm_input = self.data_dict["gf"][index].astype(np.float32)
        else:
            stm_input = np.array([0], dtype=np.float32)
        value_target = self.data_dict["vt"][index].astype(np.float32)

        if "pt" in self.data_dict:
            policy_target = self.data_dict["pt"][index].astype(np.float32)
        else:
            _, h, w = board_input.shape
            if self.has_pass_move:
                policy_target = np.zeros((h * w + 1,), dtype=np.float32)
            else:
                policy_target = np.zeros((h, w), dtype=np.float32)

        # Ignore pass move for 1d policy target
        if not self.has_pass_move and policy_target.ndim == 1:
            _, h, w = board_input.shape
            policy_target = policy_target[:-1].reshape((h, w))

        return {
            "board_size": board_size,
            "board_input": board_input,
            "stm_input": stm_input,
            "value_target": value_target,
            "policy_target": policy_target,
        }

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = self._prepare_data(index)
        return post_process_data(
            data,
            self.fixed_side_input,
            self.fixed_board_size,
            self.apply_symmetry,
            symmetry_index=_map_symmetry_index(
                self, index, tuple(int(value) for value in data["board_size"])
            ),
        )

    def map_record_ref(self, index):
        output_shape = self.fixed_board_size or self.boardsize
        return MapRecordRef(
            type(self).__name__,
            int(index),
            (
                "map-row",
                self._sample_root_digest,
                type(self).__name__,
                (int(index),),
            ),
            tuple(int(value) for value in output_shape),
        )


@DATASETS.register("iterative_processed_katago_numpy")
class IterativeProcessedKatagoNumpyDataset(PlannedBatchDataset):
    """
    Similar to ProcessedKatagoNumpyDataset but with iterative loading.
    This is useful when the dataset is too large to fit into memory.
    """

    FILE_EXTS = [".npz"]

    def __init__(
        self,
        file_list: list[str],
        boardsizes: set[tuple[int, int]],
        rules: set[str] | None = None,
        fixed_side_input: bool = False,
        fixed_board_size: None | tuple[int, int] = None,
        has_pass_move: bool = False,
        apply_symmetry: bool | str = False,
        filter_stm: int | None = None,
        filter_condition: str | None = None,
        board_input_channels: list[int] | None = None,
        stm_input_channel: int | None = None,
        value_target_channels: list[int] | None = None,
        shuffle: bool = False,
        sample_rate: float = 1.0,
        batch_size: int | None = None,
        batch_pipelines=(),
        shuffle_window_size: int = 32768,
        shuffle_buffer_bytes: int | None = None,
        steps_per_epoch: int | None = None,
    ):
        super().__init__()
        self.file_list = file_list
        self.boardsizes = boardsizes
        self.fixed_side_input = fixed_side_input
        self.shuffle = shuffle
        self.sample_rate = sample_rate
        self.batch_pipelines = tuple(batch_pipelines)
        self.extra_kwargs = {
            "fixed_board_size": fixed_board_size,
            "has_pass_move": has_pass_move,
            "apply_symmetry": apply_symmetry,
            "filter_stm": filter_stm,
            "filter_condition": filter_condition,
            "board_input_channels": board_input_channels,
            "stm_input_channel": stm_input_channel,
            "value_target_channels": value_target_channels,
            "batch_size": batch_size,
            "batch_pipelines": self.batch_pipelines,
            "shuffle_window_size": shuffle_window_size,
            "shuffle_buffer_bytes": shuffle_buffer_bytes,
            "steps_per_epoch": steps_per_epoch,
        }

    def _processed_stream_options(self):
        decoder_kwargs = dict(self.extra_kwargs)
        shuffle_window_size = decoder_kwargs.pop("shuffle_window_size", 32768)
        shuffle_buffer_bytes = decoder_kwargs.pop("shuffle_buffer_bytes", None)
        steps_per_epoch = decoder_kwargs.pop("steps_per_epoch", None)
        for option in ("rules", "batch_size", "batch_pipelines"):
            decoder_kwargs.pop(option, None)
        symmetry = decoder_kwargs.pop("apply_symmetry", False)
        planner_config = PlannerConfig(
            shuffle=self.shuffle,
            shuffle_buffer_size=shuffle_window_size,
            shuffle_buffer_bytes=shuffle_buffer_bytes,
            steps_per_epoch=steps_per_epoch,
        )
        return decoder_kwargs, symmetry, planner_config

    @staticmethod
    def _inspect_processed_manifests(decoder, paths, workers=1):
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                return list(executor.map(decoder.inspect, paths, range(len(paths))))
        return [decoder.inspect(path, ordinal) for ordinal, path in enumerate(paths)]

    def _install_processed_planner(
        self,
        runtime_context,
        decoder,
        manifests,
        planner_config,
    ):
        self._record_source = DenseNpzSource(
            manifests,
            decoder,
            seed=runtime_context.seed,
            shuffle=self.shuffle,
            sample_rate=self.sample_rate,
        )
        composer = (
            PipelineStateComposer(self.batch_pipelines)
            if self.batch_pipelines
            else None
        )
        self._partitioned_stream = DatasetPlanner(
            self._record_source,
            runtime_context,
            planner_config,
            pipeline_composer=composer,
        )
        return self._partitioned_stream

    def _build_partitioned_stream(self):
        runtime_context = getattr(self, "runtime_context", None)
        if runtime_context is None:
            raise RuntimeError(
                "iterative_processed_katago_numpy requires a DatasetRuntimeContext"
            )
        decoder_kwargs, symmetry, planner_config = self._processed_stream_options()
        decoder = ProcessedNpzDecoder(
            boardsizes=self.boardsizes,
            runtime_context=runtime_context,
            fixed_side_input=self.fixed_side_input,
            apply_symmetry=symmetry,
            **decoder_kwargs,
        )
        paths = reject_duplicate_physical_files(self.file_list)
        manifests = self._inspect_processed_manifests(decoder, paths)
        self._install_processed_planner(
            runtime_context,
            decoder,
            manifests,
            planner_config,
        )
        self._planned_decoder = SourceBatchDataset(
            self._partitioned_stream,
            self._record_source,
        )
        return self._partitioned_stream


@DATASETS.register("batched_processed_katago_numpy")
class BatchedProcessedKatagoNumpyDataset(IterativeProcessedKatagoNumpyDataset):
    """
    Planned batch-level variant of IterativeProcessedKatagoNumpyDataset.

    One deterministic global stream plans shape-homogeneous batches across
    file boundaries, then each DDP rank decodes its disjoint local slice.
    Training drops only incomplete global shape tails; evaluation pads them
    with an ``is_real`` mask. The main thread owns planning, ordered
    finalization, pipeline state, and transactional tokens. Optional worker
    threads only decode numbered batches into a bounded ordered prefetch
    queue, so thread timing cannot change the yielded order.

    Args (in addition to IterativeProcessedKatagoNumpyDataset):
        batch_size: Number of samples per yielded batch.
        apply_symmetry: Randomly transform each sample by a board symmetry
            using its epoch and stable sample key. Rows are transformed in
            one vectorized indexed gather without reordering the batch.
        prefetch_threads: Ordered decode workers (0 decodes synchronously).
        prefetch_batches: Bound on submitted but not yet yielded batches.
        pin_memory: Yield batches as pinned torch tensors for fast async H2D
            copies. Default: auto (pinned iff CUDA is available).
    """

    def __init__(
        self,
        file_list: list[str],
        boardsizes: set[tuple[int, int]],
        rules: set[str] | None = None,
        fixed_side_input: bool = False,
        fixed_board_size: None | tuple[int, int] = None,
        has_pass_move: bool = False,
        filter_stm: int | None = None,
        filter_condition: str | None = None,
        board_input_channels: list[int] | None = None,
        stm_input_channel: int | None = None,
        value_target_channels: list[int] | None = None,
        shuffle: bool = False,
        sample_rate: float = 1.0,
        batch_size: int = 1,
        apply_symmetry=False,
        batch_pipelines=(),
        prefetch_threads: int = 2,
        prefetch_batches: int = 32,
        pin_memory: bool | None = None,
        observability=False,
        autotune=False,
        shuffle_window_size: int = 32768,
        shuffle_buffer_bytes: int | None = None,
        steps_per_epoch: int | None = None,
    ):
        super().__init__(
            file_list=file_list,
            boardsizes=boardsizes,
            rules=rules,
            fixed_side_input=fixed_side_input,
            fixed_board_size=fixed_board_size,
            has_pass_move=has_pass_move,
            apply_symmetry=apply_symmetry,
            filter_stm=filter_stm,
            filter_condition=filter_condition,
            board_input_channels=board_input_channels,
            stm_input_channel=stm_input_channel,
            value_target_channels=value_target_channels,
            shuffle=shuffle,
            sample_rate=sample_rate,
            batch_size=batch_size,
            batch_pipelines=batch_pipelines,
            shuffle_window_size=shuffle_window_size,
            shuffle_buffer_bytes=shuffle_buffer_bytes,
            steps_per_epoch=steps_per_epoch,
        )
        self.batch_size = batch_size
        self.apply_symmetry = apply_symmetry
        self.batch_pipelines = tuple(batch_pipelines)
        self.prefetch_threads = prefetch_threads
        self.prefetch_batches = prefetch_batches
        self.pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory
        self.observability = observability
        self.autotune = autotune
        self.has_pass_move = self.extra_kwargs.get("has_pass_move", False)
        self._record_decoder = None

    def _finalize_planned_batch(self, data):
        if self.pin_memory:
            return {
                key: torch.from_numpy(np.ascontiguousarray(value)).pin_memory()
                for key, value in data.items()
            }
        return data

    def _build_partitioned_stream(self):
        runtime_context = getattr(self, "runtime_context", None)
        if runtime_context is None:
            raise RuntimeError(
                "batched_processed_katago_numpy requires a DatasetRuntimeContext"
            )
        paths = reject_duplicate_physical_files(self.file_list)
        telemetry_enabled = any(
            value is not None and value is not False
            for value in (self.observability, self.autotune)
        )
        stats = None
        observability_config = None
        autotune_config = None
        if telemetry_enabled:
            from .telemetry import (
                PipelineAutotuneConfig,
                PipelineObservabilityConfig,
                PipelineStats,
            )

            observability_config = PipelineObservabilityConfig.parse(
                self.observability
            )
            autotune_config = PipelineAutotuneConfig.parse(self.autotune)
            telemetry_enabled = observability_config.enabled or autotune_config.enabled
            if telemetry_enabled:
                stats = PipelineStats()
            if autotune_config.enabled:
                if self.prefetch_threads <= 0:
                    raise ValueError(
                        "pipeline autotuning requires prefetch_threads to be positive"
                    )
                if self.batch_pipelines:
                    raise ValueError(
                        "pipeline autotuning does not support stateful batch pipelines"
                    )
                if (
                    autotune_config.respect_explicit
                    and getattr(self, "_explicit_prefetch_threads", False)
                    and self.prefetch_threads > autotune_config.max_prefetch_threads
                ):
                    raise ValueError(
                        "explicit prefetch_threads exceeds autotune.max_prefetch_threads"
                    )
                if (
                    autotune_config.respect_explicit
                    and getattr(self, "_explicit_prefetch_batches", False)
                    and self.prefetch_batches > autotune_config.max_prefetch_batches
                ):
                    raise ValueError(
                        "explicit prefetch_batches exceeds autotune.max_prefetch_batches"
                    )
        decoder_kwargs, symmetry, planner_config = self._processed_stream_options()
        decoder_cls = ProcessedNpzDecoder
        if telemetry_enabled:
            from .telemetry import ObservedProcessedNpzDecoder

            decoder_cls = ObservedProcessedNpzDecoder
            decoder_kwargs["pipeline_stats"] = stats
        self._record_decoder = decoder_cls(
            boardsizes=self.boardsizes,
            runtime_context=runtime_context,
            fixed_side_input=self.fixed_side_input,
            apply_symmetry=symmetry,
            **decoder_kwargs,
        )
        # On one process, two workers overlap NPZ loading, filtering, and file
        # identity hashing. Keep distributed startup serialized per rank to
        # avoid multiplying storage I/O.
        manifest_workers = min(
            2 if runtime_context.world_size == 1 else 1,
            self.prefetch_threads,
            len(paths),
        )
        manifest_start = time.perf_counter_ns() if telemetry_enabled else None
        manifests = self._inspect_processed_manifests(
            self._record_decoder,
            paths,
            manifest_workers,
        )
        if telemetry_enabled:
            stats.record_manifest(
                time.perf_counter_ns() - manifest_start,
                len(manifests),
                sum(int(manifest["logical_row_count"]) for manifest in manifests),
            )
        self._install_processed_planner(
            runtime_context,
            self._record_decoder,
            manifests,
            planner_config,
        )
        adapter_cls = SourceBatchDataset
        adapter_kwargs = {}
        if telemetry_enabled:
            from .telemetry import ObservedSourceBatchDataset

            adapter_cls = ObservedSourceBatchDataset
            adapter_kwargs["pipeline_stats"] = stats
        if autotune_config is not None and autotune_config.enabled:
            from .telemetry import PipelineAutotuner, build_tuning_keys

            pipeline_signatures = [
                pipeline.signature_state() for pipeline in self.batch_pipelines
            ]
            exact_key, compatible_key = build_tuning_keys(
                source_manifest=self._record_source.manifest_state(),
                batch_size=runtime_context.local_batch_size,
                world_size=runtime_context.world_size,
                pin_memory=self.pin_memory,
                pipeline_signatures=pipeline_signatures,
                cache_budget_bytes=autotune_config.host_cache_budget_bytes,
                tuning_contract={
                    "initial_prefetch_threads": self.prefetch_threads,
                    "initial_prefetch_batches": self.prefetch_batches,
                    "max_prefetch_threads": autotune_config.max_prefetch_threads,
                    "max_prefetch_batches": autotune_config.max_prefetch_batches,
                    "cuda_prefetch_batches": int(
                        getattr(self, "_cuda_prefetch_batches", 0)
                    ),
                    "respect_explicit": autotune_config.respect_explicit,
                    "explicit_prefetch_threads": getattr(
                        self, "_explicit_prefetch_threads", False
                    ),
                    "explicit_prefetch_batches": getattr(
                        self, "_explicit_prefetch_batches", False
                    ),
                },
            )
            locked = set()
            if autotune_config.respect_explicit:
                if getattr(self, "_explicit_prefetch_threads", False):
                    locked.add("prefetch_threads")
                if getattr(self, "_explicit_prefetch_batches", False):
                    locked.add("prefetch_batches")
            controller = PipelineAutotuner(
                autotune_config,
                initial_workers=max(1, self.prefetch_threads),
                initial_prefetch_batches=self.prefetch_batches,
                initial_cache_entries=self._record_decoder._array_cache_capacity,
                initial_cache_bytes=self._record_decoder._array_cache_byte_capacity,
                exact_key=exact_key,
                compatible_key=compatible_key,
                locked_options=locked,
            )
            self._record_decoder.configure_cache(
                entries=controller.settings["cache_entries"],
                byte_capacity=controller.settings["cache_bytes"],
            )
            adapter_kwargs.update(
                {
                    "autotuner": controller,
                    "maximum_prefetch_workers": autotune_config.max_prefetch_threads,
                }
            )
            effective_workers = controller.settings["prefetch_threads"]
            effective_batches = controller.settings["prefetch_batches"]
        else:
            effective_workers = self.prefetch_threads
            effective_batches = self.prefetch_batches
        self._planned_decoder = adapter_cls(
            self._partitioned_stream,
            self._record_source,
            finalize_batch=self._finalize_planned_batch,
            prefetch_workers=effective_workers,
            prefetch_batches=effective_batches,
            finalize_in_prefetch=True,
            **adapter_kwargs,
        )
        if telemetry_enabled:
            self.pipeline_stats = stats
        return self._partitioned_stream

    def pipeline_metrics_snapshot(self):
        if not hasattr(self._planned_decoder, "pipeline_metrics_snapshot"):
            return None
        return self._planned_decoder.pipeline_metrics_snapshot()

    def attach_pipeline_run_dir(self, rundir):
        if hasattr(self._planned_decoder, "attach_pipeline_run_dir"):
            self._planned_decoder.attach_pipeline_run_dir(rundir)

    def pipeline_tuning_update(self, metrics, iteration):
        if not hasattr(self._planned_decoder, "pipeline_tuning_update"):
            return None
        return self._planned_decoder.pipeline_tuning_update(metrics, iteration)

    def pipeline_tuning_state_dict(self):
        if not hasattr(self._planned_decoder, "pipeline_tuning_state_dict"):
            return None
        return self._planned_decoder.pipeline_tuning_state_dict()

    def load_pipeline_tuning_state_dict(self, state):
        if not hasattr(self._planned_decoder, "load_pipeline_tuning_state_dict"):
            if state is not None:
                raise ValueError("pipeline autotuning is disabled")
            return
        self._planned_decoder.load_pipeline_tuning_state_dict(state)

    def restore_pipeline_tuning_state_dict(self, state):
        restore = getattr(
            self._planned_decoder,
            "restore_pipeline_tuning_state_dict",
            None,
        )
        return False if restore is None else restore(state)
