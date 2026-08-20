import numpy as np
import torch
import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
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
        self._pipeline_composer = (
            PipelineStateComposer(self.batch_pipelines)
            if self.batch_pipelines
            else None
        )
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
        self._partitioned_stream = DatasetPlanner(
            self._record_source,
            runtime_context,
            planner_config,
            pipeline_composer=self._pipeline_composer,
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
        adaptive_pipeline=None,
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
        self.adaptive_pipeline = adaptive_pipeline
        if (
            adaptive_pipeline is not None
            and self._pipeline_composer is not None
            and not self._pipeline_composer.is_parallel_stateless
        ):
            raise ValueError(
                "adaptive_pipeline requires parallel-stateless batch pipelines"
            )
        self.has_pass_move = self.extra_kwargs.get("has_pass_move", False)
        self._record_decoder = None
        self._adaptive_pipeline_runtime = None
        self._node_decoded_cache_catalog = None

    def _node_decoded_cache_is_supported(self):
        if self.adaptive_pipeline is None:
            return False
        if self.adaptive_pipeline.node_decoded_cache is None:
            return False
        if self.adaptive_pipeline.config.adaptation == "manual":
            return False
        return all(
            self.extra_kwargs.get(option) is None
            for option in (
                "filter_stm",
                "filter_condition",
                "board_input_channels",
                "stm_input_channel",
                "value_target_channels",
            )
        )

    def prepare_node_decoded_cache(self):
        """Build this run's node-local immutable cache on the node leader."""

        if not self._node_decoded_cache_is_supported():
            return
        paths = reject_duplicate_physical_files(self.file_list)
        cache = self.adaptive_pipeline.node_decoded_cache
        cache.prepare(
            paths,
            workers=self.adaptive_pipeline.resources.per_rank_cpu_limit,
        )

    def activate_node_decoded_cache(self):
        """Install the collectively prepared source-to-mmap catalog."""

        if not self._node_decoded_cache_is_supported():
            return
        paths = reject_duplicate_physical_files(self.file_list)
        self._node_decoded_cache_catalog = (
            self.adaptive_pipeline.node_decoded_cache.catalog(paths)
        )

    def node_decoded_cache_ready(self):
        if not self._node_decoded_cache_is_supported():
            return False
        return self.adaptive_pipeline.node_decoded_cache.is_ready()

    def set_node_decoded_cache_enabled(self, enabled):
        """Apply one globally coordinated cache availability decision."""

        if type(enabled) is not bool:
            raise TypeError("node decoded-cache enablement must be a boolean")
        if enabled:
            self.activate_node_decoded_cache()
            return
        self._node_decoded_cache_catalog = None
        cache = (
            None
            if self.adaptive_pipeline is None
            else self.adaptive_pipeline.node_decoded_cache
        )
        if cache is not None:
            cache.cleanup()

    def cleanup_node_decoded_cache(self):
        cache = (
            None
            if self.adaptive_pipeline is None
            else self.adaptive_pipeline.node_decoded_cache
        )
        if cache is not None:
            cache.cleanup()

    def _finalize_planned_batch(self, data):
        if self.pin_memory:
            return {
                key: torch.from_numpy(np.ascontiguousarray(value)).pin_memory()
                for key, value in data.items()
            }
        return data

    def _adaptive_runtime_manifests(self, manifests):
        if self._pipeline_composer is None:
            return manifests
        return [
            {
                **manifest,
                "output_row_bytes": (
                    manifest["output_row_bytes"]
                    + self._pipeline_composer.added_output_row_bytes(
                        manifest["board_size"]
                    )
                ),
            }
            for manifest in manifests
        ]

    def _build_partitioned_stream(self):
        runtime_context = getattr(self, "runtime_context", None)
        if runtime_context is None:
            raise RuntimeError(
                "batched_processed_katago_numpy requires a DatasetRuntimeContext"
            )
        paths = reject_duplicate_physical_files(self.file_list)
        adaptive_enabled = self.adaptive_pipeline is not None
        if adaptive_enabled:
            from .pipeline_runtime import AdaptivePipelineRuntimeSpec

            if not isinstance(
                self.adaptive_pipeline,
                AdaptivePipelineRuntimeSpec,
            ):
                raise TypeError(
                    "adaptive_pipeline must be AdaptivePipelineRuntimeSpec"
                )
        telemetry_enabled = adaptive_enabled or (
            self.observability is not None and self.observability is not False
        )
        stats = None
        if telemetry_enabled:
            from .telemetry import (
                PipelineObservabilityConfig,
                PipelineStats,
            )

            observability_config = PipelineObservabilityConfig.parse(
                self.observability
            )
            telemetry_enabled = adaptive_enabled or observability_config.enabled
            if telemetry_enabled:
                stats = PipelineStats()
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
        if self._node_decoded_cache_catalog is not None:
            self._record_decoder.configure_mapped_shards(
                self._node_decoded_cache_catalog
            )
        bootstrap_memory_budget = None
        if adaptive_enabled:
            from .host_memory import HostMemoryBudget

            bootstrap_memory_budget = HostMemoryBudget(
                self.adaptive_pipeline.resources.per_rank_host_budget_bytes
            )
            self._record_decoder.configure_cache(
                entries=1,
                byte_capacity=bootstrap_memory_budget.total_bytes,
                host_memory_budget=bootstrap_memory_budget,
                file_size_catalog=(),
            )
        # On one process, two workers overlap NPZ loading, filtering, and file
        # identity hashing. Adaptive inspection stays serial because transformed
        # files share the same one-file bootstrap hard-memory allowance.
        manifest_workers = (
            1
            if adaptive_enabled
            else min(
                2 if runtime_context.world_size == 1 else 1,
                self.prefetch_threads,
                len(paths),
            )
        )
        try:
            manifests = self._inspect_processed_manifests(
                self._record_decoder,
                paths,
                manifest_workers,
            )
        except BaseException:
            if adaptive_enabled:
                self._record_decoder.close()
            raise
        adaptive_runtime = None
        if adaptive_enabled:
            from .pipeline_runtime import AdaptivePipelineRuntime

            try:
                adaptive_runtime = AdaptivePipelineRuntime(
                    self.adaptive_pipeline,
                    self._adaptive_runtime_manifests(manifests),
                    local_batch_size=runtime_context.local_batch_size,
                    global_batch_size=runtime_context.global_batch_size,
                    shuffle=planner_config.shuffle,
                    shuffle_window_size=planner_config.shuffle_buffer_size,
                    memory_budget=bootstrap_memory_budget,
                    shared_decoded_cache=(
                        self._node_decoded_cache_catalog is not None
                    ),
                )
                settings = adaptive_runtime.settings
                self.pin_memory = settings.pin_memory
                self._record_decoder.configure_cache(
                    entries=max(1, len(manifests)),
                    byte_capacity=settings.decoded_cache_bytes,
                    host_memory_budget=adaptive_runtime.memory_budget,
                    file_size_catalog=manifests,
                )
                adaptive_runtime.reserve_semantic_floor()
            except BaseException:
                try:
                    self._record_decoder.close()
                finally:
                    if adaptive_runtime is not None:
                        adaptive_runtime.close()
                raise
        try:
            self._install_processed_planner(
                runtime_context,
                self._record_decoder,
                manifests,
                planner_config,
            )
        except BaseException:
            if adaptive_runtime is not None:
                try:
                    self._record_decoder.close()
                finally:
                    adaptive_runtime.close()
            raise
        adapter_cls = SourceBatchDataset
        adapter_kwargs = {}
        if telemetry_enabled:
            from .telemetry import ObservedSourceBatchDataset

            adapter_cls = ObservedSourceBatchDataset
            adapter_kwargs["pipeline_stats"] = stats
        if adaptive_runtime is not None:
            settings = adaptive_runtime.settings
            adapter_kwargs["adaptive_runtime"] = adaptive_runtime
            effective_workers = settings.decode_workers
            effective_batches = settings.ready_queue_batches
            effective_chunk_batches = settings.decode_chunk_batches
        else:
            effective_workers = self.prefetch_threads
            effective_batches = self.prefetch_batches
            effective_chunk_batches = None
        try:
            self._planned_decoder = adapter_cls(
                self._partitioned_stream,
                self._record_source,
                finalize_batch=self._finalize_planned_batch,
                prefetch_workers=effective_workers,
                prefetch_batches=effective_batches,
                prefetch_chunk_batches=effective_chunk_batches,
                finalize_in_prefetch=True,
                host_memory_budget=(
                    None
                    if adaptive_runtime is None
                    else adaptive_runtime.memory_budget
                ),
                output_batch_bytes=(
                    0
                    if adaptive_runtime is None
                    else adaptive_runtime.output_batch_bytes
                ),
                planner_token_bytes=(
                    0
                    if adaptive_runtime is None
                    else adaptive_runtime.planner_token_bytes
                ),
                output_is_pinned=(
                    False
                    if adaptive_runtime is None
                    else adaptive_runtime.settings.pin_memory
                ),
                **adapter_kwargs,
            )
        except BaseException:
            if adaptive_runtime is not None:
                try:
                    self._record_decoder.close()
                finally:
                    adaptive_runtime.close()
            raise
        self._adaptive_pipeline_runtime = adaptive_runtime
        if telemetry_enabled:
            self.pipeline_stats = stats
        return self._partitioned_stream

    def pipeline_metrics_snapshot(self):
        if not hasattr(self._planned_decoder, "pipeline_metrics_snapshot"):
            return None
        metrics = self._planned_decoder.pipeline_metrics_snapshot()
        runtime = self._adaptive_pipeline_runtime
        if runtime is not None and metrics is not None:
            memory = runtime.memory_snapshot()
            total_bytes = memory["total_bytes"]
            metrics = replace(
                metrics,
                host_memory_used_fraction=memory["used_bytes"] / total_bytes,
                host_memory_high_water_fraction=(
                    memory["high_water_bytes"] / total_bytes
                ),
                host_memory_backpressure_events=memory["backpressure_events"],
            )
        return metrics

    def close(self):
        decoder = self._record_decoder
        runtime = self._adaptive_pipeline_runtime
        self._adaptive_pipeline_runtime = None
        try:
            if decoder is not None and hasattr(decoder, "close"):
                decoder.close()
        finally:
            if runtime is not None:
                runtime.close()

    def pipeline_tuning_update(
        self,
        metrics,
        iteration,
        *,
        epoch_changed=False,
    ):
        if not hasattr(self._planned_decoder, "pipeline_tuning_update"):
            return None
        return self._planned_decoder.pipeline_tuning_update(
            metrics,
            iteration,
            epoch_changed=epoch_changed,
        )

    def pipeline_tuning_state_dict(self):
        if not hasattr(self._planned_decoder, "pipeline_tuning_state_dict"):
            return None
        return self._planned_decoder.pipeline_tuning_state_dict()

    def load_pipeline_tuning_state_dict(self, state):
        if not hasattr(self._planned_decoder, "load_pipeline_tuning_state_dict"):
            if state is not None:
                raise ValueError("adaptive data pipeline is disabled")
            return
        self._planned_decoder.load_pipeline_tuning_state_dict(state)

    def restore_pipeline_tuning_state_dict(self, state):
        restore = getattr(
            self._planned_decoder,
            "restore_pipeline_tuning_state_dict",
            None,
        )
        return False if restore is None else restore(state)
