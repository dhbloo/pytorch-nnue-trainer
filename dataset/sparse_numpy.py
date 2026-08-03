import numpy as np
from utils.data_utils import *
from . import DATASETS
from .decoder import NpzRowRecordDecoder
from .core import PipelineStateComposer
from .npz_source import IndexedNpzSource
from .planner import DatasetPlanner, PlannerConfig
from .source_dataset import PlannedBatchDataset, SourceBatchDataset
from .stream import reject_duplicate_physical_files


@DATASETS.register("iterative_sparse_numpy")
class IterativeSparseNumpyDataset(PlannedBatchDataset):
    FILE_EXTS = [".npz"]

    def __init__(
        self,
        file_list: list[str],
        boardsizes: set[tuple[int, int]],
        rules: set[str] | None = None,
        fixed_side_input: bool = False,
        fixed_board_size: None | tuple[int, int] = None,
        apply_symmetry: bool = False,
        drop_extra: bool = False,
        shuffle: bool = False,
        sample_rate: float = 1.0,
        batch_size: int | None = None,
        batch_pipelines=(),
        shuffle_window_size: int = 32768,
        shuffle_buffer_bytes: int | None = None,
        steps_per_epoch: int | None = None,
    ):
        super().__init__()
        self.batch_pipelines = tuple(batch_pipelines)
        self.file_list = file_list
        self.boardsizes = boardsizes
        self.fixed_side_input = fixed_side_input
        self.fixed_board_size = fixed_board_size
        self.apply_symmetry = apply_symmetry
        self.drop_extra = drop_extra
        self.shuffle = shuffle
        self.sample_rate = sample_rate
        self.shuffle_window_size = shuffle_window_size
        self.shuffle_buffer_bytes = shuffle_buffer_bytes
        self.steps_per_epoch = steps_per_epoch
    def _build_partitioned_stream(self):
        runtime_context = getattr(self, "runtime_context", None)
        if runtime_context is None:
            raise RuntimeError("iterative_sparse_numpy requires a DatasetRuntimeContext")

        reader = IterativeSparseNumpyDataset(
            file_list=[],
            boardsizes=self.boardsizes,
            fixed_side_input=self.fixed_side_input,
            fixed_board_size=self.fixed_board_size,
            apply_symmetry=False,
            drop_extra=self.drop_extra,
            shuffle=False,
            sample_rate=1.0,
        )

        def load(path):
            with np.load(path, allow_pickle=False) as source:
                return reader._unpack_data(
                    **{key: np.array(source[key], copy=True) for key in source.files}
                )

        decoder = NpzRowRecordDecoder(
            "sparse-katago-npz",
            runtime_context,
            load,
            reader._prepare_entry_data,
            apply_symmetry=self.apply_symmetry,
            catalog_rows=lambda data, length: (
                (
                    None
                    if tuple(data["board_input"].shape[-2:]) in self.boardsizes
                    else np.empty(0, dtype=np.int64)
                ),
                (
                    self.fixed_board_size
                    or tuple(data["board_input"].shape[-2:])
                    if tuple(data["board_input"].shape[-2:]) in self.boardsizes
                    else np.empty((0, 2), dtype=np.int64)
                ),
            ),
            semantic_state={
                "boardsizes": sorted(self.boardsizes),
                "fixed_side_input": self.fixed_side_input,
                "fixed_board_size": self.fixed_board_size,
                "drop_extra": self.drop_extra,
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
                shuffle_buffer_size=self.shuffle_window_size,
                shuffle_buffer_bytes=self.shuffle_buffer_bytes,
                steps_per_epoch=self.steps_per_epoch,
            ),
            pipeline_composer=composer,
        )
        self._planned_decoder = SourceBatchDataset(
            self._partitioned_stream,
            self._record_source,
        )
        return self._partitioned_stream

    def _unpack_global_feature(self, packed_data):
        # Channel 0: side to move (black = -1.0, white = 1.0)
        stm_input = packed_data[:, [0]].astype(np.float32)
        return stm_input

    def _unpack_board_input(self, packed_data):
        length, n_features, n_bytes = packed_data.shape
        bsize = int(np.sqrt(n_bytes * 8))

        # Channel 1: next player stones
        # Channel 2: oppo stones
        packed_data = packed_data[:, [1, 2]]

        board_input_stm = np.unpackbits(packed_data, axis=2, count=bsize * bsize, bitorder="big")
        board_input_stm = board_input_stm.reshape(length, 2, bsize, bsize).astype(np.int8)
        return board_input_stm

    def _unpack_global_target(self, packed_data):
        # Channel 0: stm win probability
        # Channel 1: stm loss probability
        # Channel 2: draw probability
        return packed_data[:, [0, 1, 2]]

    def _unpack_policy_target(self, packed_data):
        length, n_features, n_cells = packed_data.shape
        bsize = int(np.sqrt(n_cells))
        if bsize * bsize != n_cells:
            raise ValueError("packed sparse board mask has an invalid cell count")

        # Channel 0: policy target this turn
        policy_target_stm = packed_data[:, 0, :].reshape(-1, bsize, bsize)
        policy_sum = np.sum(policy_target_stm.astype(np.float32), axis=(1, 2)).reshape(-1, 1, 1)
        policy_target_stm = policy_target_stm / (policy_sum + 1e-7)
        return policy_target_stm

    def _unpack_feature_input(self, data_u8, data_u16):
        length_u8, n_feature_u8, n_cells_u8 = data_u8.shape
        length_u16, n_feature_u16, n_cells_u16 = data_u16.shape
        bsize = int(np.sqrt(n_cells_u8))
        n_feature = n_feature_u8 + n_feature_u16
        if bsize * bsize != n_cells_u8:
            raise ValueError("packed sparse feature has an invalid cell count")
        if length_u8 != length_u16 or n_cells_u8 != n_cells_u16:
            raise ValueError("u8/u16 sparse feature tensors have incompatible shapes")

        feature_input_stm = np.concatenate((data_u8.astype(np.uint16), data_u16), axis=1)
        feature_input_stm = feature_input_stm.reshape(length_u8, n_feature, bsize, bsize)
        return feature_input_stm

    def _unpack_data(
        self,
        binaryInputNCHWPacked,
        globalInputNC,
        globalTargetsNC,
        policyTargetsNCHW,
        sparseInputNCHWU8,
        sparseInputNCHWU16,
        sparseInputDim,
        **kwargs
    ):
        stm_input = self._unpack_global_feature(globalInputNC)
        board_input_stm = self._unpack_board_input(binaryInputNCHWPacked)
        value_target = self._unpack_global_target(globalTargetsNC)
        policy_target = self._unpack_policy_target(policyTargetsNCHW)
        feature_input_stm = self._unpack_feature_input(sparseInputNCHWU8, sparseInputNCHWU16)
        if sparseInputDim.ndim != 1:
            raise ValueError("sparseInputDim must be one-dimensional")
        if sparseInputDim.shape[0] != feature_input_stm.shape[1]:
            raise ValueError("sparseInputDim does not match feature channels")

        return {
            "board_input": board_input_stm,
            "sparse_feature_input": feature_input_stm,
            "sparse_feature_dim": sparseInputDim,
            "stm_input": stm_input,
            "value_target": value_target,
            "policy_target": policy_target,
        }, len(stm_input)

    def _prepare_entry_data(self, data_dict, index):
        data = {}
        for key in data_dict.keys():
            if data_dict[key].ndim == 1:
                data[key] = data_dict[key]  # for data without batch dim, use directly
            else:
                data[key] = data_dict[key][index]

        board_size = tuple(data["board_input"].shape[1:])
        if board_size not in self.boardsizes:
            return None
        data["board_size"] = np.array(board_size, dtype=np.int8)

        data = post_process_data(
            data,
            fixed_side_input=self.fixed_side_input,
            fixed_board_size=self.fixed_board_size,
            symmetry_type=self.apply_symmetry,
            drop_extra=self.drop_extra,
        )
        # Convert unsigned feature storage to PyTorch-compatible signed types.
        if data["sparse_feature_dim"].max() > np.iinfo(np.int32).max:
            raise ValueError("sparse feature dimension exceeds int32")
        data["sparse_feature_input"] = data["sparse_feature_input"].astype(np.int32)
        data["sparse_feature_dim"] = data["sparse_feature_dim"].astype(np.int32)

        return data
