import numpy as np
import torch.utils.data
from torch.utils.data.dataset import Dataset, IterableDataset
from utils.data_utils import make_subset_range, post_process_data, filter_data_by_condition
from . import DATASETS


@DATASETS.register("katago_numpy")
class KatagoNumpyDataset(Dataset):
    FILE_EXTS = [".npz"]

    def __init__(
        self,
        file_list: list[str],
        boardsizes: set[tuple[int, int]],
        fixed_side_input: bool = False,
        fixed_board_size: None | tuple[int, int] = None,
        has_pass_move: bool = False,
        apply_symmetry: bool = False,
        filter_stm: int | None = None,
        filter_condition: str | None = None,
        shuffle: bool = False,
        sample_rate: float = 1.0,
        max_worker_per_file: int = 1,
        value_td_level: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.file_list = file_list
        self.boardsizes = boardsizes
        self.fixed_side_input = fixed_side_input
        self.fixed_board_size = fixed_board_size
        self.has_pass_move = has_pass_move
        self.apply_symmetry = apply_symmetry
        self.shuffle = shuffle
        self.sample_rate = sample_rate
        self.max_worker_per_file = max_worker_per_file
        self.filter_stm = filter_stm
        self.filter_condition = filter_condition
        self.value_td_level = value_td_level
        assert filter_stm is None or isinstance(
            filter_stm, int
        ), "filter_stm should be an integer, eg. (-1 for black and 1 for white)"

        self.data_dict = {}

        # Load data tensors from npz files
        length_list = []
        for filename in self.file_list:
            data = np.load(filename)
            data_dict, length = self._unpack_data(data)
            length_list.append(length)

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

        # Get length of dataset and assert length are equal for all keys
        self.length = length_list[0]
        assert self.length > 0, f"No valid data entry in dataset: {self.file_list}"
        assert length_list.count(self.length) == len(length_list), "Unequal length of data in npz file"

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
        assert bsize * bsize + 1 == n_cells

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
                cond = f"globalInputNC[:, 0] == {self.filter_stm}"
            else:
                cond = f"globalInputNC[:, 5] > 0" if self.filter_stm == 1 else f"globalInputNC[:, 5] < 0"
            filter_data_by_condition(cond, raw_data_dict)
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
        return post_process_data(data, self.fixed_side_input, self.fixed_board_size, self.apply_symmetry)


@DATASETS.register("iterative_katago_numpy")
class IterativeKatagoNumpyDataset(IterableDataset):
    """
    Similar to KatagoNumpyDataset but with iterative loading.
    This is useful when the dataset is too large to fit into memory.
    """

    FILE_EXTS = [".npz"]

    def __init__(
        self,
        file_list: list[str],
        boardsizes: set[tuple[int, int]],
        fixed_side_input: bool = False,
        shuffle: bool = False,
        sample_rate: float = 1.0,
        max_worker_per_file: int = 1,
        **kwargs,
    ):
        self.file_list = file_list
        self.boardsizes = boardsizes
        self.fixed_side_input = fixed_side_input
        self.shuffle = shuffle
        self.sample_rate = sample_rate
        self.max_worker_per_file = max_worker_per_file
        self.extra_kwargs = kwargs

    @property
    def is_fixed_side_input(self):
        return self.fixed_side_input

    @property
    def is_internal_shuffleable(self):
        return True

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
            dataset = KatagoNumpyDataset(
                file_list=[filename],
                boardsizes=self.boardsizes,
                fixed_side_input=self.fixed_side_input,
                **self.extra_kwargs,
            )
            for index in make_subset_range(
                len(dataset),
                partition_num=worker_per_file,
                partition_idx=worker_id % worker_per_file,
                shuffle=self.shuffle,
                sample_rate=self.sample_rate,
            ):
                yield dataset[index]


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
        fixed_side_input: bool = False,
        fixed_board_size: None | tuple[int, int] = None,
        has_pass_move: bool = False,
        apply_symmetry: bool = False,
        filter_stm: int | None = None,
        filter_condition: str | None = None,
        **kwargs,
    ):
        super().__init__()
        self.file_list = file_list
        self.boardsizes = boardsizes
        self.fixed_side_input = fixed_side_input
        self.fixed_board_size = fixed_board_size
        self.has_pass_move = has_pass_move
        self.apply_symmetry = apply_symmetry

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
                    assert len(data[k]) > 0, f"Empty tensor {k} in file {filename}"
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

        # Get length of dataset and assert length are equal for all keys
        self.length = length_list[0]
        assert self.length > 0, f"No valid data entry in dataset: {self.file_list}"
        assert length_list.count(self.length) == len(length_list), "Unequal length of data in npz file"

        # Get board size
        self.boardsize = self.data_dict["bf"].shape[2:]
        assert len(self.boardsize) == 2

        if filter_stm is not None:
            assert isinstance(
                filter_stm, int
            ), "filter_stm should be an integer, eg. (-1 for black and 1 for white)"
            assert "gf" in self.data_dict, "gf tensor is required for filtering stm"
            self.length = filter_data_by_condition(f"gf[:, 0] == {filter_stm}", self.data_dict)
        if filter_condition is not None:
            self.length = filter_data_by_condition(filter_condition, self.data_dict)

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
        return post_process_data(data, self.fixed_side_input, self.fixed_board_size, self.apply_symmetry)


@DATASETS.register("iterative_processed_katago_numpy")
class IterativeProcessedKatagoNumpyDataset(IterableDataset):
    """
    Similar to ProcessedKatagoNumpyDataset but with iterative loading.
    This is useful when the dataset is too large to fit into memory.
    """

    FILE_EXTS = [".npz"]

    def __init__(
        self,
        file_list: list[str],
        boardsizes: set[tuple[int, int]],
        fixed_side_input: bool = False,
        shuffle: bool = False,
        sample_rate: float = 1.0,
        max_worker_per_file: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.file_list = file_list
        self.boardsizes = boardsizes
        self.fixed_side_input = fixed_side_input
        self.shuffle = shuffle
        self.sample_rate = sample_rate
        self.max_worker_per_file = max_worker_per_file
        self.extra_kwargs = kwargs

    @property
    def is_fixed_side_input(self):
        return self.fixed_side_input

    @property
    def is_internal_shuffleable(self):
        return True

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
            dataset = ProcessedKatagoNumpyDataset(
                file_list=[filename],
                boardsizes=self.boardsizes,
                fixed_side_input=self.fixed_side_input,
                **self.extra_kwargs,
            )
            for index in make_subset_range(
                len(dataset),
                partition_num=worker_per_file,
                partition_idx=worker_id % worker_per_file,
                shuffle=self.shuffle,
                sample_rate=self.sample_rate,
            ):
                yield dataset[index]
