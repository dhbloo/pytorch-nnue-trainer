import numpy as np
import torch.utils.data
from torch.utils.data.dataset import Dataset, IterableDataset
from utils.data_utils import Symmetry, make_subset_range
from . import DATASETS


@DATASETS.register('katago_numpy')
class KatagoNumpyDataset(IterableDataset):
    FILE_EXTS = ['.npz']

    def __init__(self,
                 file_list,
                 boardsizes,
                 fixed_side_input,
                 has_pass_move=False,
                 apply_symmetry=False,
                 shuffle=False,
                 sample_rate=1.0,
                 max_worker_per_file=2,
                 **kwargs):
        super().__init__()
        self.file_list = file_list
        self.boardsizes = boardsizes
        self.fixed_side_input = fixed_side_input
        self.has_pass_move = has_pass_move
        self.apply_symmetry = apply_symmetry
        self.shuffle = shuffle
        self.sample_rate = sample_rate
        self.max_worker_per_file = max_worker_per_file

    @property
    def is_fixed_side_input(self):
        return self.fixed_side_input

    @property
    def is_internal_shuffleable(self):
        return True

    def _unpack_global_feature(self, packed_data):
        # Channel 5: side to move (black = -1.0, white = 1.0)
        stm_input = packed_data[:, [5]]
        return stm_input

    def _unpack_board_feature(self, packed_data):
        length, n_features, n_bytes = packed_data.shape
        bsize = int(np.sqrt(n_bytes * 8))

        # Channel 1: next player stones
        # Channel 2: oppo stones
        packed_data = packed_data[:, [1, 2]]

        board_input_stm = np.unpackbits(packed_data, axis=2, count=bsize * bsize, bitorder='big')
        board_input_stm = board_input_stm.reshape(length, 2, bsize, bsize).astype(np.int8)
        return board_input_stm

    def _unpack_global_target(self, packed_data):
        # Channel 0: stm win probability
        # Channel 1: stm loss probability
        # Channel 2: draw probability
        return packed_data[:, [0, 1, 2]]

    def _unpack_policy_target(self, packed_data):
        length, n_features, n_cells = packed_data.shape
        bsize = int(np.sqrt(n_cells - 1))
        assert bsize * bsize + 1 == n_cells

        # Channel 0: policy target this turn
        policy_target_stm = packed_data[:, 0, :bsize * bsize + (1 if self.has_pass_move else 0)]
        policy_sum = np.sum(policy_target_stm.astype(np.float32), axis=1, keepdims=True)
        policy_target_stm = policy_target_stm / (policy_sum + 1e-7)
        if not self.has_pass_move:
            policy_target_stm = policy_target_stm.reshape(-1, bsize, bsize)
        return policy_target_stm  # [H, W] or [H*W+1] (append pass at last channel)

    def _unpack_data(self, binaryInputNCHWPacked, globalInputNC, globalTargetsNC,
                     policyTargetsNCMove, **kwargs):
        stm_input = self._unpack_global_feature(globalInputNC)
        board_input_stm = self._unpack_board_feature(binaryInputNCHWPacked)
        value_target = self._unpack_global_target(globalTargetsNC)
        policy_target = self._unpack_policy_target(policyTargetsNCMove)

        return {
            'board_input': board_input_stm,
            'stm_input': stm_input,
            'value_target': value_target,
            'policy_target': policy_target
        }, len(stm_input)

    def _prepare_entry_data(self, data_dict, index):
        data = {}
        for key in data_dict.keys():
            data[key] = data_dict[key][index]

        data['board_size'] = data['board_input'].shape[1:]
        if data['board_size'] not in self.boardsizes:
            return None
        data['board_size'] = np.array(data['board_size'], dtype=np.int8)

        # Flip side when stm is white
        if self.fixed_side_input and data['stm_input'] > 0:
            data['board_input'] = np.flip(data['board_input'], axis=0).copy()
            value_target = data['value_target']
            value_target[0], value_target[1] = value_target[1], value_target[0]

        if self.apply_symmetry:
            symmetry_type = self.apply_symmetry if isinstance(self.apply_symmetry, str) else "default"
            symmetries = Symmetry.available_symmetries(data['board_size'], symmetry_type)
            picked_symmetry = np.random.choice(symmetries)
            data['board_input'] = picked_symmetry.apply_to_array(data['board_input'])
            if self.has_pass_move:
                policy_target_sym = data['policy_target'][:-1].reshape((-1, *data['board_size']))
                policy_target_sym = picked_symmetry.apply_to_array(policy_target_sym)
                policy_target_sym = policy_target_sym.reshape((policy_target_sym.shape[0], -1))
                data['policy_target'] = np.append([policy_target_sym, data['policy_target'][-1:]])
            else:
                data['policy_target'] = picked_symmetry.apply_to_array(data['policy_target'])

        return data

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
            data_dict, length = self._unpack_data(**np.load(filename))

            for index in make_subset_range(length,
                                           partition_num=worker_per_file,
                                           partition_idx=worker_id % worker_per_file,
                                           shuffle=self.shuffle,
                                           sample_rate=self.sample_rate):
                data = self._prepare_entry_data(data_dict, index)
                if data is not None:
                    yield data


@DATASETS.register('processed_katago_numpy')
class ProcessedKatagoNumpyDataset(Dataset):
    FILE_EXTS = ['.npz']

    def __init__(self,
                 file_list,
                 boardsizes,
                 fixed_side_input,
                 has_pass_move=False,
                 apply_symmetry=False,
                 filter_stm=None,
                 filter_custom_condition=None,
                 **kwargs):
        super().__init__()
        self.file_list = file_list
        self.boardsizes = boardsizes
        self.fixed_side_input = fixed_side_input
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
            if data["bf"].shape[2:] not in self.boardsizes:
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
        assert length_list.count(self.length) == len(length_list), \
               "Unequal length of data in npz file"

        # Get board size
        self.boardsize = self.data_dict["bf"].shape[2:]
        assert len(self.boardsize) == 2

        if filter_stm is not None:
            self._filter_data_by_side_to_move(filter_stm)
        if filter_custom_condition is not None:
            self._filter_data_by_custom_condition(filter_custom_condition)

        if self.apply_symmetry:
            symmetry_type = self.apply_symmetry if isinstance(self.apply_symmetry, str) else "default"
            self.symmetries = Symmetry.available_symmetries(self.boardsize, symmetry_type)
        else:
            self.symmetries = [Symmetry.IDENTITY]

    @property
    def is_fixed_side_input(self):
        return self.fixed_side_input

    def _filter_data_by_side_to_move(self, side_to_move):
        assert 'gf' in self.data_dict, "No gf data in dataset"
        stm_inputs = self.data_dict['gf'][:, 0]
        selected_indices = np.nonzero(stm_inputs == side_to_move)[0]

        for k in self.data_dict.keys():
            self.data_dict[k] = self.data_dict[k][selected_indices, :]
        self.length = len(next(iter(self.data_dict.values())))

    def _filter_data_by_custom_condition(self, filter_custom_condition):
        try:
            condition = eval(filter_custom_condition, {'np': np, **self.data_dict})
        except Exception as e:
            assert 0, f"Invalid custom condition: {filter_custom_condition}, error: {e}"
        selected_indices = np.nonzero(condition)[0]

        for k in self.data_dict.keys():
            self.data_dict[k] = self.data_dict[k][selected_indices, :]
        self.length = len(next(iter(self.data_dict.values())))

    def _prepare_data(self, index):
        board_size = np.array(self.boardsize, dtype=np.int8)
        board_input = self.data_dict['bf'][index].astype(np.int8)
        if 'gf' in self.data_dict:
            stm_input = self.data_dict['gf'][index].astype(np.int8)
        else:
            stm_input = np.array([0], dtype=np.int8)
        value_target = self.data_dict['vt'][index].astype(np.float32)

        if 'pt' in self.data_dict:
            policy_target = self.data_dict['pt'][index].astype(np.float32)
        else:
            _, h, w = board_input.shape
            if self.has_pass_move:
                policy_target = np.zeros((h * w + 1, ), dtype=np.float32)
            else:
                policy_target = np.zeros((h, w), dtype=np.float32)

        # Ignore pass move for 1d policy target
        if not self.has_pass_move and policy_target.ndim == 1:
            policy_target = policy_target[:-1].reshape(*board_input.shape[1:])

        return {
            'board_size': board_size,
            'board_input': board_input,
            'stm_input': stm_input,
            'value_target': value_target,
            'policy_target': policy_target,
        }

    def __len__(self):
        return self.length * len(self.symmetries)

    def __getitem__(self, index):
        if self.apply_symmetry:
            sym_index = index // self.length
            index = index % self.length

        data = self._prepare_data(index)

        # Flip side when stm is white
        if self.fixed_side_input and data['stm_input'] > 0:
            data['board_input'] = np.flip(data['board_input'], axis=0).copy()
            value_target = data['value_target']
            value_target[0], value_target[1] = value_target[1], value_target[0]

        if self.apply_symmetry:
            picked_symmetry = self.symmetries[sym_index]
            data['board_input'] = picked_symmetry.apply_to_array(data['board_input'])
        if self.apply_symmetry and 'pt' in self.data_dict:
            if self.has_pass_move:
                policy_target_sym = data['policy_target'][:-1].reshape((-1, *data['board_size']))
                policy_target_sym = picked_symmetry.apply_to_array(policy_target_sym)
                policy_target_sym = policy_target_sym.reshape((policy_target_sym.shape[0], -1))
                data['policy_target'] = np.append([policy_target_sym, data['policy_target'][-1:]])
            else:
                data['policy_target'] = picked_symmetry.apply_to_array(data['policy_target'])

        return data


@DATASETS.register('iterative_processed_katago_numpy')
class IterativeProcessedKatagoNumpyDataset(IterableDataset):
    FILE_EXTS = ['.npz']

    def __init__(self,
                 file_list,
                 boardsizes,
                 fixed_side_input,
                 apply_symmetry=False,
                 shuffle=False,
                 sample_rate=1.0,
                 max_worker_per_file=2,
                 **kwargs):
        super().__init__()
        self.file_list = file_list
        self.boardsizes = boardsizes
        self.fixed_side_input = fixed_side_input
        self.apply_symmetry = apply_symmetry
        self.shuffle = shuffle
        self.sample_rate = sample_rate
        self.max_worker_per_file = max_worker_per_file
        self.kwargs = kwargs

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

        for file_index in make_subset_range(len(self.file_list),
                                            partition_num=worker_num // worker_per_file,
                                            partition_idx=worker_id // worker_per_file,
                                            shuffle=self.shuffle):
            filename = self.file_list[file_index]
            dataset = ProcessedKatagoNumpyDataset(file_list=[filename],
                                                  boardsizes=self.boardsizes,
                                                  fixed_side_input=self.fixed_side_input,
                                                  apply_symmetry=self.apply_symmetry,
                                                  **self.kwargs)
            for index in make_subset_range(len(dataset),
                                           partition_num=worker_per_file,
                                           partition_idx=worker_id % worker_per_file,
                                           shuffle=self.shuffle,
                                           sample_rate=self.sample_rate):
                yield dataset[index]