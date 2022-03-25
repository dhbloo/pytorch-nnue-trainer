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
        policy_target_stm = packed_data[:, 0, :bsize * bsize].reshape(-1, bsize, bsize)
        policy_sum = np.sum(policy_target_stm.astype(np.float32), axis=(1, 2)).reshape(-1, 1, 1)
        policy_target_stm = policy_target_stm / (policy_sum + 1e-7)
        return policy_target_stm

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
            symmetries = Symmetry.available_symmetries(data['board_size'])
            picked_symmetry = np.random.choice(symmetries)
            data['board_input'] = picked_symmetry.apply_to_array(data['board_input'])
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

    def __init__(self, file_list, boardsizes, fixed_side_input, apply_symmetry=False, **kwargs):
        super().__init__()
        self.file_list = file_list
        self.boardsizes = boardsizes
        self.fixed_side_input = fixed_side_input
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

            for k, tensor in data.items():
                if k in self.data_dict:
                    self.data_dict[k].append(tensor)

        # Concatenate tensors across files
        length_list = []
        for k, tensor_list in self.data_dict.items():
            if len(tensor_list) > 1:
                self.data_dict[k] = np.concatenate(tensor_list, axis=0)
            elif len(tensor_list) > 0:
                self.data_dict[k] = tensor_list[0]
            length_list.append(len(self.data_dict[k]))

        # Get length of dataset and assert length are equal for all keys
        self.length = length_list[0]
        assert length_list.count(self.length) == len(length_list), \
               "Unequal length of data in npz file"

        # Get board size
        self.boardsize = self.data_dict["bf"].shape[2:]
        assert len(self.boardsize) == 2

        if self.apply_symmetry:
            self.symmetries = Symmetry.available_symmetries(self.boardsize)
        else:
            self.symmetries = [Symmetry.IDENTITY]

    @property
    def is_fixed_side_input(self):
        return self.fixed_side_input

    def __len__(self):
        return self.length * len(self.symmetries)

    def __getitem__(self, index):
        if self.apply_symmetry:
            sym_index = index // self.length
            index = index % self.length

        data = {
            'board_size': np.array(self.boardsize, dtype=np.int8),
            'board_input': self.data_dict['bf'][index].astype(np.int8),
            'stm_input': self.data_dict['gf'][index].astype(np.float32),
            'value_target': self.data_dict['vt'][index].astype(np.float32),
            'policy_target': self.data_dict['pt'][index].astype(np.float32),
        }

        # Flip side when stm is white
        if self.fixed_side_input and data['stm_input'] > 0:
            data['board_input'] = np.flip(data['board_input'], axis=0).copy()
            value_target = data['value_target']
            value_target[0], value_target[1] = value_target[1], value_target[0]

        if self.apply_symmetry:
            symmetries = Symmetry.available_symmetries(self.boardsize)
            picked_symmetry = symmetries[sym_index]
            data['board_input'] = picked_symmetry.apply_to_array(data['board_input'])
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
                                                  apply_symmetry=self.apply_symmetry)
            for index in make_subset_range(len(dataset),
                                           partition_num=worker_per_file,
                                           partition_idx=worker_id % worker_per_file,
                                           shuffle=self.shuffle,
                                           sample_rate=self.sample_rate):
                yield dataset[index]