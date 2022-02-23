import numpy as np
from torch.utils.data.dataset import Dataset, IterableDataset
from utils.data_utils import Symmetry
import random


class KatagoNumpyDataset(IterableDataset):
    def __init__(self,
                 file_list,
                 rules,
                 boardsizes,
                 fixed_side_input,
                 apply_symmetry=False,
                 shuffle=False,
                 sample_rate=1.0,
                 **kwargs):
        super().__init__()
        self.file_list = file_list
        self.rules = rules
        self.boardsizes = boardsizes
        self.fixed_side_input = fixed_side_input
        self.apply_symmetry = apply_symmetry
        self.shuffle = shuffle
        self.sample_rate = sample_rate

    @property
    def is_fixed_side_input(self):
        return self.fixed_side_input

    def _unpack_global_feature(self, packed_data):
        # Channel 5: side to move (black = -1, white = 1)
        stm_input = packed_data[:, 5]
        return stm_input

    def _unpack_board_feature(self, packed_data):
        length, n_features, n_bytes = packed_data.shape
        bsize = int(np.sqrt(n_bytes * 8))

        # Channel 1: next player stones
        # Channel 2: oppo stones
        packed_data = packed_data[:, [1, 2]]

        board_input_stm = np.unpackbits(packed_data, axis=2, count=bsize * bsize, bitorder='big')
        board_input_stm = board_input_stm.reshape(length, 2, bsize, bsize)
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
        if self.fixed_side_input and data['stm_input'] == 1:
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
        file_list = [fn for fn in self.file_list]
        if self.shuffle:
            random.shuffle(file_list)
        for filename in file_list:
            data_dict, length = self._unpack_data(**np.load(filename))
            for idx in range(length):
                data = self._prepare_entry_data(data_dict, idx)
                if data is not None and random.random() < self.sample_rate:
                    yield data


class ProcessedKatagoNumpyDataset(Dataset):
    def __init__(self,
                 file_list,
                 rules,
                 boardsizes,
                 fixed_side_input,
                 apply_symmetry=False,
                 **kwargs):
        super().__init__()
        self.file_list = file_list
        self.rules = rules
        self.boardsizes = boardsizes
        self.fixed_side_input = fixed_side_input
        self.apply_symmetry = apply_symmetry

        self.tensor_lists = {
            "bf": [],
            "gf": [],
            "vt": [],
            "pt": [],
        }
        for filename in self.file_list:
            data = np.load(filename)
            for k, tensor in data.items():
                self.tensor_lists[k].append(tensor)
        for k, tensor_list in self.tensor_lists.items():
            self.tensor_lists[k] = np.concatenate(tensor_list, axis=0)

        print(self.tensor_lists["bf"].shape)
        self.boardsize = self.tensor_lists["bf"].shape[1:3]
        if self.apply_symmetry:
            self.symmetries = Symmetry.available_symmetries(self.boardsize)
        else:
            self.symmetries = [Symmetry.IDENTITY]

    @property
    def is_fixed_side_input(self):
        return self.fixed_side_input

    def __len__(self):
        return len(self.tensor_lists) * len(self.symmetries)

    def __getitem__(self, index):
        raise NotImplementedError()
