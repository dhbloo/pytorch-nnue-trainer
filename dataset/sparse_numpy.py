import numpy as np
from torch.utils.data.dataset import IterableDataset
from utils.data_utils import Symmetry
import random


class SparseNumpyDataset(IterableDataset):
    def __init__(self,
                 file_list,
                 boardsizes,
                 fixed_side_input,
                 apply_symmetry=False,
                 shuffle=False,
                 sample_rate=1.0,
                 **kwargs):
        super().__init__()
        self.file_list = file_list
        self.boardsizes = boardsizes
        self.fixed_side_input = fixed_side_input
        self.apply_symmetry = apply_symmetry
        self.shuffle = shuffle
        self.sample_rate = sample_rate

    @property
    def is_fixed_side_input(self):
        return self.fixed_side_input

    def _unpack_global_feature(self, packed_data):
        # Channel 0: side to move (black = -1.0, white = 1.0)
        stm_input = packed_data[:, [0]]
        return stm_input

    def _unpack_board_input(self, packed_data):
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
        bsize = int(np.sqrt(n_cells))
        assert bsize * bsize == n_cells

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
        assert bsize * bsize == n_cells_u8
        assert length_u8 == length_u16
        assert n_cells_u8 == n_cells_u16

        feature_input_stm = np.concatenate((data_u8.astype(np.uint16), data_u16), axis=1)
        feature_input_stm = feature_input_stm.reshape(length_u8, n_feature, bsize, bsize)
        return feature_input_stm

    def _unpack_data(self, binaryInputNCHWPacked, globalInputNC, globalTargetsNC,
                     policyTargetsNCHW, sparseInputNCHWU8, sparseInputNCHWU16, sparseInputDim,
                     **kwargs):
        stm_input = self._unpack_global_feature(globalInputNC)
        board_input_stm = self._unpack_board_input(binaryInputNCHWPacked)
        value_target = self._unpack_global_target(globalTargetsNC)
        policy_target = self._unpack_policy_target(policyTargetsNCHW)
        feature_input_stm = self._unpack_feature_input(sparseInputNCHWU8, sparseInputNCHWU16)
        assert sparseInputDim.ndim == 1
        assert sparseInputDim.shape[0] == feature_input_stm.shape[1]

        return {
            'board_input': board_input_stm,
            'sparse_feature_input': feature_input_stm,
            'sparse_feature_dim': sparseInputDim,
            'stm_input': stm_input,
            'value_target': value_target,
            'policy_target': policy_target
        }, len(stm_input)

    def _prepare_entry_data(self, data_dict, index):
        data = {}
        for key in data_dict.keys():
            if data_dict[key].ndim == 1:
                data[key] = data_dict[key]  # for data without batch dim, use directly
            else:
                data[key] = data_dict[key][index]

        data['board_size'] = data['board_input'].shape[1:]
        if data['board_size'] not in self.boardsizes:
            return None
        data['board_size'] = np.array(data['board_size'], dtype=np.int8)

        # Flip side when stm is white
        if self.fixed_side_input and data['stm_input'] == 1:
            data['board_input'] = np.flip(data['board_input'], axis=0).copy()
            data['sparse_feature_input'] = np.take(data['sparse_feature_input'],
                                                   [4, 5, 6, 7, 0, 1, 2, 3, 9, 8, 11, 10],
                                                   axis=0)
            value_target = data['value_target']
            value_target[0], value_target[1] = value_target[1], value_target[0]

        if self.apply_symmetry:
            symmetries = Symmetry.available_symmetries(data['board_size'])
            picked_symmetry = np.random.choice(symmetries)
            data['board_input'] = picked_symmetry.apply_to_array(data['board_input'])
            data['sparse_feature_input'] = picked_symmetry.apply_to_array(
                data['sparse_feature_input'])
            data['policy_target'] = picked_symmetry.apply_to_array(data['policy_target'])

        # convert uint16 to int16, uint32 to int32 due to pytorch requirement
        assert data['sparse_feature_input'].max() <= np.iinfo(np.int16).max
        assert data['sparse_feature_dim'].max() <= np.iinfo(np.int32).max
        data['sparse_feature_input'] = data['sparse_feature_input'].astype(np.int16)
        data['sparse_feature_dim'] = data['sparse_feature_dim'].astype(np.int32)

        return data

    def __iter__(self):
        # randomly shuffle file list
        file_list = [fn for fn in self.file_list]
        if self.shuffle:
            random.shuffle(file_list)

        for filename in file_list:
            data_dict, length = self._unpack_data(**np.load(filename))

            if self.shuffle:
                indices = list(range(length))
                random.shuffle(indices)
            else:
                indices = None

            for idx in range(length):
                # random skip data according to sample rate
                if random.random() >= self.sample_rate:
                    continue

                index = idx if indices is None else indices[idx]
                data = self._prepare_entry_data(data_dict, index)
                if data is not None:
                    yield data