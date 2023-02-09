import torch
import numpy as np
from accelerate import Accelerator
import configargparse
import yaml
import os

from model import build_model


def parse_args_and_init():
    parser = configargparse.ArgParser(description="Test Debug",
                                      config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add('-c', '--config', is_config_file=True, help='Config file path')
    parser.add('-p', '--checkpoint', required=True, help="Model checkpoint file to test")
    parser.add('--use_cpu', action='store_true', help="Use cpu only")
    parser.add('--model_type', required=True, help="Model type")
    parser.add('--model_args', type=yaml.safe_load, default={}, help="Extra model arguments")
    parser.add('--board_width', type=int, default=15, help="Board width")
    parser.add('--board_height', type=int, default=15, help="Board height")

    args, _ = parser.parse_known_args()  # parse args
    parser.print_values()  # print out values
    print('-' * 60)

    return args


class Board():
    def __init__(self, board_width, board_height, fixed_side_input=False):
        self.board = np.zeros((2, board_height, board_width), dtype=np.int8)
        self.side_to_move = 0
        self.move_history = []
        self.fixed_side_input = fixed_side_input

    @property
    def ply(self):
        return len(self.move_history)

    @property
    def width(self):
        return self.board.shape[2]

    @property
    def height(self):
        return self.board.shape[1]

    def flip_side(self):
        self.side_to_move = 1 - self.side_to_move

    def move(self, x, y):
        assert self.is_legal(x, y), "Pos is not legal!"
        self.board[self.side_to_move, y, x] = 1
        self.move_history.append((x, y, self.side_to_move))
        self.flip_side()

    def undo(self):
        assert len(self.move_history) > 0, "Can not undo when board is empty!"
        x, y, stm = self.move_history.pop()
        self.board[stm, y, x] = 0
        self.side_to_move = stm

    def is_legal(self, x, y):
        if x < 0 or x >= self.board.shape[2] or y < 0 or y >= self.board.shape[1]:
            return False
        return self.board[0, y, x] == 0 and self.board[1, y, x] == 0

    def get_data(self):
        if not self.fixed_side_input and self.side_to_move == 1:
            board_input = np.flip(self.board, axis=0).copy()
        else:
            board_input = self.board

        return {
            'board_size': torch.tensor(self.board.shape, dtype=torch.int8),
            'board_input': torch.from_numpy(board_input),
            'stm_input': torch.FloatTensor([-1 if self.side_to_move == 0 else 1])
        }

    def __str__(self):
        s = '   '
        for x in range(self.board.shape[2]):
            s += chr(x + ord('A')) + ' '
        s += '\n'
        for y in range(self.board.shape[1]):
            s += f'{y + 1:2d} '
            for x in range(self.board.shape[2]):
                if self.board[0, y, x]:
                    s += 'X '
                elif self.board[1, y, x]:
                    s += 'O '
                else:
                    s += '. '
            s += '\n'
        return s


def debug_print(board, model, data):
    # add batch dimension
    for k in data:
        if isinstance(data[k], torch.Tensor):
            data[k] = torch.unsqueeze(data[k], dim=0)

    if hasattr(model, 'forward_debug_print'):
        torch.set_printoptions(precision=3, linewidth=120, sci_mode=False)
        value, policy, *retvals = model.forward_debug_print(data)
    else:
        value, policy, *retvals = model(data)

    # remove batch dimension
    value = value.squeeze(0)
    policy = policy.squeeze(0)
    with np.printoptions(precision=2, linewidth=120, suppress=True):
        print(f'Raw Value: {value.cpu().numpy()}')
        print(f'Raw Policy: \n{policy.cpu().numpy()}')

    # apply activation function
    value = torch.softmax(value, dim=0)
    policy = torch.softmax(policy.flatten(), dim=0).reshape(policy.shape)
    with np.printoptions(precision=2, linewidth=120, suppress=True):
        print(f'Softmaxed Value: {value.cpu().numpy()}')
        print(f'Softmaxed Policy: \n{policy.cpu().numpy()}')


def input_move():
    input_str = input("Input your move: ")
    x, y = input_str[0].upper(), input_str[1:]
    return ord(x) - ord('A'), int(y) - 1


def test_play(checkpoint, use_cpu, model_type, model_args, board_width, board_height, **kwargs):
    if not os.path.exists(checkpoint) or not os.path.isfile(checkpoint):
        raise RuntimeError(f'Checkpoint {checkpoint} must be a valid file')

    # use accelerator
    accelerator = Accelerator(cpu=use_cpu)

    # build model
    model = build_model(model_type, **model_args)

    # load checkpoint if exists
    state_dicts = torch.load(checkpoint, map_location=accelerator.device)
    model.load_state_dict(state_dicts['model'])
    epoch, it = state_dicts.get('epoch', 0), state_dicts.get('iteration', 0)
    accelerator.print(f'Loaded from checkpoint: {checkpoint}, epoch: {epoch}, it: {it}')

    # accelerate model testing
    model = accelerator.prepare(model)

    # test play loop
    board = Board(board_width, board_height)

    with torch.no_grad():
        model.eval()
        while board.ply + 1 < board.width * board.height:
            print(board)

            data = board.get_data()
            for key in data.keys():
                data[key] = data[key].to(accelerator.device)
            debug_print(board, model, data)

            move = input_move()
            board.move(*move)


if __name__ == "__main__":
    args = parse_args_and_init()
    test_play(**vars(args))
