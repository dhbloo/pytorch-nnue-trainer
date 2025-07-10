import os
import yaml
import torch
import numpy as np
import configargparse
from accelerate import PartialState
from accelerate.utils import send_to_device

from model import build_model
from dataset.pipeline import build_data_pipeline
from utils.file_utils import load_torch_ckpt
from utils.misc_utils import deep_update_dict


def parse_args_and_init():
    parser = configargparse.ArgParser(
        description="Test Play (and Debug)",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
    )
    parser.add("-c", "--config", is_config_file=True, help="Config file path")
    parser.add("-p", "--checkpoint", required=True, help="Model checkpoint file to test")
    parser.add("--debug", action="store_true", help="Enable debug printing")
    parser.add("--use_cpu", action="store_true", help="Use cpu only")
    parser.add("--model_type", required=True, help="Model type")
    parser.add("--model_args", type=yaml.safe_load, default={}, help="Extra model arguments")
    parser.add("--test_model_args", type=yaml.safe_load, default={}, help="Override model args for testing")
    parser.add("--board_width", type=int, default=15, help="Board width")
    parser.add("--board_height", type=int, default=15, help="Board height")
    parser.add("--dataset_args", type=yaml.safe_load, default={}, help="Extra dataset arguments")
    parser.add("--data_pipelines", type=yaml.safe_load, default={}, help="Data-pipeline arguments")

    args, _ = parser.parse_known_args()  # parse args

    if PartialState(cpu=args.use_cpu).is_local_main_process:
        parser.print_values()
        print("-" * 60)

    return args


class Board:
    BLACK_PLANE = 0
    WHITE_PLANE = 1

    def __init__(self, board_width: int, board_height: int, dataset_args: dict, data_pipeline_args: dict):
        self.board = np.zeros((2, board_height, board_width), dtype=np.int8)
        self.side_to_move = self.BLACK_PLANE
        self.move_history = []
        self.fixed_side_input = dataset_args.get("fixed_side_input", False)
        self.data_pipelines = build_data_pipeline(data_pipeline_args)

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
        if not self.is_legal(x, y):
            raise ValueError("Pos is not legal!")
        self.board[self.side_to_move, y, x] = 1
        self.move_history.append((x, y, self.side_to_move))
        self.flip_side()

    def undo(self):
        if len(self.move_history) == 0:
            raise ValueError("Can not undo when board is empty!")
        x, y, side_to_move = self.move_history.pop()
        self.board[side_to_move, y, x] = 0
        self.side_to_move = side_to_move

    def is_legal(self, x, y):
        if x < 0 or x >= self.board.shape[2] or y < 0 or y >= self.board.shape[1]:
            return False
        return self.board[0, y, x] == 0 and self.board[1, y, x] == 0

    def get_data(self):
        if not self.fixed_side_input and self.side_to_move == 1:
            board_input = np.flip(self.board, axis=0).copy()
        else:
            board_input = self.board.copy()

        data = {
            "board_input": board_input,
            "board_size": np.array(self.board.shape, dtype=np.int8),
            "stm_input": np.array([-1 if self.side_to_move == 0 else 1], dtype=np.float32),
        }

        for pipeline in self.data_pipelines:
            data = pipeline(data)

        data = {
            k: torch.from_numpy(v) if isinstance(v, torch.Tensor) else torch.tensor(v)
            for k, v in data.items()
        }
        return data

    def __str__(self):
        s = "   "
        for x in range(self.board.shape[2]):
            s += chr(x + ord("A")) + " "
        s += "\n"
        for y in range(self.board.shape[1]):
            s += f"{y + 1:2d} "
            for x in range(self.board.shape[2]):
                if self.board[0, y, x]:
                    s += "X "
                elif self.board[1, y, x]:
                    s += "O "
                else:
                    s += ". "
            s += "\n"
        return s


def next_move(board, model, data):
    # add batch dimension
    for k in data:
        if isinstance(data[k], torch.Tensor):
            data[k] = torch.unsqueeze(data[k], dim=0)

    with torch.no_grad():
        value, policy, *retvals = model(data)

    # remove batch dimension
    value = value.squeeze(0)
    policy = policy.squeeze(0)

    # apply activation function
    value = torch.softmax(value, dim=0)
    policy = torch.softmax(policy.flatten(), dim=0)

    # calc winrate, drawrate and best valid move
    winrate = (value[0] - value[1] + 1) / 2
    drawrate = value[2]

    sorted_policy, sorted_moves = torch.sort(policy, descending=True)
    bestmove = None
    bestmove_policy = None
    for move, move_policy in zip(sorted_moves, sorted_policy):
        move = move.cpu().item()
        move_y, move_x = divmod(move, board.width)
        if board.is_legal(move_x, move_y):
            bestmove = move
            bestmove_policy = move_policy.cpu().item()
            break
    bestmove_y, bestmove_x = divmod(bestmove, board.width)

    return winrate, drawrate, (bestmove_x, bestmove_y), bestmove_policy


def debug_print(board, model, data):
    # add batch dimension
    for k in data:
        if isinstance(data[k], torch.Tensor):
            data[k] = torch.unsqueeze(data[k], dim=0)

    # get predicted value and policy from model results
    if hasattr(model, "forward_debug_print"):
        torch.set_printoptions(precision=4, linewidth=120, sci_mode=False)
        with torch.no_grad():
            value, policy, *retvals = model.forward_debug_print(data)
    else:
        with torch.no_grad():
            value, policy, *retvals = model(data)
    aux_losses = retvals[0] if len(retvals) >= 1 else None
    aux_outputs = retvals[1] if len(retvals) >= 2 else None

    def print_value(value: torch.Tensor) -> torch.Tensor:
        value = value.squeeze(0)  # remove batch dimension

        # apply activation function
        if value.shape[0] == 1:
            value_softmaxed = torch.sigmoid(value)
        elif value.shape[0] == 3:
            value_softmaxed = torch.softmax(value, dim=0)
        else:
            raise ValueError(f"Invalid value shape: {value.shape}")

        with np.printoptions(precision=4, linewidth=120, suppress=True):
            print(f"Raw Value: {value.cpu().numpy()}")
        with np.printoptions(precision=4, linewidth=120, suppress=True):
            print(f"Softmaxed Value: {value_softmaxed.cpu().numpy()}")

    def print_policy(policy: torch.Tensor) -> torch.Tensor:
        policy = policy.squeeze(0)  # remove batch dimension
        policy_softmaxed = torch.softmax(policy.flatten(), dim=0).reshape_as(policy)

        with np.printoptions(precision=2, linewidth=120, suppress=True):
            print(f"Raw Policy: \n{policy.cpu().numpy()}")
        with np.printoptions(precision=2, linewidth=120, suppress=True):
            print(f"Softmaxed Policy: \n{policy_softmaxed.cpu().numpy()}")

    print("=====Main Value Output=====")
    print_value(value)
    print("=====Main Policy Output=====")
    print_policy(policy)

    if aux_losses:
        for aux_name, aux_loss in aux_losses.items():
            if isinstance(aux_loss, tuple) and len(aux_loss) == 2:
                aux_loss_type, aux_loss_input = aux_loss
                if aux_loss_type == "value_loss":
                    print(f"=====Aux Value Output: {aux_name}=====")
                    print_value(aux_loss_input)
                elif aux_loss_type == "policy_loss":
                    print(f"=====Aux Policy Output: {aux_name}=====")
                    print_policy(aux_loss_input)


def input_move():
    input_str = input("Input your move (empty for AI move): ")
    if input_str == "":
        return None
    x, y = input_str[0].upper(), input_str[1:]
    return ord(x) - ord("A"), int(y) - 1


def output_move(move):
    return f"{chr(move[0] + ord('A'))}{move[1] + 1}"


def test_play(
    checkpoint,
    debug,
    use_cpu,
    model_type,
    model_args,
    test_model_args,
    board_width,
    board_height,
    dataset_args,
    data_pipelines,
    **kwargs,
):
    if not os.path.exists(checkpoint) or not os.path.isfile(checkpoint):
        raise RuntimeError(f"Checkpoint {checkpoint} must be a valid file")

    # build model
    if test_model_args:
        model_args = deep_update_dict(model_args, test_model_args)
    model = build_model(model_type, **model_args)

    # load checkpoint if exists
    model_state_dict, _, metadata = load_torch_ckpt(checkpoint)
    model.load_state_dict(model_state_dict)
    epoch, it = metadata.get("epoch", "?"), metadata.get("iteration", "?")
    print(f"Loaded from {checkpoint}, epoch: {epoch}, it: {it}")

    # move model to device
    device = PartialState(cpu=use_cpu).device
    print(f"Test playing on device: {device}")
    model.to(device)
    model.eval()

    # construct the test board
    board = Board(board_width, board_height, dataset_args, data_pipelines)

    # test play loop
    while board.ply + 1 < board.width * board.height:
        print(board)

        if debug:
            data = board.get_data()
            data = send_to_device(data, device)
            debug_print(board, model, data)

        move = input_move()
        if move is None:
            data = board.get_data()
            data = send_to_device(data, device)
            winrate, drawrate, move, move_policy = next_move(board, model, data)
            print(
                f"winrate: {winrate:.4f}, drawrate: {drawrate:.4f}, "
                f"move: {output_move(move)} (prob={move_policy:.4f})"
            )
        board.move(*move)


if __name__ == "__main__":
    args = parse_args_and_init()
    test_play(**vars(args))
