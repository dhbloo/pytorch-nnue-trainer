from torch.utils.data.dataloader import DataLoader
from configargparse import ArgParser
from dataset import build_dataset
import numpy as np
import matplotlib.pyplot as plt


def process_bin(index, result, ply, boardsize, rule, move, position):
    print('-' * 50)
    print(f'index: {index}')
    print(f'result: {result}')
    print(f'ply: {ply}')
    print(f'boardsize: {boardsize}')
    print(f'rule: {rule}')
    print(f'move: {move}')
    print(f'position: {"".join([str(m) for m in position])}')

    # Add process logic here......
    pass


def visualize_entry(board_size,
                    board_input,
                    stm_input=None,
                    policy_target=None,
                    value_target=None,
                    last_move=None,
                    **kwargs):
    H, W = board_size

    fig = plt.figure(figsize=[5, 5])
    fig.patch.set_facecolor((1, 1, 1))
    ax = fig.add_subplot(111)

    # draw the grid
    for x in range(W):
        ax.plot([x, x], [0, H - 1], 'k')
    for y in range(H):
        ax.plot([0, W - 1], [y, y], 'k')

    # scale the axis area to fill the whole figure
    ax.set_position([0, 0, 1, 1])
    # get rid of axes and everything (the figure background will show through)
    ax.set_axis_off()
    # scale the plot area conveniently (the board is in 0,0..18,18)
    ax.set_xlim(-1, W)
    ax.set_ylim(-1, H)

    # draw stones and policy
    for y in range(H):
        for x in range(W):
            if board_input[0, 0, y, x]:
                color = 'k'
                edgewidth = 2
            elif board_input[0, 1, y, x]:
                color = 'w'
                edgewidth = 2
            elif policy_target is not None:
                color = [1, 0, 0, float(policy_target[0, y, x])**0.5]
                edgewidth = 0
            else:
                continue
            ax.plot(x,
                    y,
                    'o',
                    markersize=20,
                    markerfacecolor=color,
                    markeredgecolor=(0, 0, 0),
                    markeredgewidth=edgewidth)

    if last_move is not None:
        ax.plot(*last_move, '+', markersize=10, markeredgecolor='g', markeredgewidth=2)

    texts = []
    if stm_input is not None:
        texts += [f'stm={stm_input[0].item()}({"white" if stm_input[0] > 0 else "black"})']
    if value_target is not None:
        texts += [f'vt={value_target[0].cpu().numpy()}']
    if texts:
        ax.text(0, -0.5, ' '.join(texts))

    plt.show()


def visualize_dataset(dataset):
    dataloader = DataLoader(dataset, batch_size=1)
    for index, data in enumerate(dataloader):
        bs = data['board_size']
        optional_texts = []
        if 'ply' in data:
            optional_texts += [f"ply={data['ply'][0]}"]
        if 'position_string' in data:
            optional_texts += [f"pos={data['position_string']}"]
        print(f"Data Entry[{index}]: bs={(bs[0][0], bs[1][0])} {' '.join(optional_texts)}")
        visualize_entry(**data)


if __name__ == "__main__":
    parser = ArgParser(description="Dataset visualizer")
    parser.add('-c', '--config', is_config_file=True, help='Config file path')
    parser.add('data_paths', nargs='+', help="Dataset file or directory paths")
    parser.add('--dataset_type', required=True, help="Dataset type")
    parser.add('--max_show_num', type=int, default=None, help="Max number of entries to show")
    options = parser.parse_args()
    options_dict = {k: v for k, v in vars(options).items() if v is not None}
    dataset = build_dataset(**options_dict)
    visualize_dataset(dataset)
