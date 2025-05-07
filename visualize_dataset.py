import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import configargparse
import yaml
import os
from dataset import build_dataset
from utils.file_utils import make_dir
from utils.misc_utils import seed_everything
from utils.training_utils import build_data_loader


def visualize_entry(
    board_size,
    board_input,
    stm_input,
    value_target,
    policy_target,
    last_move=None,
    forbidden_point=None,
    raw_eval=None,
    fixed_side_input=False,
    save_fig_path=None,
    **kwargs,
):
    H, W = board_size
    markersize = 300 / max(H, W)

    if not fixed_side_input and stm_input > 0:
        board_input = board_input[[1, 0]]
        value_target = value_target[[1, 0, 2]]

    if policy_target.ndim == 1:
        pass_target = policy_target[-1]
        policy_target = policy_target[:-1].reshape(H, W)
    elif policy_target.ndim == 2:
        pass_target = None
    else:
        raise ValueError(f"Invalid policy_target, ndim={policy_target.ndim}")

    fig = plt.figure(figsize=(5, 5))
    fig.patch.set_facecolor((1, 1, 1))
    ax = fig.add_subplot(111)

    # draw the grid
    for x in range(W):
        ax.plot([x, x], [0, H - 1], "k")
    for y in range(H):
        ax.plot([0, W - 1], [y, y], "k")

    # scale the axis area to fill the whole figure
    ax.set_position((0, 0, 1, 1))
    # get rid of axes and everything (the figure background will show through)
    ax.set_axis_off()
    # scale the plot area conveniently (the board is in 0,0..H,W)
    ax.set_xlim(-1, W)
    ax.set_ylim(-1, H)

    # draw stones and policy
    for y in range(H):
        for x in range(W):
            if board_input[0, y, x]:
                color = "k"
                edgewidth = 2
            elif board_input[1, y, x]:
                color = "w"
                edgewidth = 2
            elif policy_target is not None:
                color = [1, 0, 0, float(policy_target[y, x]) ** 0.5]
                edgewidth = 0
            else:
                continue
            ax.plot(
                x,
                y,
                "o",
                markersize=markersize,
                markerfacecolor=color,
                markeredgecolor=(0, 0, 0),
                markeredgewidth=edgewidth,
            )

    # highlight last move if exists
    if last_move is not None:
        print(last_move)
        ax.plot(
            *last_move,
            "+",
            markersize=markersize / 2,
            markeredgecolor="g",
            markeredgewidth=2,
        )

    # plot forbidden points if exists
    if forbidden_point is not None:
        for y in range(H):
            for x in range(W):
                if forbidden_point[y, x] != 0:
                    ax.plot(
                        x,
                        y,
                        "x",
                        markersize=markersize / 2,
                        markeredgecolor="r",
                        markeredgewidth=3,
                    )

    # plot pass move if exists
    if pass_target is not None and pass_target > 0:
        ax.plot(
            W - 1,
            -0.5,
            "o",
            markersize=markersize * 0.5,
            markerfacecolor=[1, 0, 0, float(pass_target) ** 0.5],
            markeredgecolor=(0, 0, 0),
            markeredgewidth=0,
        )
        ax.text(W - 2.6, -0.66, "(pass)")

    # =============================================================================
    # Add horizontal probability bar
    #
    # We assume that value_target is a 3-element array in the order:
    #   [win, loss, draw]  (after the potential swap above).
    # For the bar we want segments in the order: win (green), draw (yellow), loss (red).
    #
    # The bar is drawn along the full board width (from x=0 to x=W) at a fixed y location
    # (in this example, from y=-0.3 to y=-0.2). Adjust these values as desired.
    # =============================================================================
    if value_target is not None:
        bar_height = 0.3
        bar_y = H - 0.6
        rect_width = (W - 3) * value_target[[0, 2, 1]]
        x_start = 1 + np.cumsum(rect_width)

        ax.text(0, bar_y, "VT")
        ax.text(W - 1.6, bar_y, ["B", "W", "D"][np.argmax(value_target)])
        ax.add_patch(
            patches.Rectangle((1, bar_y), rect_width[0], bar_height, facecolor="black", edgecolor="black")
        )
        ax.add_patch(
            patches.Rectangle(
                (x_start[0], bar_y), rect_width[1], bar_height, facecolor="gray", edgecolor="black"
            )
        )
        ax.add_patch(
            patches.Rectangle(
                (x_start[1], bar_y), rect_width[2], bar_height, facecolor="white", edgecolor="black"
            )
        )

    # =============================================================================
    # Continue with the rest of the drawing (pass move marker and texts)
    # =============================================================================

    texts = []
    if stm_input is not None:
        texts += [f'stm={stm_input}({"white" if stm_input > 0 else "black"})']
        if value_target is not None:
            texts += [f"vt={value_target}(B/W/D)"]
        if raw_eval is not None and raw_eval == raw_eval:  # raw_eval is not nan
            texts += [f"eval={raw_eval}"]
    if texts:
        ax.text(0, -0.7, " ".join(texts))

    if save_fig_path is not None:
        plt.savefig(save_fig_path)
    plt.show()
    plt.close(fig)


def visualize_dataset(dataset, dataloader, max_entries=0, save_fig_dir=None):
    get_save_fig_path = lambda i: None
    if save_fig_dir is not None:
        make_dir(save_fig_dir)
        get_save_fig_path = lambda i: os.path.join(save_fig_dir, f"{i}.png")
    index = 0
    for batch_data in dataloader:
        for batch_idx in range(len(batch_data["board_size"])):
            data = {
                k: v[batch_idx].numpy() if isinstance(v[batch_idx], torch.Tensor) else v[batch_idx]
                for k, v in batch_data.items()
            }
            display_texts = [f"bs={tuple(data['board_size'])}"]
            if "rule_index" in data:
                display_texts += [f"rule={data['rule_index']}"]
            if "stm_input" in data:
                display_texts += [f"stm={data['stm_input']}"]
            if "value_target" in data:
                display_texts += [f"vt={data['value_target']}"]
            if "ply" in data:
                display_texts += [f"ply={data['ply']}"]
            if "last_move" in data:
                display_texts += [f"lastmove={data['last_move']}"]
            if "position_string" in data:
                display_texts += [f"pos={data['position_string']}"]
            print(f"Data Entry[{index}]: {' '.join(display_texts)}")
            visualize_entry(
                **data,
                fixed_side_input=dataset.is_fixed_side_input,
                save_fig_path=get_save_fig_path(index),
            )
            index += 1
            if max_entries > 0 and index >= max_entries:
                return


if __name__ == "__main__":
    parser = configargparse.ArgParser(
        description="Dataset visualizer", config_file_parser_class=configargparse.YAMLConfigFileParser
    )
    parser.add("data_paths", nargs="+", help="Dataset file or directory paths")
    parser.add("--dataset_type", required=True, help="Dataset type")
    parser.add("--dataset_args", type=yaml.safe_load, default={}, help="Extra dataset arguments")
    parser.add("--dataloader_args", type=yaml.safe_load, default={}, help="Extra dataloader arguments")
    parser.add(
        "--data_pipelines", type=yaml.safe_load, default=None, help="Data-pipeline type and arguments"
    )
    parser.add("--shuffle", action="store_true", help="Shuffle dataset")
    parser.add("--seed", type=int, default=42, help="Random seed")
    parser.add("--batch_size", type=int, default=1, help="Batch size")
    parser.add("--num_worker", type=int, default=0, help="Number of workers")
    parser.add("--max_entries", type=int, default=30, help="Max entries to visualize (0 for all)")
    parser.add("--save_fig_dir", type=str, default=None, help="Path to save figure")
    args = parser.parse_args()
    seed_everything(args.seed)

    np.set_printoptions(precision=4)
    dataset = build_dataset(
        args.dataset_type,
        args.data_paths,
        shuffle=args.shuffle,
        pipeline_args=args.data_pipelines,
        **args.dataset_args,
    )
    dataloader = build_data_loader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_worker,
        shuffle=args.shuffle,
        **args.dataloader_args,
    )
    visualize_dataset(
        dataset,
        dataloader,
        max_entries=args.max_entries,
        save_fig_dir=args.save_fig_dir,
    )
