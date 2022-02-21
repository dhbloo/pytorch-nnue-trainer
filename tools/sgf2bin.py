import os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

from utils import sgf
from dataset.packed_binary import *
from utils.data_utils import Rule
import argparse
import os
import lz4.frame
from tqdm import tqdm


def list_files_in_dir(dir, exts=None):
    files = []
    for item in os.listdir(dir):
        filepath = os.path.join(dir, item)
        if os.path.isfile(filepath):
            ext = os.path.splitext(item)[1][1:]
            if exts is not None and not ext in exts:
                continue
            files.append(filepath)
    return files


def interleave_lists(list1, list2):
    newlist = []
    a1, a2 = len(list1), len(list2)
    for i in range(max(a1, a2)):
        if i < a1: newlist.append(list1[i])
        if i < a2: newlist.append(list2[i])
    return newlist


def convert_result_bpov(result_text):
    if result_text.startswith('B+'):
        return Result.WIN
    elif result_text.startswith('W+'):
        return Result.LOSS
    else:
        return Result.DRAW


def convert_move(move_text):
    x = ord(move_text[0]) - ord('a')
    y = ord(move_text[1]) - ord('a')
    return Move(x, y)


def write_game_entries(output_f, game, rule):
    rprop = game.root.properties
    boardsize = int(rprop['SZ'][0])
    result_bpov = convert_result_bpov(rprop['RE'][0])
    if 'AW' in rprop:
        position = list(map(convert_move, interleave_lists(rprop['AB'], rprop['AW'])))
    elif 'AB' in rprop:
        position = [convert_move(rprop['AB'][0])]
    else:
        position = []

    node = game.root
    entries_written = 0
    while node.next:
        node = node.next

        ply = len(position)
        side = 'B' if ply % 2 == 0 else 'W'
        result = result_bpov if side == 'B' else Result.opposite(result_bpov)
        move = convert_move(node.properties[side][0])

        write_entry(output_f, result, boardsize, rule, move, position)
        #print(side, result, boardsize, move, position)

        entries_written += 1
        position.append(move)
    return entries_written


def parse_sgf_and_write_entries(output_f, sgf_f, rule):
    collection = sgf.parse(sgf_f.read())
    entries_written = 0
    for game in tqdm(collection):
        entries_written += write_game_entries(output_f, game, rule)
    return entries_written


#------------------------------------------------


def sgf_to_bin():
    parser = argparse.ArgumentParser(description="SGF to BIN training format converter")
    parser.add_argument('indir', type=str, help="Input directory")
    parser.add_argument('output', type=str, help="Output file name")
    parser.add_argument('--rule',
                        default='freestyle',
                        choices=['freestyle', 'standard', 'renju'],
                        help="Rule of this dataset")
    parser.add_argument('--not_compressed',
                        action='store_true',
                        default=False,
                        help="Input is not compressed with lz4")
    parser.add_argument('--no_output_compress',
                        action='store_true',
                        default=False,
                        help="Do not compress output with lz4")
    args = parser.parse_args()

    if args.rule == 'freestyle': rule = Rule.FREESTYLE
    elif args.rule == 'standard': rule = Rule.STANDARD
    elif args.rule == 'renju': rule = Rule.RENJU
    print(f'rule: {rule}')

    def open_file(fn, mode, compress):
        if compress:
            return lz4.frame.open(fn, mode, compression_level=lz4.frame.COMPRESSIONLEVEL_MINHC)
        else:
            return open(fn, mode)

    sgf_files = list_files_in_dir(args.indir, exts=['sgf', 'sgfs'])
    with open_file(args.output, 'wb', not args.no_output_compress) as output_f:
        for sgf_file in sgf_files:
            with open(sgf_file, 'r', not args.not_compressed) as sgf_f:
                entries_written = parse_sgf_and_write_entries(output_f, sgf_f, rule)

    print(f'{entries_written} entries have been written.')


if __name__ == "__main__":
    sgf_to_bin()
