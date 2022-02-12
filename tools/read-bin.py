import os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

from utils.bin import *
import argparse
import lz4.frame


def process_bin(index, result, ply, boardsize, rule, move, position):
    print('-' * 50)
    print(f'index: {index}')
    print(f'result: {result}')
    print(f'ply: {ply}')
    print(f'boardsize: {boardsize}')
    print(f'rule: {rule}')
    print(f'move: {move}')
    print(f'position: {"".join([m.to_pos() for m in position])}')

    # Add process logic here......
    pass


def read_bin():
    parser = argparse.ArgumentParser(description="BIN training format reader")
    parser.add_argument('input', type=str, help="Input binary file name")
    parser.add_argument('--not_compressed',
                        action='store_true',
                        default=False,
                        help="Input is not compressed with lz4")
    parser.add_argument('--max_show_num',
                        type=int,
                        default=10,
                        help="Max number of entries to show")
    parser.add_argument('--show_indices', nargs='+', type=int, help='Show indices', required=False)
    args = parser.parse_args()

    def open_file(fn, mode):
        if not args.not_compressed:
            return lz4.frame.open(fn, mode, compression_level=lz4.frame.COMPRESSIONLEVEL_MINHC)
        else:
            return open(fn, mode)

    total = 0
    with open_file(args.input, 'rb') as input_f:
        input_f.seek(0, 2)
        size = input_f.tell()
        input_f.seek(0, 0)

        while input_f.peek() != b'':
            data = read_entry(input_f)
            total += 1
            if total <= args.max_show_num or (args.show_indices and total in args.show_indices):
                process_bin(total, *data)

            if total % 10000 == 0:
                print(f'total read: {total} ({input_f.tell() * 100 / size: 0.2f}%)')

        print('=' * 50)
        print(f'total read: {total}')


if __name__ == "__main__":
    read_bin()
