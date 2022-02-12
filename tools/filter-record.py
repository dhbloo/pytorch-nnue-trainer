import os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

from utils.bin import *
import argparse
import lz4.frame


def filter_bin():
    parser = argparse.ArgumentParser(
        description="BIN training format reader")
    parser.add_argument('input',
                        type=str,
                        help="Input binary file name")
    parser.add_argument('output',
                        type=str,
                        help="Output binary file name")
    parser.add_argument('--not_compressed', action='store_true', default=False,
                        help="Input is not compressed with lz4")
    parser.add_argument('-f','--filter_indices', nargs='+', type=int, help='Filtered indices', required=True)
    args = parser.parse_args()

    def open_file(fn, mode, compress=not args.not_compressed):
        if compress:
            return lz4.frame.open(fn, mode, compression_level=lz4.frame.COMPRESSIONLEVEL_MINHC)
        else:
            return open(fn, mode)

    index = 0
    filtered = 0
    with open_file(args.output, 'wb', True) as output_f:
        with open_file(args.input, 'rb') as input_f:
            input_f.seek(0, 2)
            size = input_f.tell()
            input_f.seek(0, 0)

            while input_f.peek() != b'':
                result, ply, boardsize, rule, move, position = read_entry(input_f)
                index += 1
                if index not in args.filter_indices:
                    write_entry(output_f, result, boardsize, rule, move, position)
                else:
                    filtered += 1

                if index % 10000 == 0:
                    print(f'total read: {index} ({input_f.tell() * 100 / size: 0.2f}%), filtered: {filtered}')

            print('=' * 50)
            print(f'total read: {index}')


if __name__ == "__main__":
    filter_bin()
