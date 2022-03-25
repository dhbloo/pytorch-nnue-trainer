import numpy as np
import argparse
import sys
import os
from tqdm import tqdm


def npz_split():
    parser = argparse.ArgumentParser(description="Split npz file into multiple npz chunks")
    parser.add_argument('file', type=str, help="Npz file path")
    parser.add_argument('-o', '--outdir', type=str, help="Output directory")
    parser.add_argument('-s', '--splits', type=int, required=True, help="Number of splits")
    parser.add_argument('--shuffle', action='store_true', help="Shuffle data before saving")
    args = parser.parse_args()

    outdir = args.outdir or os.path.dirname(args.file)
    os.makedirs(outdir, exist_ok=True)  # make run directory
    filename_noext, file_ext = os.path.splitext(os.path.basename(args.file))

    def make_filename(idx):
        filename = f"{filename_noext}_{idx:04d}" + file_ext
        return os.path.join(outdir, filename)

    data = np.load(args.file)

    for key, array in data.items():
        assert args.splits <= len(array), "splits must be less than length of data array"
        if len(array) % args.splits != 0:
            print(
                f"Warning: array {key} of length {len(array)} is not divisible by splits {args.splits}",
                file=sys.stderr)

    if args.shuffle:
        for array in data.values():
            np.random.shuffle(array)

    for idx in tqdm(range(args.splits)):
        data_chunk = {}
        for key, array in data.items():
            data_chunk[key] = array[idx::args.splits]
        np.savez_compressed(make_filename(idx), **data_chunk)


if __name__ == '__main__':
    npz_split()