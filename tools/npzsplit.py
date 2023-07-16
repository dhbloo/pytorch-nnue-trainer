import numpy as np
import argparse
import os
from tqdm import tqdm


def npz_split():
    parser = argparse.ArgumentParser(description="Split npz file into multiple npz chunks")
    parser.add_argument('file', type=str, help="Npz file path")
    parser.add_argument('-o', '--outdir', type=str, help="Output directory")
    parser.add_argument('-s', '--splits', type=int, required=True, help="Number of splits")
    parser.add_argument('--shuffle', action='store_true', help="Shuffle data before saving")
    parser.add_argument('--exclude_keys', type=str, nargs='+', help="Exclude keys from splitting")
    args = parser.parse_args()

    outdir = args.outdir or os.path.dirname(args.file)
    os.makedirs(outdir, exist_ok=True)  # make run directory
    filename_noext, file_ext = os.path.splitext(os.path.basename(args.file))

    def make_filename(idx):
        filename = f"{filename_noext}_{idx:04d}" + file_ext
        return os.path.join(outdir, filename)

    assert os.path.exists(args.file), f"File {args.file} does not exist"
    assert os.path.isfile(args.file), f"Path {args.file} is not a file"
    data = np.load(args.file)

    # load data and print data overview
    print(f"Data overview: {len(data)} arrays")
    data_to_split = {}
    for key in data:
        if args.exclude_keys and key in args.exclude_keys:
            print(f"{key}: (excluded)")
            continue
        print(f"{key}: {data[key].shape} {data[key].dtype}")
        assert args.splits <= len(data[key]), "splits must be less than length of data array"
        if len(data[key]) % args.splits != 0:
            print(f"Warning: array {key} of length {len(data[key])} "
                  f"is not divisible by splits {args.splits}")
        data_to_split[key] = data[key]
    print()

    # shuffle data array if requested
    if args.shuffle:
        print("Shuffling data arrays...")
        for key, array in data_to_split.items():
            np.random.shuffle(array)

    # split data and save
    print(f"Splitting data into {args.splits} chunks...")
    for idx in tqdm(range(args.splits)):
        data_chunk = {}
        for key, array in data_to_split.items():
            data_chunk[key] = array[idx::args.splits]
        np.savez_compressed(make_filename(idx), **data_chunk)


if __name__ == '__main__':
    npz_split()