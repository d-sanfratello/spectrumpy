import argparse as ag
import h5py
import numpy as np
import os

from pathlib import Path


def main():
    parser = ag.ArgumentParser(
        prog='sp-correct-image',
        description='This script corrects a given image for bias, dark and '
                    'flat.',
    )
    parser.add_argument('image_path')
    parser.add_argument("-b", "--bias", dest='bias_path',
                        default=None,
                        help="The path to the bias frame image. If a dark "
                             "frame is provided, this is ignored.")
    parser.add_argument("-d", "--dark", dest='dark_path',
                        default=None,
                        help="The path to the dark frame image. If both this "
                             "and a bias frame are provided, the bias frame "
                             "is ignored.")
    parser.add_argument("-f", "--flat", dest='flat_path',
                        default=None,
                        help="The path to the flat frame image.")
    parser.add_argument("-o", "--output", dest='output_name',
                        default=None,
                        help="")
    args = parser.parse_args()

    if args.dark_path is None:
        offset_path = Path(args.bias_path)
    else:
        offset_path = Path(args.dark_path)

    flat_path = None
    if args.flat_path is not None:
        flat_path = Path(args.flat_path)

    image_path = Path(args.image_path)
    with h5py.File(image_path, 'r') as f:
        light = f['image'][:]

    with h5py.File(offset_path, 'r') as f:
        offset = f['image'][:]

    if flat_path is not None:
        with h5py.File(flat_path, 'r') as f:
            flat = f['image'][:]

        data = (light - offset) / (flat - offset)
    else:
        data = light - offset

    out_path = args.output_name
    if out_path is None:
        out_path = Path(os.getcwd()).joinpath('corrected_image.h5')
    else:
        out_path = Path(out_path)

    with h5py.File(out_path, 'w') as f:
        f.create_dataset('image', data=data)


if __name__ == "__main__":
    main()
