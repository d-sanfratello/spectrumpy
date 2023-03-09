import os

import h5py
import argparse as ag

from pathlib import Path

from spectrumpy.io import parse_image_path


def main():
    parser = ag.ArgumentParser()
    parser.add_argument('image_path')
    parser.add_argument("-I", "--image", type=int, dest='image',
                        default=0,
                        help="")
    parser.add_argument("-o", "--output", dest='output_data',
                        default=None,
                        help="")
    parser.add_argument("-l", "--lamp", action='store_true', dest='is_lamp',
                        default=False,
                        help="")
    args = parser.parse_args()

    image = parse_image_path(
        args,
        data_name='image_path',
        is_lamp=args.is_lamp,
        image=args.image,
        save_output=False
    )

    spectrum = image.run_integration()

    if args.output_data is None:
        new_filename = "integrated_spectrum.h5"
        path = Path(os.getcwd()).joinpath(new_filename)
    else:
        path = Path(args.output_data)

    if path.suffix != ".h5":
        path = path.with_suffix('.h5')

    with h5py.File(path, 'w') as f:
        f.create_dataset('spectrum', data=spectrum.spectrum)


if __name__ == "__main__":
    main()
