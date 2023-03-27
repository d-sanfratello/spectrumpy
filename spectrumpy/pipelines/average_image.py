import argparse as ag
import h5py
import numpy as np
import os

from astropy.io import fits
from pathlib import Path


def main():
    parser = ag.ArgumentParser(
        prog='sp-average',
        description='This script averages a the given fits files into a '
                    'single image and saves it.',
    )
    parser.add_argument('image_path', nargs='+')
    parser.add_argument("-I", "--image", type=int, dest='image',
                        default=0,
                        help="")
    parser.add_argument("-o", "--output", dest='output_name',
                        default='avg_image',
                        help="")
    parser.add_argument("-e", "--exposure", type=float, dest='exp_time',
                        default=-1,
                        help="The needed exposure time (for dark and flat "
                             "frames). If the images are longer than this "
                             "parameter, it performs a time average. It "
                             "raises an exception otherwise.")
    args = parser.parse_args()

    data = []
    exposures = []
    for img in args.image_path:
        path = Path(img)

        with fits.open(path) as f:
            data.append(f[args.image].data)
            exposures.append(
                float(f[args.image].header['EXPOSURE'])
            )

    exposures = np.asarray(exposures)
    if np.any(exposures < args.exp_time):
        raise ValueError(
            "Cannot perform a time average. At least one image is has a "
            "shorter exposure than requested."
        )

    if not np.all(exposures == exposures[0]):
        raise ValueError(
            "Not all images have the same exposure time."
        )

    data = np.asarray(data).mean(axis=0)
    exp = ''
    if args.exp_time > 0:
        factor = args.exp_time / exposures[0]
        data *= factor
        exp = f'_{args.exp_time:.2e}s'
        
    out_path = Path(os.getcwd())
    out_path = out_path.joinpath(
        args.output_name + exp + '.h5'
    )
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('image', data=data)
        f.create_dataset('exposure', data=args.exp_time)


if __name__ == "__main__":
    main()
