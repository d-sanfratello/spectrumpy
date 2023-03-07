import os

import h5py
import numpy as np
import optparse as op

from pathlib import Path

from spectrumpy.core import SpectrumImage
from spectrumpy.io import SpectrumPath
from spectrumpy.io import parse_image_path


def main():
    parser = op.OptionParser()
    parser.disable_interspersed_args()
    parser.add_option("-I", "--image", type='int', dest='image',
                      default=0,
                      help="")
    parser.add_option("-o", "--output", type='string', dest='output_data',
                      default=None)
    parser.add_option("-l", "--lamp", action='store_true', dest='is_lamp',
                      default=False,
                      help="")

    (options, args) = parser.parse_args()

    image = parse_image_path(
        args,
        missing_arg_msg="I need a fits or h5 file to integrate on.",
        is_lamp=options.is_lamp,
        image=options.image,
        save_output=False
    )

    spectrum = image.run_integration()

    if options.output_data is None:
        new_filename = "integrated_spectrum.h5"
        path = Path(os.getcwd()).joinpath(new_filename)
    else:
        path = Path(options.output_data)

    if path.suffix != ".h5":
        path = path.with_suffix('.h5')

    with h5py.File(path, 'w') as f:
        f.create_dataset('spectrum', data=spectrum.spectrum)


if __name__ == "__main__":
    main()
