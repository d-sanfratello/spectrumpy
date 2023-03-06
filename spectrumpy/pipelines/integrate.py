import os

import h5py
import numpy as np
import optparse as op

from pathlib import Path

from spectrumpy.core import SpectrumImage
from spectrumpy.io import SpectrumPath


def main():
    parser = op.OptionParser()
    parser.add_option("-i", "--input", type='string', dest='image_file',
                      help="")
    parser.add_option("-I", "--image", type='int', dest='image',
                      default=0,
                      help="")
    parser.add_option("-o", "--output", type='string', dest='output_data',
                      default=None)
    parser.add_option("-l", "--lamp", action='store_true', dest='is_lamp',
                      default=False,
                      help="")

    (options, args) = parser.parse_args()

    if options.image_file is None:
        raise AttributeError(
            "I need a fits or h5 file to crop."
        )

    try:
        image_file = SpectrumPath(Path(options.image_file),
                                  is_lamp=options.is_lamp)
        image = image_file.images[str(options.image)]

        if options.output_image is not None:
            path = options.output_image

            if path.find(".h5") < 0:
                path += '.h5'
            path = Path(path)

            with h5py.File(path, 'w') as f:
                f.create_dataset('image', data=image.image)
    except OSError:
        with h5py.File(Path(options.image_file), 'r') as f:
            image = np.asarray(f['image'])

        image = SpectrumImage(image, is_lamp=options.is_lamp)
    finally:
        spectrum = image.run_integration()

        path = options.output_data
        if path is None:
            new_filename = "integrated_spectrum.h5"
            path = Path(os.getcwd()).joinpath(new_filename)
        else:
            path = Path(path)

        if path.suffix != ".h5":
            path = path.with_suffix('.h5')

        with h5py.File(path, 'w') as f:
            f.create_dataset('spectrum', data=spectrum.spectrum)


if __name__ == "__main__":
    main()
