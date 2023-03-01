import h5py
import numpy as np
import optparse as op

from pathlib import Path

from spectrumpy.io import SpectrumPath
from spectrumpy.core import SpectrumImage

# FIXME: test me


if __name__ == "__main__":
    parser = op.OptionParser()
    parser.add_option("-i", "--input", type='string', dest='image_file',
                      help="")
    parser.add_option("-I", "--image", type='int', dest='image',
                      default=0,
                      help="")
    parser.add_option("-r", "--rotate", type='float', dest='rot_angle',
                      default=0,
                      help="")
    parser.add_option("-o", "--output", type='string', dest='output_image',
                      default='./rotated_image.h5',
                      help="")

    (options, args) = parser.parse_args()

    if options.image_file is None:
        raise AttributeError(
            "I need a fits or h5 file to rotate."
        )

    try:
        image_file = SpectrumPath(Path(options.image_file),
                                  is_lamp=options.is_lamp)
        image = image_file.images[str(options.image)]
    except OSError:
        with h5py.File(Path(options.image_file), 'r') as f:
            image_array = np.asarray(f['image'])

        image = SpectrumImage(image, is_lamp=False)
    finally:
        rot_image = image.rotate_image(options.rot_angle)

        path = options.output_image
        if path.find(".h5") < 0:
            path += '.h5'
        path = Path(path)

        with h5py.File(path, 'w') as f:
            f.create_dataset('image', data=rot_image.image)
