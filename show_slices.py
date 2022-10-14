import h5py
import matplotlib.pyplot as plt
import numpy as np
import optparse as op

from pathlib import Path

from spectrumpy.io import SpectrumPath


if __name__ == "__main__":
    parser = op.OptionParser()
    parser.add_option("-i", "--input", type='string', dest='image_file',
                      help="")
    parser.add_option("-I", "--image", type='int', dest='image',
                      default=0,
                      help="")
    parser.add_option("-o", "--output", type='string', dest='output_image',
                      default=None)
    parser.add_option("-s", "--slice", type='string', dest="slice",
                      default=None,
                      help="")
    parser.add_option("-w", "--width", type='int', dest="width",
                      default=1,
                      help="")
    parser.add_option("-c", "--crop", type='string', dest="crop",
                      default=None,
                      help="")

    (options, args) = parser.parse_args()

    if options.image_file is None:
        raise AttributeError(
            "I need a fits or h5 file to show you."
        )
    if options.slice is None and options.crop is None:
        raise AttributeError(
            "I need either some slices or crop limits to work on the image."
        )

    try:
        image_file = SpectrumPath(Path(options.image_file),
                                  is_lamp=options.is_lamp)
        image = image_file.images[str(options.image)]

        image.show(
            log=True,
            show=True,
            save=False,
            legend=False
        )

        if options.output_image is not None:
            path = options.output_image

            if path.find(".h5") < 0:
                path += '.h5'
            path = Path(path)

            with h5py.File(path, 'w') as f:
                f.create_dataset('image', data=image.image)
    except OSError:
        try:
            with h5py.File(Path(options.image_file), 'r') as f:
                image = np.asarray(f['image'])

            fig = plt.figure()
            ax = fig.gca()

            ax.imshow(np.log10(image),
                      origin='lower',
                      cmap='Greys')

            ax.set_xlabel('[px]')
            ax.set_ylabel('[px]')

            plt.show()
        finally:
            pass
