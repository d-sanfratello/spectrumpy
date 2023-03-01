import h5py
import matplotlib.pyplot as plt
import numpy as np
import optparse as op

from pathlib import Path

from spectrumpy.core import SpectrumImage
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
    parser.add_option("-v", "--vertical", action='store_true',
                      dest='vertical_slice', default=False)

    (options, args) = parser.parse_args()

    slices = eval(options.slice)

    if slices is None:
        raise ValueError(
            "I need at least a slice."
        )
    if not isinstance(slices, list):
        slices = [slices]

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
                                  is_lamp=False)
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

        image = SpectrumImage(image, is_lamp=False)
    finally:
        if options.vertical_slice:
            # FIXME: implement.
            raise ValueError("Not yet implemented.")
        else:
            slices_dict = {
                line: image.slice_image(line) for line in slices
            }

            int_models = {
                line: sl.run_integration() for line, sl in slices_dict.items()
            }

            for line in int_models.keys():
                exec(
                    f"def slice_{line}(x): return int_models[{line}].spectrum"
                )

            models = [
                eval(f"slice_{line}") for line in int_models.keys()
            ]

            first_model = int_models[0]
            x_len = len(first_model.spectrum)
            x_sup = x_len - 1
            name = models[0].__name__
            models.pop(slices.index(0))
            first_model.show(model=models,
                             x=np.linspace(0, x_sup, x_len),
                             legend=True,
                             show=True,
                             save=False,
                             label=name)
