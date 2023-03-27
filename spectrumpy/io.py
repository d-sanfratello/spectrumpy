import h5py
import numpy as np

from astropy.io import fits
from pathlib import Path

from spectrumpy.core.spectrum_image import SpectrumImage
from spectrumpy.core.spectrum import Spectrum


class SpectrumPath:
    def __init__(self,
                 spectrum_path,
                 info=None,
                 is_lamp=False):
        with fits.open(spectrum_path) as s_file:
            self.hdu_list = [hdu for hdu in s_file]
            self.headers = [hdu.header for hdu in self.hdu_list]
            self.data = [hdu.data for hdu in self.hdu_list]

        self.images = {
            f'{i}': SpectrumImage(img, is_lamp, info)
            for i, img in enumerate(self.data)
            if self.headers[i]['NAXIS'] == 2
        }


def parse_image_path(args,
                     data_name,
                     wrong_type_msg="Invalid file extension.",
                     save_output=True,
                     **kwargs):

    image_path = Path(getattr(args, data_name))

    if image_path.suffix.lower() in ['.fit', '.fits']:
        image_file = SpectrumPath(image_path, is_lamp=kwargs['is_lamp'])
        image = image_file.images[str(kwargs['image'])]

        if save_output and kwargs['output_image'] is not None:
            path = Path(kwargs['output_image'])

            if path.suffix != ".h5":
                path = path.with_suffix('.h5')

            with h5py.File(path, 'w') as f:
                f.create_dataset('image', data=image.image)
    elif image_path.suffix.lower() in ['.h5', '.hdf5']:
        with h5py.File(image_path, 'r') as f:
            image = np.asarray(f['image'])

        image = SpectrumImage(image, is_lamp=False)
    else:
        raise ValueError(wrong_type_msg)

    return image


def parse_spectrum_path(args,
                        data_name,
                        wrong_type_msg="Invalid file extension.",
                        **kwargs):
    spectrum_path = Path(getattr(args, data_name))
    if spectrum_path.suffix.lower() in ['.h5', '.hdf5']:
        with h5py.File(spectrum_path, 'r') as f:
            spectrum_data = np.asarray(f['spectrum'])

        spectrum = Spectrum(spectrum_data)
    else:
        raise ValueError(wrong_type_msg)

    return spectrum


def parse_data_path(arg_list,
                    data_name,
                    **kwargs):
    data_file = Path(getattr(arg_list, data_name))
    data = np.genfromtxt(data_file, names=True)

    names = data.dtype.names

    x = data[names[0]]
    dx = data[names[1]]
    y = data[names[2]]
    dy = data[names[3]]

    return x, dx, y, dy


def parse_bounds(args):
    if not args.postprocess and args.mod_bounds is None:
        raise ValueError(
            "When trying a new inference, bounds must be set."
        )

    if args.bounds is not None:
        x_bounds = eval(args.bounds)
        x_bounds = np.array(x_bounds).flatten()
    else:
        x_bounds = args.bounds

    if args.mod_bounds is not None:
        mod_bounds = eval(args.mod_bounds)

    return x_bounds, mod_bounds


def parse_tentative(args):
    if args.tentative:
        return 100, 500
    else:
        return 1000, 5000
