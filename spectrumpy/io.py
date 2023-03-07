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


def parse_image_path(arg_list,
                     missing_arg_msg="Missing positional argument.",
                     wrong_type_msg="Invalid file extension.",
                     save_output=True,
                     **kwargs):

    if len(arg_list) == 0:
        raise ValueError(missing_arg_msg)

    image_path = Path(arg_list[0])
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


def parse_spectrum_path(arg_list,
                        missing_arg_msg="Missing positional argument.",
                        wrong_type_msg="Invalid file extension.",
                        **kwargs):
    if len(arg_list) == 0:
        raise ValueError(missing_arg_msg)

    spectrum_path = Path(arg_list[0])
    if spectrum_path.suffix.lower() in ['.h5', '.hdf5']:
        with h5py.File(spectrum_path, 'r') as f:
            spectrum_data = np.asarray(f['spectrum'])

        spectrum = Spectrum(spectrum_data)
    else:
        raise ValueError(wrong_type_msg)

    return spectrum
