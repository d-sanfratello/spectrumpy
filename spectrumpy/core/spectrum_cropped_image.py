import numpy as np

from astropy.io import fits

from .core_abc_classes import SpectrumCroppedImageABC

from spectrum_rotated_image import SpectrumRotatedImage
from spectrum_image import SpectrumImage
from spectrum import Spectrum


class SpectrumCroppedImage(SpectrumCroppedImageABC, SpectrumRotatedImage):
    def __init__(self, image, lr_idx, ab_idx, info):
        SpectrumRotatedImage.__init__(self, image, info['angle'], info)
        self.__info['lr_idx'] = lr_idx
        self.__info['ab_idx'] = ab_idx

    def run_integration(self):
        integrated = np.sum(self.image, axis=0)

        return Spectrum(integrated, self.info)


class SpectrumFile:
    def __init__(self, spectrum_path):
        with fits.open(spectrum_path) as s_file:
            self.hdu_list = [hdu for hdu in s_file]
            self.headers = [hdu.header for hdu in self.hdu_list]
            self.data = [hdu.data for hdu in self.hdu_list]

        self.images = {
            f'{i}': SpectrumImage(data) for i, data in enumerate(self.data)
            if self.headers[i]['NAXIS'] == 2
        }