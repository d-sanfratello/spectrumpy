import numpy as np

from .core_abc_classes import SpectrumCroppedImageABC

from .spectrum_rotated_image import SpectrumRotatedImage
from .spectrum import Spectrum


class SpectrumCroppedImage(SpectrumCroppedImageABC, SpectrumRotatedImage):
    def __init__(self, image, crop_x, crop_y, info):
        SpectrumRotatedImage.__init__(self, image, info['angle'], info)
        self._info['crop_x'] = crop_x
        self._info['crop_y'] = crop_y

    def run_integration(self, info=None):
        integrated = np.sum(self.image, axis=0)

        if info is None:
            info = self.info

        return Spectrum(integrated, info)

    @property
    def info(self):
        return self._info

