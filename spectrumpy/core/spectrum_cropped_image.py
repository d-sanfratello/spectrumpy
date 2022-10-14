import numpy as np

from .spectrum_rotated_image import SpectrumRotatedImage
from .spectrum import Spectrum


class SpectrumCroppedImage(SpectrumRotatedImage):
    def __init__(self, image, crop_x, crop_y, info):
        SpectrumRotatedImage.__init__(self, image, info['angle'], info)
        self._info['crop_x'] = crop_x
        self._info['crop_y'] = crop_y

    @property
    def info(self):
        return self._info

