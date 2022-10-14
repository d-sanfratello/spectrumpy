import numpy as np

from .spectrum_image import SpectrumImage


class SpectrumRotatedImage(SpectrumImage):
    def __init__(self, image, angle, info):
        SpectrumImage.__init__(self, image,
                               info['lamp'], info)
        self._info = info
        self._info['angle'] = angle

    @property
    def info(self):
        return self._info
