import numpy as np

from core_abc_classes import SpectrumRotatedImageABC

from spectrum_image import SpectrumImage
from spectrum_cropped_image import SpectrumCroppedImage


class SpectrumRotatedImage(SpectrumRotatedImageABC, SpectrumImage):
    def __init__(self, image, angle, info):
        SpectrumImage.__init__(self, image)
        self.__info = info
        self.__info['angle'] = angle

    def crop(self, info):
        # FIXME: add things to crop image
        crop = np.image(self.image)  # placeholder

        return SpectrumCroppedImage(crop, 0, 0, self.info)
