import numpy as np

from .core_abc_classes import SpectrumRotatedImageABC

from .spectrum_image import SpectrumImage


class SpectrumRotatedImage(SpectrumRotatedImageABC, SpectrumImage):
    def __init__(self, image, angle, info):
        SpectrumImage.__init__(self, image,
                               info['lamp'], info['calibration'])
        self._info = info
        self._info['angle'] = angle

    def crop_image(self, crop_x=None, crop_y=None, info=None):
        from .spectrum_cropped_image import SpectrumCroppedImage

        if crop_x is None:
            crop_x = slice(0, self.image.shape[1])
        elif not isinstance(crop_x, slice):
            crop_x = slice(*crop_x)

        if crop_y is None:
            crop_y = slice(0, self.image.shape[0])
        elif not isinstance(crop_y, slice):
            crop_y = slice(*crop_y)

        crop = np.copy(self.image[crop_y, crop_x])

        if info is None:
            info = self.info.copy()

        return SpectrumCroppedImage(crop, crop_x, crop_y, info)

    @property
    def info(self):
        return self._info
