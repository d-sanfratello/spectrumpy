import numpy as np

from .spectrum_image import SpectrumImage


class SpectrumRotatedImage(SpectrumImage):
    def __init__(self, image, angle, info):
        SpectrumImage.__init__(self, image,
                               info['lamp'], info)
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

    def slice_image(self, level, info=None):
        if not 0 <= level < self.image.shape[0]:
            raise ValueError(f"Slice {level} is outside of image edges.")

        if info is None:
            info = self.info.copy()

        if level > 0:
            return self.crop_image(crop_y=slice(level-1, level), info=info)
        else:
            return self.crop_image(crop_y=slice(level, level+1), info=info)

    @property
    def info(self):
        return self._info
