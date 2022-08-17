from astropy.io import fits

from spectrumpy.core.spectrum_image import SpectrumImage


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
