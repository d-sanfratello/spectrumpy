import numpy as np

from scipy.ndimage import median_filter


class CalibratedSpectrum:
    def __init__(self, wavelength, spectrum):
        self.wl = wavelength
        self.sp = spectrum

    def normalize(self):
        min_sp = np.min(self.spectrum)
        max_sp = np.max(self.spectrum)
        weight = max_sp - min_sp

        normalized = (self.spectrum - min_sp) / weight

        return CalibratedSpectrum(
            wavelength=np.asarray(self.wl),
            spectrum=np.asarray(normalized)
        )

    def smooth(self, size):
        smoothed = median_filter(self.spectrum, size=size)

        return CalibratedSpectrum(
            wavelength=np.asarray(self.wl),
            spectrum=np.asarray(smoothed)
        )

    def cut(self, low=None, high=None):
        good_idx = np.argwhere((self.wl >= low) & (self.wl <= high))

        return CalibratedSpectrum(
            wavelength=np.asarray(self.wl[good_idx]),
            spectrum=np.asarray(self.sp[good_idx])
        )

    @classmethod
    def compare(cls, spectrum1, spectrum2):
        return spectrum1 / spectrum2
