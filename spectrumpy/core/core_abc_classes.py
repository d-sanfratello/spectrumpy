from abc import ABC, abstractmethod


class SpectrumImageABC(ABC):
    @abstractmethod
    def show(self, log, show, save, name, *args, **kwargs):
        pass

    @abstractmethod
    def find_rotation_angle(self, x, y,
                            error_y, error_x,
                            bounds, verbose, nlive, maxmcmc, nnest,
                            nensemble, seed,
                            show, save, name):
        pass

    @abstractmethod
    def rotate_image(self, angle, info):
        pass

    @abstractmethod
    def info(self):
        pass


class SpectrumRotatedImageABC(ABC):
    @abstractmethod
    def crop_image(self, info):
        pass

    @abstractmethod
    def info(self):
        pass


class SpectrumCroppedImageABC(ABC):
    @abstractmethod
    def run_integration(self, info):
        pass

    @abstractmethod
    def info(self):
        pass


class SpectrumABC(ABC):
    @abstractmethod
    def show(self, show, save, name, **kwargs):
        pass

    @abstractmethod
    def smooth(self, size, lamp):
        pass

    @abstractmethod
    def weight(self, smoothed, lamp):
        pass

    @abstractmethod
    def assign_dataset(self, *, lines, px, errpx, errlines, names):
        pass

    @abstractmethod
    def calibrate(self, order, method, bounds_pars, verbose, n, delta):
        pass

    @abstractmethod
    def compare(self, spectrum):
        pass

    @abstractmethod
    def info(self):
        pass


class Calibration(ABC):
    @abstractmethod
    def show_calibration(self, lamp, exclude, **kwargs):
        pass

