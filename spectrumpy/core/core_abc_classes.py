from abc import ABC, abstractmethod


class SpectrumImageABC(ABC):
    @abstractmethod
    def show(self, log, *args, **kwargs):
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
    def crop(self, info):
        pass


class SpectrumCroppedImageABC(ABC):
    @abstractmethod
    def run_integration(self, info):
        pass
