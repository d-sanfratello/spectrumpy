import numpy as np

from cpnest.model import Model

from . import models as mod
from .models import linear

from .constants import LOGSQRT2PI


class RotationFit(Model):
    def __init__(self,
                 x, dx, y, dy,
                 x_bounds=None, mod_bounds=None):
        self.x = x
        self.dx = dx
        self.y = y
        self.dy = dy

        self.n_pts = len(x)

        if mod_bounds is None:
            raise ValueError(
                "I need bounds of model parameters to perform the inference."
            )

        if x_bounds is None:
            c_level = 5
            x_bounds = [
                [d - c_level * s, d + c_level * s]
                for d, s in zip(self.x, self.dx)
            ]
        else:
            x_bounds = [x_bounds[:] for _ in range(self.n_pts)]

        self.bounds = x_bounds + mod_bounds
        self.names = [
            f'pt_{i}' for i in range(self.n_pts)
        ] + mod.names['linear']

    def log_prior(self, param):
        log_p = super(RotationFit, self).log_prior(param)
        if np.isfinite(log_p):
            log_p = 0.

        return log_p

    def log_likelihood(self, param):
        x_hat = param.values[:self.n_pts]
        pars = param.values[self.n_pts:]

        likel = -0.5 * (
                ((self.y - linear(x_hat, *pars)) / self.dy) ** 2
                + ((self.x - x_hat) / self.dx) ** 2
        ) - np.log(self.dx) - np.log(self.dy) - 2 * LOGSQRT2PI

        return likel.sum()
