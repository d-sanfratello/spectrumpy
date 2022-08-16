import numpy as np

from cpnest import model


class LinearPosterior(model.Model):
    def __init__(self, x, y, sx, sy, bounds):
        self.x = np.array(x)
        self.y = np.array(y)

        if hasattr(sx, "__iter__"):
            self.sx = np.array(sx)
        else:
            self.sx = np.ones(x.shape[0]) * sx

        if hasattr(sy, "__iter__"):
            self.sy = np.array(sy)
        else:
            self.sy = np.ones(y.shape[0]) * sy

        self.bounds = bounds
        self.names = ['m', 'q']

    def log_likelihood(self, param):
        m = param['m']
        q = param['q']
        s2 = m**2 * self.sx**2 + self.sy**2

        exp = m * self.x + q
        log_l = -0.5 * (self.y - exp)**2 / s2
        log_l -= 2 * np.pi
        log_l -= np.sqrt(s2)

        return log_l.sum()

    def log_prior(self, param):
        logP = super(LinearPosterior, self).log_prior(param)
        if np.isfinite(logP):
            logP = 0

        return logP

    @staticmethod
    def __linear(x, param):
        return x * param['m'] + param['q']
