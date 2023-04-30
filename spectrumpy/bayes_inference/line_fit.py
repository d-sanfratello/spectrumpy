import numpy as np

from cpnest.model import Model

from spectrumpy.line_fitting.voigt import models as v_mod
from spectrumpy.line_fitting.lorentz import models as l_mod
from .constants import LOGSQRT2PI


class LineFit(Model):
    def __init__(self,
                 lam, cnt, dcnt,
                 mod_bounds=None,
                 absorption=True,
                 use_lorentz=False):
        self.use_lorentz = use_lorentz

        self.lam = lam
        self.cnt = cnt
        self.dcnt = dcnt

        self.n_pts = len(lam)

        self.bounds = mod_bounds
        self.names = [
            'loc',
            'sigma',
            'gamma',
            'scale',
            'offset'
        ]

        if self.use_lorentz:
            mod = l_mod
            self.names.remove('sigma')
        else:
            mod = v_mod

        if absorption:
            self.model = mod['absorption']
        else:
            self.model = mod['emission']

    def log_prior(self, param):
        log_p = super(LineFit, self).log_prior(param)
        if np.isfinite(log_p):
            log_p = 0. - np.log(param['gamma'])

            if not self.use_lorentz:
                log_p -= np.log(param['sigma'])

        return log_p

    def log_likelihood(self, param):
        pars = param.values[:]
        log_likel = -0.5 * (
            (self.cnt - self.model(self.lam, *pars)) / self.dcnt) ** 2 - \
                    LOGSQRT2PI - np.log(self.dcnt)

        return log_likel.sum()
