import numpy as np

from cpnest.model import Model

from . import models as mod

from .constants import LOGSQRT2PI

class CalibrationFit(Model):
    def __init__(self,
                 px, dpx, l, dl, x_bounds, mod_bounds, model, prior=None):
        self.px = px
        self.dpx = dpx
        self.l = l
        self.dl = dl

        self.n_pts = len(px)

        self.prior = prior

        if model in mod.available_models:
            self.bounds = [
                              x_bounds[:] for _ in range(self.px.shape[0])
                          ] + mod_bounds
            self.names = [
                f'px_{i}' for i in range(self.n_pts)
            ] + mod.names[model]
            self.fit_model = mod.models[model]
        else:
            raise ValueError(
                "Unsupported model. Please choose among one of these: "
                + " ".join(mod.available_models)
            )

    def log_prior(self, param):
        log_p = super(CalibrationFit, self).log_prior(param)
        if np.isfinite(log_p):
            if self.prior is None:
                log_p = 0.
            else:
                par = param[len(self.px) + 1:]
                log_p = self.prior.logpdf(np.asarray(par))

        return log_p

    def log_likelihood(self, param):
        px_hat = param.values[:self.n_pts]
        pars = param.values[self.n_pts:]

        likel = -0.5 * (
                ((self.l - self.fit_model(px_hat, *pars)) / self.dl) ** 2
                + ((self.px - px_hat) / self.dpx) ** 2
        ) - np.log(self.dpx) - np.log(self.dl) - 2 * LOGSQRT2PI

        return likel.sum()
