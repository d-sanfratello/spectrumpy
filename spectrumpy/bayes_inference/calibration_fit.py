import numpy as np

from cpnest.model import Model

from . import models as mod

from .constants import LOGSQRT2PI


class CalibrationFit(Model):
    def __init__(self,
                 px, dpx, l, dl,
                 model,
                 sigma_prior=None,
                 x_bounds=None, mod_bounds=None, prior=None):
        self.px = px
        self.dpx = dpx
        self.l = l
        self.dl = dl

        self.sigma_prior = sigma_prior

        self.n_pts = len(px)

        self.prior = prior

        if model not in mod.available_models:
            raise ValueError(
                "Unsupported model. Please choose among one of these: "
                + " ".join(mod.available_models)
            )

        if mod_bounds is None:
            raise ValueError(
                "I need bounds of model parameters to perform the inference."
            )

        if x_bounds is None:
            x_bounds = [
                [d - self.sigma_prior * s,
                 d + self.sigma_prior * s]
                for d, s in zip(self.px, self.dpx)
            ]
        else:
            x_bounds = [x_bounds[:] for _ in range(self.n_pts)]

        self.mod_bounds = mod_bounds
        self.bounds = x_bounds + mod_bounds
        self.names = [
            f'px_{i}' for i in range(self.n_pts)
        ] + mod.names[model]
        self.fit_model = mod.models[model]

    def log_prior(self, param):
        log_p = super(CalibrationFit, self).log_prior(param)
        if np.isfinite(log_p):
            if self.prior is None:
                log_p = 0.
            else:
                par = param.values[len(self.px) + 1:]

                log_p_old = self.prior.fast_logpdf(par)
                new_bounds = self.mod_bounds[0]
                log_new = - np.log(new_bounds[1] - new_bounds[0])

                log_p = log_p_old + log_new
        return log_p

    def log_likelihood(self, param):
        px_hat = param.values[:self.n_pts]
        pars = param.values[self.n_pts:]

        likel = -0.5 * (
                ((self.l - self.fit_model(px_hat, *pars)) / self.dl) ** 2
                + ((self.px - px_hat) / self.dpx) ** 2
        ) - np.log(self.dpx) - np.log(self.dl) - 2 * LOGSQRT2PI

        return likel.sum()
