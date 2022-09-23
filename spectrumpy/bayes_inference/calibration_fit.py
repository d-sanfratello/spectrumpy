import corner
import cpnest
import h5py
import numpy as np
import os
import optparse as op

from cpnest.model import Model
from pathlib import Path

import models as mod

from .constants import LOGSQRT2PI


class CalibrationFit(Model):
    def __init__(self, px, dpx, l, dl, x_bounds, mod_bounds, model):
        self.px = px
        self.dpx = dpx
        self.l = l
        self.dl = dl

        self.n_pts = len(px)

        if model in mod.available_models:
            self.bounds = [_ for _ in x_bounds] + mod_bounds
            self.names = [
                f'px_{i}' for i in range(self.n_pts)
            ] + mod.names[model]
            self.model = mod.models[model]
        else:
            raise ValueError(
                "Unsupported model. Please choose among one of these: "
                + " ".join(mod.available_models)
            )

    def log_prior(self, param):
        log_p = super(CalibrationFit, self).log_prior(param)
        if np.isfinite(log_p):
            log_p = 0.

        return log_p

    def log_likelihood(self, param):
        px_hat = param.values[:self.n_pts]
        pars = param.values[self.n_pts:]

        likel = -0.5 * (
                ((self.l - self.model(px_hat, *pars)) / self.dl) ** 2
                + ((self.px - px_hat) / self.dpx) ** 2
        ) - np.log(self.dpx) - np.log(self.dl) - 2 * LOGSQRT2PI

        return likel


if __name__ == "__main__":
    parser = op.OptionParser()
    parser.add_option("-i", "--input", type="string", dest="data_file",
                      default=None)
    parser.add_option("-m", "--model", type="string", dest="model")
    parser.add_option("-b", "--bounds", type="string", dest="bounds")
    parser.add_option("-B", "--model-bounds", type="string", dest="mod_bounds")
    parser.add_option("-p", "--postprocess", action='store_true',
                      dest="postprocess", default=False)
    parser.add_option("-o", "--output-folder", dest="out_folder",
                      default=Path(os.getcwd()))
    (options, args) = parser.parse_args()

    bounds = np.atleast_1d(eval(options.bounds))
    mod_bounds = np.atleast_2d(eval(options.mod_bounds))

    if options.data_file is not None:
        data = np.genfromtxt(options.data_file, names=True)
        px = data['px']
        dpx = data['dpx']
        l = data['l']
        dl = data['dl']
    else:
        true_vals = [1e-3, 1e-2, 10, 1]
        px = np.random.uniform(low=bounds[0], high=bounds[1], size=8)
        l = mod.cubic(px, *true_vals)
        dpx = np.ones(len(px)) * (bounds[1] - bounds[0]) / 100.
        dl = np.ones(len(l))
        px = np.random.normal(px, dpx)
        l = np.random.normal(l, dl)

    if not options.postprocess:
        fit_model = CalibrationFit(
            px, dpx, l, dl, bounds, mod_bounds, options.model
        )

        work = cpnest.CPNest(
            fit_model,
            verbose=2,
            nlive=1000,
            maxmcmc=5000,
            nensemble=1,
            out_folder=options.out_folder.joinpath(options.model)
        )
        work.run()
        post = work.get_posterior_samples().ravel()
    else:
        with h5py.File(
                options.out_folder.joinpath(
                    options.model, 'cpnest.h5'), 'r'
        ) as f:
            post = np.array([
                np.array([x for x in p])
                for p in f['combined']['posterior_samples']
            ]).T

    post = post[len(l):len(l) + len(mod.names[options.model])].T

    if options.data_file is not None:
        c = corner.corner(
            post,
            labels=['${0}$'.format(l) for l in mod.names[options.model]],
            quantiles=[0.16, 0.5, 0.84]
        )
    else:
        c = corner.corner(
            post,
            labels=['${0}$'.format(l) for l in mod.names[options.model]],
            quantiles=[0.16, 0.5, 0.84],
            truths=true_vals
        )

    c.savefig(
        f'joint_posterior_{options.model}.pdf',
        bbox_inches='tight'
    )
