import corner
import cpnest
import h5py
import numpy as np
import os
import optparse as op

from figaro.mixture import DPGMM
from figaro.load import save_density, load_density
from figaro.utils import get_priors
from pathlib import Path

from spectrumpy.bayes_inference import CalibrationFit
from spectrumpy.bayes_inference import models as mod


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
    parser.add_option("-d", "--save-density", action='store_true',
                      dest="density", default=False)
    parser.add_option("-P", "--prior", dest="prior_density", default=None)
    (options, args) = parser.parse_args()

    bounds = eval(options.bounds)
    mod_bounds = eval(options.mod_bounds)

    prior_density = None
    if options.prior_density is not None:
        prior_density = load_density(options.prior_density)

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
            px, dpx, l, dl, bounds, mod_bounds, options.model,
            prior=prior_density
        )

        work = cpnest.CPNest(
            fit_model,
            verbose=2,
            nlive=1000,  # 1000
            maxmcmc=5000,  # 5000
            nensemble=1,
            output=Path(options.out_folder).joinpath(options.model)
        )
        work.run()
        post = work.posterior_samples.ravel()
    else:
        with h5py.File(
                Path(options.out_folder).joinpath(
                    options.model, 'cpnest.h5'), 'r'
        ) as f:
            post = np.array([
                np.array([x for x in p])
                for p in f['combined']['posterior_samples']
            ]).T

    post = post[len(l):len(l) + len(mod.names[options.model])].T
    samples = np.column_stack((post['x_0'], post['x_1']))

    if options.data_file is not None:
        c = corner.corner(
            samples,
            labels=['${0}$'.format(l) for l in mod.names[options.model]],
            quantiles=[0.16, 0.5, 0.84]
        )
    else:
        c = corner.corner(
            samples,
            labels=['${0}$'.format(l) for l in mod.names[options.model]],
            quantiles=[0.16, 0.5, 0.84],
            truths=true_vals
        )

    c.savefig(
        Path(options.out_folder).joinpath(
            options.model,
            f'joint_posterior_{options.model}.pdf'
        ),
        bbox_inches='tight'
    )

    if options.postprocess and options.density:
        d_bounds = [
            [samples[:, _].min(), samples[:, _].max()]
            for _ in range(samples.shape[0])
        ]
        prior = get_priors(d_bounds, samples)

        mix = DPGMM(bounds=d_bounds, prior_pars=prior)
        density = mix.density_from_samples(samples)

        save_density(density,
                     folder=Path(options.out_folder).joinpath(options.model),
                     name=f'params_posterior_density_{options.model}.json')
