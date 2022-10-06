import corner
import cpnest
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import optparse as op

from corner import quantile
from figaro.mixture import DPGMM
from figaro.load import save_density, load_density
from figaro.utils import get_priors
from pathlib import Path

from spectrumpy.bayes_inference import CalibrationFit
from spectrumpy.bayes_inference import models as mod


if __name__ == "__main__":
    parser = op.OptionParser()
    parser.add_option("-i", "--input", type="string", dest="data_file",
                      default=None,
                      help="The file to input the calibration lines from. If "
                           "none is provided the dataset is simulated.")
    parser.add_option("-o", "--output-folder", dest="out_folder",
                      default=Path(os.getcwd()),
                      help="The folder where to save the output of this "
                           "command.")
    parser.add_option("-p", "--postprocess", action='store_true',
                      dest="postprocess", default=False,
                      help="Whether to do a new inference or to open a file "
                           "of samples.")
    parser.add_option("-m", "--model", type="string", dest="model",
                      help="The type of model to do inference on.")
    parser.add_option("-b", "--bounds", type="string", dest="bounds",
                      help="The bounds for the pixels.")
    parser.add_option("-B", "--model-bounds", type="string",
                      dest="mod_bounds",
                      help="The bounds for the model parameters. Write from "
                           "highest to lowest order.")
    parser.add_option("-P", "--prior", dest="prior_density", default=None,
                      help="The path to a 'figaro'-compatible .json file "
                           "containing a pdf posterior from another sample "
                           "to use as a prior on a following calibration.")
    parser.add_option("-d", "--save-density", action='store_true',
                      dest="density", default=False,
                      help="Whether to run 'figaro' to create an analytic "
                           "pdf and store it in a .json file.")
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
            post = np.asarray(f['combined']['posterior_samples'])

    columns = [post[par] for par in mod.names[options.model]]
    samples = np.column_stack(columns)

    title = 'Model: $f(p) = '
    for par in mod.names[options.model]:
        deg = par[-1]
        title += rf'{par}\,p^{deg} + '
    title = title[:-3] + '$'

    fig = plt.figure()
    fig.suptitle(title)
    if options.data_file is not None:
        c = corner.corner(
            samples,
            labels=[f'${l}$' for l in mod.names[options.model]],
            quantiles=[0.16, 0.5, 0.84],
            fig=fig,
            use_math_text=True,
        )
    else:
        c = corner.corner(
            samples,
            labels=[f'${l}$' for l in mod.names[options.model]],
            quantiles=[0.16, 0.5, 0.84],
            truths=true_vals,
            fig=fig,
            use_math_text=True,
        )

    c.savefig(
        Path(options.out_folder).joinpath(
            options.model,
            f'joint_posterior_{options.model}.pdf'
        ),
        bbox_inches='tight'
    )
    with open(Path(options.out_folder).joinpath(
            options.model,
            f'quantiles_{options.model}.txt'), 'w+') as f:
        for _, l in enumerate(mod.names[options.model]):
            x = samples[:, _]
            q_16, q_50, q_84 = quantile(x, [0.16, 0.5, 0.84])
            q_m, q_p = q_50 - q_16, q_84 - q_50

            f.write(f'{l} =\t{q_50} +{q_p} -{q_m}\n')

    if options.density:
        d_bounds = [
            [samples[:, _].min(), samples[:, _].max()]
            for _ in range(samples.shape[1])
        ]
        del_idx = []
        for idx in range(samples.shape[1]):
            del_idx.append(np.argmin(samples[:, idx]))
            del_idx.append(np.argmax(samples[:, idx]))
        del_idx_ = []
        [del_idx_.append(x) for x in del_idx if x not in del_idx_]

        samples = np.delete(samples, del_idx_, axis=0)
        prior = get_priors(d_bounds, samples)

        mix = DPGMM(bounds=d_bounds, prior_pars=prior)
        density = mix.density_from_samples(samples)

        save_density(density,
                     folder=Path(options.out_folder).joinpath(options.model),
                     name=f'params_posterior_density_{options.model}')
