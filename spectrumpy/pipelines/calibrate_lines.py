import argparse as ag
import corner
import cpnest
import h5py
import numpy as np
import os

from corner import quantile
from figaro.mixture import DPGMM
from figaro.load import save_density, load_density
from figaro.plot import plot_multidim
from figaro.utils import get_priors
from pathlib import Path

from spectrumpy.core import Spectrum
from spectrumpy.bayes_inference import CalibrationFit
from spectrumpy.bayes_inference import models as mod
from spectrumpy.io import parse_data_path, parse_bounds


def main():
    parser = ag.ArgumentParser(
            prog='sp-calibrate',
            description='Pipeline to do calibration inference on a spectrum.',
    )
    parser.add_argument('data_path')
    parser.add_argument("-m", "--model", dest="model", default=None,
                        required=True,
                        help="The type of model to do inference on.")
    parser.add_argument("-b", "--bounds", dest="bounds", default=None,
                        help="The bounds for the pixels.")
    parser.add_argument("-B", "--model-bounds", dest="mod_bounds",
                        required=True,
                        help="The bounds for the model parameters. Write "
                             "from highest to lowest order.")
    parser.add_argument("-P", "--prior", dest="prior_density",
                        help="The path to a 'figaro'-compatible .json file "
                             "containing a pdf posterior from another sample "
                             "to use as a prior on a following calibration.")
    parser.add_argument("-o", "--output-folder", dest="out_folder",
                        help="The folder where to save the output of this "
                             "command.")
    parser.add_argument("-p", "--postprocess", action='store_true',
                        dest="postprocess", default=False,
                        help="Whether to do a new inference or to open a "
                             "file of samples.")
    parser.add_argument("-d", "--density", action='store_true',
                        dest="generate_density", default=False,
                        help="With this flag set, the pipeline will generate a"
                             " new density from the given samples.")
    parser.add_argument("--log-scale", action='store_true',
                        dest="logscale", default=False,
                        help="Setting the log scale on the calibration fit "
                             "plot.")
    parser.add_argument("--error-scale", dest="error_scale", type=float,
                        default=None,
                        help="A number to scale the errorbars in the plot.")
    args = parser.parse_args()

    bounds, mod_bounds = parse_bounds(args)

    px, dpx, l, dl = parse_data_path(args, data_name='data_path')

    prior_density = None
    if args.prior_density is not None:
        prior_density = load_density(Path(args.prior_density))

    out_folder = args.out_folder
    if out_folder is None:
        out_folder = os.getcwd()
    out_folder = Path(out_folder)

    fit_model = CalibrationFit(
        px, dpx, l, dl,
        model=args.model,
        x_bounds=bounds, mod_bounds=mod_bounds,
        prior=prior_density
    )

    if not args.postprocess:
        work = cpnest.CPNest(
            fit_model,
            verbose=2,
            nlive=1000,  # 1000
            maxmcmc=5000,  # 5000
            nensemble=1,
            output=out_folder.joinpath(args.model)
        )
        work.run()
        post = work.posterior_samples.ravel()
    else:
        with h5py.File(
                out_folder.joinpath(args.model, 'cpnest.h5'), 'r'
        ) as f:
            post = np.asarray(f['combined']['posterior_samples'])

    columns = [post[par] for par in mod.names[args.model]]
    samples = np.column_stack(columns)

    labels = []
    for par in mod.names[args.model]:
        deg = par[-1]

        if deg == '0':
            units = '[nm]'
        elif deg == '1':
            units = '[nm / px]'
        else:
            units = f'[nm / px^{deg}]'
        labels.append(f'{par} {units}')

    c = corner.corner(
        samples,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        use_math_text=True,
    )

    c.savefig(
        out_folder.joinpath(
            args.model,
            f'joint_posterior_{args.model}.pdf'
        ),
        bbox_inches='tight'
    )

    medians = []
    for _, lam in enumerate(mod.names[args.model]):
        x = samples[:, _]
        q_16, q_50, q_84 = quantile(x, [0.16, 0.5, 0.84])
        q_m, q_p = q_50 - q_16, q_84 - q_50

        print(f'{lam} =\t{q_50:.5e} +{q_p:.3e} -{q_m:.3e}')

        medians.append(q_50)
    medians = np.asarray(medians)

    model_name = args.model
    if model_name is None:
        model_name = Path(os.getcwd())
    with h5py.File(
            Path(out_folder).joinpath(
                model_name,
                'median_params.h5'
            ), 'w') as f:
        f.create_dataset('params', data=medians)
        f.create_dataset('bounds', data=mod_bounds)

    px_width = max(px) - min(px)
    l_width = max(l) - min(l)

    xlims = [
        min(px) - 0.1 * px_width,
        max(px) + 0.1 * px_width
    ]

    llims = [
        min(l) - 0.1 * l_width,
        max(l) + 0.1 * l_width
    ]

    x = np.linspace(xlims[0], xlims[1], 1000)
    models = [
        fit_model.fit_model(x, *s) for s in samples
    ]
    Spectrum.show_calibration_fit(
        px, l, dpx, dl,
        x=x, model=models,
        show=True, save=True,
        name=out_folder.joinpath(args.model, 'calibration_fit.pdf'),
        legend=False,
        xlim=xlims,
        ylim=llims,
        units='nm',
        logscale=args.logscale,
        errorscale=args.error_scale
    )

    if args.generate_density:
        d_bounds = [
            [samples[:, _].min(), samples[:, _].max()]
            for _ in range(samples.shape[1])
        ]
        for _, b in enumerate(d_bounds):
            length = b[1] - b[0]
            b[0] -= length * 1e-2
            b[1] += length * 1e-2

        prior = get_priors(d_bounds, samples, probit=True)

        mix = DPGMM(bounds=d_bounds, prior_pars=prior, probit=True)
        density = mix.density_from_samples(samples)

        save_density([density],
                     folder=out_folder.joinpath(args.model),
                     name=f'params_posterior_density_{args.model}',
                     ext='json')

        labs = [lab.split(' ')[0] for lab in labels]
        uts = [lab.split(' ')[1][1:-1] for lab in labels]

        plot_multidim([density],
                      out_folder=out_folder.joinpath(args.model),
                      name=args.model,
                      labels=labs,
                      units=uts,
                      show=True, save=True, subfolder=False,
                      figsize=c.get_size_inches()[0],
                      samples=samples,
                      bounds=d_bounds,
                      median_label='DPGMM',
                      levels=[0.5, 0.68, 0.9, 0.95],
                      )


if __name__ == "__main__":
    main()
