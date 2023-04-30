import corner
import cpnest
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse as ag

from pathlib import Path

from spectrumpy.core import CalibratedSpectrum
from spectrumpy.bayes_inference import LineFit
from spectrumpy.bayes_inference import models as calib_mod
from spectrumpy.io import parse_spectrum_path, parse_bounds, parse_tentative


def main():
    parser = ag.ArgumentParser(
        prog='sp-find-line',
        description='This script performs an inference on an interval of '
                    'points to fit a Voigt profile on a line.',
    )
    parser.add_argument('spectrum_path')
    parser.add_argument('calibration')
    parser.add_argument("-i", "--interval",
                        dest="bounds", required=True,
                        help="the bounds between which to look for a line.")
    parser.add_argument("-o", "--output-folder", dest="out_folder",
                        default=Path(os.getcwd()),
                        help="The folder where to save the output of this "
                             "command.")
    parser.add_argument("-p", "--postprocess", action='store_true',
                        dest="postprocess",
                        default=False,
                        help="Whether to do a new inference or to open a "
                             "file of samples.")
    parser.add_argument("-B", "--model-bounds",
                        dest="mod_bounds",
                        default=None,
                        help="The bounds for the model parameters. Write "
                             "from highest to lowest order.")
    parser.add_argument("-u", "--units", dest="units", default='nm',
                        help="The units of the calibrated wavelengths.")
    parser.add_argument("--use-lorentz", dest='lorentz',
                        action='store_true', default=False,
                        help="if this flag is set, a Lorentz profile is used "
                             "instead of a Voigt profile.")
    parser.add_argument("--absorption", action='store_true', default=False,
                        dest='absorption',
                        help="if this flag is set, the script tries to fit "
                             "an absorption line.")
    parser.add_argument("--emission", action='store_true', default=False,
                        dest='emission',
                        help="if this flag is set, the script tries to fit "
                             "an emission line.")
    parser.add_argument("--tentative", action='store_true',
                        dest='tentative', default=False,
                        help="if this flag is set, the script performs "
                             "only a tentative fit using a smaller number of "
                             "samples.")
    args = parser.parse_args()

    calibrated = False

    if not args.emission and not args.absorption:
        parser.error(
            "You need to specify either an absorption or an emission line fit."
        )
    elif args.emission:
        abs_ = False
    elif args.absorption:
        abs_ = True

    bounds, mod_bounds = parse_bounds(args)

    with h5py.File(Path(args.calibration).joinpath(
            'median_params.h5'), 'r') as f:
        calib_parameters = f['params'][:]

    mod_name = calib_mod.available_models[len(calib_parameters) - 2]
    calibration = calib_mod.models[mod_name]

    cnt = parse_spectrum_path(
        args,
        data_name='spectrum_path'
    )
    if isinstance(cnt, CalibratedSpectrum):
        calibrated = True

    if not calibrated:
        cnt.assign_calibration(
            calibration=calibration,
            pars=calib_parameters,
            units=args.units
        )
        cnt = cnt.return_calibrated()
    else:
        cnt = cnt.cut(
            low=bounds[0],
            high=bounds[1]
        )

    fit_model = LineFit(
        cnt.wl,
        cnt.sp,
        np.sqrt(cnt.sp),
        mod_bounds=mod_bounds, absorption=abs_,
        use_lorentz=args.lorentz
    )

    if not args.postprocess:
        nlive, maxmcmc = parse_tentative(args)

        work = cpnest.CPNest(
            fit_model,
            verbose=2,
            nlive=nlive,  # 1000
            maxmcmc=maxmcmc,  # 5000
            nensemble=1,
            output=Path(args.out_folder).joinpath('line_fit')
        )
        work.run()
        post = work.posterior_samples.ravel()

        if args.lorentz:
            samples = np.column_stack([
                post['loc'],
                post['gamma'],
                post['scale'],
                post['offset']
            ])
        else:
            samples = np.column_stack([
                post['loc'],
                post['sigma'],
                post['gamma'],
                post['scale'],
                post['offset']
            ])

        with h5py.File(
                Path(args.out_folder).joinpath(
                    'line_fit_samples.h5'), 'w'
        ) as f:
            f.create_dataset('line_fit_params',
                             data=samples)
    else:
        with h5py.File(
                Path(args.out_folder).joinpath(
                    'line_fit_samples.h5'), 'r'
        ) as f:
            post = np.asarray(f['line_fit_params'])

        if args.lorentz:
            samples = np.column_stack([
                post.T[0],
                post.T[1],
                post.T[2],
                post.T[3],
            ])
        else:
            samples = np.column_stack([
                post.T[0],
                post.T[1],
                post.T[2],
                post.T[3],
                post.T[4],
            ])

    if args.lorentz:
        len_q = 4
    else:
        len_q = 5

    quantiles = [
        corner.quantile(samples.T[i], [0.16, 0.5, 0.84])
        for i in range(len_q)
    ]
    q = quantiles

    errorbars = [
        [_[1] - _[0], _[2] - _[1]] for _ in quantiles
    ]
    eb = errorbars

    print()
    print(f'loc [{args.units}] ='
          f'\t{q[0][1]:.8e} +{eb[0][1]:.3e} - {eb[0][0]:.3e}')
    if not args.lorentz:
        print(f'sigma [{args.units}] ='
              f'\t{q[1][1]:.8e} +{eb[1][1]:.3e} - {eb[1][0]:.3e}')
        print(f'gamma [{args.units}] ='
              f'\t{q[2][1]:.8e} +{eb[2][1]:.3e} - {eb[2][0]:.3e}')
        print(f'scale [a.u.] =\t{q[3][1]:.3e} +{eb[3][1]:.3e} - {eb[3][0]:.3e}')
        print(f'offset [a.u.] =\t{q[4][1]:.5e} +{eb[4][1]:.3e} -'
              f' {eb[4][0]:.3e}')
    else:
        print(f'gamma [{args.units}] = \t{q[1][1]:.8e} +{eb[1][1]:.3e} - '
              f'{eb[1][0]:.3e}')
        print(f'scale [a.u.] =\t{q[2][1]:.3e} +{eb[2][1]:.3e} -'
              f' {eb[2][0]:.3e}')
        print(f'offset [a.u.] =\t{q[3][1]:.5e} +{eb[3][1]:.3e} -'
              f' {eb[3][0]:.3e}')

    labels = [
        rf'$\mu_\lambda$ [{args.units}]',
        rf'$\sigma$ [{args.units}]',
        rf'$\gamma$ [{args.units}]',
        r'$S$ [arb.unit]',
        rf'$offset$ [arb.unit]'
    ]

    if args.lorentz:
        labels.remove(rf'$\sigma$ [{args.units}]',)

    c = corner.corner(
        samples,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        use_math_text=True,
    )
    c.savefig(
        Path(args.out_folder).joinpath(
            f'joint_posterior_line_fit.pdf'
        ),
        bbox_inches='tight'
    )

    fig = plt.figure(figsize=(7, 7))
    ax = fig.gca()
    ax.grid(axis='both', which='major')

    ax.errorbar(
        cnt.wl.flatten(),
        cnt.sp.flatten(),
        np.sqrt(cnt.sp.flatten()),
        capsize=2, linestyle='', ms=2, marker='.',
    )

    x = np.linspace(
        cnt.wl.flatten().min(),
        cnt.wl.flatten().max(),
        1000
    )

    model = np.asarray([
        fit_model.model(x, *samples[_])
        for _ in range(samples.shape[0])
    ])
    l, m, h = np.percentile(model,
                            [16, 50, 95], axis=0)

    ax.plot(x, m, lw=0.8, color='r')
    ax.fill_between(x, l, h, facecolor='red', alpha=0.25)

    ax.set_xlabel(f'[{args.units}]')
    ax.set_ylabel('[arb.units]')

    fig.savefig('line_fit.pdf')
    plt.show()


if __name__ == "__main__":
    main()
