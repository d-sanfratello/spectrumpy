import corner
import cpnest
import h5py
import numpy as np
import os
import argparse as ag

from pathlib import Path

from spectrumpy.bayes_inference import RotationFit
from spectrumpy.io import parse_data_path, parse_bounds, parse_tentative


def main():
    parser = ag.ArgumentParser(
        prog='sp-find-angle',
        description='This script performs an inference on some given data to '
                    'find the correct rotation angle for a spectrum image.',
    )
    parser.add_argument('data_path')
    parser.add_argument("-o", "--output-folder", dest="out_folder",
                        default=Path(os.getcwd()),
                        help="The folder where to save the output of this "
                             "command.")
    parser.add_argument("-p", "--postprocess", action='store_true',
                        dest="postprocess",
                        default=False,
                        help="Whether to do a new inference or to open a "
                             "file of samples.")
    parser.add_argument("-b", "--bounds", dest="bounds",
                        default=None,
                        help="The bounds for the pixels. If unset, "
                             "it selects an interval plus or minus 5 sigma "
                             "around each point.")
    parser.add_argument("-B", "--model-bounds",
                        dest="mod_bounds",
                        default=None,
                        help="The bounds for the model parameters. Write "
                             "from highest to lowest order.")
    parser.add_argument("--tentative", action='store_true',
                        dest='tentative', default=False,
                        help="if this flag is set, the script performs "
                             "only a tentative fit using a smaller number of "
                             "samples.")
    args = parser.parse_args()

    bounds, mod_bounds = parse_bounds(args)

    x, dx, y, dy = parse_data_path(
        args,
        data_name='data_path'
    )

    if mod_bounds is not None:
        angles = mod_bounds[0]
        low_m = np.tan(np.deg2rad(angles[0]))
        high_m = np.tan(np.deg2rad(angles[1]))

        mod_bounds[0] = [low_m, high_m]

    if not args.postprocess:
        nlive, maxmcmc = parse_tentative(args)

        fit_model = RotationFit(
            x, dx, y, dy, x_bounds=bounds, mod_bounds=mod_bounds
        )

        work = cpnest.CPNest(
            fit_model,
            verbose=2,
            nlive=nlive,  # 1000
            maxmcmc=maxmcmc,  # 5000
            nensemble=1,
            output=Path(args.out_folder).joinpath('rotation_fit')
        )
        work.run()
        post = work.posterior_samples.ravel()

        samples = np.column_stack([
            np.rad2deg(np.arctan(post['x_1'])),
            post['x_0']
        ])

        with h5py.File(
            Path(args.out_folder).joinpath(
                'rotation_samples.h5'), 'w'
        ) as f:
            f.create_dataset('rotation_params',
                             data=samples)
    else:
        with h5py.File(
                Path(args.out_folder).joinpath(
                    'rotation_samples.h5'), 'r'
        ) as f:
            post = np.asarray(f['rotation_params'])

        samples = np.column_stack([
            post.T[0],
            post.T[1]
        ])

    a_16, a_50, a_84 = corner.quantile(
        samples.T[0], [0.16, 0.5, 0.84])
    a_m, a_p = a_50 - a_16, a_84 - a_50

    q_16, q_50, q_84 = corner.quantile(
        samples.T[1], [0.16, 0.5, 0.84])
    q_m, q_p = q_50 - q_16, q_84 - q_50

    print()
    print(f'angle [deg] =\t{a_50:.3e} +{a_p:.3e} -{a_m:.3e}')
    print(f'offset [px] =\t{q_50:.3e} +{q_p:.3e} - {q_m:.3e}')

    labels = [r'$\alpha$ [deg]', '$x_0$ [px]']
    c = corner.corner(
        samples,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        use_math_text=True,
    )
    c.savefig(
        Path(args.out_folder).joinpath(
            f'joint_posterior_rotation.pdf'
        ),
        bbox_inches='tight'
    )


if __name__ == "__main__":
    main()
