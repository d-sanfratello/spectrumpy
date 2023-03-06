import corner
import cpnest
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import optparse as op

from pathlib import Path

from spectrumpy.bayes_inference import RotationFit

# FIXME: Test


def main():
    parser = op.OptionParser()
    parser.add_option("-i", "--input", type="string", dest="data_file",
                      default=None,
                      help="The file to input the point on which the "
                           "inference is carried out.")
    parser.add_option("-o", "--output-folder", dest="out_folder",
                      default=Path(os.getcwd()),
                      help="The folder where to save the output of this "
                           "command.")
    parser.add_option("-p", "--postprocess", action='store_true',
                      dest="postprocess", default=False,
                      help="Whether to do a new inference or to open a file "
                           "of samples.")
    parser.add_option("-b", "--bounds", type="string",
                      dest="bounds", default=None,
                      help="The bounds for the pixels. If unset, it selects "
                           "an interval plus or minus 5 sigma around each "
                           "point.")
    parser.add_option("-B", "--model-bounds", type="string",
                      dest="mod_bounds", default=None,
                      help="The bounds for the model parameters. Write from "
                           "highest to lowest order.")
    (options, args) = parser.parse_args()

    if not options.postprocess and options.mod_bounds is None:
        raise ValueError("When trying a new inference, bounds must be set.")

    if options.bounds is not None:
        bounds = eval(options.bounds)
        bounds = np.array(bounds).flatten()
    else:
        bounds = options.bounds

    if options.mod_bounds is not None:
        mod_bounds = eval(options.mod_bounds)

    if options.data_file is not None:
        data = np.genfromtxt(options.data_file, names=True)
        x = data['x']
        dx = data['dx']
        y = data['y']
        dy = data['dy']
    else:
        raise AttributeError(
            "A dataset must be provided."
        )

    if not options.postprocess:
        fit_model = RotationFit(
            x, dx, y, dy, x_bounds=bounds, mod_bounds=mod_bounds
        )

        work = cpnest.CPNest(
            fit_model,
            verbose=2,
            nlive=1000,  # 1000
            maxmcmc=5000,  # 5000
            nensemble=1,
            output=Path(options.out_folder).joinpath('rotation_fit')
        )
        work.run()
        post = work.posterior_samples.ravel()

        samples = np.column_stack([
            np.rad2deg(np.arctan(post['x_1'])),
            post['x_0']
        ])

        with h5py.File(
            Path(options.out_folder).joinpath(
                'rotation_samples.h5'), 'w'
        ) as f:
            f.create_dataset('rotation_params',
                             data=samples)
    else:
        with h5py.File(
                Path(options.out_folder).joinpath(
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
        Path(options.out_folder).joinpath(
            f'joint_posterior_rotation.pdf'
        ),
        bbox_inches='tight'
    )


if __name__ == "__main__":
    main()
