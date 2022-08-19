import corner
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

from cpnest import CPNest
from pathlib import Path

from spectrumpy.dataset import Dataset
from spectrumpy.io import SpectrumPath
from spectrumpy.bayes_inference import (LinearPosterior,
                                        RotationPosterior)
from spectrumpy.function_models import Linear

cwd = Path(os.getcwd())

figsize_sbs = (7, 7)

sp_lamps_folder = cwd.joinpath('data')
ext = '_ATIK.fit'

lamps_path = [l for l in sp_lamps_folder.glob(f'*{ext}')]
names = [l.name.split(f'{ext}')[0] for l in lamps_path]
lamps = {n: l for n, l in zip(names, lamps_path)}

exe_data = cwd.joinpath('exercise_data')
samples_folder = exe_data.joinpath('1_hydrogen')
rot_samples = samples_folder.joinpath('rot_samples.h5')
rot_corr_samples = samples_folder.joinpath('rot_corr_samples.h5')
calibration_samples = samples_folder.joinpath('hydr_calibration_samples.h5')

find_angle = False
show_fitted_lamp = False
show_rotated_lamp = False
show_slices = False
find_angle_correction = False
show_corr_rotated_lamp = False
check_alignment = False
show_cropped_lamp = False
show_top_and_bottom = False
show_lamp_spectrum = False
calibrate_lines = False
show_calibration_fit = True


if __name__ == "__main__":
    hydr_file = SpectrumPath(lamps['hydrogen'], is_lamp=True)
    hydr = hydr_file.images['0']

    if find_angle:
        # upper line
        x = [100, 399, 906, 1570, 1769, 1860]
        sx = [2, 2, 4, 4, 3, 2]
        y = [1821, 1727, 1571, 1369, 1309, 1280]
        sy = [2, 2, 3, 1, 1, 1]

        x = np.array(x)
        y = np.array(y)
        sy = np.array(sy)
        sx = np.array(sx)

        rot_angle = RotationPosterior(x, y, sx, sy,
                                      bounds=[[-0.4, -0.1],
                                              [1778, 1880]])

        job = CPNest(
            rot_angle,
            verbose=1,
            nlive=1000,
            maxmcmc=1500,
            nnest=4,
            nensemble=4,
            seed=1234,
            output='./exercise_data/1_hydrogen/angle_rotation'
        )

        job.run()

        post = job.posterior_samples.ravel()
        samples = np.column_stack([np.rad2deg(np.arctan(post['m'])),
                                   post['q']])
        a_16, a_50, a_84 = corner.quantile(samples.T[0], [0.16, 0.5, 0.84])
        a_m, a_p = a_50 - a_16, a_84 - a_50
        alpha = (a_50, a_m, a_p)

        l_pars = np.column_stack([post['m'], post['q']])

        with h5py.File(rot_samples, 'w') as hf:
            hf.create_dataset('line params', data=l_pars)
            hf.create_dataset('alpha', data=np.asarray(alpha))

        fig = corner.corner(samples, labels=[r'$\alpha$', 'q'],
                            quantiles=[.05, .95],
                            filename='./exercise_data/1_hydrogen/'
                                     'joint_rotation.pdf',
                            show_titles=True,
                            title_fmt='.3e',
                            title_kwargs={'fontsize': 8},
                            label_kwargs={'fontsize': 8},
                            use_math_text=True)

        fig.savefig('./exercise_data/1_hydrogen/joint_rotation.pdf')
        plt.show()
        plt.close()
    else:
        with h5py.File(rot_samples, 'r') as hf:
            l_pars = hf['line params'][:]
            alpha = hf['alpha'][:]

    print(f"alpha = {alpha[0]:.3e} (+){alpha[2]:.3e} (-){alpha[1]:.3e}")
    alpha = alpha[0]

    ylims = hydr.image.shape[1]
    x = np.linspace(-0.5, ylims - 0.5, ylims + 1)
    models = np.array(
        [Linear.func(x, p[0], p[1]) for p in l_pars]
    )
    hydr.show(figsize=figsize_sbs,
              model=models,
              x=x,
              show=show_fitted_lamp,
              save=True,
              name='./exercise_data/1_hydrogen/image_hydr.pdf',
              title="Hydrogen lamp image")

    # Rotating image by alpha
    hydr_rotated = hydr.rotate_image(alpha)
    hydr_rotated.show(figsize=figsize_sbs,
                      show=show_rotated_lamp,
                      save=True,
                      name='./exercise_data/1_hydrogen/rotated_hydr.pdf',
                      title="Rotated hydrogen lamp image")

    # Checking if there are still residual effects taking three slices
    # at the top, middle and bottom of the spectrum.
    slices = [1150, 1250, 1350, 1450, 1550, 1650, 1768]
    mid_line = 1450
    hydr_corr_slices = {
        line: hydr_rotated.slice_image(line) for line in slices
    }

    int_models = {
        line: hydr_slice.run_integration()
        for line, hydr_slice in hydr_corr_slices.items()
    }

    for line in int_models.keys():
        exec(f"def slice_{line}(x): return int_models[{line}].spectrum")

    models = [
        eval(f"slice_{line}") for line in int_models.keys()
    ]

    mid_model = int_models[mid_line]
    x_len = len(mid_model.spectrum)
    x_sup = x_len - 1
    models.pop(slices.index(mid_line))
    mid_model.show(figsize=figsize_sbs,
                   model=models,
                   x=np.linspace(0, x_sup, x_len),
                   legend=True,
                   show=show_slices,
                   save=True,
                   name='./exercise_data/1_hydrogen/hydr_align_slices.pdf',
                   title="Slices of rotated hydrogen spectrum image")

    if find_angle_correction:
        y = [1744, 1745, 1746, 1747, 1750, 1752, 1754]
        sy = [2, 4, 4, 3, 3, 2, 2]
        x = [1768, 1650, 1550, 1450, 1350, 1250, 1150]
        sx = [0, 0, 0, 0, 0, 0, 0]

        x = np.array(x)
        y = np.array(y)
        sy = np.array(sy)
        sx = np.array(sx)

        rot_angle = RotationPosterior(x, y, sx, sy,
                                      bounds=[[-2, 2],
                                              [1e3, 2e3]])

        job = CPNest(
            rot_angle,
            verbose=1,
            nlive=1000,
            maxmcmc=1500,
            nnest=4,
            nensemble=4,
            seed=1234,
            output='./exercise_data/1_hydrogen/angle_correction/'
        )

        job.run()

        post = job.posterior_samples.ravel()
        samples = np.column_stack([np.rad2deg(np.arctan(post['m'])),
                                      post['q']])

        a_16, a_50, a_84 = corner.quantile(samples.T[0], [0.16, 0.5, 0.84])
        a_m, a_p = a_50 - a_16, a_84 - a_50
        alpha_corr = (a_50, a_m, a_p)

        l_pars = np.column_stack([post['m'], post['q']])

        with h5py.File(rot_corr_samples, 'w') as hf:
            hf.create_dataset('line params', data=l_pars)
            hf.create_dataset('alpha_corr',
                              data=np.asarray(alpha_corr))

        fig = corner.corner(samples, labels=[r'$\alpha$', 'q'],
                            quantiles=[.05, .95],
                            filename='./exercise_data/1_hydrogen/'
                                     'joint_rotation_correction_alpha.pdf',
                            show_titles=True,
                            title_fmt='.3e',
                            title_kwargs={'fontsize': 8},
                            label_kwargs={'fontsize': 8},
                            use_math_text=True)

        fig.savefig('./exercise_data/1_hydrogen/'
                    'joint_rotation_correction_alpha.pdf')
        plt.show()
        plt.close()
    else:
        with h5py.File(rot_corr_samples, 'r') as hf:
            l_pars = hf['line params'][:]
            alpha_corr = hf['alpha_corr'][:]

    print(f"alpha_corr = {alpha_corr[0]:.3e} "
          f"(+){alpha_corr[2]:.3e} (-){alpha_corr[1]:.3e}")
    alpha_corr = alpha_corr[0]

    # Rotating image by alphacorr
    hydr_rotated = hydr.rotate_image(alpha-alpha_corr)
    hydr_rotated.show(figsize=figsize_sbs,
                      show=show_corr_rotated_lamp,
                      save=True,
                      name='./exercise_data/1_hydrogen/corr_rotated_hydr.pdf',
                      title="Rotated hydrogen lamp image")

    if check_alignment:
        slices = [1130, 1230, 1330, 1400, 1430, 1480, 1530, 1630, 1730]
        mid_line = 1430
        hydr_corr_slices = {
            line: hydr_rotated.slice_image(line) for line in slices
        }

        int_models = {
            line: hydr_slice.run_integration()
            for line, hydr_slice in hydr_corr_slices.items()
        }

        for line in int_models.keys():
            exec(f"def slice_{line}(x): return int_models[{line}].spectrum")

        models = [
            eval(f"slice_{line}") for line in int_models.keys()
        ]

        mid_model = int_models[mid_line]
        x_len = len(mid_model.spectrum)
        x_sup = x_len - 1
        models.pop(slices.index(mid_line))
        mid_model.show(figsize=figsize_sbs,
                       model=models,
                       x=np.linspace(0, x_sup, x_len),
                       legend=True,
                       show=True,
                       save=True,
                       title="Slices of rotated hydrogen spectrum image"
                             " with correction")

    crops_y = [1400, 1480]
    hydr_cropped = hydr_rotated.crop_image(crop_y=crops_y)
    hydr_cropped.show(figsize=figsize_sbs,
                      show=show_cropped_lamp,
                      save=True,
                      name='./exercise_data/1_hydrogen/cropped_hydr.pdf',
                      title="Cropped hydrogen lamp image")

    if show_top_and_bottom:
        ylims = hydr_cropped.image.shape[0]
        hydr_cropped_up = hydr_cropped.crop_image(crop_y=[ylims-1, ylims])
        hydr_cropped_dw = hydr_cropped.crop_image(crop_y=[21, 22])

        sp_ref_up = hydr_cropped_up.run_integration()
        sp_ref_dw = hydr_cropped_dw.run_integration()

        def sp_slice_dw(x):
            return sp_ref_dw.spectrum

        sp_ref_up.show(figsize=figsize_sbs,
                       model=sp_slice_dw,
                       x=np.linspace(0,
                                     len(sp_ref_dw.spectrum) - 1,
                                     len(sp_ref_dw.spectrum)),
                       show=True,
                       save=True,
                       name='./exercise_data/1_hydrogen/'
                            'hydr_spectrum_slices.pdf',
                       title="Slices of cropped hydrogen spectrum image")

    sp = hydr_cropped.run_integration()
    sp.show(figsize=figsize_sbs,
            show=show_lamp_spectrum,
            save=True,
            name='./exercise_data/1_hydrogen/int_spectrum_hydr.pdf',
            title="Hydrogen lamp spectrum")

    # Calibration lines
    px = [1041, 1736, 1945, 2040]
    s_px = [5, 3, 3, 3]
    lam = [656.279, 486.135, 434.0472, 410.1734]  # nm
    s_lam = [3e-3, 5e-3, 6e-4, 6e-4]


    if calibrate_lines:
        bounds = [[-0.5, 0], [800, 1000]]
        linear_posterior = LinearPosterior(px, lam, s_px, s_lam, bounds)

        job = CPNest(
            linear_posterior,
            verbose=1,
            nlive=1000,
            maxmcmc=1500,
            nnest=4,
            nensemble=4,
            seed=1234,
            output='./exercise_data/1_hydrogen/line_calibration/'
        )

        job.run()

        post = job.posterior_samples.ravel()
        samples = np.column_stack([post['m'], post['q']])
        
        fig = corner.corner(samples, labels=['m', 'q'],
                            quantiles=[.05, .95],
                            filename='joint_calibration.pdf',
                            show_titles=True,
                            title_fmt='.3e',
                            title_kwargs={'fontsize': 8},
                            label_kwargs={'fontsize': 8},
                            use_math_text=True)

        fig.savefig('./exercise_data/joint_calibration.pdf')
        plt.show()
        plt.close()
    
        with h5py.File(calibration_samples, 'w') as hf:
            hf.create_dataset('line params', data=samples)
    else:
        with h5py.File(calibration_samples, 'r') as hf:
            samples = hf['line params'][:]

    m_16, m_50, m_84 = corner.quantile(samples.T[0], [0.16, 0.5, 0.84])
    m_m, m_p = m_50 - m_16, m_84 - m_50

    q_16, q_50, q_84 = corner.quantile(samples.T[1], [0.16, 0.5, 0.84])
    q_m, q_p = q_50 - q_16, q_84 - q_50

    print(f"m = {m_50:.3e} (+){m_p:.3e} (-){m_m:.3e}")
    print(f"q = {q_50:.3e} (+){q_p:.3e} (-){q_m:.3e}")

    # Plot calibration
    x = np.linspace(1000, 2100, 1000)
    models = [
        Linear.func(x, s[0], s[1]) for s in samples
    ]

    sp.show_calibration_fit(px, lam, s_px, s_lam=s_lam,
                            model=models,
                            x=x,
                            title="Spectrum calibration",
                            units='nm',
                            xlim=[1000, 2100],
                            ylim=[400, 700],
                            show=show_calibration_fit,
                            save=True,
                            name='./exercise_data/1_hydrogen/'
                                 'calibration_fit.pdf',
                            legend=False)

    # Save calibration
    dataset = Dataset(lines=lam, errlines=s_lam,
                      px=px, errpx=s_px,
                      names=[r'H-$\alpha$', r'H-$\beta$', r'H-$\gamma$',
                             r'H-$\delta$'])
    sp.assign_dataset(dataset)
    sp.assign_calibration(Linear(m_50, q_50), units='nm')

    sp.show(show=True,
            save=True,
            name='./exercise_data/1_hydrogen/calibrated_H-I.pdf',
            legend=False,
            calibration=True,
            title='Calibrated H-I spectrum'
            )

    sp.save_info(filename='./exercise_data/1_hydrogen/hydr_calibration.json')
