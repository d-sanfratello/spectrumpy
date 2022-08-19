import corner
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

from cpnest import CPNest
from pathlib import Path

from spectrumpy.core.spectrum import Spectrum
from spectrumpy.dataset import Dataset
from spectrumpy.io import SpectrumPath
from spectrumpy.bayes_inference import QuadraticPosterior
from spectrumpy.function_models import Quadratic

cwd = Path(os.getcwd())

figsize_sbs = (7, 7)

sp_lamps_folder = cwd.joinpath('data')
ext = '_ATIK.fit'

lamps_path = [l for l in sp_lamps_folder.glob(f'*{ext}')]
names = [l.name.split(f'{ext}')[0] for l in lamps_path]
lamps = {n: l for n, l in zip(names, lamps_path)}

exe_data = cwd.joinpath('exercise_data')
samples_folder = exe_data.joinpath('2_Neon')
calibration_samples = samples_folder.joinpath('Ne_calib_samples.h5')

calibrate_lines = False

if __name__ == "__main__":
    info = Spectrum.load_info(
        './exercise_data/1_hydrogen/hydr_calibration.json'
    )
    neon_file = SpectrumPath(lamps['neon'], is_lamp=True, info=info)
    neon = neon_file.images['0']

    # Rotating image by known angle
    neon_rotated = neon.rotate_image(info['angle'])
    neon_rotated.show(figsize=figsize_sbs,
                      show=False,
                      save=True,
                      name='./exercise_data/2_Neon/rotated_neon.pdf',
                      title="Rotated neon lamp image")

    crops_y = info['crop_y']
    neon_cropped = neon_rotated.crop_image(crop_y=crops_y)
    neon_cropped.show(figsize=figsize_sbs,
                      show=False,
                      save=True,
                      name='./exercise_data/2_Neon/cropped_neon.pdf',
                      title="Cropped neon lamp image")

    sp = neon_cropped.run_integration()
    sp.show(show=True,
            save=False,
            name='./exercise_data/2_Neon/calibrated_Ne-I.pdf',
            legend=False,
            calibration=True,
            title='Calibrated Ne-I spectrum',
            overlay_pixel=True, inverted_overlay=True)

    # Calibration lines - Hydrogen
    px_prev = px = [1041, 1736, 1945, 2040]
    s_px_prev = [5, 3, 3, 3]
    lam_prev = [656.279, 486.135, 434.0472, 410.1734]  # nm
    s_lam_prev = [3e-3, 5e-3, 6e-4, 6e-4]
    names_prev = [r'H-$\alpha$', r'H-$\beta$', r'H-$\gamma$', r'H-$\delta$']
    # Calibration lines
    px = [584.0, 639.3, 695.0, 706.1]  # nm
    s_px = [1.5, 3.8, 1.2, 1.7]
    lam = [585.24878, 640.22480, 692.94672, 703.24128]  # nm
    s_lam = [5e-5, 10e-5, 4e-5, 4e-5]
    names = [r'Ne-I', r'Ne-I', r'Ne-I', r'Ne-I']

    for p, s, l, s_l, n in zip(px_prev, s_px_prev, lam_prev, s_lam_prev,
                               names_prev):
        px.append(p)
        s_px.append(s)
        lam.append(l)
        s_lam.append(s_l)
        names.append(n)

    dataset = Dataset(px=px, errpx=s_px, lines=lam, errlines=s_lam,
                      names=names)

    if calibrate_lines:
        bounds = [[-10, 10], [-2, 5], [-20, 20]]
        quadratic_posterior = QuadraticPosterior(px, lam, s_px, s_lam, bounds)

        job = CPNest(
            quadratic_posterior,
            verbose=1,
            nlive=1000,
            maxmcmc=1500,
            nnest=4,
            nensemble=4,
            seed=1234,
            output='./exercise_data/2_Neon/'
        )

        job.run()

        post = job.posterior_samples.ravel()
        samples = np.column_stack([post['a'], post['b'], post['c']])

        with h5py.File(calibration_samples, 'w') as hf:
            hf.create_dataset('params', data=samples)

        fig = corner.corner(samples, labels=['a', 'b' 'c'],
                            quantiles=[.05, .95],
                            filename='Ne_calibration.pdf',
                            show_titles=True,
                            title_fmt='.3e',
                            title_kwargs={'fontsize': 8},
                            label_kwargs={'fontsize': 8},
                            use_math_text=True)

        fig.savefig('./exercise_data/2_Neon/Ne_calibration.pdf')
        plt.show()
        plt.close()
    else:
        with h5py.File(calibration_samples, 'r') as hf:
            samples = hf['params'][:]

    a_16, a_50, a_84 = corner.quantile(samples.T[0], [0.16, 0.5, 0.84])
    a_m, a_p = a_50 - a_16, a_84 - a_50

    b_16, b_50, b_84 = corner.quantile(samples.T[1], [0.16, 0.5, 0.84])
    b_m, b_p = b_50 - b_16, b_84 - b_50

    c_16, c_50, c_84 = corner.quantile(samples.T[2], [0.16, 0.5, 0.84])
    c_m, c_p = c_50 - c_16, c_84 - c_50

    print(f"a = {a_50:.3e} (+){a_p:.3e} (-){a_m:.3e}")
    print(f"b = {b_50:.3e} (+){b_p:.3e} (-){b_m:.3e}")
    print(f"c = {c_50:.3e} (+){c_p:.3e} (-){c_m:.3e}")

    # Plot calibration
    asc = np.linspace(400, 710, 1000)
    models = [
        Quadratic.func(asc, s[0], s[1], s[2]) for s in samples
    ]

    sp.show_calibration_fit(px, lam, s_px, s_lam=s_lam,
                            model=models,
                            x=asc,
                            title="Spectrum calibration",
                            units='nm',
                            xlim=[400, 710],
                            ylim=[400, 710],
                            show=True,
                            save=False,
                            name='./exercise_data/2_Neon/'
                                 'Ne_calibration_fit.pdf',
                            legend=False)

    # Save calibration
    # dataset = Dataset(lines=lam, errlines=s_lam,
    #                   px=px, errpx=s_px,
    #                   names=[r'H-$\alpha$', r'H-$\beta$', r'H-$\gamma$',
    #                          r'H-$\delta$'])
    # sp.assign_dataset(dataset)
    # sp.assign_calibration(Linear(m_50, q_50), units='nm')

    # sp.show(show=True, save=False,
    #         name='./exercise_data/2_Neon/calibrated_H-I.pdf',
    #         legend=False,
    #         calibration=True,
    #         title='Calibrated H-I spectrum')

    # sp.save_info(filename='./exercise_data/2_Neon/neon_calibration.json')
