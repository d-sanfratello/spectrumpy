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
from spectrumpy.bayes_inference import LinearPosterior
from spectrumpy.function_models import Linear

cwd = Path(os.getcwd())

figsize_sbs = (7, 7)

sp_lamps_folder = cwd.joinpath('data')
ext = '_ATIK.fit'

lamps_path = [l for l in sp_lamps_folder.glob(f'*{ext}')]
names = [l.name.split(f'{ext}')[0] for l in lamps_path]
lamps = {n: l for n, l in zip(names, lamps_path)}

samples_folder = cwd.joinpath('exercise_data')
rot_samples = samples_folder.joinpath('rot_samples.h5')
rot_corr_samples = samples_folder.joinpath('rot_corr_samples.h5')
calibration_samples = samples_folder.joinpath('calibration_samples.h5')

calibrate_lines = False

if __name__ == "__main__":
    info = Spectrum.load_info('./exercise_data/hydr_calibration.json')
    neon_file = SpectrumPath(lamps['neon'], is_lamp=True, info=info)
    neon = neon_file.images['0']

    # Rotating image by known angle
    neon_rotated = neon.rotate_image(info['angle'])
    neon_rotated.show(figsize=figsize_sbs,
                      show=False,
                      save=True,
                      name='./exercise_data/rotated_neon.pdf',
                      title="Rotated neon lamp image")

    crops_y = info['crop_y']
    neon_cropped = neon_rotated.crop_image(crop_y=crops_y)
    neon_cropped.show(figsize=figsize_sbs,
                      show=False,
                      save=True,
                      name='./exercise_data/cropped_neon.pdf',
                      title="Cropped neon lamp image")

    sp = neon_cropped.run_integration()
    sp.show(show=True, save=False,
            name='./exercise_data/calibrated_Ne-I.pdf',
            legend=False,
            calibration=True,
            title='Calibrated Ne-I spectrum')


    # Calibration lines
    px = [1041, 1737, 1945, 2041]
    s_px = [4, 3, 3, 3]
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
            seed=1234
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
    # else:
        # with h5py.File(calibration_samples, 'r') as hf:
        #     samples = hf['line params'][:]

    # m_16, m_50, m_84 = corner.quantile(samples.T[0], [0.16, 0.5, 0.84])
    # m_m, m_p = m_50 - m_16, m_84 - m_50
    #
    # q_16, q_50, q_84 = corner.quantile(samples.T[1], [0.16, 0.5, 0.84])
    # q_m, q_p = q_50 - q_16, q_84 - q_50
    #
    # print(f"m = {m_50:.3e} (+){m_p:.3e} (-){m_m:.3e}")
    # print(f"q = {q_50:.3e} (+){q_p:.3e} (-){q_m:.3e}")

    # Plot calibration
    # x = np.linspace(1000, 2100, 1000)
    # models = [
    #     Linear.func(x, s[0], s[1]) for s in samples
    # ]

    # sp.show_calibration_fit(px, lam, s_px, s_lam=s_lam,
    #                         model=models,
    #                         x=x,
    #                         title="Spectrum calibration",
    #                         units='nm',
    #                         xlim=[1000, 2100],
    #                         ylim=[400, 700],
    #                         show=True, save=False,
    #                         name='./exercise_data/calibration_fit.pdf',
    #                         legend=False)

    # Save calibration
    # dataset = Dataset(lines=lam, errlines=s_lam,
    #                   px=px, errpx=s_px,
    #                   names=[r'H-$\alpha$', r'H-$\beta$', r'H-$\gamma$',
    #                          r'H-$\delta$'])
    # sp.assign_dataset(dataset)
    # sp.assign_calibration(Linear(m_50, q_50), units='nm')

    # sp.show(show=True, save=False,
    #         name='./exercise_data/calibrated_H-I.pdf',
    #         legend=False,
    #         calibration=True,
    #         title='Calibrated H-I spectrum')

    # sp.save_info(filename='./exercise_data/neon_calibration.json')