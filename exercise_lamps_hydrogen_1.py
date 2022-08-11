import corner
import h5py
import numpy as np
import os

from pathlib import Path

from spectrumpy.io import SpectrumPath
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

find_angle = False


if __name__ == "__main__":
    hydr_file = SpectrumPath(lamps['hydrogen'], is_lamp=True)
    hydr = hydr_file.images['0']

    # hydr.show(figsize=figsize_sbs, save=False, show=False)

    if find_angle:
        x = [100, 399, 906, 1570, 1769, 1860]
        sx = [2, 2, 4, 4, 3, 2]
        y = [1821, 1727, 1571, 1369, 1309, 1280]
        sy = [2, 2, 3, 1, 1, 1]

        job, alpha = hydr.find_rotation_angle(
            x, y, sy, sx,
            bounds=[[-0.4, -0.1], [1840, 1880]],
            verbose=1,
            show=False,
            save=True,
            nlive=1000
            )

        post = job.posterior_samples.ravel()
        l_pars = np.column_stack([post['m'], post['q']])

        with h5py.File(rot_samples, 'w') as hf:
            hf.create_dataset('line params', data=l_pars)
            hf.create_dataset('alpha', data=np.asarray(alpha))
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
    hydr.show(figsize=figsize_sbs, model=models, x=x,
              save=True, name='./exercise_data/image_hydr.pdf')

    hydr_rotated = hydr.rotate_image(alpha)
    hydr_rotated.show(figsize=figsize_sbs,
                      save=True, name='./exercise_data/rotated_hydr.pdf')

    crops_y = [1125, 1778]
    hydr_cropped = hydr_rotated.crop_image(crop_y=crops_y)
    hydr_cropped.show(figsize=figsize_sbs,
                      save=True, name='./exercise_data/cropped_hydr.pdf')

    sp = hydr_cropped.run_integration()
    sp.show(figsize=figsize_sbs,
            save=True, name='./exercise_data/int_spectrum_hydr.pdf')
