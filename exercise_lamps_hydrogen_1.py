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
rot_corr_samples = samples_folder.joinpath('rot_corr_samples.h5')

find_angle = False


if __name__ == "__main__":
    hydr_file = SpectrumPath(lamps['hydrogen'], is_lamp=True)
    hydr = hydr_file.images['0']

    # hydr.show(figsize=figsize_sbs, save=False, show=False)

    if find_angle:
        # upper line
        x = [100, 399, 906, 1570, 1769, 1860]
        sx = [2, 2, 4, 4, 3, 2]
        y = [1821, 1727, 1571, 1369, 1309, 1280]
        sy = [2, 2, 3, 1, 1, 1]
        # lower line
        # x += [214, 723, 1392, 1593, 1686]
        # y += [1095, 947, 756, 700, 674]
        # sx += [4, 4, 3, 3, 2]
        # sy += [4, 2, 2, 2, 1]

        job, alpha = hydr.find_rotation_angle(
            x, y, sy, sx,
            bounds=[[-0.4, -0.1], [1778, 1880]],
            verbose=1,
            show=True,
            save=True,
            nlive=1000,
            name='./exercise_data/joint_rotation.pdf'
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
    hydr.show(figsize=figsize_sbs,
              model=models,
              x=x,
              show=False,
              save=True,
              name='./exercise_data/image_hydr.pdf',
              title="Hydrogen lamp image")

    # Rotating image by alpha
    hydr_rotated = hydr.rotate_image(alpha)
    hydr_rotated.show(figsize=figsize_sbs,
                      show=False,
                      save=True,
                      name='./exercise_data/rotated_hydr.pdf',
                      title="Rotated hydrogen lamp image")

    # Checking if there are still residual effects taking three slices
    # at the top, middle and bottom of the spectrum.
    slices = [1150, 1250, 1350, 1450, 1550, 1650, 1768]
    mid_line = 1450
    hydr_corr_slices = {
        line: hydr_rotated.crop_image(crop_y=[slices[n_line] - 1,
                                      slices[n_line]])
        for n_line, line in enumerate(slices)
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
                   show=False,
                   save=True,
                   name='./exercise_data/hydr_align_slices.pdf',
                   title="Slices of rotated hydrogen spectrum image")

    angle_correction = False
    if angle_correction:
        y = [1744, 1745, 1746, 1747, 1750, 1752, 1754]
        sy = [2, 4, 4, 3, 3, 2, 2]
        x = [1768, 1650, 1550, 1450, 1350, 1250, 1150]
        sx = [0, 0, 0, 0, 0, 0, 0]

        job, alpha_corr = hydr.find_rotation_angle(
            x, y, sy, sx,
            bounds=[[-2, 2], [1e3, 2e3]],
            verbose=1,
            show=True,
            save=True,
            nlive=1000,
            name='./exercise_data/joint_rotation_correction.pdf'
        )

        post = job.posterior_samples.ravel()
        l_pars = np.column_stack([post['m'], post['q']])

        with h5py.File(rot_corr_samples, 'w') as hf:
            hf.create_dataset('line params', data=l_pars)
            hf.create_dataset('alpha_corr',
                              data=np.asarray(alpha_corr))
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
                      show=False,
                      save=True,
                      name='./exercise_data/corr_rotated_hydr.pdf',
                      title="Rotated hydrogen lamp image")

    check_alignment = False
    if check_alignment:
        slices = [1130, 1230, 1330, 1430, 1530, 1630, 1730]
        mid_line = 1430
        hydr_corr_slices = {
            line: hydr_rotated.crop_image(crop_y=[slices[n_line] - 1,
                                                  slices[n_line]])
            for n_line, line in enumerate(slices)
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
                       save=False,
                       title="Slices of rotated hydrogen spectrum image"
                             " with correction")

    crops_y = [1330, 1530]
    hydr_cropped = hydr_rotated.crop_image(crop_y=crops_y)
    hydr_cropped.show(figsize=figsize_sbs,
                      show=False,
                      save=True,
                      name='./exercise_data/cropped_hydr.pdf',
                      title="Cropped hydrogen lamp image")

    show_top_and_bottom = False
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
                       name='./exercise_data/hydr_spectrum_slices.pdf',
                       title="Slices of cropped hydrogen spectrum image")

    sp = hydr_cropped.run_integration()
    sp.show(figsize=figsize_sbs,
            show=True,
            save=True,
            name='./exercise_data/int_spectrum_hydr.pdf',
            title="Hydrogen lamp spectrum")
