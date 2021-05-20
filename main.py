import numpy as np

from os.path import join
from path import Path

from spectrumpy import Spectrum


if __name__ == '__main__':
    moon_fits_path = Path('C:/Users/Daniele/Desktop/fits/alpy_spectra_repository/27 marzo 2018/')
    extension = '.fit'
    moon_path = join(moon_fits_path, Path('moon-001' + extension))
    moon_lamp_path = join(moon_fits_path, Path('lamp_moon_venus' + extension))

    moon = Spectrum(moon_path, moon_lamp_path)
    moon.show_base_image()
    moon.run_integration()
    moon.smooth(50)

    lin = np.array([5852.4878, 6143.0627, 6402.2480, 6677.282])
    px = np.array([894, 1029, 1148, 1275])
    err = np.array([3, 3, 3, 3])
    names = np.array(['NeI', 'NeI', 'NeI', 'ArI'])

    moon.assign_dataset(lin, px, err, names)

    print(moon.run_calibration(1, 'ls'))
    moon.show_calibration(figsize=(8, 6))
