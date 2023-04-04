import argparse as ag
import os

import h5py
import numpy as np

from astropy import units as u
from pathlib import Path

from spectrumpy.io import parse_spectrum_path, parse_data_path
from spectrumpy.bayes_inference import models as mod


def main():
    parser = ag.ArgumentParser(
        prog='sp-apply-doppler-shift',
        description='Script to apply a doppler shift to the calibrated '
                    'spectrum.'
    )
    parser.add_argument('spectrum_path')
    parser.add_argument('calibration',
                        help="The folder containing the calibration model "
                             "parameters.")
    parser.add_argument("-o", "--output-folder", dest="out_folder",
                        default=None,
                        help="The folder where to save the output of this "
                             "command.")
    parser.add_argument("-u", "--units", dest="units", default='nm',
                        help="The units of the calibrated wavelengths.")

    args = parser.parse_args()

    out_folder = args.out_folder
    if out_folder is None:
        out_folder = os.getcwd()
    out_folder = Path(out_folder)

    spectrum = parse_spectrum_path(
        args,
        data_name='spectrum_path'
    )

    with h5py.File(Path(args.calibration).joinpath(
            'median_params.h5'), 'r') as f:
        calib_parameters = f['params'][:]

    mod_name = mod.available_models[len(calib_parameters) - 2]
    calibration = mod.models[mod_name]

    spectrum.assign_calibration(
        calibration=calibration,
        pars=calib_parameters,
        units=args.units
    )

    calib_model, calib_pars = spectrum.calibration
    x_clb = np.linspace(0, len(spectrum.spectrum) - 1, len(spectrum.spectrum))
    x_clb = calib_model(x_clb, *calib_pars)

    # units = getattr(u, args.units)
    # print((656*units).to(u.Hz, equivalencies=u.spectral()))


if __name__ == "__main__":
    main()
