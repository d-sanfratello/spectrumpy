import argparse as ag
import os

import h5py

from pathlib import Path

from spectrumpy.io import parse_spectrum_path, parse_data_path
from spectrumpy.bayes_inference import models as mod


def main():
    parser = ag.ArgumentParser(
        prog='sp-show-calibrated',
        description='Script to show the spectrum with calibrated wavelengths.'
    )
    parser.add_argument('spectrum_path')
    parser.add_argument('calibration',
                        help="The folder containing the calibration model "
                             "parameters.")
    parser.add_argument('-s', '--shift', dest='shift', type=int,
                        default=0,
                        help="the shift, in pixels, to be applied to the "
                             "main spectrum.")
    parser.add_argument('-c', '--cut', dest='cut', default=None,
                        help="a list containing the number of pixels to be "
                             "cut from the spectra from each side.")
    parser.add_argument("-o", "--output", dest="out_name",
                        default=None,
                        help="The name of the output file where to save the "
                             "output of this command.")
    parser.add_argument("-u", "--units", dest="units", default='nm',
                        help="The units of the calibrated wavelengths.")

    args = parser.parse_args()

    out_name = args.out_name
    if out_name is None:
        out_name = Path(os.getcwd()).joinpath('calibrated_spectrum.h5')
    out_name = Path(out_name)

    spectrum = parse_spectrum_path(
        args,
        data_name='spectrum_path'
    )

    cut = None
    if args.cut is not None:
        cut = eval(args.cut)

    with h5py.File(Path(args.calibration).joinpath(
            'median_params.h5'), 'r') as f:
        calib_parameters = f['params'][:]

    mod_name = mod.available_models[len(calib_parameters) - 2]
    calibration = mod.models[mod_name]

    spectrum.apply_shift(args.shift)
    spectrum.assign_calibration(
        calibration=calibration,
        pars=calib_parameters,
        units=args.units
    )

    cal_spectrum = spectrum.return_calibrated()
    if cut is not None:
        cal_spectrum = cal_spectrum.cut(*cut)

    with h5py.File(out_name, 'w') as f:
        f.create_dataset('wavelength', data=cal_spectrum.wl)
        f.create_dataset('spectrum', data=cal_spectrum.sp)
        units = cal_spectrum.units.astype(
            h5py.string_dtype(encoding='utf-8')
        )
        f.create_dataset('units', data=units)


if __name__ == "__main__":
    main()
