import argparse as ag
import os

import h5py

from pathlib import Path

from spectrumpy.core import Spectrum
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
    parser.add_argument('-S', '--shift2', dest='shift2', type=int,
                        default=0,
                        help="the shift, in pixels, to be applies to the "
                             "secondary spectrum, if shown.")
    parser.add_argument('-c', '--cut', dest='cut', default=None,
                        help="a list containing the number of pixels to be "
                             "cut from the spectra from each side.")
    parser.add_argument("-l", "--lines", dest="lines", default=None,
                        help="The path to a file containing lines to be "
                             "plotted over the spectrum.")
    parser.add_argument("-n", "--line-names", dest='line_names', default=None,
                        help="The names of the lines, if they are identified "
                             "by name.")
    parser.add_argument("-o", "--output-folder", dest="out_folder",
                        default=None,
                        help="The folder where to save the output of this "
                             "command.")
    parser.add_argument("-u", "--units", dest="units", default='nm',
                        help="The units of the calibrated wavelengths.")
    parser.add_argument('--overlay-spectrum', dest='added_spectrum',
                        default=None,
                        help="a spectrum to overlay to the current spectrum "
                             "dataset.")
    parser.add_argument('--labels', dest="labels", default=None,
                        help="A comma-separated-string containing the labels "
                             "for the spectrum(a).")
    parser.add_argument("--normalized", dest='normalized',
                        action='store_true', default=False,
                        help="if set, this flag normalizes the spectra "
                             "before plotting them.")
    parser.add_argument("--calibrated", dest='calibrated', default=False,
                        action='store_true',
                        help="if set, this flag treats the given spectrum as "
                             "already calibrated.")
    #FIXME: Quando non si passa --calibrated è in grado di aprire il file lo
    # stesso. Magari rinominare il dataset?

    args = parser.parse_args()

    out_folder = args.out_folder
    if out_folder is None:
        out_folder = os.getcwd()
    out_folder = Path(out_folder)

    spectrum = parse_spectrum_path(
        args,
        data_name='spectrum_path'
    )
    add_spectrum = None
    if args.added_spectrum is not None:
        add_spectrum = parse_spectrum_path(
            args,
            data_name='added_spectrum'
        )

    cut = None
    if args.cut is not None:
        cut = eval(args.cut)

    if not args.calibrated and args.lines is not None:
        px, dpx, l, dl = parse_data_path(args, data_name='lines')

        spectrum.assign_dataset(
            px=px + args.shift, errpx=dpx,
            lines=l, errlines=dl,
            names=args.line_names
        )

    labels = None
    if args.labels is not None:
        labels = args.labels.split(',')

    if not args.calibrated:
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

        if add_spectrum is not None:
            add_spectrum.apply_shift(args.shift2)
            add_spectrum.assign_calibration(
                calibration=calibration,
                pars=calib_parameters,
                units=args.units
            )

            Spectrum.even_spectra(spectrum, add_spectrum)

            if args.normalized:
                spectrum = spectrum.normalize()
                add_spectrum = add_spectrum.normalize()

            cal_add_spectrum = add_spectrum.return_calibrated()
            if cut is not None:
                cal_add_spectrum = cal_add_spectrum.cut(*cut)
        else:
            cal_add_spectrum = None

        cal_spectrum = spectrum.return_calibrated()
        if cut is not None:
            cal_spectrum = cal_spectrum.cut(*cut)
    else:
        cal_spectrum = spectrum
        cal_add_spectrum = None

        if add_spectrum is not None:
            cal_add_spectrum = add_spectrum
            if cut is not None:
                cal_add_spectrum = cal_add_spectrum.cut(*cut)

        if cut is not None:
            cal_spectrum = cal_spectrum.cut(*cut)

    lines = None
    if args.lines is not None:
        lines = spectrum.dataset

    cal_spectrum.show(
        show=True, save=True,
        name=out_folder.joinpath(
            'calibrated_spectrum.pdf'
        ),
        legend=True,
        overlay_spectrum=cal_add_spectrum,
        labels=labels,
        lines=lines
    )


if __name__ == "__main__":
    main()
