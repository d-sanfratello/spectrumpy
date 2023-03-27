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
    parser.add_argument("--overlay-pixel", dest="overlay",
                        action="store_true", default=False,
                        help="If this flag is set, a second axis with pixels "
                             "is displayed below the main axis.")
    parser.add_argument('--overlay-spectrum', dest='added_spectrum',
                        default=None,
                        help="a spectrum to overlay to the current spectrum "
                             "dataset.")

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
    if args.lines is not None:
        px, dpx, l, dl = parse_data_path(args, data_name='lines')

        spectrum.assign_dataset(
            px=px, errpx=dpx,
            lines=l, errlines=dl,
            names=args.line_names
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

    if add_spectrum is not None:
        if args.lines is not None:
            add_spectrum.assign_dataset(
                px=px, errpx=dpx,
                lines=l, errlines=dl,
                names=args.line_names
            )

        add_spectrum.assign_calibration(
            calibration=calibration,
            pars=calib_parameters,
            units=args.units
        )

        spectrum = spectrum.normalize()
        add_spectrum = add_spectrum.normalize()

    show_lines = True
    if args.lines is None:
        show_lines = False

    spectrum.show(
        model=None,
        show=True, save=(not args.overlay),
        name=out_folder.joinpath(
            mod_name + '_calibrated_spectrum.pdf'
        ),
        legend=False,
        calibration=True,
        overlay_pixel=args.overlay,
        overlay_spectrum=add_spectrum,
        label=None,
        model_label=None,
        show_lines=show_lines
    )


if __name__ == "__main__":
    main()
