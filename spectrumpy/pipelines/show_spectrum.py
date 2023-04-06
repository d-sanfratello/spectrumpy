import argparse as ag

from spectrumpy.core import Spectrum
from spectrumpy.io import parse_spectrum_path


def main():
    parser = ag.ArgumentParser(
        prog='sp-show-spectrum',
        description='This script shows a plot with the integrated spectrum '
                    'from an image.',
    )
    parser.add_argument('spectrum_path')
    parser.add_argument('-s', '--shift', dest='shift', type=int,
                        default=0,
                        help="the shift, in pixels, to be applied to the "
                             "main spectrum.")
    parser.add_argument('-S', '--shift2', dest='shift2', type=int,
                        default=0,
                        help="the shift, in pixels, to be applies to the "
                             "secondary spectrum, if shown.")
    parser.add_argument("-L", "--limits", action='store_true',
                        dest='show_limits',
                        default=False,
                        help="")
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
    args = parser.parse_args()

    spectrum = parse_spectrum_path(
        args,
        data_name='spectrum_path'
    )

    if args.show_limits:
        print(f"Spectrum size: {spectrum.spectrum.shape[0]}")
        exit(0)

    from spectrumpy.bayes_inference import models as mod
    calib_model = mod.models['linear']
    calib_pars = [1, 0]

    spectrum.apply_shift(args.shift)
    spectrum.assign_calibration(
        calibration=calib_model,
        pars=calib_pars,
        units='px'
    )

    add_spectrum = None
    if args.added_spectrum is not None:
        add_spectrum = parse_spectrum_path(
            args,
            data_name='added_spectrum'
        )

        add_spectrum.apply_shift(args.shift2)
        add_spectrum.assign_calibration(
            calibration=calib_model,
            pars=calib_pars,
            units='px'
        )

        Spectrum.even_spectra(spectrum, add_spectrum)

        if args.normalized:
            spectrum = spectrum.normalize()
            add_spectrum = add_spectrum.normalize()

        cal_add_spectrum = add_spectrum.return_calibrated()
        cal_spectrum = spectrum.return_calibrated()

        cal_spectrum.show(
            show=True, save=False,
            legend=True,
            overlay_spectrum=cal_add_spectrum,
            label=None,
        )
    else:
        spectrum.show(
            show=True,
            save=False,
            legend=False,
            overlay_spectrum=add_spectrum,
            show_lines=False
        )


if __name__ == "__main__":
    main()
