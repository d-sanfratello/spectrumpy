import argparse as ag

import h5py

from pathlib import Path

from spectrumpy.core import Spectrum, CalibratedSpectrum
from spectrumpy.io import parse_spectrum_path


def main():
    parser = ag.ArgumentParser(
        prog='sp-compare',
        description='Compare two spectra by dividing the first by the second.'
    )
    parser.add_argument('spectrum_path_1')
    parser.add_argument('spectrum_path_2')
    parser.add_argument("-o", "--output", dest="out_file",
                        default=None,
                        help="The name of the file where to save the output.")
    parser.add_argument('--label', dest="label", default=None,
                        help="A comma-separated-string containing the label "
                             "for the spectra comparison.")
    parser.add_argument("--normalized", dest='normalized',
                        action='store_true', default=False,
                        help="if set, this flag normalizes the spectra "
                             "before plotting them.")
    parser.add_argument("--ymin", dest='y_min', default=None, type=float,
                        help="the minimum y of the plot.")
    parser.add_argument("--ymax", dest='y_max', default=None, type=float,
                        help="the maximum y of the plot.")
    args = parser.parse_args()

    spectrum_1 = parse_spectrum_path(
        args,
        data_name='spectrum_path_1'
    )
    spectrum_2 = parse_spectrum_path(
        args,
        data_name='spectrum_path_2'
    )
    if isinstance([spectrum_1, spectrum_2], Spectrum):
        raise ValueError(
            "sp-compare requires a CalibratedSpectrum instance."
        )

    label = None
    if args.label is not None:
        label = args.label.split(',')

    if args.normalized:
        spectrum_1 = spectrum_1.normalize()
        spectrum_2 = spectrum_2.normalize()

    wl, ratio_1_2, units = CalibratedSpectrum.compare(spectrum_1, spectrum_2)
    ratio = CalibratedSpectrum(
        wavelength=wl,
        spectrum=ratio_1_2,
        units=units
    )

    out_file = None
    if args.out_file is not None:
        out_file = Path(args.out_file).with_suffix('.pdf')
        out_compared = out_file.with_suffix('.h5')

        with h5py.File(out_compared, 'w') as f:
            f.create_dataset('wavelength', data=ratio.wl)
            f.create_dataset('ratio', data=ratio_1_2)
            units = ratio.units.astype(
                h5py.string_dtype(encoding='utf-8')
            )
            f.create_dataset('units', data=units)

    ymin = args.y_min
    ymax = args.y_max

    if ymin is None and ymax is None:
        ylim = None
    else:
        step = (ratio.sp.max() - ratio.sp.min()) / 20
        if ymin is None:
            ymin = ratio.sp.min() - step
        if ymax is None:
            ymax = ratio.sp.max() - step

        ylim = [ymin, ymax]

    ratio.show(
        show=True,
        save=(args.out_file is not None),
        name=out_file,
        legend=(label is not None),
        labels=label,
        ylabel="",
        ylim=ylim,
    )


if __name__ == "__main__":
    main()
