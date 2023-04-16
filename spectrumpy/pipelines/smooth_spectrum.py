import argparse as ag

import h5py

from pathlib import Path

from spectrumpy.core import Spectrum
from spectrumpy.io import parse_spectrum_path


def main():
    parser = ag.ArgumentParser(
        prog='sp-smooth',
        description='Smooth the spectrum to retrieve the continuum.'
    )
    parser.add_argument('spectrum_path')
    parser.add_argument('frac', nargs='?',
                        type=float, default=0.5,
                        help="the fraction of points for the robust loess "
                             "filter to be applied to smooth the spectrum.")
    parser.add_argument("-o", "--output", dest="out_file",
                        default=None,
                        help="The name of the file where to save the output.")
    args = parser.parse_args()

    spectrum = parse_spectrum_path(
        args,
        data_name='spectrum_path'
    )

    if isinstance(spectrum, Spectrum):
        raise ValueError(
            "sp-smooth requires a CalibratedSpectrum instance."
        )

    spectrum_smoothed = spectrum.smooth(frac=args.frac)

    out_file = None
    if args.out_file is not None:
        out_file = Path(args.out_file).with_suffix('.pdf')
        out_spectrum = out_file.with_suffix('.h5')

        with h5py.File(out_spectrum, 'w') as f:
            f.create_dataset('wavelength', data=spectrum_smoothed.wl)
            f.create_dataset('spectrum', data=spectrum_smoothed.sp)
            units = spectrum_smoothed.units.astype(
                h5py.string_dtype(encoding='utf-8')
            )
            f.create_dataset('units', data=units)

    spectrum_smoothed.show(
        show=True,
        save=(args.out_file is not None),
        name=out_file,
        legend=False,
    )


if __name__ == "__main__":
    main()
