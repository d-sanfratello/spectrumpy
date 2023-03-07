import h5py
import matplotlib.pyplot as plt
import numpy as np
import optparse as op

from pathlib import Path

from spectrumpy.core import Spectrum
from spectrumpy.io import parse_spectrum_path


def main():
    parser = op.OptionParser()
    parser.disable_interspersed_args()
    parser.add_option("-L", "--limits", action='store_true',
                      dest='show_limits', default=False,
                      help="")

    (options, args) = parser.parse_args()

    spectrum = parse_spectrum_path(
        args,
        missing_arg_msg="I need a h5 spectrum file to show you."
    )

    if options.show_limits:
        print(f"Spectrum size: {spectrum.spectrum.shape[0]}")
        exit(0)

    spectrum.show(
        show=True,
        save=False,
        legend=False,
        calibration=False
    )


if __name__ == "__main__":
    main()
