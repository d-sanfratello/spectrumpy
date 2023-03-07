import h5py
import matplotlib.pyplot as plt
import numpy as np
import optparse as op

from pathlib import Path

from spectrumpy.core import Spectrum


def main():
    parser = op.OptionParser()
    parser.add_option("-i", "--input", type='string', dest='spectrum_file',
                      help="")
    parser.add_option("-L", "--limits", action='store_true',
                      dest='show_limits', default=False,
                      help="")

    (options, args) = parser.parse_args()

    if options.spectrum_file is None:
        raise AttributeError(
            "I need a h5 spectrum file to show you."
        )

    with h5py.File(Path(options.spectrum_file), 'r') as f:
        spectrum_data = np.asarray(f['spectrum'])

    spectrum = Spectrum(spectrum_data)

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
