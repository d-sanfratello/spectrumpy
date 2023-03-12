import argparse as ag

from spectrumpy.io import parse_spectrum_path


def main():
    parser = ag.ArgumentParser()
    parser.add_argument('spectrum_path')
    parser.add_argument("-L", "--limits", action='store_true',
                        dest='show_limits',
                        default=False,
                        help="")
    args = parser.parse_args()

    spectrum = parse_spectrum_path(
        args,
        data_name='spectrum_path'
    )

    if args.show_limits:
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
