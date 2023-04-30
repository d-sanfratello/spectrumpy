import argparse as ag


def main():
    parser = ag.ArgumentParser(
        prog='sp-help',
        description='This script helps navigate the order of operations to '
                    'execute over a spectral image to process it. Calling '
                    'the different options does not return anything other '
                    'than this docstring.'
    )
    parser.add_argument("--simulate-spectrum", action='store_true',
                        help="simulates a few lines from a polynomial "
                             "calibration function and returns them.")
    parser.add_argument("--show-image", action='store_true',
                        help="shows a fit or h5 image file.")
    parser.add_argument("--average-image", action='store_true',
                        help="performs an average between different files. "
                             "It can be used to perform a time average on "
                             "flat/dark/bias frames, if they have a longer "
                             "exposure than the one requested.")
    parser.add_argument("--correct-image", action='store_true',
                        help="corrects an image for dark, flat and bias "
                             "frames.")
    parser.add_argument("--find-angle", action='store_true',
                        help="given a set of data, finds the rotation angle "
                             "to be applied to the image.")
    parser.add_argument("--rotate", action='store_true',
                        help="rotates the image of the given angle.")
    parser.add_argument("--show-slices", action='store_true',
                        help="shows different slices of a spectrum from a "
                             "given list.")
    parser.add_argument("--crop-image", action='store_true',
                        help="crops an image between the given bounds and "
                             "saves it.")
    parser.add_argument("--integrate", action='store_true',
                        help="integrates a given spectrum and saves the "
                             "corresponding dataset.")
    parser.add_argument("--show-spectrum", action='store_true',
                        help="shows the plot of an integrated spectrum, "
                             "without calibration.")
    parser.add_argument("--calibrate", action='store_true',
                        help="given a set of data and a model, calibrates "
                             "the spectrum over the model.")
    parser.add_argument("--show-calibrated", action='store_true',
                        help="shows a calibrated spectrum.")
    parser.add_argument("--save-calibrated", action='store_true',
                        help="saves the calibrated spectrum into a file.")
    parser.add_argument("--smooth", action='store_true',
                        help="smooths a spectrum to retrieve the continuum.")
    parser.add_argument("--compare", action='store_true',
                        help="compares two spectra by evaluating their ratio.")
    parser.add_argument("--find-line", action='store_true',
                        help="Finds the Voigt profile of a line within a "
                             "spectrum.")
    parser.add_argument("--velocity-resolution", action='store_true',
                        help="given the distance between two lines and their "
                             "region, it estimates the radial velocity "
                             "resolution of the spectrum.")
    args = parser.parse_args()

    parser.print_help()


if __name__ == "__main__":
    main()
