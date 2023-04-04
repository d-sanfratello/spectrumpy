import argparse as ag

from astropy import units as u
from astropy.constants import c


def main():
    parser = ag.ArgumentParser(
        prog='sp-velocity-resolution',
        description='Script to find the resolution in radial velocity given '
                    'a certain wavelength interval.'
    )
    parser.add_argument('delta_lambda', type=float)
    parser.add_argument('ref_lambda', type=float)
    parser.add_argument("-u", "--units", dest="units", default='nm',
                        help="The units of the calibrated wavelengths.")

    args = parser.parse_args()

    units = getattr(u, args.units)
    c_unit = c.to(units/u.s)

    resolution = c_unit * args.delta_lambda/args.ref_lambda
    print(f'{resolution.to(u.km/u.s).value} km/s')


if __name__ == "__main__":
    main()
