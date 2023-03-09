import h5py
import argparse as ag

from pathlib import Path

from spectrumpy.io import parse_image_path


def main():
    parser = ag.OptionParser()
    parser.add_argument("image_path")
    parser.add_argument("-I", "--image", type=int, dest='image',
                        default=0,
                        help="")
    parser.add_argument("-r", "--rotate", type=float, dest='rot_angle',
                        required=True,
                        help="")
    parser.add_argument("-o", "--output", dest='output_image',
                        default='./rotated_image.h5',
                        help="")
    parser.add_argument("-l", "--lamp", action='store_true', dest='is_lamp',
                        default=False,
                        help="")
    args = parser.parse_args()

    image = parse_image_path(
        args,
        data_name='image_path',
        save_output=False,
        is_lamp=args.is_lamp,
        image=args.image
    )

    rot_image = image.rotate_image(args.rot_angle)

    path = Path(args.output_image)
    if path.suffix != ".h5":
        path = path.with_suffix('.h5')

    with h5py.File(path, 'w') as f:
        f.create_dataset('image', data=rot_image.image)


if __name__ == "__main__":
    main()
