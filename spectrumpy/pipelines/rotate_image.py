import h5py
import optparse as op

from pathlib import Path

from spectrumpy.io import parse_image_path


def main():
    parser = op.OptionParser()
    parser.disable_interspersed_args()
    parser.add_option("-I", "--image", type='int', dest='image',
                      default=0,
                      help="")
    parser.add_option("-r", "--rotate", type='float', dest='rot_angle',
                      default=0,
                      help="")
    parser.add_option("-o", "--output", type='string', dest='output_image',
                      default='./rotated_image.h5',
                      help="")
    parser.add_option("-l", "--lamp", action='store_true', dest='is_lamp',
                      default=False,
                      help="")

    (options, args) = parser.parse_args()

    if options.image_file is None:
        raise AttributeError(
            "I need a fits or h5 file to rotate."
        )

    image = parse_image_path(
        args,
        missing_arg_msg="I need a fits or h5 file to rotate.",
        save_output=False,
        is_lamp=options.is_lamp,
        image=options.image
    )

    rot_image = image.rotate_image(options.rot_angle)

    path = Path(options.output_image)
    if path.suffix != ".h5":
        path = path.with_suffix('.h5')

    with h5py.File(path, 'w') as f:
        f.create_dataset('image', data=rot_image.image)


if __name__ == "__main__":
    main()
