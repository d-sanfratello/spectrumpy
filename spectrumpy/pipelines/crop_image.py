import os

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
    parser.add_option("-o", "--output", type='string', dest='output_image',
                      default=None)
    parser.add_option("-c", "--crop", type='string', dest="crop",
                      default=None,
                      help="")
    parser.add_option("-v", "--vertical", action='store_true',
                      dest='vertical_crop', default=False)
    parser.add_option("-l", "--lamp", action='store_true', dest='is_lamp',
                      default=False,
                      help="")

    (options, args) = parser.parse_args()

    crop = eval(options.crop)
    if crop is None:
        raise ValueError(
            "I need two rows or columns to crop the image."
        )
    if not isinstance(crop, list):
        crop = [crop]

    image = parse_image_path(
        args,
        missing_arg_msg="I need a fits or h5 file to crop.",
        is_lamp=options.is_lamp,
        image=options.image,
        save_output=False
    )

    if options.vertical_crop:
        cropped_image = image.crop_image(crop_x=crop)
    else:
        cropped_image = image.crop_image(crop_y=crop)

    if options.output_image is None:
        new_filename = f"cropped_{crop[0]}-{crop[1]}.h5"
        path = Path(os.getcwd()).joinpath(new_filename)
    else:
        path = Path(options.output_image)

    if path.suffix != ".h5":
        path = path.with_suffix('.h5')

    with h5py.File(path, 'w') as f:
        f.create_dataset('image', data=cropped_image.image)


if __name__ == "__main__":
    main()
