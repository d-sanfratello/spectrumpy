import os

import h5py
import argparse as ag

from pathlib import Path

from spectrumpy.io import parse_image_path


def main():
    parser = ag.ArgumentParser(
        prog='sp-crop-image',
        description='Pipeline to crop a spectrum image.',
    )
    parser.add_argument('image_path')
    parser.add_argument("-I", "--image", type=int, dest='image',
                        default=0,
                        help="The image to be selected in a fits file.")
    parser.add_argument("-o", "--output", dest='output_image',
                        default=None,
                        help="")
    parser.add_argument("-c", "--crop", dest="crop", default=None,
                        required=True,
                        help="")
    parser.add_argument("-v", "--vertical", action='store_true',
                        dest='vertical_crop',
                        default=False,
                        help="")
    parser.add_argument("-l", "--lamp", action='store_true', dest='is_lamp',
                        default=False,
                        help="")
    args = parser.parse_args()

    crop = eval(args.crop)
    if crop is None:
        raise ValueError(
            "I need two rows or columns to crop the image."
        )
    if not isinstance(crop, list):
        crop = [crop]

    image = parse_image_path(
        args,
        data_name='image_path',
        is_lamp=args.is_lamp,
        image=args.image,
        save_output=False
    )

    if args.vertical_crop:
        cropped_image = image.crop_image(crop_x=crop)
    else:
        cropped_image = image.crop_image(crop_y=crop)

    if args.output_image is None:
        new_filename = f"cropped_{crop[0]}-{crop[1]}.h5"
        path = Path(os.getcwd()).joinpath(new_filename)
    else:
        path = Path(args.output_image)

    if path.suffix != ".h5":
        path = path.with_suffix('.h5')

    with h5py.File(path, 'w') as f:
        f.create_dataset('image', data=cropped_image.image)


if __name__ == "__main__":
    main()
