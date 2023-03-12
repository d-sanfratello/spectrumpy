import argparse as ag

from spectrumpy.io import parse_image_path


def main():
    parser = ag.ArgumentParser(
        prog='sp-show-image',
        description='This script shows a fits or h5 image of a spectrum.',
    )
    parser.add_argument('image_path')
    parser.add_argument("-l", "--lamp", action='store_true', dest='is_lamp',
                        default=False,
                        help="")
    parser.add_argument("-I", "--image", type=int, dest='image',
                        default=0,
                        help="")
    parser.add_argument("-o", "--output", dest='output_image',
                        default=None,
                        help="")
    parser.add_argument("-L", "--limits", action='store_true',
                        dest='show_limits',
                        default=False,
                        help="")
    args = parser.parse_args()

    image = parse_image_path(
        args,
        data_name='image_path',
        is_lamp=args.is_lamp,
        image=args.image,
        output_image=args.output_image
    )

    if args.show_limits:
        print(f"Image size (y, x): {image.image.shape}")
        exit(0)

    image.show(
        log=True,
        show=True,
        save=False,
        legend=False
    )


if __name__ == "__main__":
    main()
