import optparse as op

from spectrumpy.io import parse_image_path


def main():
    parser = op.OptionParser()
    parser.disable_interspersed_args()
    parser.add_option("-l", "--lamp", action='store_true', dest='is_lamp',
                      default=False,
                      help="")
    parser.add_option("-I", "--image", type='int', dest='image',
                      default=0,
                      help="")
    parser.add_option("-o", "--output", type='string', dest='output_image',
                      default=None)
    parser.add_option("-L", "--limits", action='store_true',
                      dest='show_limits', default=False,
                      help="")

    (options, args) = parser.parse_args()

    image = parse_image_path(
        args,
        missing_arg_msg="I need a fits or h5 file to show you.",
        is_lamp=options.is_lamp,
        image=options.image,
        output_image=options.output_image
    )

    if options.show_limits:
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
