import argparse as ag

from spectrumpy.io import parse_image_path


def main():
    parser = ag.ArgumentParser(
        prog='sp-show-slices',
        description='This script shows some given slices of a spectrum image.',
    )
    parser.add_argument("image_path")
    parser.add_argument("-I", "--image", type=int, dest='image',
                        default=0,
                        help="")
    parser.add_argument("-o", "--output", dest='output_image',
                        default=None,
                        help="")
    parser.add_argument("-s", "--slice", dest="slice",
                        default=None,
                        required=True,
                        help="")
    parser.add_argument("-v", "--vertical", action='store_true',
                        dest='vertical_slice',
                        default=False,
                        help="")
    parser.add_argument("-l", "--lamp", action='store_true', dest='is_lamp',
                        default=False,
                        help="")
    args = parser.parse_args()

    slices = eval(args.slice)
    if slices is None:
        raise ValueError(
            "I need at least a slice."
        )
    if not isinstance(slices, list):
        slices = [slices]

    image = parse_image_path(
        args,
        data_name='image_path',
        is_lamp=args.is_lamp,
        image=args.image,
        output_image=args.output_image,
    )

    if args.vertical_slice:
        # FIXME: implement.
        raise ValueError("Not yet implemented.")
    else:
        models = [
            image.slice_image(line).run_integration() for line in slices
        ]
        first_model = models[0]
        name = f"slice {slices[0]}"
        models.pop(0)
        labels = [f"slice {sl}" for sl in slices[1:]]

        first_model.show(model=models,
                         legend=True,
                         show=True,
                         save=False,
                         label=name,
                         model_label=labels)


if __name__ == "__main__":
    main()
