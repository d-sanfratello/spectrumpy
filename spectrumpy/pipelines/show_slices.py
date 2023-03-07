import numpy as np
import optparse as op

from spectrumpy.io import parse_image_path


def main():
    parser = op.OptionParser()
    parser.disable_interspersed_args()
    parser.add_option("-I", "--image", type='int', dest='image',
                      default=0,
                      help="")
    parser.add_option("-o", "--output", type='string', dest='output_image',
                      default=None)
    parser.add_option("-s", "--slice", type='string', dest="slice",
                      default=None,
                      help="")
    parser.add_option("-v", "--vertical", action='store_true',
                      dest='vertical_slice', default=False)
    parser.add_option("-l", "--lamp", action='store_true', dest='is_lamp',
                      default=False,
                      help="")

    (options, args) = parser.parse_args()

    slices = eval(options.slice)
    if slices is None:
        raise ValueError(
            "I need at least a slice."
        )
    if not isinstance(slices, list):
        slices = [slices]

    image = parse_image_path(
        args,
        missing_arg_msg="I need a fits or h5 file to show you.",
        is_lamp=options.is_lamp,
        image=options.image,
        output_image=options.output_image,
    )

    if options.vertical_slice:
        # FIXME: implement.
        raise ValueError("Not yet implemented.")
    else:
        models = [
            image.slice_image(line).run_integration() for line in slices
        ]
        first_model = models[0]
        name = f"slice {slices[0]}"
        models.pop(0)
        x_len = len(first_model.spectrum)
        x_sup = x_len - 1
        labels = [f"slice {sl}" for sl in slices[1:]]

        first_model.show(model=models,
                         x=np.linspace(0, x_sup, x_len),
                         legend=True,
                         show=True,
                         save=False,
                         label=name,
                         model_label=labels)


if __name__ == "__main__":
    main()
