import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import optparse as op

from pathlib import Path

from spectrumpy.io import SpectrumPath


if __name__ == "__main__":
    parser = op.OptionParser()
    parser.add_option("-i", "--input", type='string', dest='image_file',
                      help="")
    parser.add_option("-l", "--lamp", action='store_true', dest='is_lamp',
                      default=False,
                      help="")
    parser.add_option("-I", "--image", type='int', dest='image',
                      default=0,
                      help="")

    (options, args) = parser.parse_args()

    if options.image_file is None:
        raise AttributeError(
            "I need a fits or h5 file to show you."
        )

    try:
        image_file = SpectrumPath(options.image_file, is_lamp=options.is_lamp)
        image = image_file.images[options.image]
    except FileNotFoundError:
        # try:
        # FIXME: need to try to open a fits and then an h5.
        pass
