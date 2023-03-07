import corner
import cpnest
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import optparse as op

from corner import quantile
from figaro.mixture import DPGMM
from figaro.load import save_density, load_density
from figaro.utils import get_priors
from pathlib import Path

from spectrumpy.bayes_inference import models as mod

# FIXME: Remove simulation. Maybe add it to another pipeline.


def main():
    parser = op.OptionParser()
    parser.disable_interspersed_args()
    parser.add_option("-o", "--output-folder", dest="out_folder",
                      default=None,
                      help="The folder where to save the simulated lines.")
    (options, args) = parser.parse_args()

    if options.out_folder is None:
        out_folder = os.getcwd()
    out_folder = Path(options.out_folder)

    true_vals = [1e-3, 1e-2, 10, 1]
    px = np.random.uniform(low=0, high=3000, size=8)
    l = mod.cubic(px, *true_vals)
    dpx = np.ones(len(px)) * 3000 * 0.01
    dl = np.ones(len(l))

    px = np.random.normal(px, dpx)
    l = np.random.normal(l, dl)

    dtype = np.dtype([
        ('px', float),
        ('dpx', float),
        ('l', float),
        ('dl', float)
    ])

    data_array = np.zeros(px.shape, dtype=dtype)
    data_array['px'] = px
    data_array['dpx'] = dpx
    data_array['l'] = l
    data_array['dl'] = dl

    output_file = out_folder.joinpath('simulated_data.h5')
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('lines', data=data_array)

    print()
    print(f"Simulation true parameters: {true_vals}")


if __name__ == "__main__":
    main()
