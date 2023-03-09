import h5py
import numpy as np
import os
import argparse as ag

from pathlib import Path

from spectrumpy.bayes_inference import models as mod


def main():
    parser = ag.ArgumentParser()
    parser.add_argment("-o", "--output-folder", dest="out_folder",
                       default=None,
                       help="The folder where to save the simulated lines.")
    args = parser.parse_args()

    out_folder = args.out_folder
    if out_folder is None:
        out_folder = os.getcwd()
    out_folder = Path(out_folder)

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
