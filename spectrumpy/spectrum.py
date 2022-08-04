import matplotlib.pyplot as plt
import numpy as np
import warnings

from astropy.io import fits
from numbers import Number
from os.path import join
from pathlib import Path

from scipy.ndimage import rotate
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from scipy.optimize import minimize

from .datasets import DataSets
from .fit import FitStatus
from .mhspectrasampler import functions as fs
from .mhspectrasampler import MHsampler


class Spectrum:
    def __init__(self, spectrum_path, lamp_path, data_ext=0):
        if not isinstance(spectrum_path, (Path, str)):
            raise TypeError("`{}` is not a valid path.".format(spectrum_path))
        if not isinstance(lamp_path, (Path, str)):
            raise TypeError("`{}` is not a valid path.".format(lamp_path))

        if not isinstance(data_ext, int):
            raise TypeError("`data_ext` must be an integer.")
        elif data_ext < 0:
            raise ValueError("`data_ext` must be positive.")

        with fits.open(spectrum_path) as s_file:
            self.s_header = s_file[data_ext].header
            self.s_image = s_file[data_ext].data

            self.s_fits_hdu = s_file

        with fits.open(lamp_path) as l_file:
            self.l_header = l_file[data_ext].header
            self.l_image = l_file[data_ext].data

            self.l_fits_hdu = l_file

        self.smoothed = None
        self.weighted_lamp = None

        self.l_int = None
        self.s_int = None

        self.dataset = None
        self.model = None
        self.calibration = None
        self.sampler = None

    def show_base_image(self, log=True, *args, **kwargs):
        if not isinstance(log, bool):
            raise TypeError("`log` must be a bool.")

        fig = plt.figure(*args, **kwargs)
        ax = fig.gca()
        if log:
            ax.imshow(np.log10(self.s_image))
        else:
            ax.imshow(self.s_image)

        fig = plt.figure(*args, **kwargs)
        ax = fig.gca()
        if log:
            ax.imshow(np.log10(self.l_image))
        else:
            ax.imshow(self.l_image)

        plt.show()

    def run_integration(self):
        self.s_int = np.sum(self.s_image, axis=0)
        self.l_int = np.sum(self.l_image, axis=0)

    def show_integration(self, **kwargs):
        fig_l = plt.figure(**kwargs)
        ax = fig_l.gca()
        ax.grid()
        ax.plot(self.l_int, linestyle='solid', color='black', linewidth=0.5)
        ax.set_xlim(0, len(self.l_int) - 1)
        ax.set_xlabel(r'[px]')

        fig_s = plt.figure(**kwargs)
        ax = fig_s.gca()
        ax.grid()
        ax.plot(self.s_int, linestyle='solid', color='black', linewidth=0.5)
        ax.set_xlim(0, len(self.s_int) - 1)
        ax.set_xlabel(r'[px]')

        plt.show()

    def smooth(self, size):
        if self.s_int is None:
            raise AttributeError("Integration has not been performed, yet.")

        self.smoothed = median_filter(self.s_int, size=size)
        self.weighted_lamp = self.l_int / self.smoothed

    def assign_dataset(self, lines, px, errpx, names):
        self.dataset = DataSets(lines, px, errpx, names)

    def run_calibration(self, order, method='ls', bounds_pars=None,
                        verbose=False, n=1000, delta=0.1):
        if order not in [1, 2, 3, 4]:
            raise ValueError("Unknown order.")
        if method not in ['ls', 'bayes']:
            raise ValueError("Unknown calibration method.")

        if order == 1:
            self.model = fs.Linear
        elif order == 2:
            self.model = fs.Quadratic
        elif order == 3:
            self.model = fs.Cubic
        elif order == 4:
            self.model = fs.Quartic

        if method == 'ls':
            popt, pcov = curve_fit(self.model.func, self.dataset.px,
                                   self.dataset.lines)

            model = self.model(*popt)
            errs = abs(model.derivative(self.dataset.px)) * self.dataset.errpx

            popt, pcov = curve_fit(self.model.func, self.dataset.px,
                                   self.dataset.lines,
                                   sigma=errs,
                                   absolute_sigma=True,
                                   p0=popt)

            self.calibration = self.model(*popt)

            chisq = ((self.dataset.lines - self.calibration(
                self.dataset.px)) ** 2 / self.dataset.errpx ** 2).sum()
            dof = len(self.dataset.lines) - len(popt)

            fit_status = FitStatus(popt, pcov, chisq, dof,
                                   order=self.calibration.order)
            return fit_status

        elif method == 'bayes':
            # warnings.warn("Not yet implemented, running with `method='ls'`.")
            # return self.run_calibration(order=order, method='ls')

            if bounds_pars is None:
                raise ValueError("Must define bounds for bayes estimation.")

            self.sampler = MHsampler(self.dataset.px, self.dataset.lines,
                                     self.dataset.errpx,
                                     bounds_pars=bounds_pars,
                                     bounds_x=[0, len(self.s_int) - 1],
                                     n=n, delta=delta)

            def log_likelihood(x, y, errx, x_hat, theta):
                model = self.model(*theta)
                zeros = model.zeros()

                if zeros is None:
                    return -np.inf

                term1 = np.array([np.exp(
                    -0.5 * (x - z) ** 2 / errx ** 2) / abs(model.derivative(z))
                                  for z in zeros])
                term1 = term1.sum(axis=0)

                logs = np.log(term1).sum()

                if not np.isfinite(logs):
                    return -np.inf
                else:
                    return logs

            self.sampler.log_likelihood = log_likelihood

            samples, x_hat_s, s_orig, x_orig, acc, rej = self.sampler.run(
                verbose=verbose)

            self.calibration = self.model(*samples.mean(axis=1))

            return samples, x_hat_s, s_orig, x_orig, acc, rej

    def show_calibration(self, exclude=None, **kwargs):
        if self.model is None:
            raise AttributeError("Calibration has not been run, yet.")

        fig = plt.figure(**kwargs)
        ax = plt.subplot2grid((4, 1), (0, 0), rowspan=3, fig=fig)
        ax.grid()
        ax.errorbar(self.dataset.px, self.dataset.lines,
                    xerr=self.dataset.errpx,
                    linestyle='', capsize=2, marker=',')

        x = np.linspace(0, len(self.l_int) - 1, len(self.l_int))
        ax.plot(x, self.calibration(x),
                linestyle='solid', color='black', linewidth=0.5)
        ax.set_xlim(0, x[-1])
        ax.set_ylabel(r'$\lambda$ [$\mathring{A}$]')
        ax.set_xticklabels([])
        ax1 = plt.subplot2grid((4, 1), (3, 0), fig=fig)
        ax1.grid()
        errs = abs(
            self.calibration.derivative(self.dataset.px)) * self.dataset.errpx
        ax1.scatter(self.dataset.px, (self.dataset.lines - self.calibration(
            self.dataset.px)) / errs,
                    marker='+', color='black')
        ax1.set_xlim(0, x[-1])
        ax1.set_ylabel(r'Norm.Res.')
        ax1.set_xlabel('[px]')
        fig.tight_layout()

        fig_l = plt.figure(**kwargs)
        ax = fig_l.gca()
        ax.grid()
        line = self.calibration(x)
        ax.plot(line, self.l_int, linestyle='solid', color='black',
                linewidth=0.5)
        for lam in self.dataset.lines:
            if lam == self.dataset.lines.min() or lam == self.dataset.lines.max():
                ax.axvline(lam, ymin=0, ymax=1, linewidth=0.5, color='red',
                           linestyle='dotted')
            else:
                ax.axvline(lam, ymin=0, ymax=1, linewidth=0.5, color='navy',
                           linestyle='dashed')
                ax.text(lam, self.l_int.max() / 2,
                        '{:s}'.format(self.dataset.names[lam]),
                        rotation=90, verticalalignment='center',
                        horizontalalignment='left',
                        size=7.5, color='navy')
        ax.set_xlim(line.min(), line.max())
        ax.set_xlabel(r'$\lambda$ [$\mathring{A}$]')

        fig_s = plt.figure(**kwargs)
        ax = fig_s.gca()
        ax.grid()
        line = self.calibration(x)
        ax.plot(line, self.s_int, linestyle='solid', color='black',
                linewidth=0.5)
        for lam in self.dataset.lines:
            if lam == self.dataset.lines.min() or lam == self.dataset.lines.max():
                ax.axvline(lam, ymin=0, ymax=1, linewidth=0.5, color='red',
                           linestyle='dotted')
            elif exclude is not None and self.dataset.names[
                lam] not in exclude:
                ax.axvline(lam, ymin=0, ymax=1, linewidth=0.5, color='navy',
                           linestyle='dashed')
                ax.text(lam, (self.s_int.max() + self.s_int.min()) / 2,
                        '{:s}'.format(self.dataset.names[lam]),
                        rotation=90, verticalalignment='center',
                        horizontalalignment='left',
                        size=7.5, color='navy')
        ax.set_xlim(line.min(), line.max())
        ax.set_xlabel(r'$\lambda$ [$\mathring{A}$]')

        plt.show()

    def compare(self, spectrum):
        eq_spectrum = spectrum.s_int / spectrum.s_int.max() * self.s_int.max()

        x = np.linspace(0, len(self.s_int) - 1, len(self.s_int))
        ref_calibration = self.calibration(x)

        x_cmp = np.linspace(0, len(spectrum.s_int) - 1, len(spectrum.s_int))
        cmp_calibration = spectrum.calibration(x_cmp)

        if len(ref_calibration) == len(cmp_calibration) and np.all(
                self.calibration == eq_spectrum):
            return self.s_int / eq_spectrum

        spectra_ratio = np.zeros(len(ref_calibration), dtype=np.float64)

        for _, pt in enumerate(ref_calibration):
            if pt < cmp_calibration.min():
                low_x = cmp_calibration[0]
                upp_x = cmp_calibration[1]

                low_y = eq_spectrum[0]
                upp_y = eq_spectrum[1]
            elif cmp_calibration.min() <= pt < cmp_calibration.max():
                low_x = cmp_calibration[cmp_calibration <= pt]
                idx_low = len(low_x)
                low_x = low_x.max()
                low_y = eq_spectrum[idx_low - 1]

                upp_x = cmp_calibration[cmp_calibration > pt].min()
                upp_y = eq_spectrum[idx_low]
            elif pt >= cmp_calibration.max():
                low_x = cmp_calibration[-2]
                upp_x = cmp_calibration[-1]

                low_y = eq_spectrum[-2]
                upp_y = eq_spectrum[-1]

            l_approx = fs.Linear((upp_y - low_y) / (upp_x - low_x),
                                 low_y - (upp_y - low_y) / (
                                             upp_x - low_x) * low_x)

            spectra_ratio[_] = self.s_int[_] / l_approx(pt)

        return spectra_ratio
