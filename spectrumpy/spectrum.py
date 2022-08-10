import corner
import matplotlib.pyplot as plt
import numpy as np
import warnings

from astropy.io import fits
from cpnest import CPNest
from numbers import Number
from pathlib import Path

from scipy.ndimage import rotate
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from scipy.optimize import minimize

from .datasets import DataSets
from .fit import FitStatus
from .mhspectrasampler import functions as fs
from .mhspectrasampler import MHsampler
from spectrumpy.bayes_inference import RotationPosterior


class Spectrum:
    def __init__(self,
                 spectrum_path, data_ext=0,
                 calibration=None,
                 is_lamp=False):
        with fits.open(spectrum_path) as s_file:
            self.header = s_file[data_ext].header
            self.image = s_file[data_ext].data

            self.fits_hdu = s_file

        self.is_lamp = is_lamp

        self.smoothed = None
        self.weighted_lamp = None

        self.int = None

        self.dataset = None
        self.model = None
        self.calibration = calibration
        self.sampler = None

    def show_base_image(self, log=True, *args, **kwargs):
        if not isinstance(log, bool):
            raise TypeError("`log` must be a bool.")

        fig = plt.figure(*args, **kwargs)
        ax = fig.gca()
        if log:
            ax.imshow(np.log10(self.image), origin='lower')
        else:
            ax.imshow(self.image, origin='lower')

        ax.set_xlabel('[px]')
        ax.set_ylabel('[px]')
        plt.show()

    def find_rotation_angle(self,
                            x, y,
                            error_y, error_x=None,
                            bounds=[[-np.inf, np.inf],
                                    [-np.inf, np.inf]],
                            verbose=1, nlive=1000, maxmcmc=1500, nnest=4,
                            nensemble=4, seed=1234,
                            show=True, save=True, name='./joint_rotation.pdf'):

        if error_x is None:
            error_x = np.zeros_like(x)

        x = np.array(x)
        y = np.array(y)
        error_y = np.array(error_y)

        rot_angle = RotationPosterior(x, y,
                                      error_x, error_y,
                                      bounds)

        job = CPNest(
            rot_angle,
            verbose=verbose,
            nlive=nlive,
            maxmcmc=maxmcmc,
            nnest=nnest,
            nensemble=nensemble,
            seed=seed
        )

        job.run()

        return job, self.__angle_from_m(
            job, show=show, save=save, name=name)

    def run_integration(self):
        self.int = np.sum(self.image, axis=0)

    def show_integration(self, **kwargs):
        fig = plt.figure(**kwargs)
        ax = fig.gca()
        ax.grid()
        ax.plot(self.int, linestyle='solid', color='black', linewidth=0.5)
        ax.set_xlim(0, len(self.int) - 1)
        ax.set_xlabel(r'[px]')

        plt.show()

    def smooth(self, size, lamp):
        if not isinstance(lamp, Spectrum):
            raise TypeError("`lamp` must be a `Spectrum` instance.")
        if self.int is None:
            raise AttributeError("Integration has not been performed, yet.")

        self.smoothed = median_filter(self.int, size=size)
        self.weighted_lamp = lamp.int / self.smoothed

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
                                     bounds_x=[0, len(self.int) - 1],
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

    def show_calibration(self,
                         lamp,
                         exclude=None,
                         **kwargs):
        # FIXME: need to delete all lamp references
        if not isinstance(lamp, Spectrum):
            raise TypeError("`lamp` must be a `Spectrum` instance.")
        if self.model is None:
            raise AttributeError("Calibration has not been run, yet.")

        fig = plt.figure(**kwargs)
        ax = plt.subplot2grid((4, 1), (0, 0), rowspan=3, fig=fig)
        ax.grid()
        ax.errorbar(self.dataset.px, self.dataset.lines,
                    xerr=self.dataset.errpx,
                    linestyle='', capsize=2, marker=',')

        x = np.linspace(0, len(lamp.int) - 1, len(lamp.int))
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
        ax.plot(line, lamp.int, linestyle='solid', color='black',
                linewidth=0.5)
        for lam in self.dataset.lines:
            if lam == self.dataset.lines.min() or lam == self.dataset.lines.max():
                ax.axvline(lam, ymin=0, ymax=1, linewidth=0.5, color='red',
                           linestyle='dotted')
            else:
                ax.axvline(lam, ymin=0, ymax=1, linewidth=0.5, color='navy',
                           linestyle='dashed')
                ax.text(lam, lamp.int.max() / 2,
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
        ax.plot(line, self.int, linestyle='solid', color='black',
                linewidth=0.5)
        for lam in self.dataset.lines:
            if lam == self.dataset.lines.min() or lam == self.dataset.lines.max():
                ax.axvline(lam, ymin=0, ymax=1, linewidth=0.5, color='red',
                           linestyle='dotted')
            elif exclude is not None and self.dataset.names[
                lam] not in exclude:
                ax.axvline(lam, ymin=0, ymax=1, linewidth=0.5, color='navy',
                           linestyle='dashed')
                ax.text(lam, (self.int.max() + self.int.min()) / 2,
                        '{:s}'.format(self.dataset.names[lam]),
                        rotation=90, verticalalignment='center',
                        horizontalalignment='left',
                        size=7.5, color='navy')
        ax.set_xlim(line.min(), line.max())
        ax.set_xlabel(r'$\lambda$ [$\mathring{A}$]')

        plt.show()

    def compare(self, spectrum):
        eq_spectrum = spectrum.int / spectrum.int.max() * self.int.max()

        x = np.linspace(0, len(self.int) - 1, len(self.int))
        ref_calibration = self.calibration(x)

        x_cmp = np.linspace(0, len(spectrum.int) - 1, len(spectrum.int))
        cmp_calibration = spectrum.calibration(x_cmp)

        if len(ref_calibration) == len(cmp_calibration) and np.all(
                self.calibration == eq_spectrum):
            return self.int / eq_spectrum

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

            spectra_ratio[_] = self.int[_] / l_approx(pt)

        return spectra_ratio

    @staticmethod
    def __angle_from_m(job,
                       show=True,
                       save=True, name='./joint_rotation.pdf'):
        post = job.posterior_samples.ravel()
        samples = np.column_stack([np.rad2deg(np.arctan(post['m'])),
                                   post['q']])
        fig = corner.corner(samples, labels=[r'$\alpha$', 'q'],
                            quantiles=[.05, .95],
                            filename=name, show_titles=True,
                            title_fmt='.3e',
                            title_kwargs={'fontsize': 8},
                            label_kwargs={'fontsize': 8},
                            use_math_text=True)

        if save:
            fig.savefig(name)
        if show:
            plt.show()

        # as in corner.core.corner_impl function.
        q_16, q_50, q_84 = corner.quantile(
            samples.T[0], [0.16, 0.5, 0.84])
        q_m, q_p = q_50 - q_16, q_84 - q_50

        return q_50, q_m, q_p
