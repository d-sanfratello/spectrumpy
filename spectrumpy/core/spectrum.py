import matplotlib.pyplot as plt
import numpy as np

from spectrumpy.core import CalibratedSpectrum
from spectrumpy.dataset import Dataset


class Spectrum:
    def __init__(self, int_spectrum, info=None):
        self.spectrum = int_spectrum
        self._info = info

        self.dataset = None
        self.calibration = None

        if info is None:
            self._info = {}
        else:
            if 'dataset' in self.info.keys():
                self.dataset = self.info['dataset']
            if 'calibration' in self.info.keys():
                self.calibration = self.info['calibration']

        self._info['uncalib spectrum'] = np.copy(self.spectrum)

    def show(self,
             model=None,
             show=False, save=True,
             name=None,
             legend=False,
             # calibration=True,
             overlay_spectrum=None,
             label=None,
             model_label=None,
             show_lines=True,
             *args, **kwargs):

        fig = plt.figure(*args)
        ax = fig.gca()

        if 'title' in kwargs.keys():
            ax.set_title(kwargs['title'])

        ax.grid()

        fig = self._show(
            fig,
            model=model,
            label=label,
            model_label=model_label,
            **kwargs
        )

        if name is None:
            name = './spectrum_show.pdf',

        if legend:
            ax.legend(loc='best')

        if save:
            fig.savefig(name)
        if show:
            plt.show()

        plt.close()

    def _show(self,
              fig,
              model=None,
              label=None,
              model_label=None,
              **kwargs):

        ax = fig.gca()

        calib_model, calib_pars = self.calibration

        x = np.linspace(0,
                        len(self.spectrum) - 1,
                        len(self.spectrum))
        ax.plot(x, self.spectrum,
                linestyle='solid', color='black', linewidth=0.5,
                label=label)

        if model is not None:
            if hasattr(model, '__iter__') and \
                    isinstance(model[0], Spectrum):
                for mdl, label in zip(model, model_label):
                    sp = mdl.spectrum
                    ax.plot(x, sp,
                            linestyle='dashed', linewidth=0.5,
                            label=label)
            elif hasattr(model, '__iter__') \
                    and not hasattr(model[0], '__call__'):
                model = np.asarray(model)
                l, m, h = np.percentile(model, [5, 50, 95], axis=0)

                ax.plot(x, m, lw=0.5, color='r')
                ax.fill_between(x, l, h, facecolor='red', alpha=0.5)
            elif hasattr(model, '__iter__') \
                    and hasattr(model[0], '__call__'):
                for mdl in model:
                    ax.plot(x, mdl(x),
                            linestyle='dashed', linewidth=0.5,
                            label=mdl.__name__)
            else:
                ax.plot(x, model(x),
                        linestyle='solid', color='red', linewidth=0.5)

        if 'xlim' in kwargs.keys():
            ax.set_xlim(kwargs['xlim'][0], kwargs['xlim'][1])
        else:
            ax.set_xlim(0, len(self.spectrum)-1)

        if 'ylim' in kwargs.keys():
            ax.set_ylim(kwargs['ylim'][0], kwargs['ylim'][1])

        ax.set_xlabel(r'[px]')

        return fig

    @staticmethod
    def show_calibration_fit(px, lam, s_px, s_lam=None,
                             model=None, x=None,
                             show=False, save=True,
                             name='./show_calibration_fit.pdf',
                             legend=False,
                             logscale=False,
                             errorscale=None,
                             *args, **kwargs):
        if model is None:
            raise ValueError("You need to pass a model, to show the fit.")
        if 'units' not in kwargs.keys():
            raise KeyError("Wavelength units must be passed.")

        fig = plt.figure(*args)
        ax = fig.gca()
        plt.grid(axis='both', which='major')
        ax.grid(axis='both', which='major')

        if 'title' in kwargs.keys():
            ax.set_title(kwargs['title'])

        if errorscale is None:
            ax.errorbar(px, lam, s_lam, s_px,
                        capsize=2, linestyle='',
                        ms=2)
        else:
            ax.errorbar(px, lam, s_lam*errorscale, s_px*errorscale,
                        label=f"Errorbars are scaled by a factor of"
                              f" {errorscale}",
                        capsize=2, linestyle='',
                        ms=2)
            legend = True

        if model is not None:
            if hasattr(model, '__iter__') \
                    and not hasattr(model[0], '__call__'):
                model = np.asarray(model)
                l, m, h = np.percentile(model, [5, 50, 95], axis=0)

                ax.plot(x, m, lw=0.8, color='r')
                ax.fill_between(x, l, h, facecolor='red', alpha=0.25)
            elif hasattr(model, '__iter__') \
                    and hasattr(model[0], '__call__'):
                for mdl in model:
                    ax.plot(x, mdl(x),
                            linestyle='dashed', linewidth=0.5,
                            label=mdl.__name__)
            else:
                ax.plot(x, model(x),
                        linestyle='solid', color='red', linewidth=0.5)

        if 'xlim' in kwargs.keys():
            ax.set_xlim(kwargs['xlim'][0], kwargs['xlim'][1])
        if 'ylim' in kwargs.keys():
            ax.set_ylim(kwargs['ylim'][0], kwargs['ylim'][1])

        ax.set_xlabel('[px]')
        ax.set_ylabel(f"[{kwargs['units']}]")

        if legend:
            ax.legend(loc='best')

        if logscale:
            ax.set_xscale('log')
            ax.set_yscale('log')

        if save:
            fig.savefig(name)
        if show:
            plt.show()

        plt.close()

    def normalize(self):
        min_sp = np.min(self.spectrum)
        max_sp = np.max(self.spectrum)
        weight = max_sp - min_sp

        normalized = (self.spectrum - min_sp) / weight

        return Spectrum(int_spectrum=normalized, info=self.info)

    def assign_dataset(self, dataset=None, *,
                       lines=None, px=None, errpx=None, errlines=None,
                       names=None):
        if dataset is None:
            dataset = Dataset(lines, px, errpx, errlines=errlines, names=names)

        self.dataset = dataset
        self._info['dataset'] = dataset

    def assign_calibration(self, calibration, pars, units):
        if not hasattr(calibration, '__call__'):
            raise TypeError("'calibration' must be a callable.")

        self.calibration = (calibration, pars)

        self._info['calibration'] = self.calibration
        self._info['calib units'] = units

    def return_calibrated(self):
        if self.calibration is None:
            raise AttributeError(
                "Unknown calibration."
            )

        calib_model, calib_pars = self.calibration
        px = np.linspace(0,
                         len(self.spectrum) - 1,
                         len(self.spectrum))
        x_clb = calib_model(px, *calib_pars)

        return CalibratedSpectrum(
            x_clb,
            self.spectrum,
            self.info['calib units']
        )

    def apply_shift(self, shift=0):
        if shift > 0:
            self.spectrum = np.concatenate(
                (np.zeros(shift), self.spectrum[:-shift])
            )
        elif shift < 0:
            shift = -shift
            self.spectrum = self.spectrum[shift - 1:]

    @classmethod
    def even_spectra(cls, spectrum1, spectrum2):
        diff = len(spectrum1.spectrum) - len(spectrum2.spectrum)

        if diff > 0:  # sp1 is longer than sp2
            spectrum2.spectrum = spectrum2.spectrum[0:-diff]
        elif diff < 0:  # sp1 is shorter than sp2
            spectrum1.spectrum = spectrum1.spectrum[0:diff]

    @property
    def info(self):
        return self._info
