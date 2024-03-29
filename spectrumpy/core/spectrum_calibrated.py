import matplotlib.pyplot as plt
import numpy as np
import warnings

from loess.loess_1d import loess_1d


class CalibratedSpectrum:
    def __init__(self, wavelength, spectrum, units):
        self.wl = wavelength
        self.sp = spectrum
        self.units = units

    def show(self,
             model=None,
             show=False, save=True,
             name=None,
             legend=False,
             overlay_spectrum=None,
             labels=None,
             lines=None,
             ylabel=None,
             *args, **kwargs):

        fig = plt.figure(*args)
        ax = fig.gca()

        if 'title' in kwargs.keys():
            ax.set_title(kwargs['title'])

        # ax.grid()

        fig = self._show_calibrated(fig, labels=labels, ylabel=ylabel,
                                    overlay_spectrum=overlay_spectrum,
                                    lines=lines, model=model, **kwargs)

        if name is None:
            name = './spectrum_calibrated_show.pdf'

        if legend:
            ax.legend(loc='best')

        if save:
            fig.savefig(name)
        if show:
            plt.show()

        plt.close()

    def _show_calibrated(self,
                         fig,
                         labels=None,
                         ylabel=None,
                         overlay_spectrum=None,
                         lines=None,
                         **kwargs):

        label1, label2 = None, None
        if labels is not None:
            label1 = labels[0]
            if len(labels) == 2:
                label2 = labels[1]

        ax = fig.gca()

        if lines is not None:
            for lam in lines:
                ax.axvline(lam,
                           ymin=0, ymax=1, linewidth=0.5, color='navy',
                           linestyle='-', alpha=0.5)

        ax.plot(self.wl, self.sp,
                linestyle='solid', color='black', linewidth=0.5,
                label=label1)

        if 'xlim' in kwargs.keys() and kwargs['xlim'] is not None:
            ax.set_xlim(kwargs['xlim'][0], kwargs['xlim'][1])
        else:
            ax.set_xlim(self.wl.min(), self.wl.max())

        if 'ylim' in kwargs.keys() and kwargs['ylim'] is not None:
            ax.set_ylim(kwargs['ylim'][0], kwargs['ylim'][1])

        ax.set_xlabel(f"[{self.units}]")
        if ylabel is None:
            ax.set_ylabel(r'[arb.units]')
        else:
            ax.set_ylabel(f"{ylabel}")

        if overlay_spectrum is not None:
            ax.plot(
                overlay_spectrum.wl,
                overlay_spectrum.sp,
                linestyle='solid', color='orange', linewidth=0.5,
                label=label2
            )

        return fig

    def normalize(self):
        min_sp = np.min(self.sp)
        max_sp = np.max(self.sp)
        weight = max_sp - min_sp

        normalized = (self.sp - min_sp) / weight

        return CalibratedSpectrum(
            wavelength=np.asarray(self.wl),
            spectrum=np.asarray(normalized),
            units=self.units
        )

    def smooth(self, frac=0.2):
        wl, smoothed, w = loess_1d(
            self.wl,
            self.sp,
            degree=2,
            frac=frac
        )

        return CalibratedSpectrum(
            wavelength=np.asarray(wl),
            spectrum=np.asarray(smoothed),
            units=self.units
        )

    def cut(self, low=None, high=None):
        good_idx = np.argwhere((self.wl >= low) & (self.wl <= high))

        return CalibratedSpectrum(
            wavelength=np.asarray(self.wl[good_idx]),
            spectrum=np.asarray(self.sp[good_idx]),
            units=self.units
        )

    @classmethod
    def compare(cls, spectrum1, spectrum2):
        if not np.all(spectrum1.wl == spectrum2.wl):
            raise ValueError(
                "Spectra must have the same calibration."
            )
        if not spectrum1.units == spectrum2.units:
            raise ValueError(
                "Spectra must have the same units."
            )

        idx_2_is_zero = np.argwhere(spectrum2.sp == 0)
        mask = np.ones(spectrum1.wl.shape, dtype=bool)
        mask[idx_2_is_zero] = False

        wl = spectrum1.wl[mask]
        sp1 = spectrum1.sp[mask]
        sp2 = spectrum2.sp[mask]
        units = spectrum1.units

        inverse_mask = np.logical_not(mask.copy())
        removed_points = len(np.argwhere(inverse_mask is False))

        if removed_points > 0:
            warnings.warn(
                f"Removed {removed_points} point(s) as they where 0 in divisor"
                f" array. Removed wavelengths are at: "
                f"{spectrum1.wl[inverse_mask]} {units}."
            )

        return wl, sp1 / sp2, units
