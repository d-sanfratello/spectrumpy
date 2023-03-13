import json
import matplotlib.pyplot as plt
import numpy as np
import warnings

from matplotlib.ticker import Formatter
from scipy.ndimage import median_filter

from spectrumpy.dataset import Dataset
from spectrumpy import function_models as fs


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
             calibration=True,
             overlay_pixel=False,
             overlay_spectrum=None,
             label=None,
             model_label=None,
             *args, **kwargs):

        fig = plt.figure(*args)
        ax = fig.gca()

        if 'title' in kwargs.keys():
            ax.set_title(kwargs['title'])

        ax.grid()

        if calibration:
            if overlay_pixel:
                fig = self._show_with_overlay(
                    fig,
                    model=model,
                    overlay_pixel=overlay_pixel,
                    overlay_spectrum=overlay_spectrum,
                    label=label,
                    model_label=model_label,
                    **kwargs
                )

                if name is None:
                    name = './spectrum_doubleaxis_show.pdf'
            else:
                fig = self._show_calibrated(
                    fig,
                    model=model,
                    overlay_pixel=overlay_pixel,
                    overlay_spectrum=overlay_spectrum,
                    label=label,
                    model_label=model_label,
                    **kwargs
                )

                if name is None:
                    name = './spectrum_calibrated_show.pdf'
        else:
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

        x = np.linspace(0, len(self.spectrum) - 1, len(self.spectrum))
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

    def _show_calibrated(self,
                         fig,
                         model=None,
                         label=None,
                         model_label=None,
                         **kwargs):
        if self.calibration is None:
            raise AttributeError(
                "Unknown calibration."
            )

        calib_model, calib_pars = self.calibration
        ax = fig.gca()

        x_clb = np.linspace(0, len(self.spectrum) - 1, len(self.spectrum))
        x_clb = calib_model(x_clb, *calib_pars)
        ax.plot(x_clb, self.spectrum,
                linestyle='solid', color='black', linewidth=0.5,
                label=label)

        if model is not None:
            if hasattr(model, '__iter__') and \
                    isinstance(model[0], Spectrum):
                for mdl, label in zip(model, model_label):
                    sp = mdl.spectrum
                    ax.plot(x_clb, sp,
                            linestyle='dashed', linewidth=0.5,
                            label=label)
            elif hasattr(model, '__iter__') \
                    and hasattr(model[0], '__call__'):
                for mdl in model:
                    ax.plot(x_clb, mdl(x_clb),
                            linestyle='dashed', linewidth=0.5,
                            label=mdl.__name__)
            else:
                ax.plot(x_clb, model(x_clb),
                        linestyle='solid', color='red', linewidth=0.5)

        if 'xlim' in kwargs.keys():
            ax.set_xlim(kwargs['xlim'][0], kwargs['xlim'][1])
        else:
            ax.set_xlim(x_clb.min(), x_clb.max())

        if 'ylim' in kwargs.keys():
            ax.set_ylim(kwargs['ylim'][0], kwargs['ylim'][1])

        ax.set_xlabel(f"[{self._info['calib units']}]")

        dset = self.dataset
        for lam in dset.lines:
            ax.axvline(lam,
                       ymin=0, ymax=1, linewidth=0.5, color='navy',
                       linestyle='dashed')
            # ax.text(lam, self.spectrum.max() / 2,
            #         f'{dset.names[lam]}',
            #         rotation=90, verticalalignment='center',
            #         horizontalalignment='left',
            #         size=7.5, color='navy')

        return fig

    def _show_with_overlay(self,
                           fig,
                           model=None,
                           overlay_spectrum=None,
                           label=None,
                           model_label=None,
                           **kwargs):
        if self.calibration is None:
            raise AttributeError(
                "Unknown calibration."
            )

        calib_model, calib_pars = self.calibration

        ax = fig.gca()

        px = np.linspace(0, len(self.spectrum) - 1, len(self.spectrum))
        x_clb = calib_model(px, *calib_pars)
        ax.plot(px, self.spectrum,
                linestyle='solid', color='black', linewidth=0.5,
                label=label, picker=True)

        if model is not None:
            if hasattr(model, '__iter__') and \
                    isinstance(model[0], Spectrum):
                for mdl, label in zip(model, model_label):
                    sp = mdl.spectrum
                    ax.plot(px, sp,
                            linestyle='dashed', linewidth=0.5,
                            label=label)
            elif hasattr(model, '__iter__') \
                    and hasattr(model[0], '__call__'):
                for mdl in model:
                    ax.plot(px, mdl(px),
                            linestyle='dashed', linewidth=0.5,
                            label=mdl.__name__)
            else:
                ax.plot(px, model(px),
                        linestyle='solid', color='red', linewidth=0.5)

        if 'xlim' in kwargs.keys():
            ax.set_xlim(kwargs['xlim'][0], kwargs['xlim'][1])
        else:
            ax.set_xlim(px.min(), px.max())

        if 'ylim' in kwargs.keys():
            ax.set_ylim(kwargs['ylim'][0], kwargs['ylim'][1])

        ax.set_xlabel("[px]")

        dset = self.dataset
        for lam in dset.px:
            ax.axvline(lam,
                       ymin=0, ymax=1, linewidth=0.5, color='navy',
                       linestyle='dashed')

        fig.subplots_adjust(bottom=0.3)
        ax2 = ax.twiny()

        if overlay_spectrum is not None:
            ax.plot(px, overlay_spectrum,
                    linestyle='solid', color='orange', linewidth=0.5)

        tick_formatter = PxFormatter(ax2, px, self.calibration)
        ax2.xaxis.set_major_formatter(tick_formatter)

        ax2.set_xlabel(f"[{self._info['calib units']}]")

        ax2.spines["bottom"].set_position(("axes", -0.2))
        ax2.spines["bottom"].set_edgecolor('blue')

        ax2.xaxis.label.set_color('blue')
        ax2.xaxis.set_label_position("bottom")

        ax2.tick_params(which='both', axis='x', colors='blue', rotation=45)
        ax2.xaxis.set_ticks_position("bottom")

        ax.locator_params(nbins=15)
        ax2.set_visible(True)

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

    def smooth(self, size, lamp):
        if self.info['lamp']:
            raise ValueError("It makes no sense to smooth the lamp.")
        if not isinstance(lamp, Spectrum):
            raise TypeError("`lamp` must be a `Spectrum` instance.")

        smoothed = median_filter(self.spectrum, size=size)

        return Spectrum(smoothed, self.info)

    def weight(self, smoothed, lamp):
        if not self.info['lamp']:
            warnings.warn("You are weighting a spectrum containing actual "
                          "data. Proceed with caution.")
        if not isinstance(lamp, Spectrum):
            raise TypeError("`lamp` must be a `Spectrum` instance.")

        weighted = lamp.spectrum / smoothed

        return Spectrum(weighted, self.info)

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

        self._info['calibration'] = calibration
        self._info['calib units'] = units

    def save_info(self, filename='./info.json'):
        # FIXME: check this
    
        dict_ = self.info.copy()
        dict_['original'] = None

        calibration = dict_.pop('calibration')
        dict_['calib order'] = calibration.order
        dict_['calib pars'] = calibration.pars

        dataset = dict_.pop('dataset')
        dataset_dict = dataset.__dict__.copy()
        for key, value in dataset_dict.items():
            if isinstance(value, np.ndarray):
                value = value.tolist()
            dict_[key] = value

        dict_['uncalib spectrum'] = dict_['uncalib spectrum'].tolist()

        crop_x = dict_.pop('crop_x')
        crop_y = dict_.pop('crop_y')

        if not isinstance(crop_x, slice):
            crop_x = slice(*crop_x)
        if not isinstance(crop_y, slice):
            crop_y = slice(*crop_y)

        dict_['crop_x'] = [crop_x.start, crop_x.stop, crop_x.step]
        dict_['crop_y'] = [crop_y.start, crop_y.stop, crop_y.step]

        s = json.dumps(dict_, indent=4)

        with open(filename, 'w') as f:
            json.dump(s, f)

    @staticmethod
    def load_info(filename):
        with open(filename, 'r') as fjson:
            dictjson = json.load(fjson)

        dict_ = json.loads(dictjson)

        order = dict_.pop('calib order')
        pars = dict_.pop('calib pars')

        px = dict_.pop('px')
        errpx = dict_.pop('errpx')
        lines = dict_.pop('lines')
        errlines = dict_.pop('errlines')

        names = dict_.pop('names')
        str_keys = list(names.keys())
        for key in str_keys:
            new_key = float(key)
            label = names.pop(key)
            names[new_key] = label

        crop_x = dict_.pop('crop_x')
        crop_y = dict_.pop('crop_y')

        uncalib_spectrum = dict_.pop('uncalib spectrum')

        info_dict = dict_.copy()

        info_dict['crop_x'] = slice(*crop_x)
        info_dict['crop_y'] = slice(*crop_y)

        info_dict['uncalib spectrum'] = np.array(uncalib_spectrum)

        f_gen = fs.FunctionGenerator(order=order, pars=pars)
        info_dict['calibration'] = f_gen.assign()
        info_dict['dataset'] = Dataset(px=px, errpx=errpx,
                                       lines=lines, errlines=errlines,
                                       names=names)

        return info_dict

    def compare(self, spectrum):
        # FIXME: low cohesion
        calib_model, calib_pars = self.calibration

        eq_spectrum = spectrum.int / spectrum.int.max() * self.int.max()

        x = np.linspace(0, len(self.int) - 1, len(self.int))
        ref_calibration = calib_model(x, *calib_pars)

        x_cmp = np.linspace(0, len(spectrum.int) - 1, len(spectrum.int))
        cmp_calibration = spectrum.calibration(x_cmp)

        if len(ref_calibration) == len(cmp_calibration) and np.all(
                ref_calibration == eq_spectrum):
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

    @property
    def info(self):
        return self._info


class PxFormatter(Formatter):
    def __init__(self,
                 ax,
                 px_range=None,
                 calibration=None):
        super().__init__()
        self.set_axis(ax)
        self.min = px_range.min()
        self.width = px_range.max() - px_range.min()

        self.model, self.pars = calibration

    def __call__(self, x, pos=None):
        vmin, vmax = self.axis.get_view_interval()

        x_ = x * self.width + self.min
        l = self.model(x_, *self.pars)

        return f"{l:.2f}"
