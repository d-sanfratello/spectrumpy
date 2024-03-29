import corner
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import warnings

from spectrumpy.core.spectrum import Spectrum


class SpectrumImage:
    def __init__(self, image, is_lamp, info=None):
        self.image = image

        if info is None:
            self._info = {'original': np.copy(image),
                          'lamp': is_lamp}
        else:
            self._info = info.copy()
            self._info['original'] = np.copy(image)

    def show(self,
             fig=None,
             log=True, model=None, x=None,
             show=False, save=True, name='./image_show.pdf',
             legend=False,
             *args, **kwargs):
        if not isinstance(log, bool):
            raise TypeError("`log` must be a bool.")

        if model is not None and x is None:
            raise ValueError("Must pass `x` array-like together with a model.")

        if not np.all(self.image):
            warnings.warn("Image contains zeros, using log my create "
                          "artifacts in the representation. This should not "
                          "affect data.")

        user_defined_fig = True
        if fig is None:
            fig = plt.figure(*args)
            user_defined_fig = False

        ax = fig.gca()

        if 'title' in kwargs.keys() and not user_defined_fig:
            ax.set_title(kwargs['title'])

        if log:
            plot_image = np.log10(self.image)
        else:
            plot_image = self.image
        ax.imshow(plot_image,
                  origin='lower',
                  cmap='Greys')

        if model is not None:
            if hasattr(model, '__iter__') \
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

        ax.set_xlabel('[px]')
        ax.set_ylabel('[px]')

        if legend:
            ax.legend(loc='best')

        img_size = self.image.shape
        wh_ratio = img_size[1] / img_size[0]

        if wh_ratio > 5:
            wh_ratio = 5

        if img_size[1] >= img_size[0]:  # wider than tall
            new_height = fig.get_figwidth() / wh_ratio
            fig.set_figheight(new_height)
        else:  # taller than wide
            new_width = fig.get_figwidth() * wh_ratio
            fig.set_figwidth(new_width)

        fig.tight_layout()

        if save:
            fig.savefig(name)
        if show:
            plt.show()

        if not user_defined_fig:
            plt.close()

    def rotate_image(self, angle, info=None):
        rotated = scipy.ndimage.rotate(self.image, angle, reshape=True)

        if info is None:
            info = self.info.copy()
        is_lamp = info['lamp']
        return SpectrumImage(rotated, is_lamp, info)

    def crop_image(self, crop_x=None, crop_y=None, info=None):
        if crop_x is None:
            crop_x = slice(0, self.image.shape[1])
        elif not isinstance(crop_x, slice):
            crop_x = slice(*crop_x)

        if crop_y is None:
            crop_y = slice(0, self.image.shape[0])
        elif not isinstance(crop_y, slice):
            crop_y = slice(*crop_y)

        crop = np.copy(self.image[crop_y, crop_x])

        if info is None:
            info = self.info.copy()
        is_lamp = info['lamp']
        return SpectrumImage(crop, is_lamp, info)

    def slice_image(self, level, info=None):
        if not 0 <= level < self.image.shape[0]:
            raise ValueError(f"Slice {level} is outside of image edges.")

        if info is None:
            info = self.info.copy()

        if level > 0:
            return self.crop_image(crop_y=slice(level-1, level), info=info)
        else:
            return self.crop_image(crop_y=slice(level, level+1), info=info)

    def run_integration(self, info=None):
        integrated = np.sum(self.image, axis=0)

        if info is None:
            info = self.info.copy()

        return Spectrum(integrated, info)

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

    @property
    def info(self):
        return self._info
