import corner
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import warnings

from cpnest import CPNest

from .core_abc_classes import SpectrumImageABC

from spectrumpy.bayes_inference import RotationPosterior


class SpectrumImage(SpectrumImageABC):
    def __init__(self, image, is_lamp, calibration):
        self.image = image
        self._info = {'original': np.copy(image),
                      'lamp': is_lamp,
                      'calibration': calibration}

    def show(self,
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

        fig = plt.figure(*args)
        ax = fig.gca()

        if 'title' in kwargs.keys():
            ax.set_title(kwargs['title'])

        if log:
            ax.imshow(np.log10(self.image), origin='lower')
        else:
            ax.imshow(self.image, origin='lower')

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

        if save:
            fig.savefig(name)
        if show:
            plt.show()

        plt.close()

    def find_rotation_angle(self,
                            x, y,
                            error_y, error_x=None,
                            bounds=[[-np.inf, np.inf],
                                    [-np.inf, np.inf]],
                            verbose=1, nlive=1000, maxmcmc=1500, nnest=4,
                            nensemble=4, seed=1234,
                            show=True, save=True, name='./joint_rotation.pdf'):
        # FIXME: strong coupling

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

    def rotate_image(self, angle, info=None):
        from .spectrum_rotated_image import SpectrumRotatedImage

        rotated = scipy.ndimage.rotate(self.image, angle, reshape=True)

        if info is None:
            info = self.info.copy()

        return SpectrumRotatedImage(rotated, angle, info)

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
