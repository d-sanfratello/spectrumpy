import numpy as np

from cpnest import model


class QuadraticPosterior(model.Model):
    def __init__(self, x, y, sx, sy, bounds):
        self.x = np.array(x)
        self.y = np.array(y)

        if hasattr(sx, "__iter__"):
            self.sx = np.array(sx)
        else:
            self.sx = np.ones(x.shape[0]) * sx

        if hasattr(sy, "__iter__"):
            self.sy = np.array(sy)
        else:
            self.sy = np.ones(y.shape[0]) * sy

        self.bounds = bounds
        sigma_lvl = 2
        for x, sx in zip(self.x, self.sx):
            x_min = x - sx * sigma_lvl
            x_max = x + sx * sigma_lvl
            self.bounds.append([x_min, x_max])

        self.names = ['a', 'b', 'c']
        for idx, x in enumerate(self.x):
            self.names.append(f'x_{idx}')

    def log_likelihood(self, param):
        # a = param['a']
        # b = param['b']
        # c = param['c']
        for par in param.names:
            exec(f"{par} = {param[par]}")

        sx2 = self.sx**2
        sy2 = self.sy**2
        x = self.x
        y = self.y
        x_hat = [param[x_par] for x_par in param.names
                 if x_par not in ['a', 'b', 'c']]
        x_hat = np.array(x_hat)

        exp_f = self.__quadratic(x_hat, param)

        log_l = -0.5 * ((y - exp_f)**2 / sy2 + (x - x_hat)**2 / sx2)
        log_l -= np.log(2 * np.pi * self.sx * self.sy)

        return log_l.sum()

    def log_prior(self, param):
        logP = super(QuadraticPosterior, self).log_prior(param)
        if np.isfinite(logP):
            logP = 0

        return logP

    @staticmethod
    def __quadratic(x, param):
        return param['a'] * x**2 + param['b'] * x + param['c']
