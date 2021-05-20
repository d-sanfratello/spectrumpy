import numpy as np
import matplotlib.pyplot as plt

from numbers import Number
from scipy.stats import norm
import logging


class MHsampler:
    def __init__(self, x, y, errx, bounds_pars, bounds_x, n=1000, delta=0.1, seed=None):
        self.log = logging.getLogger()
        console = logging.StreamHandler()
        self.log.addHandler(console)
        self.log.setLevel(logging.INFO)

        self.x = x
        self.y = y
        self.errx = errx
        self.n = n

        if isinstance(delta, Number):
            self.delta = delta * np.ones(len(bounds_pars) + 1)
        else:
            self.delta = delta

        self.bounds = bounds_pars
        self.bounds_x = bounds_x

        self.rng = np.random.default_rng(seed)

    def run(self, verbose=True):
        x_hat_0 = np.array([self.rng.uniform(self.bounds_x[0], self.bounds_x[1]) for _ in range(len(self.x))])
        theta_0 = np.array([self.rng.uniform(b_par[0], b_par[1]) for b_par in self.bounds])

        log_prob0 = self.log_likelihood(self.x, self.y, self.errx, x_hat_0, theta_0) + self.log_prior(theta_0) + \
                    self.log_prior_x(x_hat_0)

        # log_mean0 = log_prob0 + np.log(theta_0)

        samples = [theta_0]
        x_hat = [x_hat_0]
        # log_mean = [log_mean0]

        accepted = 0
        rejected = 0

        i = 0
        while i < self.n:
            if verbose and i % 1000 == 0:
                self.log.info('{:.1f}%'.format(i / self.n * 100))

            while True:
                x_hat_p = np.array([self.__proposal(_, self.delta[-1]) for _ in x_hat_0])

                prior_x = self.log_prior_x(x_hat_p)

                if np.isfinite(prior_x):
                    break

            while True:
                theta_p = np.array([self.__proposal(theta_0[_], self.delta[_]) for _ in range(len(theta_0))])

                prior = self.log_prior(theta_p)

                if np.isfinite(prior):
                    break

            log_prob = self.log_likelihood(self.x, self.y, self.errx, x_hat_p, theta_p) + prior + prior_x

            log_r = log_prob - log_prob0

            if np.log(self.rng.uniform(0, 1)) <= log_r:
                theta_0 = theta_p
                x_hat_0 = x_hat_p
                log_prob0 = log_prob
                accepted += 1
            else:
                rejected += 1

            samples.append(theta_0)
            x_hat.append(x_hat_0)
            # log_mean0 +=
            i += 1

        return np.array(samples[::self.thin]), np.array(x_hat[::self.thin]), \
            np.array(samples), np.array(x_hat), \
            accepted, rejected

    @property
    def thin(self):
        return 1

    def __proposal(self, loc, scale):
        return norm.rvs(loc=loc, scale=scale, random_state=self.rng)

    def log_prior(self, theta):
        log_prior = 0.

        for _, _theta in enumerate(theta):
            if self.bounds[_, 0] <= _theta[_] <= self.bounds[_, 1]:
                log_prior += 0.
            else:
                return -np.inf

        return log_prior

    def log_prior_x(self, x_hat):
        log_prior = 0.

        for _x in x_hat:
            if self.bounds_x[0] <= _x <= self.bounds_x[1]:
                log_prior += 0.
            else:
                return -np.inf

        return log_prior

    def log_likelihood(self, x, y, errx, x_hat, theta):
        raise AttributeError("Not defined!")
