from numpy import sqrt
from scipy.stats import chi2


class FitStatus:
    def __init__(self, popt, pcov, chisq, dof, order=None):
        self.popt = popt
        self.pcov = pcov
        self.chisq = chisq
        self.dof = dof
        self.order = order

        self.sigma = sqrt(pcov.diagonal())
        self.pvalue = chi2.sf(chisq, df=dof)

        self.status = {}
        if order is not None:
            self.status['fit ord'] = self.order
        for _ in range(len(self.popt)):
            self.status['par{:d}'.format(_)] = f'{self.popt[_]} +- {self.sigma[_]}'
        self.status['cov'] = self.pcov
        self.status['chi2_R'] = self.chisq / self.dof
        self.status['dof'] = self.dof
        self.status['p-value'] = self.pvalue

    def __repr__(self):
        string = '\n'
        for key in self.status.keys():
            string += f'{key}\t:\t{self.status[key]}\n'

        return string

    def __str__(self):
        return self.__repr__()
