import numpy as np

from abc import ABC, abstractmethod


class FunctionABC(ABC):
    @abstractmethod
    def derivative(self):
        pass

    @abstractmethod
    def func(self, x, *args):
        pass

    @abstractmethod
    def order(self):
        pass

    @abstractmethod
    def pars(self):
        pass

    @abstractmethod
    def zeros(self):
        pass

    @abstractmethod
    def __call__(self, x):
        pass


class Constant(FunctionABC):
    def __init__(self, *args):
        if len(args) > 1:
            raise ValueError("Constant can only have a single value.")

        missing = self.order + 1 - len(args)
        self.__pars = [*args]

        if missing > 0:
            self.__pars.append([0 for _ in range(missing)])

    def zeros(self):
        return None

    def derivative(self):
        return Constant(0)

    @property
    def order(self):
        return 0

    @property
    def pars(self):
        return tuple(self.__pars)

    def __call__(self, x):
        a = self.pars[0]
        return a * np.ones_like(x)

    @classmethod
    def func(cls, x, *args):
        a = args[0]

        return a * np.ones_like(x)


class Linear(FunctionABC):
    def __init__(self, *args):
        if len(args) > 2:
            raise ValueError("Linear can only have two parameters.")

        missing = self.order + 1 - len(args)
        self.__pars = [*args]

        if missing > 0:
            self.__pars.append([0 for _ in range(missing)])

    def zeros(self):
        a, b = self.pars
        return -b/a,

    def derivative(self):
        a, b = self.pars
        return Constant(a)

    @property
    def order(self):
        return 1

    @property
    def pars(self):
        return tuple(self.__pars)

    def __call__(self, x):
        a, b = self.pars
        return a*x + b

    @classmethod
    def func(cls, x, *args):
        x = np.asarray(x)

        a, b = args

        return a * x + b


class Quadratic(FunctionABC):
    def __init__(self, *args):
        if len(args) > 3:
            raise ValueError("Quadratic can only have three parameters.")

        missing = self.order + 1 - len(args)
        self.__pars = [*args]

        if missing > 0:
            self.__pars.append([0 for _ in range(missing)])

    def zeros(self):
        a, b, c = self.pars
        delta = (b/a)**2 - 4 * c/a

        if delta < 0:
            return None
        elif delta == 0:
            return - 0.5 * b/a
        else:
            return 0.5 * (-b/a + np.sqrt(delta)), 0.5 * (-b/a - np.sqrt(delta))

    def derivative(self):
        a, b, c = self.pars
        return Linear(2 * a, b)

    @property
    def order(self):
        return 2

    @property
    def pars(self):
        return tuple(self.__pars)

    def __call__(self, x):
        a, b, c = self.pars
        return a * x**2 + b * x + c

    @classmethod
    def func(cls, x, *args):
        x = np.asarray(x)

        a, b, c = args

        return a * x**2 + b * x + c


class Cubic(FunctionABC):
    def __init__(self, *args):
        if len(args) > 4:
            raise ValueError("Cubic can only have four parameters.")

        missing = self.order + 1 - len(args)
        self.__pars = [*args]

        if missing > 0:
            self.__pars.append([0 for _ in range(missing)])

    def zeros(self):
        a, b, c, d = self.pars
        q = (3 * c/a - (b/a)**2) / 9
        r = (9 * b*c/a**2 - 27 * d/a - 2 * (b/a)**3) / 54

        s = (r/2 + np.sqrt(q**3 / 27 + r**2 / 4))**(1/3)
        t = (r/2 - np.sqrt(q**3 / 27 + r**2 / 4))**(1/3)

        z1 = s + t - b * a / 3
        z2 = -0.5 * (s + t) - b / (3 * a) + np.sqrt(3) / 2 * (s - t) * 1j
        z3 = -0.5 * (s + t) - b / (3 * a) - np.sqrt(3) / 2 * (s - t) * 1j

        if abs(z2.imag) < np.finfo(np.float64).eps \
                and abs(z3.imag) < np.finfo(np.float64).eps:
            return z1, z2, z3
        else:
            return z1

    def derivative(self):
        a, b, c, d = self.pars
        return Quadratic(3 * a, 2 * b, c)

    @property
    def order(self):
        return 3

    @property
    def pars(self):
        return tuple(self.__pars)

    def __call__(self, x):
        a, b, c, d = self.pars
        return a * x ** 3 + b * x**2 + c * x + d

    @classmethod
    def func(cls, x, *args):
        x = np.asarray(x)

        a, b, c, d = args
        return a * x ** 3 + b * x**2 + c * x + d


class Quartic(FunctionABC):
    def __init__(self, *args):
        if len(args) > 5:
            raise ValueError("Quartic can only have five parameters.")

        missing = self.order + 1 - len(args)
        self.__pars = [*args]

        if missing > 0:
            self.__pars.append([0 for _ in range(missing)])

    def zeros(self):
        a, b, c, d, e = self.pars

        p = (8 * c/a - 3 * (b/a) ** 2) / 8
        S = (8 * d/a - 4 * b * c/a ** 2 + (b / a) ** 3) / 8

        q = 12 * e / a - 3 * b*d/a**2 + (c/a)**2
        s = 27 * (d/a) ** 2 - 72 * c*e/a ** 2 + 27 * b ** 2 * e/a**3 \
            - 9 * b * c * d / a ** 3 + 2 * (c / a) ** 3

        Delta0 = (0.5 * (s + np.sqrt(s ** 2 - 4 * q ** 3))) ** (1 / 3)
        Q = 0.5 * np.sqrt(-2 * p / 3 + (Delta0 + q / Delta0) / (3 * a))

        z1 = -.25 * b / a - Q + 0.5 * np.sqrt(-4 * Q ** 2 - 2 * p + S / Q)
        z2 = -.25 * b / a - Q - 0.5 * np.sqrt(-4 * Q ** 2 - 2 * p + S / Q)
        z3 = -.25 * b / a + Q + 0.5 * np.sqrt(-4 * Q ** 2 - 2 * p + S / Q)
        z4 = -.25 * b / a + Q - 0.5 * np.sqrt(-4 * Q ** 2 - 2 * p + S / Q)

        zeros = []
        if abs(z1.imag) < np.finfo(np.float64).eps:
            zeros.append(z1)
        elif abs(z2.imag) < np.finfo(np.float64).eps:
            zeros.append(z2)
        elif abs(z3.imag) < np.finfo(np.float64).eps:
            zeros.append(z3)
        elif abs(z4.imag) < np.finfo(np.float64).eps:
            zeros.append(z4)

        zeros = np.array(zeros)

        if len(zeros) == 0:
            return None
        else:
            return zeros

    def derivative(self):
        a, b, c, d, e = self.pars
        return Cubic(4 * a, 3 * b, 2 * c, d)

    @property
    def order(self):
        return 4

    @property
    def pars(self):
        return tuple(self.__pars)

    def __call__(self, x):
        a, b, c, d, e = self.pars
        return a * x**4 + b * x**3 + c * x**2 + d * x + e

    @classmethod
    def func(cls, x, *args):
        x = np.asarray(x)

        a, b, c, d, e = args
        return a * x**4 + b * x**3 + c * x**2 + d * x + e


class FunctionGenerator:
    def __init__(self, order, pars):
        self.order = order
        self.pars = pars

    def assign(self):
        if self.order == 0:
            return Constant(*self.pars)
        elif self.order == 1:
            return Linear(*self.pars)
        elif self.order == 2:
            return Quadratic(*self.pars)
        elif self.order == 3:
            return Cubic(*self.pars)
        elif self.order == 4:
            return Quartic(*self.pars)
        else:
            raise ValueError("Unknown order.")
