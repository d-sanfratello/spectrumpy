import numpy as np


class Function(object):
    def __init__(self, order):
        self.order = order

    def zeros(self):
        pass

    def derivative(self, x):
        pass

    def __call__(self, x):
        pass


class Constant(Function):
    def __init__(self, a):
        super(Constant, self).__init__(0)
        self.a = a

    def zeros(self):
        return None

    def derivative(self, x):
        try:
            l = len(x)
        except TypeError:
            return self.a
        else:
            return np.zeros(l)

    def __call__(self, x):
        try:
            l = len(x)
        except TypeError:
            return self.a
        else:
            return self.a * np.ones(l)

    @classmethod
    def func(cls, x, a):
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        return a * np.ones(len(x))


class Linear(Function):
    def __init__(self, a, b):
        super(Linear, self).__init__(1)
        self.a = a
        self.b = b

    def zeros(self):
        return - self.b / self.a,

    def derivative(self, x):
        df_dx = Constant(self.a)
        return df_dx(x)

    def __call__(self, x):
        return self.a * x + self.b

    @classmethod
    def func(cls, x, a, b):
        return a * x + b


class Quadratic(Function):
    def __init__(self, a, b, c):
        super(Quadratic, self).__init__(2)
        self.a = a
        self.b = b
        self.c = c

    def zeros(self):
        delta = (self.b/self.a)**2 - 4 * self.c/self.a

        if delta < 0:
            return None
        elif delta == 0:
            return - 0.5 * self.b / self. a
        else:
            return 0.5 * (-self.b/self.a + np.sqrt(delta)), 0.5 * (-self.b/self.a - np.sqrt(delta))

    def derivative(self, x):
        x = np.squeeze(np.array(x))
        df_dx = Linear(2 * self.a, self.b)
        return df_dx(x)

    def __call__(self, x):
        return self.a * x**2 + self.b * x + self.c

    @classmethod
    def func(cls, x, a, b, c):
        return a * x**2 + b * x + c


class Cubic(Function):
    def __init__(self, a, b, c, d):
        super(Cubic, self).__init__(3)
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def zeros(self):
        q = (3 * self.c/self.a - (self.b/self.a)**2) / 9
        r = (9 * self.b*self.c/self.a**2 - 27 * self.d/self.a - 2 * (self.b/self.a)**3) / 54

        s = (r/2 + np.sqrt(q**3 / 27 + r**2 / 4))**(1/3)
        t = (r/2 - np.sqrt(q**3 / 27 + r**2 / 4))**(1/3)

        z1 = s + t - self.b * self.a / 3
        z2 = -0.5 * (s + t) - self.b / (3 * self.a) + np.sqrt(3) / 2 * (s - t) * 1j
        z3 = -0.5 * (s + t) - self.b / (3 * self.a) - np.sqrt(3) / 2 * (s - t) * 1j

        if abs(z2.imag) < np.finfo(np.float64).eps and abs(z3.imag) < np.finfo(np.float64).eps:
            return z1, z2, z3
        else:
            return z1

    def derivative(self, x):
        df_dx = Quadratic(3 * self.a, 2 * self.b, self.c)
        return df_dx(x)

    def __call__(self, x):
        return self.a * x ** 3 + self.b * x**2 + self.c * x + self.d

    @classmethod
    def func(cls, x, a, b, c, d):
        return a * x ** 3 + b * x**2 + c * x + d


class Quartic(Function):
    def __init__(self, a, b, c, d, e):
        super(Quartic, self).__init__(4)
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e

    def zeros(self):
        p = (8 * self.c / self.a - 3 * (self.b / self.a) ** 2) / 8
        S = (8 * self.d / self.a - 4 * self.b * self.c / self.a ** 2 + (self.b / self.a) ** 3) / 8

        q = 12 * self.e / self.a - 3 * self.b*self.d/self.a**2 + (self.c/self.a)**2
        s = 27 * (self.d / self.a) ** 2 - 72 * self.c * self.e / self.a ** 2 + 27 * self.b ** 2 * self.e / self.a ** 3 \
            - 9 * self.b * self.c * self.d / self.a ** 3 + 2 * (self.c / self.a) ** 3

        Delta0 = (0.5 * (s + np.sqrt(s ** 2 - 4 * q ** 3))) ** (1 / 3)
        Q = 0.5 * np.sqrt(-2 * p / 3 + (Delta0 + q / Delta0) / (3 * self.a))

        z1 = -.25 * self.b / self.a - Q + 0.5 * np.sqrt(-4 * Q ** 2 - 2 * p + S / Q)
        z2 = -.25 * self.b / self.a - Q - 0.5 * np.sqrt(-4 * Q ** 2 - 2 * p + S / Q)
        z3 = -.25 * self.b / self.a + Q + 0.5 * np.sqrt(-4 * Q ** 2 - 2 * p + S / Q)
        z4 = -.25 * self.b / self.a + Q - 0.5 * np.sqrt(-4 * Q ** 2 - 2 * p + S / Q)

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

    def derivative(self, x):
        df_dx = Cubic(4 * self.a, 3 * self.b, 2 * self.c, self.d)
        return df_dx(x)

    def __call__(self, x):
        return self.a * x ** 4 + self.b * x**3 + self.c * x**2 + self.d * x + self.e

    @classmethod
    def func(cls, x, a, b, c, d, e):
        return a * x ** 4 + b * x ** 3 + c * x**2 + d * x + e
