available_models = ['linear', 'quadratic', 'cubic', 'quartic']

names = {
    'linear': ['x_1', 'x_0'],
    'quadratic': ['x_2', 'x_1', 'x_0'],
    'cubic': ['x_3', 'x_2', 'x_1', 'x_0'],
    'quartic': ['x_4', 'x_3', 'x_2', 'x_1', 'x_0']
}


def linear(x, a, b):
    return a * x + b


def quadratic(x, a, b, c):
    return a * x**2 + b * x + c


def cubic(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d


def quartic(x, a, b, c, d, e):
    return a * x**4 + b * x**3 + c * x**2 + d * x + e


models = {
    'linear': linear,
    'quadratic': quadratic,
    'cubic': cubic,
    'quartic': quartic
}
