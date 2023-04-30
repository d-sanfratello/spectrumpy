from .voigt import voigt_ab, voigt_em


def lorentz_ab(x, loc, gamma, scale, offset):
    return voigt_ab(x, loc, 0, gamma, scale, offset)


def lorentz_em(x, loc, gamma, scale, offset):
    return voigt_em(x, loc, 0, gamma, scale, offset)


models = {
    'absorption': lorentz_ab,
    'emission': lorentz_em
}
