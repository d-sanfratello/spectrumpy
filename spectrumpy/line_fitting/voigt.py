from scipy.special import voigt_profile


def voigt_ab(x, loc, sigma, gamma, scale, offset):
    return offset - scale * voigt_profile(x - loc, sigma, gamma)


def voigt_em(x, loc, sigma, gamma, scale, offset):
    return offset + scale * voigt_profile(x - loc, sigma, gamma)


models = {
    'absorption': voigt_ab,
    'emission': voigt_em
}
