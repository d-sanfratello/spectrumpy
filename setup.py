import numpy as np
from setuptools import setup
# from setuptools.command.build_ext import build_ext as _build_ext
from codecs import open

try:
    import cpnest
except ImportError:
    raise Exception("This package needs `cpnest`. To install it follow "
    "instructions at "
    "https://github.com/johnveitch/cpnest/tree/massively_parallel.")

#
# # see https://stackoverflow.com/a/21621689/1862861 for why this is here
# class build_ext(_build_ext):
#     def finalize_options(self):
#         _build_ext.finalize_options(self)
#         # Prevent numpy from thinking it is still in its setup process:
#         __builtins__.__NUMPY_SETUP__ = False
#         self.include_dirs.append(np.get_include())


with open("requirements.txt") as requires_file:
    requirements = requires_file.read().split("\n")

setup(
    name='spectrumpy',
    use_scm_version=True,
    description='A package to do spectral calibration and analysis on python.',
    author='Daniele Sanfratello',
    author_email='d.sanfratello@studenti.unipi.it',
    url='https://gitlab.com/da.sanfratello/spectrumpy',
    python_requires='>=3.10.4',
    packages=['spectrumpy'],
    install_requires=requirements,
    include_dirs=[np.get_include()],
    setup_requires=['numpy>=1.23.1'],
    entry_points={},
    )
