import numpy as np
from setuptools import setup
from codecs import open

try:
    import cpnest
except ImportError:
    raise Exception(
        "This package needs `cpnest`. To install it follow instructions at"
        "https://github.com/johnveitch/cpnest/tree/massively_parallel."
    )

try:
    import figaro
except ImportError:
    raise Exception(
        "This package needs `figaro`. To install it follow instructions at"
        "https://github.com/sterinaldi/figaro."
    )


with open("requirements.txt") as requires_file:
    requirements = requires_file.read().split("\n")

scripts = [
    'sp-help=spectrumpy.pipelines.help:main',
    'sp-show-image=spectrumpy.pipelines.show_image:main',
    'sp-find-angle=spectrumpy.pipelines.find_rotation_angle:main',
    'sp-rotate=spectrumpy.pipelines.rotate_image:main',
    'sp-show-slices=spectrumpy.pipelines.show_slices:main',
    'sp-crop-image=spectrumpy.pipelines.crop_image:main',
    'sp-integrate=spectrumpy.pipelines.integrate:main',
    'sp-show-spectrum=spectrumpy.pipelines.show_spectrum:main',
    'sp-simulate-spectrum=spectrumpy.pipelines.simulate_spectrum:main',
    'sp-calibrate=spectrumpy.pipelines.calibrate_lines:main',
    'sp-show-calibrated=spectrumpy.pipelines.show_calibrated_spectrum:main',
    'sp-average-image=spectrumpy.pipelines.average_image:main',
    'sp-correct-image=spectrumpy.pipelines.correct_image:main',
    'sp-velocity-resolution=spectrumpy.pipelines'
    '.evaluate_velocity_resolution:main',
    'sp-save-calibrated=spectrumpy.pipelines.save_calibrated_spectrum:main',
    'sp-smooth=spectrumpy.pipelines.smooth_spectrum:main',
    'sp-compare=spectrumpy.pipelines.compare_spectrum:main',
    'sp-find-line=spectrumpy.pipelines.fit_line:main',
]
pymodules = [
    'spectrumpy/pipelines/help',
    'spectrumpy/pipelines/show_image',
    'spectrumpy/pipelines/find_rotation_angle',
    'spectrumpy/pipelines/rotate_image',
    'spectrumpy/pipelines/show_slices',
    'spectrumpy/pipelines/crop_image',
    'spectrumpy/pipelines/integrate',
    'spectrumpy/pipelines/show_spectrum',
    'spectrumpy/pipelines/simulate_spectrum',
    'spectrumpy/pipelines/calibrate_lines',
    'spectrumpy/pipelines/show_calibrated_spectrum',
    'spectrumpy/pipelines/average_image',
    'spectrumpy/pipelines/correct_image',
    'spectrumpy/pipelines/evaluate_velocity_resolution',
    'spectrumpy/pipelines/save_calibrated_spectrum',
    'spectrumpy/pipelines/smooth_spectrum',
    'spectrumpy/pipelines/compare_spectrum',
    'spectrumpy/pipelines/fit_line',
]

setup(
    name='spectrumpy',
    use_scm_version=True,
    description='A package to do spectral calibration and analysis on python.',
    author='Daniele Sanfratello',
    author_email='d.sanfratello@studenti.unipi.it',
    url='https://github.com/d-sanfratello/spectrumpy',
    python_requires='~=3.8.15',
    packages=['spectrumpy'],
    install_requires=requirements,
    include_dirs=[np.get_include()],
    setup_requires=['numpy~=1.21.5', 'setuptools_scm'],
    py_modules=pymodules,
    entry_points={
        'console_scripts': scripts
    },
)
