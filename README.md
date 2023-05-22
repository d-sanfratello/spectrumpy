# Spectrumpy
`spectrumpy` is a python package to analyze FITS files containing spectra, 
calibrate them and perform some minor tasks to analyze the data from the 
spectra.

It was developed as a piece of software to be used for the Astrofisica 
Osservativa exam at Università di Pisa.

## Installation
Installation requires `python 3.8.15`. To run the installation just run 
`python setup.py install` in your terminal. The use of an environment is 
strongly advised.

## Pipelines
At any time, a more detailed explaination of the functionalities of each 
pipeline may be obtained with `sp-[COMMAND] -h`.  

`sp-help`: A simple script that helps navigate the different available 
operations with this package. The output also represents the ideal order 
commands should be called to analyse a spectral image.

`sp-simulate-spectrum`: This simple script simulates a few lines from a 
polynomical calibration function and returns them.  
`sp-show-image`: Shows a fit or h5 image file.  
`sp-average-image`: Performs an average between different files. It can be 
used to perform a time average on flat/dark/bias frames, if they have a 
longer exposure than the one requested.  
`sp-correct-image`: Corrects an image for dark, flat and bias frames.  
`sp-find-angle`: Given a set of data, it finds the rotation angle to be 
applied to the image.  
`sp-rotate`: Rotates the image of the given angle.  
`sp-show-slices`: Shows different slices of a spectrum from a given list.  
`sp-crop-image`: Crops an image between the given bounds and saves it.  
`sp-integrate`: Integrates a given spectrum and saves the corresponding 
dataset.  
`sp-show-spectrum`: Shows the plot of an integrated spectrum, without 
calibration.  
`sp-calibrate`: Given a set of data and a model, it calibrates the spectrum 
over the model.  
`sp-show-calibrated`: Shows a calibrated spectrum.  
`sp-save-calibrated`: Saves the calibrated spectrum into a file.  
`sp-smooth`: Smooths a spectrum to retrieve the continuum.  
`sp-compare`: Compares two spectra by evaluating their ratio.  
`sp-find-line`: Finds the Voigt profile of a line within a spectrum.  
`sp-velocity-resolution`: Given the distance between two lines and their 
region, it estimates the radial velocity resolution of the spectrum.