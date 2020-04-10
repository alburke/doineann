# Hagelslag

[![Build Status](https://travis-ci.org/djgagne/hagelslag.svg?branch=master)](https://travis-ci.org/djgagne/hagelslag)

Hagelslag is an object-based severe storm forecasting system that utilizing image processing and machine learning tools
to derive calibrated probabilities of severe hazards from convection-allowing numerical weather prediction model output.
The package contains modules for storm identification and tracking, spatio-temporal data extraction, and 
machine learning model training to predict hazard intensity as well as space and time translations.

### Citation
If you employ hagelslag in your research, please acknowledge its use with the following citation:
    
    Gagne II, D. J., A. McGovern, N. Snook, R. Sobash, J. Labriola, J. K. Williams, S. E. Haupt, and M. Xue, 2016: 
    Hagelslag: Scalable object-based severe weather analysis and forecasting. Proceedings of the Sixth Symposium on 
    Advances in Modeling and Analysis Using Python, New Orleans, LA, Amer. Meteor. Soc., 447.

If you discover any issues, please post them to the Github issue tracker page. Questions and comments should be sent to
djgagne at ou dot edu.

### Requirements

Hagelslag is compatible with Python 2.7 and 3.5. Hagelslag is easiest to install with the help of the Anaconda Python Distribution, but it should work with other
Python setups as well. Hagelslag requires the following packages and recommends the following versions:

* numpy >= 1.10
* scipy >= 0.15
* matplotlib >= 1.4
* scikit-learn >= 0.16
* pandas >= 0.15
* arrow >= 0.8.0
* basemap
* netCDF4-python

### Installation

To install hagelslag, enter the top-level directory of the package and run the standard python setup command: 

    python setup.py install

Hagelslag will install the libraries in site-packages and will also install 3 applications into the `bin` directory
of your Python installation.

### Use
A Jupyter notebook is located in the demos directory that showcases the functionality of the package. For larger scale 
use, 3 scripts are provided in the bin directory. 

* `hsdata` performs object tracking and matching as well as data processing.
* `hsfore` trains and applies machine learning models.
* `hseval` performs forecast verification.

All scripts take input from a config file. The config file should be valid Python code and contain a dictionary called
config. Custom machine learning models and parameters should be contained within the config files. Examples of them can
be found in the config directory.

### Documentation
API Documentation is available [here](http://hagelslag.readthedocs.io/en/latest/).
