#This is the setup.py, dthe scripts are inside src/hydrodynamic_int
from setuptools import setup, find_packages
import os

#install the package
setup(
    
    name='hydrodynamic_int',
    version='0.1',
    description='Hydrodynamic interaction between two particles',
    author='Joan',

    #
    packages=find_packages('src'),
    package_dir={'': 'src'},
    )