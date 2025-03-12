# Setup file for the modules in src

from setuptools import setup, find_packages

setup(
    name='hydrodynamic_interactions',
    version='0.2',
    description='Hydrodynamic interactions between particles',
    author='Joan Ronquillo',

    packages=find_packages('src'),
    package_dir={'': 'src'},
)

