#!/usr/bin/env python3

# Set this to True to enable building extensions using Cython.
# Set it to False to build extensions from the C file (that
# was previously created using Cython).
# Set it to 'auto' to build with Cython if available, otherwise
# from the C file.
USE_CYTHON = 'auto'


import sys

from setuptools import setup
from setuptools import Extension
import numpy

import _version

if USE_CYTHON:
    try:
        from Cython.Distutils import build_ext
    except ImportError:
        if USE_CYTHON=='auto':
            USE_CYTHON=False
        else:
            raise

cmdclass = { }
ext_modules = [ ]

if sys.version_info[0] == 2:
    raise Exception('Python 2.x is no longer supported')

if USE_CYTHON:
    ext_modules += [
        Extension("tyme.base_forecasters.exponential_smoothing_cy", [ "src/cython/exponential_smoothing_cy.pyx" ]),
        Extension("tyme.base_forecasters.robust_exponential_smoothing_cy", [ "src/cython/robust_exponential_smoothing_cy.pyx" ])
    ]
    cmdclass.update({ 'build_ext': build_ext })
else:
    ext_modules += [
        Extension("tyme.base_forecasters.exponential_smoothing_cy", [ "src/cython/exponential_smoothing_cy.c" ]),
        Extension("tyme.base_forecasters.robust_exponential_smoothing_cy", [ "src/cython/robust_exponential_smoothing_cy.c" ])
    ]

setup(
    name='tyme',
    version=_version.__version__,
    description='A timeseries forecasting package, specialised in forecasting grouped timeseries',
    author='Sam Bailey',
    author_email='samcbailey90@gmail.com',
    url='http://github.com/sam-bailey/tyme',
    packages=[ 'tyme'],
    package_dir={
        '' : 'src'
    },
    cmdclass = cmdclass,
    ext_modules=ext_modules,
    include_dirs=[numpy.get_include()],

    long_description=open('README.md').read(),

    license="MIT",
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.1',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Cython',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    keywords='timeseries forecast forecasting time',
)
