#!/usr/bin/env python3

# Set this to True to enable building extensions using Cython.
# Set it to False to build extensions from the C file (that
# was previously created using Cython).
# Set it to 'auto' to build with Cython if available, otherwise
# from the C file.
import sys

from setuptools import setup, find_packages, Extension
from distutils.command.sdist import sdist as _sdist

import numpy

USE_CYTHON = "auto"

if USE_CYTHON:
    try:
        from Cython.Distutils import build_ext
        from Cython.Build import cythonize
    except ImportError:
        if USE_CYTHON == "auto":
            USE_CYTHON = False
        else:
            raise


class CythonModule(object):
    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    @property
    def path(self) -> str:
        return self._path

    @path.setter
    def path(self, path: str) -> None:
        self._path = path

    @property
    def pyx(self) -> str:
        return self.path + ".pyx"

    @property
    def c(self) -> str:
        return self.path + ".c"


cython_modules = [
    CythonModule(
        name="tyme.base_forecasters.exponential_smoothing_cy",
        path="src/cython/exponential_smoothing_cy",
    ),
    CythonModule(
        name="tyme.base_forecasters.robust_exponential_smoothing_cy",
        path="src/cython/robust_exponential_smoothing_cy",
    ),
]

if sys.version_info[0] == 2:
    raise Exception("Python 2.x is no longer supported")

if USE_CYTHON:

    class sdist(_sdist):
        def run(self):
            # Make sure the compiled Cython files in the distribution are up-to-date
            cythonize([module.pyx for module in cython_modules])
            _sdist.run(self)

    ext_modules = [
        Extension(module.name, [module.pyx]) for module in cython_modules
    ]
    cmdclass = dict(build_ext=build_ext, sdist=sdist)
else:
    ext_modules = [
        Extension(module.name, [module.c]) for module in cython_modules
    ]
    cmdclass = {}

requirements = [
    "Bottleneck",
    "cycler",
    "kiwisolver",
    "numpy",
    "pandas",
    "Pillow",
    "pyparsing",
    "python-dateutil",
    "pytz",
    "six",
    "scipy",
    "Cython",
]

requirements_dev = ["pytest", "Cython", "pre-commit", "tox"]

setup(
    name="tyme",
    version="0.1.0",
    description="A timeseries forecasting package, specialised in forecasting grouped timeseries",
    author="Sam Bailey",
    author_email="samcbailey90@gmail.com",
    url="http://github.com/sam-bailey/tyme",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    include_dirs=[numpy.get_include()],
    long_description=open("README.md").read(),
    install_requires=requirements,
    extras_require={"dev": requirements_dev},
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.1",
        "Programming Language :: Python :: 3.2",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Cython",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    keywords="timeseries forecast forecasting time",
)
