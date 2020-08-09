from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize([
        "tyme/base_forecasters/exponential_smoothing_cy.pyx",
        "tyme/base_forecasters/robust_exponential_smoothing_cy.pyx"
    ], language_level=3, annotate=True),
    include_dirs=[numpy.get_include()]
)
