#cython: language_level=3

import numpy as np
cimport numpy as np
import bottleneck as bn
cimport cython

from libc.math cimport fabs, copysign, sqrt

DTYPE = np.float
ctypedef np.float_t DTYPE_t

@cython.boundscheck(False)
cdef DTYPE_t std_cy(np.ndarray[DTYPE_t, ndim=1] a):
    cdef Py_ssize_t i
    cdef Py_ssize_t n = a.shape[0]
    cdef DTYPE_t m = 0.0
    for i in range(n):
        m += a[i]
    m /= n
    cdef DTYPE_t v = 0.0
    for i in range(n):
        v += (a[i] - m) ** 2
    return sqrt(v / n)

# @cython.boundscheck(False)
# cdef DTYPE_t mean_cy(np.ndarray[DTYPE_t, ndim=1] a):
#     cdef Py_ssize_t i
#     cdef Py_ssize_t n = a.shape[0]
#     cdef DTYPE_t m = 0.0
#     for i in range(n):
#         m += a[i]
#     return m / n

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)
cdef (DTYPE_t, DTYPE_t) starting_params(np.ndarray[DTYPE_t, ndim=1] x0):
    cdef DTYPE_t trend_0, level_0
    cdef int i, j
    cdef int xmax = x0.shape[0]

    trend_0 = 0.0
    for i in range(xmax):
        for j in range(xmax):
            if i != j:
                trend_0 += (x0[i] - x0[j]) / (i - j)
    trend_0 /= (xmax*(xmax-1))

    level_0 = 0.0
    for i in range(xmax):
        level_0 += x0[i] - trend_0 * i
    level_0 /= xmax

    return trend_0, level_0

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)
cpdef (DTYPE_t, DTYPE_t) exp_smoothing_filter(np.ndarray[DTYPE_t, ndim=1] x, DTYPE_t alpha, DTYPE_t beta,
                                                       DTYPE_t phi):
    # Initialize
    cdef DTYPE_t trend, level, forecast
    cdef DTYPE_t new_trend, new_level, new_sigma

    cdef DTYPE_t lam_s = 0.1
    cdef int i
    cdef int xmax = x.shape[0]

    trend, level = starting_params(x[:10])

    for i in range(xmax):
        forecast = level + phi * trend

        new_level = level + alpha * (x[i] - forecast)
        new_trend = beta * (new_level - level) + (1.0 - beta) * phi * trend

        level = new_level
        trend = new_trend

    return level, trend

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim=1] exp_smoothing_forecaster(DTYPE_t level, DTYPE_t trend, DTYPE_t phi,
                                                           int n_steps_min = 1, int n_steps_max = 1):
    cdef np.ndarray[DTYPE_t, ndim=1] _forecast = np.zeros(n_steps_max - n_steps_min + 1, dtype=DTYPE)
    cdef int k, i, j

    for i in range(n_steps_max - n_steps_min + 1):
        k = n_steps_min + i
        _forecast[i] = level
        for j in range(1, k + 1):
            _forecast[i] += trend * phi ** j

    return _forecast
