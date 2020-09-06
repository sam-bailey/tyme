#cython: language_level=3

import numpy as np
cimport numpy as np
import bottleneck as bn
cimport cython

from libc.math cimport fabs, copysign, sqrt

DTYPE = np.float
ctypedef np.float_t DTYPE_t

@cython.boundscheck(False)
def std_cy(np.ndarray[DTYPE_t, ndim=1] a not None):
    cdef Py_ssize_t i
    cdef Py_ssize_t n = a.shape[0]
    cdef double m = 0.0
    for i in range(n):
        m += a[i]
    m /= n
    cdef double v = 0.0
    for i in range(n):
        v += (a[i] - m) ** 2
    return sqrt(v / n)

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)
cdef DTYPE_t mad(np.ndarray[DTYPE_t, ndim=1] x):
    cdef DTYPE_t med, std_mad, std, output
    cdef np.ndarray[DTYPE_t, ndim=1] err = np.zeros(x.shape[0], dtype=DTYPE)
    cdef DTYPE_t tol = 0.000000001
    cdef int i
    cdef int xmax = x.shape[0]

    med = bn.median(x)
    for i in range(xmax):
        err[i] = fabs(x[i] - med)

    std_mad = 1.4826 * bn.median(err)
    std = std_cy(x)

    if std_mad > tol:
        output = std_mad
    elif std > tol:
        output = std
    else:
        output = tol

    return output

cdef DTYPE_t huber(DTYPE_t x):
    cdef DTYPE_t output
    cdef DTYPE_t k = 3.0
    if fabs(x) < k:
        output = x
    else:
        output = copysign(k, x)

    return output

cdef DTYPE_t bi_weight(DTYPE_t x):
    cdef DTYPE_t output
    cdef DTYPE_t k = 3.0
    cdef DTYPE_t c_k = 4.12

    if fabs(x) < k:
        output = c_k * (1.0 - (1.0 - (x / k) ** 2) ** 3)
    else:
        output = c_k

    return output

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)
cdef (DTYPE_t, DTYPE_t, DTYPE_t) robust_starting_params(np.ndarray[DTYPE_t, ndim=1] x0):
    cdef DTYPE_t trend_0, level_0, sigma_0
    cdef np.ndarray[DTYPE_t, ndim=1] x0_diff = np.zeros(x0.shape[0] - 1, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] x0_diff_medians = np.zeros(x0.shape[0], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] levels = np.zeros(x0.shape[0], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] sigmas = np.zeros(x0.shape[0], dtype=DTYPE)
    cdef int i, j
    cdef int xmax = x0.shape[0]

    for i in range(xmax):
        for j in range(xmax):
            if i > j:
                x0_diff[j] = (x0[i] - x0[j]) / (i - j)
            elif i < j:
                x0_diff[j - 1] = (x0[i] - x0[j]) / (i - j)
        x0_diff_medians[i] = bn.median(x0_diff)

    trend_0 = bn.median(x0_diff_medians)

    for i in range(xmax):
        levels[i] = x0[i] - trend_0 * i
    level_0 = bn.median(levels)

    for i in range(xmax):
        sigmas[i] = x0[i] - level_0 - trend_0 * i
    sigma_0 = mad(sigmas)

    return trend_0, level_0, sigma_0

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)
cpdef (DTYPE_t, DTYPE_t, DTYPE_t) robust_exp_smoothing_filter(np.ndarray[DTYPE_t, ndim=1] x, DTYPE_t alpha,
                                                              DTYPE_t beta, DTYPE_t phi):
    # Initialize
    cdef DTYPE_t trend, level, sigma, robust_x_i, forecast
    cdef DTYPE_t new_trend, new_level, new_sigma

    cdef DTYPE_t lam_s = 0.1
    cdef int i
    cdef int xmax = x.shape[0]

    trend, level, sigma = robust_starting_params(x[:10])

    for i in range(xmax):
        forecast = level + phi * trend

        new_sigma = sigma * sqrt(
            lam_s * bi_weight((x[i] - forecast) / sigma) +
            (1.0 - lam_s)
        )
        robust_x_i = forecast + huber((x[i] - forecast) / new_sigma) * new_sigma
        new_level = level + alpha * (robust_x_i - forecast)
        new_trend = beta * (new_level - level) + (1.0 - beta) * phi * trend

        level = new_level
        trend = new_trend
        sigma = new_sigma

    return level, trend, sigma

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim=1] robust_exp_smoothing_forecaster(DTYPE_t level, DTYPE_t trend, DTYPE_t phi,
                                                           int n_steps_min = 1, int n_steps_max = 1):
    cdef np.ndarray[DTYPE_t, ndim=1] _forecast = np.zeros(n_steps_max - n_steps_min + 1, dtype=DTYPE)
    cdef int k, i, j

    for i in range(n_steps_max - n_steps_min + 1):
        k = n_steps_min + i
        _forecast[i] = level
        for j in range(1, k + 1):
            _forecast[i] += trend * phi ** j

    return _forecast
