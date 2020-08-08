import numpy as np
from typing import List, Union, Mapping, Tuple
from .cython_exponential_smoothing import exp_smoothing_filter, exp_smoothing_forecaster


class ExponentialSmoothing:
    def __init__(self, alpha, beta, phi):
        self._alpha = np.float(alpha)
        self._beta = np.float(beta)
        self._phi = np.float(phi)
        self._state = None

    def __repr__(self):
        strings = [
            "ExponentialSmoothingObject",
            "Params:",
            f"\t Alpha = {self._alpha}",
            f"\t Beta = {self._beta}",
            f"\t Phi = {self._phi}"
        ]
        if self._state is None:
            strings += ["No State Yet"]
        else:
            level = self._state["level"]
            trend = self._state["trend"]
            sigma = self._state["sigma"]
            strings += [
                "Current State:",
                f"\t Level={level}",
                f"\t Trend={trend}",
                f"\t Sigma={sigma}"
            ]

        return "\n".join(strings)

    def set_state(self, level, trend, sigma):
        self._state = dict(
            level=np.float(level),
            trend=np.float(trend),
            sigma=np.float(sigma)
        )

    def get_state(self):
        return self._state

    def filter(self, x):
        # Initialize
        x_np = np.array(x)
        level, trend, sigma = exp_smoothing_filter(x, self._alpha, self._beta, self._phi)
        self.set_state(level, trend, sigma)

    def forecast(self, n_steps_min = 1, n_steps_max = 1):
        assert self._state is not None, "You have not set the state yet. Use either set_state() or filter() first."

        return exp_smoothing_forecaster(
            level = self._state["level"],
            trend = self._state["trend"],
            phi = self._phi,
            n_steps_min = n_steps_min,
            n_steps_max = n_steps_max
        )