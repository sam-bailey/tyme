import numpy as np
from typing import Mapping
from .robust_exponential_smoothing_cy import robust_exp_smoothing_filter, robust_exp_smoothing_forecaster


class RobustExponentialSmoothing:
    def __init__(self, alpha: float, beta: float, phi: float) -> None:
        self._alpha = np.float(alpha)
        self._beta = np.float(beta)
        self._phi = np.float(phi)
        self._state = None

    def __repr__(self) -> str:
        strings = [
            "RobustExponentialSmoothingObject",
            "\t Params:",
            f"\t\t Alpha = {self._alpha}",
            f"\t\t Beta = {self._beta}",
            f"\t\t Phi = {self._phi}"
        ]
        if self._state is None:
            strings += ["\t No State Yet"]
        else:
            level = self._state["level"]
            trend = self._state["trend"]
            sigma = self._state["sigma"]
            strings += [
                "\t Current State:",
                f"\t\t Level={level}",
                f"\t\t Trend={trend}",
                f"\t\t Sigma={sigma}"
            ]

        return "\n".join(strings)

    def set_state(self, level: float, trend: float, sigma: float) -> None:
        self._state = dict(
            level=np.float(level),
            trend=np.float(trend),
            sigma=np.float(sigma)
        )

    def get_state(self) -> Mapping[str,float]:
        return self._state

    def filter(self, x: np.ndarray) -> None:
        x_np = np.array(x).astype(float)
        level, trend, sigma = robust_exp_smoothing_filter(x, self._alpha, self._beta, self._phi)
        self.set_state(level, trend, sigma)

    def forecast(self, n_steps_min: int = 1, n_steps_max: int = 1) -> np.ndarray:
        assert self._state is not None, "You have not set the state yet. Use either set_state() or filter() first."

        return robust_exp_smoothing_forecaster(
            level = self._state["level"],
            trend = self._state["trend"],
            phi = self._phi,
            n_steps_min = n_steps_min,
            n_steps_max = n_steps_max
        )
