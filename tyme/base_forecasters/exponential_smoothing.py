import numpy as np
from typing import Mapping, List
from .exponential_smoothing_cy import exp_smoothing_filter, exp_smoothing_forecaster


class ExponentialSmoothing:
    default_starting_params = [
        0.5,  # Alpha
        0.5,  # Beta
        0.5   # Phi
    ]

    param_bounds = [
        [0.0, 1.0],  # Alpha
        [0.0, 1.0],  # Beta
        [0.0, 1.0]   # Phi
    ]

    @classmethod
    def params_to_lst(cls, alpha: float, beta: float, phi: float) -> List[float]:
        return [alpha, beta, phi]

    @classmethod
    def lst_to_params(cls, x: List[float]) -> Mapping[str, float]:
        return {"alpha": x[0], "beta": x[1], "phi": x[2]}

    @classmethod
    def create_from_lst(cls, x: List[float]):
        return cls(**cls.lst_to_params(x))

    def __init__(self, alpha: float, beta: float, phi: float):
        self._alpha = np.float(alpha)
        self._beta = np.float(beta)
        self._phi = np.float(phi)
        self._state = None

    def __repr__(self) -> str:
        strings = [
            "ExponentialSmoothingObject",
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
            strings += [
                "\t Current State:",
                f"\t\t Level={level}",
                f"\t\t Trend={trend}"
            ]

        return "\n".join(strings)

    def set_state(self, level: float, trend: float) -> None:
        self._state = dict(
            level=np.float(level),
            trend=np.float(trend)
        )

    def get_state(self) -> Mapping[str,float]:
        return self._state

    def filter(self, x: np.ndarray) -> None:
        level, trend = exp_smoothing_filter(x.astype(float), self._alpha, self._beta, self._phi)
        self.set_state(level, trend)

    def forecast(self, n_steps_min: int = 1, n_steps_max: int = 1) -> np.ndarray:
        assert self._state is not None, "You have not set the state yet. Use either set_state() or filter() first."

        return exp_smoothing_forecaster(
            level = self._state["level"],
            trend = self._state["trend"],
            phi = self._phi,
            n_steps_min = n_steps_min,
            n_steps_max = n_steps_max
        )
