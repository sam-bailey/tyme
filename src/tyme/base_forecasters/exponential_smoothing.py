import numpy as np
from typing import Mapping, TypeVar
from .base import BaseForecaster
from .exponential_smoothing_cy import (
    exp_smoothing_filter,
    exp_smoothing_forecaster,
)

NumpyArray = TypeVar("numpy.ndarray")


class ExponentialSmoothing(BaseForecaster):
    _forecaster_name = "ExponentialSmoothing"

    _param_names = ["alpha", "beta", "phi"]

    _default_starting_params = [0.5, 0.5, 0.5]  # Alpha  # Beta  # Phi

    _param_bounds = [
        [0.0, 1.0],  # Alpha
        [0.0, 1.0],  # Beta
        [0.0, 1.0],  # Phi
    ]

    def set_state(self, level: float, trend: float) -> None:
        self._state = dict(level=np.float(level), trend=np.float(trend))

    def get_state(self) -> Mapping[str, float]:
        return self._state

    def filter(self, x: NumpyArray) -> None:
        level, trend = exp_smoothing_filter(x.astype(float), **self._params)
        self.set_state(level, trend)

    def forecast(
        self, n_steps_min: int = 1, n_steps_max: int = 1
    ) -> NumpyArray:
        assert (
            self._state is not None
        ), "You have not set the state yet. Use either set_state() or filter() first."

        return exp_smoothing_forecaster(
            level=self._state["level"],
            trend=self._state["trend"],
            phi=self._params["phi"],
            n_steps_min=n_steps_min,
            n_steps_max=n_steps_max,
        )
