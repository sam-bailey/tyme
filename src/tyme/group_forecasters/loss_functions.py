import numpy as np
from typing import TypeVar, List, Optional, Callable, Mapping

NumpyArray = TypeVar("numpy.ndarray")


def _mae(y_pred: NumpyArray, y_true: NumpyArray) -> float:
    return np.mean(np.abs(y_pred - y_true))


def _rmse(y_pred: NumpyArray, y_true: NumpyArray) -> float:
    return np.sqrt(np.mean((y_pred - y_true) ** 2))


def _smape(y_pred: NumpyArray, y_true: NumpyArray) -> float:
    err = (y_pred - y_true) / (0.5 * (np.abs(y_pred) + np.abs(y_true)))
    err[y_pred == y_true] = 0
    return np.mean(np.abs(err))


class LossFunctions:
    _loss_functions = {"mae": _mae, "rmse": _rmse, "smape": _smape}

    def __init__(
        self, primary_loss: str, monitoring_losses: Optional[List[str]] = None
    ):
        self._primary_loss = primary_loss
        if monitoring_losses is not None:
            self._monitoring_losses = monitoring_losses
        else:
            self._monitoring_losses = []

    def add_primary_loss_function(
        self, name: str, function: Callable[[NumpyArray, NumpyArray], float]
    ):
        self._loss_functions[name] = function
        self._primary_loss = name

    def add_monitoring_loss_function(
        self, name: str, function: Callable[[NumpyArray, NumpyArray], float]
    ):
        self._loss_functions[name] = function
        self._monitoring_losses.append(name)

    def set_loss_functions(
        self, primary_loss: str, monitoring_losses: Optional[List[str]] = None
    ):
        self._primary_loss = primary_loss
        if monitoring_losses is not None:
            self._monitoring_losses = monitoring_losses
        else:
            self._monitoring_losses = []

    def primary_loss(self, y_pred: NumpyArray, y_true: NumpyArray) -> float:
        return self._loss_functions[self._primary_loss](y_pred, y_true)

    def monitoring_losses(
        self, y_pred: NumpyArray, y_true: NumpyArray
    ) -> Mapping[str, float]:
        return {
            name: self._loss_functions[name](y_pred, y_true)
            for name in self._monitoring_losses
        }
