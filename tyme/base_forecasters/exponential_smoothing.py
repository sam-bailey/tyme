import numpy as np
from typing import List, Union, Mapping, Tuple


class Constants:
    k = 3
    c_k = 4.12
    starting_params_n_samples = 10
    lam_s = 0.1


def mad(x: np.ndarray) -> float:
    med = np.median(x)
    err = np.abs(x - med)
    std_mad = 1.4826 * np.median(err)

    std = np.std(x)
    if ~np.isclose(std_mad, 0.0):
        return float(std_mad)
    elif ~np.isclose(std, 0.0):
        return float(std)
    else:
        return 1.0


def biweight(x: Union[float, int]) -> float:
    if np.abs(x) < Constants.k:
        return Constants.c_k * (1.0 - (1.0 - (x / float(Constants.k)) ** 2) ** 3)
    else:
        return Constants.c_k


def huber(x: Union[float, int]) -> float:
    if np.abs(x) < Constants.k:
        return float(x)
    else:
        return float(np.sign(x) * Constants.k)


def robust_starting_params(x: np.ndarray) -> Mapping[str, float]:
    x0 = x[:Constants.starting_params_n_samples]
    trend_0 = np.median(
        [np.median([(x0[i] - x0[j]) / float(i - j) for j in np.arange(len(x0)) if i != j])
         for i in np.arange(len(x0))]
    )
    level_0 = np.median(x0 - trend_0 * np.arange(len(x0)))
    sigma_0 = mad(x0 - level_0 - trend_0 * np.arange(len(x0)))

    return {
        "level_0": float(level_0),
        "trend_0": float(trend_0),
        "sigma_0": float(sigma_0)
    }


def exp_smoothing_filter(x: List[Union[float, int]], alpha: float, beta: float, phi: float) -> Tuple[float, float]:
    # Initialize
    starting_params = robust_starting_params(x)
    level = starting_params["level_0"]
    trend = starting_params["trend_0"]
    sigma = starting_params["sigma_0"]

    for x_i in x:
        forecast = level + phi * trend

        new_sigma = sigma * np.sqrt(
            Constants.lam_s * biweight((x_i - forecast) / float(sigma)) +
            (1.0 - Constants.lam_s)
        )
        robust_x_i = forecast + huber((x_i - forecast) / float(new_sigma)) * float(new_sigma)
        new_level = level + alpha * (robust_x_i - forecast)
        new_trend = beta * (new_level - level) + (1.0 - beta) * phi * trend

        level = new_level
        trend = new_trend
        sigma = new_sigma

    return level, trend


def exp_smoothing_forecaster(level: float, trend: float, phi: float,
                             n_steps_min: int = 1, n_steps_max: int = 1) -> List[float]:

    _forecast = [
        level + np.sum(phi ** np.arange(1, i + 1)) * trend for i in
        range(n_steps_min, n_steps_max + 1)
    ]

    return _forecast
