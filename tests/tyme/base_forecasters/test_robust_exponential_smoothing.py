import numpy as np
import pytest

from tyme.base_forecasters.robust_exponential_smoothing import (
    RobustExponentialSmoothing,
)


@pytest.fixture()
def timeseries():
    return np.arange(20)


def test_robust_exponential_smoothing_classmethods():
    params = {"alpha": 0.5, "beta": 0.6, "phi": 0.7}

    default_starting_params = (
        RobustExponentialSmoothing.get_default_starting_params()
    )
    assert default_starting_params == [0.5, 0.5, 0.5]

    param_bounds = RobustExponentialSmoothing.get_param_bounds()
    assert param_bounds == [[0.0, 1.0]] * 3

    params_lst = RobustExponentialSmoothing.params_to_lst(**params)
    assert params_lst == [0.5, 0.6, 0.7]

    params_dict = RobustExponentialSmoothing.lst_to_params(params_lst)
    for param_name, param_value in params.items():
        assert param_value == params_dict[param_name]

    forecaster = RobustExponentialSmoothing.create_from_lst(params_lst)
    for param_name, param_value in params.items():
        assert param_value == forecaster._params[param_name]

    forecaster = RobustExponentialSmoothing.default_params()
    for param_name, param_value in forecaster._params.items():
        assert param_value == 0.5

    forecaster = RobustExponentialSmoothing(**params)
    str(forecaster)  # Tests __str__ runs, doesn't test the contents
    assert (
        repr(forecaster)
        == "RobustExponentialSmoothing(alpha=0.5, beta=0.6, phi=0.7)"
    )  # Tests __repr__

    forecaster.set_state(1.0, 2.0, 0.5)
    str(
        forecaster
    )  # Tests __str__ runs with the state defined, doesn't test the contents


@pytest.mark.parametrize(
    "alpha,beta,phi",
    [(0.5, 0.6, 0.7), (0.5, 0.6, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)],
)
def test_robust_exponential_smoothing_filter(timeseries, alpha, beta, phi):
    forecaster = RobustExponentialSmoothing(alpha=alpha, beta=beta, phi=phi)
    forecaster.filter(timeseries)
    forecaster.get_state()  # We don't actually the output...


@pytest.mark.parametrize(
    "n_steps_min,n_steps_max", [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3)]
)
@pytest.mark.parametrize("alpha,beta,phi", [(0.5, 0.6, 0.5), (0.5, 0.6, 1.0)])
@pytest.mark.parametrize(
    "level,trend,sigma", [(3.0, 2.0, 0.5), (3.0, 0.0, 10.0), (3.0, -1.0, 1.0)]
)
def test_robust_exponential_smoothing_forecast(
    alpha, beta, phi, n_steps_min, n_steps_max, level, trend, sigma
):
    forecaster = RobustExponentialSmoothing(alpha=alpha, beta=beta, phi=phi)
    forecaster.set_state(level=level, trend=trend, sigma=sigma)
    output_forecast = forecaster.forecast(n_steps_min, n_steps_max)

    steps = np.arange(n_steps_min, n_steps_max + 1)
    expected_forecast = level * np.ones_like(steps)
    for i, step in enumerate(steps):
        expected_forecast[i] += trend * np.sum(phi ** np.arange(1, step + 1))

    np.testing.assert_array_almost_equal(expected_forecast, output_forecast)
