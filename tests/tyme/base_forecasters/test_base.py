import pytest

from tyme.base_forecasters.base import BaseForecaster


def test_subclass_nothing_defined():
    class MyForecaster(BaseForecaster):
        pass

    with pytest.raises(Exception):
        MyForecaster()


def test_subclass_no_filter():
    class MyForecaster(BaseForecaster):
        _forecaster_name = "MyForecaster"
        _param_names = ["alpha"]
        _default_starting_params = [0.5]  # Alpha
        _param_bounds = [
            [0.0, 1.0],  # Alpha
        ]

    base_forecaster = MyForecaster(alpha=0.5)
    with pytest.raises(Exception):
        base_forecaster.filter([0.1, 0.2, 0.3])


def test_subclass_no_forecast():
    class MyForecaster(BaseForecaster):
        _forecaster_name = "MyForecaster"
        _param_names = ["alpha"]
        _default_starting_params = [0.5]  # Alpha
        _param_bounds = [
            [0.0, 1.0],  # Alpha
        ]

    base_forecaster = MyForecaster(alpha=0.5)
    with pytest.raises(Exception):
        base_forecaster.forecast(1, 1)
