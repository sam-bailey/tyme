from typing import Mapping, List, TypeVar

NumpyArray = TypeVar("numpy.ndarray")


class BaseForecaster:
    """
    This is the base class for a simple forecaster. It has to have a filter method (basically fit the model
    to the timeseries), and a forecast method (forecast the fitted model for a range of days into the future).

    It must also have the following class parameters:
        _param_names
        _default_starting_params
        _param_bounds
        _forecaster_name
    """
    _param_names: List[str] = None
    _default_starting_params: List[float] = None
    _param_bounds: List[List[float]] = None
    _forecaster_name = None

    @classmethod
    def get_default_starting_params(self):
        return self._default_starting_params

    @classmethod
    def get_param_bounds(self):
        return self._param_bounds

    @classmethod
    def params_to_lst(cls, **params: Mapping[str, float]) -> List[float]:
        return [params[_n] for _n in cls._param_names]

    @classmethod
    def lst_to_params(cls, x: List[float]) -> Mapping[str, float]:
        assert len(x) == len(cls._param_names)
        return {cls._param_names[i]: x[i] for i in range(len(cls._param_names))}

    @classmethod
    def create_from_lst(cls, x: List[float]):
        return cls(**cls.lst_to_params(x))

    @classmethod
    def default_params(cls, x: List[float]):
        return cls(**cls.lst_to_params(cls._default_starting_params))

    def __init__(self, **params: Mapping[str, float]):
        assert self._param_names is not None, "Must define _param_names"
        assert self._default_starting_params is not None, "Must define _default_starting_params"
        assert self._param_bounds is not None, "Must define _param_bounds"
        assert self._forecaster_name is not None, "Must define _forecaster_name"

        self._params = {_n: params[_n] for _n in self._param_names}
        self._state = None

    def __repr__(self) -> str:
        strings = [
            self._forecaster_name,
            "\t Params:"]

        for name, value in self._params.items():
            strings.append(f"\t\t {name} = {value}")

        if self._state is None:
            strings.append("\t No State Yet")
        else:
            strings.append("\t Current State:")
            for name, value in self._state.items():
                strings.append(f"\t\t {name} = {value}")

        return "\n".join(strings)

    def filter(self, x: NumpyArray) -> None:
        raise(Exception("No filter method set"))

    def forecast(self, n_steps_min: int = 1, n_steps_max: int = 1) -> NumpyArray:
        raise(Exception("No forecast method set"))
