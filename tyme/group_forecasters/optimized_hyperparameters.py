from scipy.optimize import minimize
import numpy as np
import pandas as pd
import warnings
from typing import Optional, List, TypeVar, Union, Tuple, Mapping, Callable

from .loss_functions import LossFunctions
from ..utils.timeseries import GroupedTimeSeries
from ..base_forecasters import ExponentialSmoothing
from ..base_forecasters import RobustExponentialSmoothing
from ..base_forecasters.base import BaseForecaster

NumpyArray = TypeVar("numpy.ndarray")


class GroupRegressor:
    _base_forecaster: BaseForecaster = None

    def __init__(self, lookback_window: int, min_predict_window: int, max_predict_window: int,
                 agg_func: Callable[[NumpyArray], float],
                 loss_function: LossFunctions = LossFunctions(primary_loss="mae", monitoring_losses=["smape"]),
                 verbose: bool = False
                 ):

        assert self._base_forecaster is not None, "GroupRegressor must be subclassed and the _base_forecaster set"

        self._lookback_window = lookback_window
        self._min_predict_window = min_predict_window
        self._max_predict_window = max_predict_window
        self._agg_func = agg_func
        self._loss = loss_function
        self._verbose = verbose
        self._fitted_base_forecaster = None

    def _evaluate(self, base_forecaster, X: NumpyArray, y: NumpyArray,
                  with_cov: bool = False, time_id: Optional[NumpyArray] = None,
                  group_id: Optional[NumpyArray] = None
                  ) -> Union[Tuple[float, Mapping[str, float], NumpyArray], Tuple[float, Mapping[str, float]]]:

        def _filter_forecast_aggregate(x_history: NumpyArray) -> float:
            base_forecaster.filter(x_history)
            _fcst = base_forecaster.forecast(
                n_steps_min = self._min_predict_window,
                n_steps_max = self._max_predict_window
            )
            return self._agg_func(_fcst)

        y_pred = np.array(list(map(_filter_forecast_aggregate, X)))
        loss = self._loss.primary_loss(y_pred=y_pred, y_true=y)
        monitoring_losses = self._loss.monitoring_losses(y_pred=y_pred, y_true=y)

        if with_cov:
            df1 = pd.DataFrame({
                "group_id": group_id,
                "time_id": time_id,
                "err": y_pred - y
            })
            df2 = pd.DataFrame({
                "group_id": group_id,
                "time_id": time_id,
                "err": y_pred - y
            })
            cov_df = (
                df1.merge(df2, on="time_id")
                .groupby(["group_id_x", "group_id_y"])
                .apply(lambda x: x['err_x'].cov(x['err_y']))
                .reset_index()
            )

            return loss, monitoring_losses, cov_df
        else:
            return loss, monitoring_losses

    def fit(self, timeseries: GroupedTimeSeries, optimizer: str ="L-BFGS-B",
            starting_params: Optional[List[float]] = None) -> None:

        X, y, _, _ = timeseries.regression(
            lookback_window=self._lookback_window,
            min_predict_window=self._min_predict_window,
            max_predict_window=self._max_predict_window,
            agg_func=self._agg_func
        )

        if starting_params is None:
            starting_params = self._base_forecaster.get_default_starting_params()
        param_bounds = self._base_forecaster.get_param_bounds()

        def _minimize_me(params_lst, info):
            base_forecaster = self._base_forecaster.create_from_lst(params_lst)
            loss, monitoring_losses = self._evaluate(base_forecaster, X, y)

            if (info['Nfeval'] % 10 == 0) & self._verbose:
                print('Params={0}  Loss={1:9f}   Monitoring Losses={2}'.format(
                    str(params_lst), loss, str(monitoring_losses))
                )
            info['Nfeval'] += 1

            return loss

        self._fit_results = minimize(
            fun=_minimize_me,
            x0=starting_params,
            bounds=param_bounds,
            method=optimizer,
            args=({'Nfeval': 0},)
        )
        self._fitted_base_forecaster = self._base_forecaster.create_from_lst(self._fit_results.x)

    def forecast(self, timeseries: GroupedTimeSeries) -> pd.DataFrame:
        pred_col_name = "{}_forecast".format(timeseries._target_column)

        out_pdf = timeseries._raw_pdf[[
            timeseries._time_column, timeseries._target_column, timeseries._group_id_column
        ]]

        out_pdf.loc[pred_col_name] = None
        out_lst = [out_pdf]

        for group_id, pdf_i in timeseries._raw_pdf.groupby(timeseries._group_id_column):
            time_max = pdf_i[timeseries._time_column].values[-1]

            y = pdf_i[timeseries._target_column].values[-self._lookback_window:]
            self._fitted_base_forecaster.filter(y)
            forecast_y = self._fitted_base_forecaster.forecast(1, self._max_predict_window)
            forecast_time = [time_max + timeseries._frequency*(n+1.0) for n in range(self._max_predict_window)]

            pdf_out_i = pd.DataFrame({
                timeseries._time_column: forecast_time,
                pred_col_name: forecast_y
            })
            pdf_out_i[timeseries._target_column] = None
            pdf_out_i[timeseries._group_id_column] = group_id

            out_lst.append(pdf_out_i)

        return pd.concat(out_lst, sort=True)

    def predict(self, timeseries: GroupedTimeSeries) -> pd.DataFrame:
        pred_col_name = "{}_pred".format(timeseries._target_column)

        def _filter_forecast_aggregate(x_history):
            self._fitted_base_forecaster.filter(x_history.values)
            _fcst = self._fitted_base_forecaster.forecast(
                n_steps_min = self._min_predict_window,
                n_steps_max = self._max_predict_window
            )
            return self._agg_func(_fcst)

        output_pdf = (
            timeseries._raw_pdf
            .sort_values(timeseries._time_column)
            .groupby(timeseries._group_id_column)
            .agg({
                timeseries._target_column: _filter_forecast_aggregate
            })
            .reset_index()
            .rename({timeseries._target_column: pred_col_name}, axis='columns')
        )

        return output_pdf

    def score(self, timeseries: GroupedTimeSeries, with_cov: bool = False):
        X, y, time_id, group_id = timeseries.regression(
            lookback_window=self._lookback_window,
            min_predict_window=self._min_predict_window,
            max_predict_window=self._max_predict_window,
            agg_func=self._agg_func
        )
        return self._evaluate(self._fitted_base_forecaster, X, y,
                              with_cov=with_cov, time_id=time_id, group_id=group_id)


class RobustExponentialSmoothingGroupRegressor(GroupRegressor):
    _base_forecaster = RobustExponentialSmoothing


class ExponentialSmoothingGroupRegressor(GroupRegressor):
    _base_forecaster = ExponentialSmoothing
