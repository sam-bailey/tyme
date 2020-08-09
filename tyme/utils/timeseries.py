import pandas as pd
import numpy as np
from typing import TypeVar, List, Callable, Tuple, Mapping, Any
import functools

NumpyArray = TypeVar("numpy.ndarray")


class GroupedTimeSeries:
    def __init__(self, time_series_pd: pd.DataFrame, time_column: str, group_columns: List[str], target_column: str,
                 group_aggregation_value: str = "all"
                 ):
        self._time_column = time_column
        self._group_columns = group_columns
        self._group_aggregation_value = group_aggregation_value # When a group has this value, it is the sum over all.
        self._group_id_column = "_group_id"
        self._target_column = target_column

        self._raw_pdf = self._process_timeseries_df(time_series_pd)
        self._frequency = self._calculate_frequency()

    def _group_cols_to_id(self, **group_cols: Mapping[str,Any]) -> str:
        for col in self._group_columns:
            assert col in group_cols.keys(), f"{col} must be included in group columns. You passed {group_cols}"

        return "__".join([
            "{}={}".format(col,str(group_cols[col])) for col in self._group_columns
        ])

    def _group_id_to_cols(self, group_id: str) -> Mapping[str,Any]:
        output = dict(map(lambda x: x.split("="), group_id.split("__")))

        for col in self._group_columns:
            assert col in output.keys(), f"{col} is missing from the group_id: {group_id}"

        return output

    def _process_timeseries_df(self, time_series_pd: pd.DataFrame) -> pd.DataFrame:
        time_series_pd[self._time_column] = pd.to_datetime(time_series_pd[self._time_column])
        time_series_pd = time_series_pd.sort_values(self._time_column)
        time_series_pd[self._group_id_column] = (
            time_series_pd[self._group_columns].apply(lambda x: self._group_cols_to_id(**x), axis=1)
        )
        return time_series_pd

    def _calculate_frequency(self) -> np.timedelta64:
        possible_frequencies = (
                self._raw_pdf.groupby(self._group_columns)[self._time_column].diff(periods=1).value_counts()
        )

        _err_msg = f"Your data seems to have variable time frequencies: {possible_frequencies}"
        assert len(possible_frequencies.index.values) == 1, _err_msg

        return possible_frequencies.index.values[0]

    @functools.lru_cache()
    def regression(self, lookback_window: int, min_predict_window: int, max_predict_window: int,
                   agg_func: Callable[[NumpyArray], float]) -> Tuple[NumpyArray, NumpyArray, NumpyArray, NumpyArray]:
        """
        Create the numpy arrays needed for regression problems. Regression problems are problems where
        you care about some aggregation over the forecast.

        For example, if you are foreacsting daily sales, a regression problem would be predicting the total sales you
        will get in the next week. For this, you would use min_predict_window=1, max_predict_window=7, agg_func=np.sum.

        :param lookback_window: The number of past observations to build the training data from
        :param min_predict_window: The first day to include in the aggregation
        :param max_predict_window: The last day to include in the aggregation
        :param agg_func: How to aggregate the
        :return: X, y, time_id, group_id
        """
        X_lst = []
        y_lst = []
        time_id_lst = []
        group_id_lst = []

        for group_id, pdf_i in self._raw_pdf.groupby(self._group_id_column):
            ts_full = pdf_i[self._target_column].values
            total_length = len(ts_full)
            max_i = total_length - max_predict_window - lookback_window + 1

            X_lst += [
                ts_full[i:(i + lookback_window)]
                for i in range(max_i)
            ]
            y_lst += [
                agg_func(
                    ts_full[(i + lookback_window + min_predict_window - 1):(
                            i + lookback_window + max_predict_window)]
                )
                for i in range(max_i)
            ]
            time_id_lst += [i for i in range(max_i)]
            group_id_lst += [group_id] * max_i

        X = np.array(X_lst)
        y = np.array(y_lst)
        time_id = np.array(time_id_lst)
        group_id = np.array(group_id_lst)

        return X, y, time_id, group_id

    @functools.lru_cache()
    def forecasting(self, lookback_window: int, max_predict_window: int, format: str = "long"):
        """
        Create the numpy arrays needed for forecasting problems. Forecasting problems are problems where
        you want to know the predicted value at every time step.

        Format: If you choose long, then the target is just one timestep, and there is a new row for each timestep
        and a new column to indicate which timestep that is. If you choose wide, then the target is an array of all the
        timesteps.

        Max_predict_window: Must be >= 1. If this = 1, then the format option doesn't matter,
        as you are doing 1 step ahead forecasting.

        :param lookback_window: The number of past observations to build the training data from
        :param max_predict_window: The number of days into the future you want to forecast for
        :param format: How to structure the target variable for multi-step forecasting. 'long' or 'wide'
        :return: X, y, time_id, group_id, step_ahead_id (if format = long)
        """
        pass

    def get_s_matrix(self):
        pass