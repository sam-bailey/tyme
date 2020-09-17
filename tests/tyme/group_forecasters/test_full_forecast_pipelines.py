import numpy as np
import pandas as pd
import itertools
import datetime as dt
import pytest

from tyme.utils import GroupedTimeSeries
from tyme.group_forecasters.optimized_hyperparameters import (
    RobustExponentialSmoothingGroupRegressor,
    ExponentialSmoothingGroupRegressor,
)


@pytest.fixture()
def input_df():
    group_a = list("abcdef")
    group_b = list(range(5))

    _pdf_lst = []
    for g_a, g_b in itertools.product(group_a, group_b):
        n_steps = 52
        _pdf_lst.append(
            pd.DataFrame(
                {
                    "group_a": [g_a] * n_steps,
                    "group_b": [g_b] * n_steps,
                    "yyyy_mm_dd": [
                        (
                            dt.date(2019, 1, 1) + dt.timedelta(days=d * 7)
                        ).strftime("%Y-%m-%d")
                        for d in range(n_steps)
                    ],
                    "y": np.random.rand(n_steps) + np.arange(n_steps),
                }
            )
        )
        _pdf_lst[-1].loc[int(0.5 * n_steps), "y"] *= 4.0  # Add an outlier

    input_df = pd.concat(_pdf_lst)
    return input_df


@pytest.mark.parametrize("verbose", [True, False])
@pytest.mark.parametrize("with_cov", [True, False])
@pytest.mark.parametrize(
    "group_forecaster_class",
    [
        RobustExponentialSmoothingGroupRegressor,
        ExponentialSmoothingGroupRegressor,
    ],
)
def test_full_pipeline_runs(
    input_df, group_forecaster_class, verbose, with_cov
):
    my_timeseries = GroupedTimeSeries(
        time_series_pd=input_df,
        group_columns=["group_a", "group_b"],
        time_column="yyyy_mm_dd",
        target_column="y",
    )

    my_model = group_forecaster_class(
        lookback_window=20,
        min_predict_window=2,
        max_predict_window=10,
        agg_func=np.sum,
        verbose=verbose,
    )
    my_model.fit(my_timeseries)
    my_model.predict(my_timeseries)
    my_model.forecast(my_timeseries)
    my_model.score(my_timeseries, with_cov=with_cov)
