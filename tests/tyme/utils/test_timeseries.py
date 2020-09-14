import numpy as np
import pandas as pd
import pytest

from tyme.utils import GroupedTimeSeries


@pytest.fixture()
def simple_a_b_input_df():
    return pd.DataFrame(
        {
            "group_a": ["a"] * 10 + ["b"] * 10,
            "group_b": [1, 2] * 10,
            "yyyy_mm_dd": [f"2020-01-{d:02}" for d in range(1, 21)],
            "y": list(range(20)),
        }
    )


@pytest.fixture()
def simple_a_b_regression_output():
    X = np.array(
        [
            [0, 2],
            [2, 4],
            [1, 3],
            [3, 5],
            [10, 12],
            [12, 14],
            [11, 13],
            [13, 15],
        ]
    )
    y = np.array([10, 14, 12, 16, 30, 34, 32, 36])
    group_idx = np.array(
        [
            "group_a=a__group_b=1",
            "group_a=a__group_b=1",
            "group_a=a__group_b=2",
            "group_a=a__group_b=2",
            "group_a=b__group_b=1",
            "group_a=b__group_b=1",
            "group_a=b__group_b=2",
            "group_a=b__group_b=2",
        ]
    )
    time_idx = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    return X, y, time_idx, group_idx


@pytest.mark.parametrize("output_idx", [0, 1, 2, 3])
def test_timeseries_regression(
    simple_a_b_input_df, simple_a_b_regression_output, output_idx
):

    my_timeseries = GroupedTimeSeries(
        time_series_pd=simple_a_b_input_df,
        group_columns=["group_a", "group_b"],
        time_column="yyyy_mm_dd",
        target_column="y",
    )

    output = my_timeseries.regression(
        lookback_window=2,
        min_predict_window=1,
        max_predict_window=2,
        agg_func=np.sum,
    )

    np.testing.assert_array_equal(
        output[output_idx], simple_a_b_regression_output[output_idx]
    )
