import numpy as np
import pandas as pd
from tyme.utils import GroupedTimeSeries


def test_timeseries_regression():
    input_df = pd.DataFrame({
        "group_a": ["a"]*15 + ["b"]*15,
        "group_b": [1,2]*15,
        "yyyy_mm_dd": [f"2020-01-{d:02}" for d in range(1, 31)],
        "y": list(range(30))
    })

    my_timeseries = GroupedTimeSeries(
        time_series_pd=input_df,
        group_columns=["group_a", "group_b"],
        time_column="yyyy_mm_dd",
        target_column="y"
    )

    out = my_timeseries.regression(
        lookback_window=3,
        min_predict_window=1,
        max_predict_window=2,
        agg_func=np.sum
    )

    # print(input_df)

    # for x_i, y_i, time_id_i, group_id_i in zip(*out):
    #     print("")
    #     print(f"group_id = {group_id_i}")
    #     print(f"time_id = {time_id_i}")
    #     print(f"x_i = {x_i}")
    #     print(f"y_i = {y_i}")
