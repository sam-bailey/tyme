import numpy as np
import pandas as pd
import pytest

from tyme.utils import reconcile_predictions


_n_bottom_level_series = 5
_n_series_total = 8


@pytest.fixture()
def predictions():
    return np.array([55.0, 32.0, 15.0, 32.0, 10.0, 90.0, 50.0, 150.0])


@pytest.fixture()
def error_cov_matrix():
    return (
        np.array(
            [
                [30.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                [0.5, 20.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                [0.5, 0.5, 8.0, 0.5, 0.5, 0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5, 10.0, 0.5, 0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5, 0.5, 2.0, 0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5, 0.5, 0.5, 70.0, 0.5, 0.5],
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 60.0, 0.5],
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 80.0],
            ]
        )
        ** 2
    )


@pytest.fixture()
def s_matrix():
    s = np.zeros((_n_series_total, _n_bottom_level_series), dtype=int)
    s[0, 0] = 1
    s[1, 1] = 1
    s[2, 2] = 1
    s[3, 3] = 1
    s[4, 4] = 1
    s[5, 0] = 1
    s[5, 1] = 1
    s[6, 2] = 1
    s[6, 3] = 1
    s[6, 4] = 1
    s[7, :] = 1
    return s


@pytest.mark.parametrize(
    "n_bottom_level_series", [None, _n_bottom_level_series]
)
@pytest.mark.parametrize(
    "method", ["ols", "wls", "nseries", "full", "__unknown__"]
)
def test_reconcile_predictions(
    predictions, error_cov_matrix, s_matrix, n_bottom_level_series, method
):
    if method == "__unknown__":
        with pytest.raises(Exception):
            reconcile_predictions(
                predictions=predictions,
                error_cov_matrix=error_cov_matrix,
                s=s_matrix,
                n_bottom_level_series=n_bottom_level_series,
                method=method,
            )
    elif (method == "nseries") & (n_bottom_level_series is None):
        with pytest.raises(AssertionError):
            reconcile_predictions(
                predictions=predictions,
                error_cov_matrix=error_cov_matrix,
                s=s_matrix,
                n_bottom_level_series=n_bottom_level_series,
                method=method,
            )
    else:
        reconciled_predictions, _ = reconcile_predictions(
            predictions=predictions,
            error_cov_matrix=error_cov_matrix,
            s=s_matrix,
            n_bottom_level_series=n_bottom_level_series,
            method=method,
        )

        np.testing.assert_approx_equal(
            np.sum(reconciled_predictions[:2]), reconciled_predictions[5]
        )
        np.testing.assert_approx_equal(
            np.sum(reconciled_predictions[2:5]), reconciled_predictions[6]
        )
        np.testing.assert_approx_equal(
            np.sum(reconciled_predictions[:5]), reconciled_predictions[7]
        )
