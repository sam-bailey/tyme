import numpy as np
from numpy.linalg import multi_dot, inv
from typing import Optional, Tuple, TypeVar

NumpyArray = TypeVar("numpy.ndarray")


def reconcile_predictions(predictions: NumpyArray, error_cov_matrix: NumpyArray, s: NumpyArray,
                          n_bottom_level_series: Optional[int] = None,
                          method: str = "wls") -> Tuple[NumpyArray, NumpyArray]:
    """
    Method from here: https://otexts.com/fpp2/reconciliation.html
    """
    n_predictions = predictions.shape[0]
    if method == "wls":
        w = error_cov_matrix * np.identity(n_predictions)  # Only take diagonal elements, for stability
    elif method == "nseries":
        w = np.dot(s, np.ones((n_bottom_level_series, 1))) * np.identity(n_predictions)
    elif method == "ols":
        w = np.identity(n_predictions)
    elif method == "full":
        w = error_cov_matrix  # Can be unstable...
    else:
        raise Exception(ValueError("Incorrect method, but be either: wls, nseries, ols, full. You entered "+method))

    w_inv = np.linalg.inv(w)
    reconciliation_matrix = multi_dot([
        s,
        inv(multi_dot(
            [s.T, w_inv, s]
        )),
        s.T,
        w_inv
    ])
    reconciled_predictions = np.dot(reconciliation_matrix, predictions)

    return reconciled_predictions, reconciliation_matrix
