import numpy as np
import pytest

from tyme.group_forecasters.loss_functions import LossFunctions


_expected_losses = {
    "mae": np.mean([0.5, 0.5, 3.5, 9.0, 4.0, 0.0]),
    "rmse": np.sqrt(
        np.mean([0.5 ** 2, 0.5 ** 2, 3.5 ** 2, 9.0 ** 2, 4.0 ** 2, 0.0 ** 2])
    ),
    "smape": np.mean(
        [
            0.5 / (0.5 * 2.5),
            0.5 / (0.5 * 3.5),
            3.5 / (0.5 * 3.5),
            9.0 / (0.5 * 9.0),
            4.0 / (0.5 * 4.0),
            0.0 / (0.5 * 4.0),
        ]
    ),
}


@pytest.fixture()
def y_true():
    return np.array([1.0, 2.0, 0.0, 4.0, 4.0, 2.0])


@pytest.fixture()
def y_pred():
    return np.array([1.5, 1.5, 3.5, -5.0, 0.0, 2.0])


@pytest.mark.parametrize("monitoring", [["rmse"], [], None])
def test_loss_functions_set_loss_functions(monitoring):
    loss = LossFunctions(primary_loss="rmse")
    loss.set_loss_functions("mae", monitoring)

    assert loss._primary_loss == "mae"
    if monitoring is not None:
        assert loss._monitoring_losses == monitoring
    else:
        assert loss._monitoring_losses == []


def test_loss_functions_add_primary_loss(y_true, y_pred):
    loss = LossFunctions(primary_loss="rmse")
    loss.add_primary_loss_function(
        "my_mae", lambda x, y: np.mean(np.abs(x - y))
    )

    assert loss._primary_loss == "my_mae"
    assert (
        loss.primary_loss(y_pred=y_pred, y_true=y_true)
        == _expected_losses["mae"]
    )


def test_loss_functions_add_monitoring_loss(y_true, y_pred):
    loss = LossFunctions(primary_loss="rmse")
    loss.add_monitoring_loss_function(
        "my_mae", lambda x, y: np.mean(np.abs(x - y))
    )

    assert loss._monitoring_losses == ["my_mae"]
    assert (
        loss.monitoring_losses(y_pred=y_pred, y_true=y_true)["my_mae"]
        == _expected_losses["mae"]
    )


@pytest.mark.parametrize(
    "primary,monitoring",
    [
        ("mae", ["rmse", "smape"]),
        ("rmse", ["mae", "smape"]),
        ("smape", ["rmse", "mae"]),
        ("smape", []),
    ],
)
def test_loss_functions_built_in(y_true, y_pred, primary, monitoring):
    loss = LossFunctions(primary_loss=primary, monitoring_losses=monitoring)

    primary_loss = loss.primary_loss(y_pred=y_pred, y_true=y_true)
    monitoring_losses = loss.monitoring_losses(y_pred=y_pred, y_true=y_true)

    assert primary_loss == _expected_losses[primary]
    for k in monitoring:
        assert monitoring_losses[k] == _expected_losses[k]
