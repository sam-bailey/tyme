import numpy as np


def mad(x):
    med = np.median(x)
    err = np.abs(x - med)
    mad = 1.4826 * np.median(err)
    std = np.std(x)
    if ~np.isclose(mad, 0):
        return mad
    elif ~np.isclose(std, 0):
        return std
    else:
        return 1.0


def biweight(x):
    k = 3
    c_k = 4.12
    if np.abs(x) < k:
        return c_k * (1.0 - (1.0 - (x / float(k)) ** 2) ** 3)
    else:
        return c_k


def huber(x):
    k = 3
    if np.abs(x) < k:
        return x
    else:
        return np.sign(x) * k


def _params_from_lst(model_type, x):
    if model_type == "level":
        return dict(alpha=x[0], beta=0, phi=0)
    elif model_type == "trend":
        return dict(alpha=x[0], beta=x[1], phi=1.0)
    elif model_type == "damped-trend":
        return dict(alpha=x[0], beta=x[1], phi=x[2])
    else:
        raise Exception("Unknown model type")

def _series_to_supervised(self, pdf, downsample_frac=1.0):
    X_lst = []
    y_lst = []
    t_lst = []
    group_lst = []

    pdf_sorted = pdf.sort_values(self._time_column)
    for group_ids, pdf_i in pdf_sorted.groupby(self._group_column):
        ts_full = pdf_i[self._target_column].values
        total_length = len(ts_full)
        max_i = total_length - self._max_predict_window - self._lookback_window + 1

        X_lst += [
            ts_full[i:(i + self._lookback_window)]
            for i in range(max_i)
        ]
        y_lst += [
            self._predict_agg(
                ts_full[(i + self._lookback_window + self._min_predict_window - 1):(
                        i + self._lookback_window + self._max_predict_window)]
            )
            for i in range(max_i)
        ]
        t_lst += [i for i in range(max_i)]
        group_lst += [group_ids] * max_i

    X = np.array(X_lst)
    y = np.array(y_lst)
    t = np.array(t_lst)
    group = np.array(group_lst)

    if downsample_frac < 1.0:
        r = np.random.rand(len(y)) < downsample_frac
        X = X[r, :]
        y = y[r]
        t = t[r]
        group = group[r]

    return X, y, t, group


def _get_starting_params(model_params, x):
    x0 = x[:10]
    if model_params["phi"] == 0:  # level only model
        trend_0 = 0
        level_0 = np.median(x0)
        sigma_0 = mad(x0 - level_0)
    else:
        trend_0 = np.median(
            [np.median([(x0[i] - x0[j]) / float(i - j) for j in np.arange(len(x0)) if i != j]) for i in
             np.arange(len(x0))])
        level_0 = np.median([x0[i] - trend_0 * float(i) for i in np.arange(len(x0))])
        sigma_0 = mad([x0[i] - level_0 - trend_0 * float(i) for i in np.arange(len(x0))])

    return {
        "level_0": level_0,
        "trend_0": trend_0,
        "sigma_0": sigma_0
    }


def _get_exp_smoothing_filter(self, model_params):
    def exp_smoothing_filter(x):
        # Initialize
        starting_params = self._get_starting_params(model_params, x)
        level = starting_params["level_0"]
        trend = starting_params["trend_0"]
        sigma = starting_params["sigma_0"]
        lam_s = 0.1

        for x_i in x:
            forecast = level + model_params["phi"] * trend

            new_sigma = sigma * np.sqrt(
                lam_s * biweight((x_i - forecast) / float(sigma)) +
                (1.0 - lam_s)
            )
            robust_x_i = forecast + huber((x_i - forecast) / float(new_sigma)) * float(new_sigma)
            new_level = level + model_params["alpha"] * (robust_x_i - forecast)
            new_trend = model_params["beta"] * (new_level - level) + (1.0 - model_params["beta"]) * model_params[
                "phi"] * trend

            level = new_level
            trend = new_trend
            sigma = new_sigma

        return level, trend

    return exp_smoothing_filter


def _get_forecaster(model_params, n_steps_min, n_steps_max):
    def forecaster(level, trend):
        return np.array([level + np.sum(model_params["phi"] ** (1.0 + np.arange(i))) * trend for i in
                         range(n_steps_min, n_steps_max)])

    return forecaster