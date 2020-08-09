from scipy.optimize import minimize
import datetime as dt
import numpy as np
import pandas as pd
import warnings


def np_to_datetime(dt64):
    ts = (dt64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    return dt.datetime.utcfromtimestamp(ts)


class RobustExponentialSmoothing:
    def __init__(self, lookback_window, min_predict_window, max_predict_window,
                 predict_agg, time_column, group_column, target_column, model_type="trend",
                 loss="mae"
                 ):
        self._lookback_window = lookback_window
        self._min_predict_window = min_predict_window
        self._max_predict_window = max_predict_window
        self._predict_agg = predict_agg
        self._time_column = time_column
        self._group_column = group_column
        self._target_column = target_column
        self._model_type = model_type
        self._loss = loss

        self._starting_model_params = {
            "level": [0.5],
            "trend": [0.5, 0.05],
            "damped-trend": [0.5, 0.05, 1.0]
        }

        self._bounds_model_params = {
            "level": [[0.0, 1.0]],
            "trend": [[0.0, 1.0], [0.0, 1.0]],
            "damped-trend": [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
        }

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def _get_forecaster(model_params, n_steps_min, n_steps_max):
        def forecaster(level, trend):
            return np.array([level + np.sum(model_params["phi"] ** (1.0 + np.arange(i))) * trend for i in
                             range(n_steps_min, n_steps_max)])

        return forecaster

    @staticmethod
    def _aic(loss, n_params):
        return loss

    def _loss_function(self, y_pred, y_true):
        if self._loss == "mae":
            return np.mean(np.abs(y_pred - y_true))
        elif self._loss == "rmse":
            return np.sqrt(np.mean((y_pred - y_true) ** 2))
        elif self._loss == "smape":
            err = (y_pred - y_true) / (0.5 * (y_pred + y_true))
            err[y_pred == y_true] = 0
            return np.mean(np.abs(err))

    def _evaluate(self, params, X, y, with_cov=False, t=None, group_lst=None):
        level_and_trend = map(self._get_exp_smoothing_filter(params), X)

        forecaster = lambda x: self._get_forecaster(params, self._min_predict_window, self._max_predict_window)(*x)
        forecasts = map(forecaster, level_and_trend)

        aggregates = np.array(list(map(self._predict_agg, forecasts)))

        if with_cov:
            df1 = pd.DataFrame({
                "group": group_lst,
                "t": t,
                "err": aggregates - y
            })
            df2 = pd.DataFrame({
                "group": group_lst,
                "t": t,
                "err": aggregates - y
            })
            cov_df = (
                df1.merge(df2, on="t")
                    .groupby(["group_x", "group_y"])
                    .apply(lambda x: x['err_x'].cov(x['err_y']))
                    .reset_index()
            )

            return self._loss_function(aggregates, y), cov_df
        else:
            return self._loss_function(aggregates, y)

    def fit(self, pdf, downsample_frac=1.0, optimizer="L-BFGS-B"):
        X, y, _, _ = self._series_to_supervised(pdf, downsample_frac)

        if self._model_type == "best-aic":
            warnings.warn(
                "Best-aic not implemented yet, instead it just chooses the model with the smallest loss. Might as well just use damped-trend")
            best_aic = np.inf
            for model_type in self._starting_model_params.keys():
                print("")
                print("Trying " + model_type)
                n_params = len(self._starting_model_params[model_type])

                def _minimize_me(x, info):

                    params = self._params_from_lst(model_type, x)
                    loss = self._evaluate(params, X, y)
                    if info['Nfeval'] % 10 == 0:
                        print('Params={0}  Loss={1:9f}'.format(str(params), loss))
                    info['Nfeval'] += 1
                    return loss

                _Nfeval = 0
                _fit_results = minimize(
                    fun=_minimize_me,
                    x0=self._starting_model_params[model_type],
                    bounds=self._bounds_model_params[model_type],
                    method=optimizer,
                    args=({'Nfeval': 0},)
                )

                aic = self._aic(_fit_results.fun, n_params)
                print("Loss = {}".format(aic))
                if aic < best_aic:
                    print("Best so far!")
                    best_aic = aic
                    self._fit_results = _fit_results
                    self.set_parameters(**self._params_from_lst(model_type, self._fit_results.x))

        else:

            def _minimize_me(x, info):

                params = self._params_from_lst(self._model_type, x)
                loss = self._evaluate(params, X, y)
                if info['Nfeval'] % 10 == 0:
                    print('Params={0}  Loss={1:9f}'.format(str(params), loss))
                info['Nfeval'] += 1
                return loss

            self._fit_results = minimize(
                fun=_minimize_me,
                x0=self._starting_model_params[self._model_type],
                bounds=self._bounds_model_params[self._model_type],
                method=optimizer,
                args=({'Nfeval': 0},)
            )
            self.set_parameters(**self._params_from_lst(self._model_type, self._fit_results.x))

    def set_parameters(self, alpha, beta, phi):
        self._model_params = {
            "alpha": alpha,
            "beta": beta,
            "phi": phi
        }

    def predict(self, pdf):
        pred_col_name = "{}_pred".format(self._target_column)

        def _predict_ts(y):
            try:
                level, trend = self._get_exp_smoothing_filter(self._model_params)(y.values)
                forecast = self._get_forecaster(self._model_params, self._min_predict_window, self._max_predict_window)(
                    level, trend)
                return self._predict_agg(forecast)
            except:
                warnings.warn("Failed for one prediction")
                return None

        output_pdf = (
            pdf
                .sort_values(self._time_column)
                .groupby(self._group_column)
                .agg({
                self._target_column: _predict_ts
            })
                .reset_index()
                .rename({self._target_column: pred_col_name}, axis='columns')
        )

        return output_pdf

    def forecast(self, pdf):
        pred_col_name = "{}_forecast".format(self._target_column)

        out_pdf = pdf[[self._time_column, self._target_column, self._group_column]].copy()

        out_pdf[self._time_column] = out_pdf[self._time_column].apply(lambda x: np_to_datetime(x).date())
        out_pdf[pred_col_name] = None
        out_lst = [out_pdf]

        pdf_sorted = pdf.sort_values(self._time_column)
        for group_ids, pdf_i in pdf_sorted.groupby(self._group_column):
            x_max = np_to_datetime(pdf_i[self._time_column].values[-1]).date()
            x_delta = (x_max - np_to_datetime(pdf_i[self._time_column].values[-2]).date()).days

            y = pdf_i[self._target_column].values[-self._lookback_window:]
            level, trend = self._get_exp_smoothing_filter(self._model_params)(y)
            forecast = self._get_forecaster(self._model_params, 0, self._max_predict_window)(level, trend)
            forecast_x = [x_max + dt.timedelta(days=int((1 + d) * x_delta)) for d in range(self._max_predict_window)]

            pdf_out_i = pd.DataFrame({
                self._time_column: forecast_x,
                pred_col_name: forecast
            })
            pdf_out_i[self._target_column] = None

            pdf_out_i[self._group_column] = group_ids

            out_lst.append(pdf_out_i)

        return pd.concat(out_lst, sort=True)

    def score(self, pdf, with_cov=False, downsample_frac=1.0):
        X, y, t, group_lst = self._series_to_supervised(pdf, downsample_frac)
        return self._evaluate(self._model_params, X, y, with_cov=with_cov, t=t, group_lst=group_lst)
