"""ARIMA с выбором порядка по AIC и ARIMAX с новостными признаками; прогноз на шаг вперёд (задача 7.2).

Ряд — часовая лог-доходность r_s (колонка ret_lag1 строки s). Модель оценивается на
последних estimation_window_hours часах перед началом фолда: на шести годах часовых данных
оценка шестнадцати порядков по каждому фолду заняла бы много часов (раздел 12 протокола).
Порядок (p, 0, q) выбирается по AIC на том же окне. Внутри фолда параметры не меняются;
прогноз для строки t — одношаговый прогноз фильтра Калмана для r_{t+1} по наблюдениям
до r_t включительно. Пропуски биржи фильтр пропускает.

ARIMAX — регрессия с ARMA-ошибками: r_s = c + β'x_{s−1} + u_s, где x_{s−1} — новостные
признаки строки s − 1 (известны к закрытию свечи s − 1). Порядок берётся у ARIMA того же фолда.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from cryptonews.validation import Window

RETURN = "ret_lag1"


def _fit(endog: np.ndarray, exog: np.ndarray | None, order: tuple[int, int, int]):
    from statsmodels.tsa.arima.model import ARIMA

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ARIMA(endog, exog=exog, order=order, trend="c").fit()


def select_order(endog: np.ndarray, p_grid, q_grid) -> tuple[tuple[int, int, int], list[dict]]:
    """Порядок с наименьшим AIC и таблица AIC по всем порядкам."""
    table = []
    for p in p_grid:
        for q in q_grid:
            try:
                table.append({"p": int(p), "q": int(q), "aic": float(_fit(endog, None, (p, 0, q)).aic)})
            except Exception as error:  # порядок, который не оценился, просто не участвует в выборе
                table.append({"p": int(p), "q": int(q), "aic": None, "error": f"{type(error).__name__}: {error}"})
    usable = [r for r in table if r["aic"] is not None and np.isfinite(r["aic"])]
    if not usable:
        raise RuntimeError("Ни один порядок ARIMA не удалось оценить")
    best = min(usable, key=lambda r: r["aic"])
    return (best["p"], 0, best["q"]), table


def _hours(start: pd.Timestamp, end: pd.Timestamp, inclusive: str) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="h", inclusive=inclusive)


def test_rows(table: pd.DataFrame, window: Window) -> pd.Index:
    mask = table["valid"] & (table.index >= window.test_start) & (table.index < window.test_end)
    return table.index[mask]


def param_table(result) -> dict:
    names = getattr(result, "param_names", None) or result.model.param_names
    return {str(k): float(v) for k, v in zip(names, np.asarray(result.params))}


def one_step(result, endog_full: np.ndarray, exog_full: np.ndarray | None, full_index: pd.DatetimeIndex,
             rows: pd.Index) -> pd.Series:
    """Прогноз r_{t+1} для строк rows по параметрам result и наблюдениям до r_t."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        applied = result.apply(endog_full, exog=exog_full)
        predicted = np.asarray(applied.predict())
    onestep = pd.Series(predicted, index=full_index)
    return pd.Series(onestep.reindex(rows + pd.Timedelta(hours=1)).to_numpy(), index=rows, name="y_pred")


class Arima:
    name, stochastic, needs_grid = "arima", False, False

    def __init__(self, p_grid, q_grid, estimation_window_hours: int):
        self.p_grid, self.q_grid, self.window_hours = list(p_grid), list(q_grid), int(estimation_window_hours)

    def fit_predict(self, table: pd.DataFrame, columns, window: Window, params: dict, seed: int):
        start = window.test_start - pd.Timedelta(hours=self.window_hours)
        returns = table[RETURN]
        endog = returns.reindex(_hours(start, window.test_start, "left")).to_numpy(float)
        order, aic_table = select_order(endog, self.p_grid, self.q_grid)
        result = _fit(endog, None, order)
        full_index = _hours(start, window.test_end, "both")      # последний час — для прогноза последней строки
        pred = one_step(result, returns.reindex(full_index).to_numpy(float), None, full_index,
                        test_rows(table, window))
        info = {"order": list(order), "aic": aic_table, "estimation_start": str(start),
                "params": param_table(result)}
        return pred, info


class Arimax:
    name, stochastic, needs_grid = "arimax", False, False

    def __init__(self, estimation_window_hours: int):
        self.window_hours = int(estimation_window_hours)

    def fit_predict(self, table: pd.DataFrame, columns, window: Window, params: dict, seed: int):
        order = tuple(int(v) for v in params["order"])
        start = window.test_start - pd.Timedelta(hours=self.window_hours)
        returns = table[RETURN]
        exog_all = table[list(columns)].shift(1)                  # x_{s−1} для r_s
        est_index = _hours(start, window.test_start, "left")
        result = _fit(returns.reindex(est_index).to_numpy(float),
                      exog_all.reindex(est_index).fillna(0.0).to_numpy(float), order)
        full_index = _hours(start, window.test_end, "both")
        pred = one_step(result, returns.reindex(full_index).to_numpy(float),
                        exog_all.reindex(full_index).fillna(0.0).to_numpy(float), full_index,
                        test_rows(table, window))
        info = {"order": list(order), "estimation_start": str(start), "aic": float(result.aic),
                "params": param_table(result)}
        return pred, info
