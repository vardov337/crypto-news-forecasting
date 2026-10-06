"""Нулевой прогноз и историческое среднее по расширяющемуся окну (задача 7.2).

Обе модели не обучаются. Историческое среднее для строки t — среднее всех часовых
доходностей, известных к закрытию свечи t, начиная с начала выборки (Campbell, Thompson,
2008): относительно него считается R²_OOS.
"""
from __future__ import annotations

import pandas as pd

from cryptonews.validation import Window


def test_rows(table: pd.DataFrame, window: Window) -> pd.Index:
    mask = table["valid"] & (table.index >= window.test_start) & (table.index < window.test_end)
    return table.index[mask]


class NaiveZero:
    name, stochastic, needs_grid = "naive_zero", False, False

    def fit_predict(self, table: pd.DataFrame, columns, window: Window, params: dict, seed: int):
        rows = test_rows(table, window)
        return pd.Series(0.0, index=rows, name="y_pred"), {}


class NaiveMean:
    name, stochastic, needs_grid = "naive_mean", False, False

    def fit_predict(self, table: pd.DataFrame, columns, window: Window, params: dict, seed: int):
        rows = test_rows(table, window)
        # ret_lag1 строки t — доходность свечи t, она известна к закрытию этой свечи
        history = table["ret_lag1"].expanding().mean()
        return history.reindex(rows).fillna(0.0).rename("y_pred"), {}
