"""Random Forest и XGBoost: подбор на окне настройки, ранняя остановка, несколько зёрен (задача 7.3).

Обе модели строятся средствами XGBoost и считаются на видеокарте, если она есть.

Random Forest — режим случайного леса XGBoost: один шаг бустинга из num_parallel_tree
параллельных деревьев с шагом 1 (прогноз — среднее деревьев; так XGBoost рекомендует строить
лес вместо устаревшей обёртки XGBRFRegressor). Каждое дерево — на случайной подвыборке
63,2% строк (как доля уникальных строк в бутстрепе),
в каждом узле — случайная доля признаков (sqrt — √k из k), без сжатия и регуляризации,
min_child_weight для квадратичной потери равен минимальному числу строк в листе.
Отличие от RandomForestRegressor из scikit-learn — гистограммные разбиения; на
процессоре Colab лес из 300 деревьев по 50 тыс. строк считался бы часами (раздел 12).

XGBoost — градиентный бустинг; ранняя остановка по последним 10% обучающего окна.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from cryptonews.features import TARGET
from cryptonews.validation import Window


def device() -> str:
    """cuda, если PyTorch видит видеокарту, иначе cpu."""
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def train_test(table: pd.DataFrame, columns: list[str], window: Window):
    train = table["valid"] & (table.index < window.train_end)
    test = table["valid"] & (table.index >= window.test_start) & (table.index < window.test_end)
    return (table.loc[train, columns].to_numpy(np.float32), table.loc[train, TARGET].to_numpy(np.float32),
            table.loc[test, columns].to_numpy(np.float32), table.index[test])


def column_fraction(max_features, n_features: int) -> float:
    """Доля признаков для colsample_bynode. XGBoost берёт int(доля × k) признаков, как
    scikit-learn; половина в числителе защищает от ошибки округления (3/9 × 9 = 2,9999…)."""
    if max_features == "sqrt":
        count = int(np.sqrt(n_features))
    else:
        count = int(float(max_features) * n_features)
    return min(1.0, (max(1, count) + 0.5) / n_features)


class RandomForest:
    name, stochastic, needs_grid = "random_forest", True, True

    def __init__(self, compute_device: str | None = None):
        self.device = compute_device or device()

    def fit_predict(self, table: pd.DataFrame, columns, window: Window, params: dict, seed: int):
        import xgboost as xgb

        X, y, X_test, rows = train_test(table, list(columns), window)
        depth = params.get("max_depth")
        # без ограничения глубины: рост по листьям; 4096 листьев недостижимы при min_child_weight ≥ 50
        growth = ({"max_depth": int(depth)} if depth is not None
                  else {"max_depth": 0, "grow_policy": "lossguide", "max_leaves": 4096})
        model = xgb.XGBRegressor(
            n_estimators=1, num_parallel_tree=int(params["n_estimators"]),
            min_child_weight=float(params["min_samples_leaf"]),
            colsample_bynode=column_fraction(params["max_features"], X.shape[1]), subsample=0.632,
            learning_rate=1.0, reg_lambda=0.0, tree_method="hist", device=self.device,
            random_state=int(seed), n_jobs=-1, **growth)
        model.fit(X, y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = model.predict(X_test)
        return pd.Series(pred.astype(float), index=rows, name="y_pred"), {"device": self.device}


class XGBoost:
    name, stochastic, needs_grid = "xgboost", True, True

    def __init__(self, n_estimators: int, early_stopping_rounds: int, early_stopping_fraction: float,
                 compute_device: str | None = None):
        self.n_estimators, self.rounds = int(n_estimators), int(early_stopping_rounds)
        self.fraction = float(early_stopping_fraction)
        self.device = compute_device or device()

    def fit_predict(self, table: pd.DataFrame, columns, window: Window, params: dict, seed: int):
        import xgboost as xgb

        X, y, X_test, rows = train_test(table, list(columns), window)
        n_val = max(1, int(round(len(X) * self.fraction)))       # последние 10% окна — для ранней остановки
        model = xgb.XGBRegressor(
            n_estimators=self.n_estimators, early_stopping_rounds=self.rounds, eval_metric="rmse",
            learning_rate=float(params["learning_rate"]), max_depth=int(params["max_depth"]),
            min_child_weight=float(params["min_child_weight"]), subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]), reg_lambda=float(params["reg_lambda"]),
            tree_method="hist", device=self.device, random_state=int(seed), n_jobs=-1)
        model.fit(X[:-n_val], y[:-n_val], eval_set=[(X[-n_val:], y[-n_val:])], verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = model.predict(X_test)
        return (pd.Series(pred.astype(float), index=rows, name="y_pred"),
                {"device": self.device, "best_iteration": int(model.best_iteration)})
