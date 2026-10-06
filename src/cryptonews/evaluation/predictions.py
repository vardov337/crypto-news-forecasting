"""Прогнозы всех моделей одного актива в одной таблице (шаги 8 и 9).

Шаг 7 пишет по файлу на модель, набор признаков, фолд и зерно — их сотни, и на Google
Диске они читаются минутами. Шаг 8 собирает их один раз: прогноз стохастических моделей —
среднее по зёрнам, строки — общие для всех моделей часы теста. Результат —
results/predictions_combined_<актив>.parquet: столбец y_true и по столбцу на модель.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cryptonews.evaluation import metrics
from cryptonews.models import run

PRETTY = {"naive_zero": "Нулевой прогноз", "naive_mean": "Историческое среднее", "arima": "ARIMA",
          "arimax": "ARIMAX", "random_forest": "Random Forest", "xgboost": "XGBoost", "lstm": "LSTM"}
BASELINES = tuple(run.FAMILIES["baselines"])
ZERO, MEAN = "naive_zero:none", "naive_mean:none"


def key_of(model: str, feature_set: str) -> str:
    return f"{model}:{feature_set}"


def label(key: str) -> str:
    model, feature_set = key.split(":")
    return PRETTY[model] if feature_set == run.NO_FEATURES else f"{PRETTY[model]} {feature_set}"


def all_keys(cfg: dict) -> list[str]:
    """Все модели в порядке групп шага 7: базовые, ARIMA, деревья, LSTM."""
    return [key_of(m, s) for family in run.FAMILIES.values() for m in family for s in run.feature_sets(cfg, m)]


def is_baseline(key: str) -> bool:
    return key.split(":")[0] in BASELINES


def combined_path(results_dir, symbol: str) -> Path:
    return Path(results_dir) / f"predictions_combined_{symbol}.parquet"


def collect_all(cfg: dict, out_dir, symbol: str, n_folds: int) -> tuple[dict, list]:
    """Прогнозы по всем моделям (среднее по зёрнам) и список моделей, посчитанных не до конца."""
    found, missing = {}, []
    for key in all_keys(cfg):
        model, feature_set = key.split(":")
        frame = run.collect(out_dir, symbol, model, feature_set, n_folds)
        if frame is None:
            missing.append(key)
        else:
            found[key] = frame
    for key in (ZERO, MEAN):
        if key not in found:
            raise SystemExit(f"[{symbol}] нет прогнозов «{label(key)}» — сначала выполните шаг 7 (группа baselines)")
    return found, missing


def combine(found: dict) -> pd.DataFrame:
    """Общие строки всех моделей; проверка, что целевая переменная у всех одна и та же."""
    rows = metrics.common_rows(found)
    truth = found[ZERO].loc[rows, "y_true"]
    table = pd.DataFrame({"y_true": truth.to_numpy(float)}, index=rows)
    for key, frame in found.items():
        if not np.allclose(frame.loc[rows, "y_true"].to_numpy(), truth.to_numpy(), rtol=0, atol=1e-12):
            raise SystemExit(f"У модели «{label(key)}» другая целевая переменная — прогнозы посчитаны по разным данным")
        table[key] = frame.loc[rows, "y_pred"].to_numpy(float)
    table.index.name = "open_time"
    return table
