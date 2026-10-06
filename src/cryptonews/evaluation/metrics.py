"""RMSE, MAE, точность направления, R²_OOS относительно исторического среднего (задача 8.1).

Все функции получают факт и прогноз на одних и тех же строках — общих для всех моделей
актива (PROTOCOL.md, раздел 6): у LSTM строк чуть меньше, сравнение идёт по пересечению.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(y, f) -> float:
    y, f = np.asarray(y, float), np.asarray(f, float)
    return float(np.sqrt(np.mean((y - f) ** 2)))


def mae(y, f) -> float:
    y, f = np.asarray(y, float), np.asarray(f, float)
    return float(np.mean(np.abs(y - f)))


def direction_mask(y, f) -> np.ndarray:
    """Строки, где знак есть и у факта, и у прогноза: часы с нулевой доходностью исключаются,
    нулевой прогноз направления не задаёт."""
    return (np.asarray(y, float) != 0) & (np.asarray(f, float) != 0)


def directional_accuracy(y, f) -> tuple[float, int]:
    """Доля совпавших знаков и число строк, на которых она считалась."""
    y, f = np.asarray(y, float), np.asarray(f, float)
    mask = direction_mask(y, f)
    if not mask.any():
        return float("nan"), 0
    return float(np.mean(np.sign(y[mask]) == np.sign(f[mask]))), int(mask.sum())


def r2_oos(y, f, benchmark) -> float:
    """R²_OOS = 1 − Σ(y − f)² / Σ(y − b)², b — прогноз историческим средним (Campbell, Thompson, 2008)."""
    y, f, b = (np.asarray(v, float) for v in (y, f, benchmark))
    return float(1.0 - np.sum((y - f) ** 2) / np.sum((y - b) ** 2))


def common_rows(frames: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """Строки, на которых есть прогнозы всех моделей."""
    index = None
    for frame in frames.values():
        index = frame.index if index is None else index.intersection(frame.index)
    return index.sort_values()
