"""Схема walk-forward (задача 7.1, PROTOCOL.md, раздел 7).

Тест — последние test_months месяцев выборки, разбитые на фолды по fold_months
месяцев. Перед каждым фолдом модель обучается на всех строках до его начала
(расширяющееся окно). Гиперпараметры выбираются один раз на окне настройки —
tuning_months месяцев непосредственно перед тестом.

Целевая переменная строки t становится известна в момент t + 2 ч (закрытие свечи t + 1),
поэтому между обучением и проверкой оставлен зазор embargo_hours: строка, чья цель
ещё не наблюдалась к началу фолда, в обучение не попадает.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass
class Window:
    name: str
    train_end: pd.Timestamp       # обучение: строки с open_time < train_end
    test_start: pd.Timestamp      # проверка: test_start ≤ open_time < test_end
    test_end: pd.Timestamp

    def as_dict(self) -> dict:
        return {k: str(v) for k, v in asdict(self).items()}

    def train_mask(self, index: pd.DatetimeIndex) -> pd.Series:
        return pd.Series(index < self.train_end, index=index)

    def test_mask(self, index: pd.DatetimeIndex) -> pd.Series:
        return pd.Series((index >= self.test_start) & (index < self.test_end), index=index)


def walk_forward(start: pd.Timestamp, end: pd.Timestamp, test_months: int = 24, fold_months: int = 3,
                 tuning_months: int = 3, embargo_hours: int = 1) -> dict:
    """Окно настройки и фолды теста для выборки [start, end)."""
    if test_months % fold_months:
        raise ValueError("Тестовый период должен делиться на фолды без остатка")
    embargo = pd.Timedelta(hours=embargo_hours)
    test_start = end - pd.DateOffset(months=test_months)
    tuning_start = test_start - pd.DateOffset(months=tuning_months)
    if tuning_start <= start:
        raise ValueError("Выборка слишком коротка для окна настройки и теста")
    tuning = Window("tuning", tuning_start - embargo, tuning_start, test_start)
    folds = []
    for i in range(test_months // fold_months):
        fold_start = test_start + pd.DateOffset(months=i * fold_months)
        fold_end = test_start + pd.DateOffset(months=(i + 1) * fold_months)
        folds.append(Window(f"fold_{i + 1}", fold_start - embargo, fold_start, fold_end))
    return {"sample_start": start, "sample_end": end, "tuning": tuning, "folds": folds,
            "embargo_hours": embargo_hours}


def from_config(cfg: dict, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    v = cfg["validation"]
    return walk_forward(start, end, test_months=int(v["test_months"]), fold_months=int(v["fold_months"]),
                        tuning_months=int(v["tuning_window_months"]), embargo_hours=int(v.get("embargo_hours", 1)))


def from_json(data: dict) -> dict:
    """Обратное к as_json: схема из results/splits.json."""
    def window(item: dict) -> Window:
        return Window(item["name"], pd.Timestamp(item["train_end"]), pd.Timestamp(item["test_start"]),
                      pd.Timestamp(item["test_end"]))
    return {"sample_start": pd.Timestamp(data["sample_start"]), "sample_end": pd.Timestamp(data["sample_end_exclusive"]),
            "embargo_hours": data["embargo_hours"], "tuning": window(data["tuning"]),
            "folds": [window(item) for item in data["folds"]]}


def as_json(splits: dict) -> dict:
    return {"sample_start": str(splits["sample_start"]), "sample_end_exclusive": str(splits["sample_end"]),
            "embargo_hours": splits["embargo_hours"], "tuning": splits["tuning"].as_dict(),
            "folds": [f.as_dict() for f in splits["folds"]]}
