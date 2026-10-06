"""Обязательные проверки из PROTOCOL.md, раздел 10 (задачи 6.4 и 7.5).

Здесь — проверки данных, которые выполняются при построении признаков. Тест сдвига,
плацебо-тест и контроль дат при масштабировании относятся к шагам моделей и оценки.
"""
from __future__ import annotations

import pandas as pd


class PipelineError(RuntimeError):
    """Ошибка данных, после которой расчёт продолжать нельзя."""


def check_monotonic(index: pd.DatetimeIndex, name: str) -> None:
    """Метки времени строго возрастают и не повторяются."""
    if index.has_duplicates:
        raise PipelineError(f"{name}: повторяющиеся метки времени, например {index[index.duplicated()][:3].tolist()}")
    if not index.is_monotonic_increasing:
        raise PipelineError(f"{name}: метки времени идут не по возрастанию")


def check_target(target: pd.Series, limit: float, name: str) -> None:
    """Скачок целевой переменной больше limit за час — ошибка пайплайна."""
    jumps = target[target.abs() > limit]
    if len(jumps):
        examples = ", ".join(f"{t:%Y-%m-%d %H:%M} ({v:+.3f})" for t, v in jumps.head(5).items())
        raise PipelineError(f"{name}: лог-доходность за час по модулю больше {limit}: {examples}. "
                            "Проверьте цены за эти часы")


def check_news_alignment(news_hours: pd.Series, published: pd.Series) -> None:
    """Каждая новость отнесена к часу, который начинается не позже публикации и не раньше чем за час до неё."""
    lag = published - news_hours
    if ((lag < pd.Timedelta(0)) | (lag >= pd.Timedelta(hours=1))).any():
        raise PipelineError("Новость отнесена не к тому часу: нарушено правило t ≤ τ < t + 1 ч")
