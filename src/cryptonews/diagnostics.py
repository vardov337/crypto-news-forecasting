"""Обязательные проверки из PROTOCOL.md, раздел 10 (задачи 6.4 и 7.5).

Здесь — проверки данных, которые выполняются при построении признаков. Тест сдвига,
плацебо-тест и контроль дат при масштабировании относятся к шагам моделей и оценки.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class PipelineError(RuntimeError):
    """Ошибка данных, после которой расчёт продолжать нельзя."""


def check_monotonic(index: pd.DatetimeIndex, name: str) -> None:
    """Метки времени строго возрастают и не повторяются."""
    if index.has_duplicates:
        raise PipelineError(f"{name}: повторяющиеся метки времени, например {index[index.duplicated()][:3].tolist()}")
    if not index.is_monotonic_increasing:
        raise PipelineError(f"{name}: метки времени идут не по возрастанию")


def simple_return(log_return: pd.Series) -> pd.Series:
    return np.expm1(log_return)


def check_target(target: pd.Series, limit: float, name: str) -> None:
    """Скачок цены больше limit за час (простая доходность) — ошибка данных, а не рынка.

    Порог рассчитан на грубые ошибки вроде смешения рядов BTC и ETH (доходность в разы),
    а не на реальные обвалы: 12.03.2020 биткоин на Binance упал за час на 18%."""
    jumps = target[simple_return(target).abs() > limit]
    if len(jumps):
        examples = ", ".join(f"{t:%Y-%m-%d %H:%M} ({np.expm1(v):+.1%})" for t, v in jumps.head(5).items())
        raise PipelineError(f"{name}: изменение цены за час по модулю больше {limit:.0%}: {examples}. "
                            "Проверьте цены за эти часы")


def large_moves(targets: dict[str, pd.Series], threshold: float) -> pd.DataFrame:
    """Часы, когда хотя бы один актив сдвинулся больше threshold, с доходностями всех активов.

    Если в тот же час сильно двигался и другой актив, это событие рынка, а не сбой данных."""
    simple = pd.DataFrame({name: simple_return(series) for name, series in targets.items()})
    flagged = simple[(simple.abs() > threshold).any(axis=1)]
    flagged.index.name = "Час (строка t; доходность свечи t + 1)"
    return flagged.round(4)


def check_news_alignment(news_hours: pd.Series, published: pd.Series) -> None:
    """Каждая новость отнесена к часу, который начинается не позже публикации и не раньше чем за час до неё."""
    lag = published - news_hours
    if ((lag < pd.Timedelta(0)) | (lag >= pd.Timedelta(hours=1))).any():
        raise PipelineError("Новость отнесена не к тому часу: нарушено правило t ≤ τ < t + 1 ч")
