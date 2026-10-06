"""Признаки (задачи 6.2–6.3, PROTOCOL.md, разделы 3–4).

Ценовые (блок P): лог-доходности с лагами 1–6 ч, реализованная волатильность за 6 и 24 ч,
изменение логарифма объёма за час.

Новостные, отдельно для каждого языка, по формулам (2)–(6) статьи:
  * средняя тональность за час s̄_t (2); в часы без новостей — 0;
  * доля негативных новостей за час (класс с максимальной вероятностью — «негатив»);
  * тональность с лагом 1 ч;
  * максимум (3), минимум (4) и стандартное отклонение (5) средней тональности в окне 6 ч;
    в формуле (5) статьи пределы суммы исправлены: σ_t = sqrt(1/5 · Σ_{k=0..5} (s̄_{t−k} − μ_t)²);
  * тренд (6): среднее s̄ за последние 3 ч минус среднее за предыдущие 3 ч;
  * интенсивность новостного потока: ln(1 + N_t) − ln(1 + среднее N за предыдущие 168 ч).
    Она заменяет сырое число новостей N_t: состав источников CryptoVision меняется во
    времени, и уровень N_t зависит от сбора данных, а не только от рынка (раздел 12 протокола).
    Сырое N_t сохраняется в таблице как описательная величина и в модели не идёт.

Целевая переменная строки t — лог-доходность следующей свечи: ln(C_{t+1} / C_t).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "target"


def target(prices: pd.DataFrame) -> pd.Series:
    """y_{t+1} = ln(C_{t+1} / C_t) в строке t."""
    close = prices["close"]
    return np.log(close.shift(-1) / close).rename(TARGET)


def price_features(prices: pd.DataFrame, return_lags=(1, 2, 3, 4, 5, 6), realized_vol_windows=(6, 24),
                   volume_change: bool = True) -> pd.DataFrame:
    """Блок P. ret_lag1 — доходность свечи t (последняя известная к её закрытию), ret_lag2 — свечи t − 1 и т. д."""
    returns = np.log(prices["close"]).diff()
    out = {f"ret_lag{lag}": returns.shift(lag - 1) for lag in return_lags}
    for window in realized_vol_windows:
        out[f"rv_{window}"] = np.sqrt((returns ** 2).rolling(window, min_periods=window).sum())
    if volume_change:
        out["volume_change"] = np.log1p(prices["volume"]).diff()
    return pd.DataFrame(out, index=prices.index)


def price_columns(cfg: dict) -> list[str]:
    p = cfg["features"]["price"]
    columns = [f"ret_lag{lag}" for lag in p["return_lags"]]
    columns += [f"rv_{w}" for w in p["realized_vol_windows"]]
    return columns + (["volume_change"] if p.get("volume_change") else [])


def news_features(hourly: pd.DataFrame, prefix: str, rolling_window: int = 6, trend_window: int = 3,
                  sentiment_lag: int = 1, no_news_fill: float = 0.0,
                  intensity_window: int = 168) -> pd.DataFrame:
    """Новостные признаки одного языка из почасовых величин (align.hourly_news)."""
    s = hourly["sent_mean"].fillna(no_news_fill)
    count = hourly["count"].astype(float)
    out = pd.DataFrame(index=hourly.index)
    out[f"{prefix}_sent_mean"] = s
    norm = count.shift(1).rolling(intensity_window, min_periods=1).mean()
    out[f"{prefix}_news_intensity"] = np.log1p(count) - np.log1p(norm.fillna(0.0))
    out[f"{prefix}_neg_share"] = hourly["neg_share"].fillna(0.0)
    out[f"{prefix}_sent_lag{sentiment_lag}"] = s.shift(sentiment_lag)
    window = s.rolling(rolling_window, min_periods=rolling_window)
    out[f"{prefix}_sent_max{rolling_window}"] = window.max()
    out[f"{prefix}_sent_min{rolling_window}"] = window.min()
    out[f"{prefix}_sent_std{rolling_window}"] = window.std(ddof=1)
    recent = s.rolling(trend_window, min_periods=trend_window).mean()
    out[f"{prefix}_sent_trend{trend_window}"] = recent - recent.shift(trend_window)
    out[f"{prefix}_news_count"] = count
    return out


def news_columns(cfg: dict, prefix: str) -> list[str]:
    """Новостные признаки языка, которые идут в модели (без описательного числа новостей)."""
    n = cfg["features"]["news"]
    w, t, lag = n["rolling_window"], n["trend_window"], n["sentiment_lag"]
    return [f"{prefix}_sent_mean", f"{prefix}_news_intensity", f"{prefix}_neg_share",
            f"{prefix}_sent_lag{lag}", f"{prefix}_sent_max{w}", f"{prefix}_sent_min{w}",
            f"{prefix}_sent_std{w}", f"{prefix}_sent_trend{t}"]


def feature_set_columns(cfg: dict, name: str) -> list[str]:
    """Колонки набора признаков P, P_EN, P_RU или P_EN_RU из конфигурации."""
    blocks = cfg["features"]["feature_sets"][name]
    columns: list[str] = []
    for block in blocks:
        columns += price_columns(cfg) if block == "price" else news_columns(cfg, block)
    return columns


def news_params(cfg: dict) -> dict:
    n = cfg["features"]["news"]
    return {"rolling_window": int(n["rolling_window"]), "trend_window": int(n["trend_window"]),
            "sentiment_lag": int(n["sentiment_lag"]), "no_news_fill": float(n["no_news_fill"]),
            "intensity_window": int(n.get("intensity_window_hours", 168))}


def warmup_hours(cfg: dict) -> int:
    """Сколько часов истории нужно новостным признакам до первой строки выборки."""
    p = news_params(cfg)
    return max(p["intensity_window"] + 1, p["rolling_window"], 2 * p["trend_window"], p["sentiment_lag"] + 1)
