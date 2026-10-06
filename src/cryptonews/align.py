"""Выравнивание новостей по часовым свечам (задача 6.1, PROTOCOL.md, раздел 3).

Свеча с временем открытия t охватывает интервал [t, t + 1 ч). Новость с временем
публикации τ относится к часу t, если t ≤ τ < t + 1 ч, то есть к часу floor(τ).
Признаки строки t строятся из новостей с τ < t + 1 ч и цен до закрытия свечи t
включительно, а целевая переменная строки t — доходность следующей свечи t + 1 ч.
Так новость, вышедшая в 10:59, попадает в признаки строки 10:00 и используется
только для прогноза доходности 11:00–12:00.
"""
from __future__ import annotations

import pandas as pd

ASSET_FLAGS = {"BTCUSDT": "mentions_btc", "ETHUSDT": "mentions_eth"}
ASSET_RULES = ("coin_or_market_wide", "coin_only", "all")


def hour_of(times: pd.Series) -> pd.Series:
    """Час свечи, к которой относится новость: t ≤ τ < t + 1 ч."""
    return times.dt.tz_convert("UTC").dt.floor("h")


def news_for_asset(news: pd.DataFrame, symbol: str, rule: str = "coin_or_market_wide") -> pd.DataFrame:
    """Новости, которые идут в признаки актива.

    coin_or_market_wide — новость упоминает актив или не упоминает ни биткоин, ни эфир
    (общерыночная новость учитывается для обоих активов); coin_only — только упоминания
    актива; all — все новости."""
    if rule not in ASSET_RULES:
        raise ValueError(f"Неизвестное правило отбора новостей: {rule}")
    if rule == "all":
        return news
    own = news[ASSET_FLAGS[symbol]].astype(bool)
    if rule == "coin_only":
        return news[own]
    any_coin = news[list(ASSET_FLAGS.values())].astype(bool).any(axis=1)
    return news[own | ~any_coin]


def hourly_news(news: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.DataFrame:
    """Почасовые величины на сетке grid: число новостей, средняя тональность, доля негативных.

    В часы без новостей число — 0, а тональность и доля — пропуск (заполняются в features)."""
    frame = news.assign(hour=hour_of(news["published_utc"]),
                        negative=news["label"].eq("negative").astype(float))
    grouped = frame.groupby("hour")
    hourly = pd.DataFrame({
        "count": grouped.size(),
        "sent_mean": grouped["score"].mean(),
        "neg_share": grouped["negative"].mean(),
    }).reindex(grid)
    hourly["count"] = hourly["count"].fillna(0).astype(int)
    hourly.index.name = grid.name
    return hourly
