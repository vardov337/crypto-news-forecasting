"""Границы общей выборки (PROTOCOL.md, раздел 2).

Начало — первые полные сутки UTC, с которых доступны все источники: цены обоих
активов, англо- и русскоязычные новости.

Конец (граница не включается) — самая ранняя из границ:
  * час после последней свечи цен;
  * граница полного покрытия каждого набора новостей — начало месяца, в котором
    обрывается хотя бы один из основных источников набора. Основной источник —
    тот, на который приходится не меньше 5% новостей за последние 12 месяцев набора.

Последняя метка набора для конца не годится: если сбор шёл до 27-го числа или один
из источников перестал собираться раньше других, последний месяц неполный, и число
новостей в нём падает не из-за рынка. Такой месяц исказил бы новостные признаки.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from cryptonews.data import news as news_rules

MIN_SOURCE_SHARE = 0.05
RECENT_MONTHS = 12


def month_start(ts: pd.Timestamp) -> pd.Timestamp:
    """Начало календарного месяца метки, 00:00 UTC."""
    return ts.tz_convert("UTC").normalize().replace(day=1)


def ceil_day(ts: pd.Timestamp) -> pd.Timestamp:
    """Ближайшая полночь UTC не раньше метки."""
    ts = ts.tz_convert("UTC")
    midnight = ts.normalize()
    return midnight if midnight == ts else midnight + pd.Timedelta(days=1)


def coverage_params(cfg: dict) -> dict:
    """Параметры правила полного покрытия из раздела time.coverage конфигурации."""
    coverage = cfg["time"].get("coverage") or {}
    return {"min_share": float(coverage.get("min_source_share", MIN_SOURCE_SHARE)),
            "recent_months": int(coverage.get("recent_months", RECENT_MONTHS))}


def coverage_end(news: pd.DataFrame, min_share: float = MIN_SOURCE_SHARE,
                 recent_months: int = RECENT_MONTHS) -> tuple[pd.Timestamp, pd.DataFrame]:
    """Граница полного покрытия набора новостей и таблица его основных источников."""
    span = news_rules.sources_span(news, recent_months=recent_months)
    share = span[f"Доля за последние {recent_months} мес."]
    major = span[share >= min_share].copy()
    major["Граница полного покрытия"] = major["Последняя"].map(month_start)
    return major["Граница полного покрытия"].min(), major


@dataclass
class SamplePeriod:
    start: pd.Timestamp                         # включается
    end: pd.Timestamp                           # не включается
    starts: dict[str, pd.Timestamp] = field(default_factory=dict)
    ends: dict[str, pd.Timestamp] = field(default_factory=dict)
    min_share: float = MIN_SOURCE_SHARE
    recent_months: int = RECENT_MONTHS

    def mask(self, times: pd.Series) -> pd.Series:
        return (times >= self.start) & (times < self.end)

    def describe(self) -> str:
        last = self.end - pd.Timedelta(hours=1)
        return f"{self.start:%d.%m.%Y %H:%M} … {last:%d.%m.%Y %H:%M} UTC (последний час включительно)"

    def as_dict(self) -> dict:
        return {
            "start_inclusive": str(self.start), "end_exclusive": str(self.end),
            "first_available": {k: str(v) for k, v in self.starts.items()},
            "end_candidates": {k: str(v) for k, v in self.ends.items()},
            "rule": ("начало — первые полные сутки UTC, когда доступны все источники; конец — самая "
                     "ранняя из границ: час после последней свечи цен, начало месяца, в котором "
                     "обрывается основной источник новостей (доля не меньше "
                     f"{self.min_share:.0%} за последние {self.recent_months} мес. набора)"),
        }


def sample_period(prices: dict[str, pd.DataFrame], news: dict[str, pd.DataFrame],
                  min_share: float = MIN_SOURCE_SHARE, recent_months: int = RECENT_MONTHS,
                  forced_start=None, forced_end=None) -> SamplePeriod:
    """Границы выборки по ценам (индекс — время открытия свечи, колонка close) и новостям.

    forced_start / forced_end — даты из конфигурации, если правило нужно переопределить
    (такое решение записывается в раздел 12 протокола)."""
    starts: dict[str, pd.Timestamp] = {}
    ends: dict[str, pd.Timestamp] = {}
    for symbol, frame in prices.items():
        valid = frame.index[frame["close"].notna()]
        starts[f"Цены {symbol}"] = valid.min()
        ends[f"Цены {symbol}"] = valid.max() + pd.Timedelta(hours=1)
    for name, frame in news.items():
        starts[f"Новости {name}"] = frame["published_utc"].min()
        ends[f"Новости {name}"], _ = coverage_end(frame, min_share=min_share, recent_months=recent_months)
    start = ceil_day(max(starts.values()))
    end = min(ends.values())
    if forced_start is not None:
        start = pd.Timestamp(forced_start, tz="UTC")
        starts["Задано в конфигурации"] = start
    if forced_end is not None:
        end = pd.Timestamp(forced_end, tz="UTC")
        ends["Задано в конфигурации"] = end
    if end <= start:
        raise ValueError(f"Источники не пересекаются по времени: начало {start}, конец {end}")
    return SamplePeriod(start=start, end=end, starts=starts, ends=ends,
                        min_share=min_share, recent_months=recent_months)


def from_config(cfg: dict, prices: dict[str, pd.DataFrame], news: dict[str, pd.DataFrame]) -> SamplePeriod:
    """sample_period с параметрами из раздела time конфигурации."""
    return sample_period(prices, news, **coverage_params(cfg),
                         forced_start=cfg["time"].get("sample_start"),
                         forced_end=cfg["time"].get("sample_end"))
