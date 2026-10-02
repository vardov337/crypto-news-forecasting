"""Границы общей выборки (PROTOCOL.md, раздел 2).

Начало — первые полные сутки UTC, с которых доступны все источники: цены обоих
активов, англо- и русскоязычные новости.

Конец (граница не включается) — самая ранняя из границ:
  * час после последней свечи цен;
  * граница полного покрытия каждого набора новостей — начало месяца, в котором
    прекращается регулярный сбор хотя бы одного основного источника набора.

Основной источник — тот, на который приходится не меньше 5% новостей за последние
12 месяцев набора. Регулярный сбор источника идёт, пока его темп — среднее число
новостей в день в окне 7 суток с центром в этом дне — не ниже половины обычного
темпа (медианы темпа за последний год набора). Последняя запись источника для этого
не годится: после прекращения сбора в наборе остаются единичные записи (у Cointelegraph
в CryptoVision 7 новостей за июль 2025 года и 2 за август при 570–790 в месяц до этого).
Неполный месяц дал бы ложное падение числа новостей, которое модель приняла бы за
событие рынка.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from cryptonews.data import news as news_rules

MIN_SOURCE_SHARE = 0.05
RECENT_MONTHS = 12
WINDOW_DAYS = 7
MIN_RATIO = 0.5
REFERENCE_DAYS = 365


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
            "recent_months": int(coverage.get("recent_months", RECENT_MONTHS)),
            "window_days": int(coverage.get("window_days", WINDOW_DAYS)),
            "min_ratio": float(coverage.get("min_ratio", MIN_RATIO)),
            "reference_days": int(coverage.get("reference_days", REFERENCE_DAYS))}


def regular_end(times: pd.Series, last_day: pd.Timestamp, window_days: int = WINDOW_DAYS,
                min_ratio: float = MIN_RATIO,
                reference_days: int = REFERENCE_DAYS) -> tuple[pd.Timestamp, float]:
    """Первый день после регулярного сбора источника и его обычный темп (новостей в день).

    Темп дня — среднее число новостей в день в окне window_days суток с центром в этом
    дне (у краёв ряда окно короче, но не меньше половины). Обычный темп — медиана
    ненулевого темпа за последние reference_days дней набора. Регулярный сбор длится
    до последнего дня, когда темп не ниже min_ratio обычного."""
    days = times.dt.tz_convert("UTC").dt.floor("D")
    daily = days.value_counts().sort_index()
    calendar = pd.date_range(daily.index.min(), last_day, freq="D")
    daily = daily.reindex(calendar, fill_value=0).astype(float)
    min_periods = min((window_days + 1) // 2, len(calendar))
    rate = daily.rolling(window_days, center=True, min_periods=min_periods).mean()
    recent = rate[rate.index > last_day - pd.Timedelta(days=reference_days)]
    reference = float(recent[recent > 0].median())
    regular = rate[rate >= min_ratio * reference]
    if regular.empty:               # слишком короткий ряд: признаков обрыва нет
        return last_day + pd.Timedelta(days=1), reference
    return regular.index.max() + pd.Timedelta(days=1), reference


def coverage_end(news: pd.DataFrame, min_share: float = MIN_SOURCE_SHARE,
                 recent_months: int = RECENT_MONTHS, window_days: int = WINDOW_DAYS,
                 min_ratio: float = MIN_RATIO,
                 reference_days: int = REFERENCE_DAYS) -> tuple[pd.Timestamp, pd.DataFrame]:
    """Граница полного покрытия набора новостей и таблица его основных источников."""
    span = news_rules.sources_span(news, recent_months=recent_months)
    share = span[f"Доля за последние {recent_months} мес."]
    major = span[share >= min_share].copy()
    last_day = news["published_utc"].max().tz_convert("UTC").floor("D")
    ends, rates = {}, {}
    for source in major.index:
        times = news.loc[news["source"] == source, "published_utc"]
        ends[source], rates[source] = regular_end(times, last_day, window_days, min_ratio, reference_days)
    major["Обычный темп, новостей в день"] = pd.Series(rates).round(1)
    major["Регулярный сбор до (не включая)"] = pd.Series(ends)
    major["Граница полного покрытия"] = major["Регулярный сбор до (не включая)"].map(month_start)
    return major["Граница полного покрытия"].min(), major


def coverage_report(table: pd.DataFrame) -> str:
    """Таблица регулярности сбора основных источников в читаемом виде (даты без часов)."""
    view = pd.DataFrame({
        "Новостей в день": table["Обычный темп, новостей в день"],
        "Последняя запись": table["Последняя"].map(lambda t: f"{t:%d.%m.%Y}"),
        "Регулярный сбор до": table["Регулярный сбор до (не включая)"].map(lambda t: f"{t:%d.%m.%Y}"),
        "Граница": table["Граница полного покрытия"].map(lambda t: f"{t:%d.%m.%Y}"),
    })
    return view.to_string()


@dataclass
class SamplePeriod:
    start: pd.Timestamp                         # включается
    end: pd.Timestamp                           # не включается
    starts: dict[str, pd.Timestamp] = field(default_factory=dict)
    ends: dict[str, pd.Timestamp] = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    coverage: dict[str, pd.DataFrame] = field(default_factory=dict)

    def mask(self, times: pd.Series) -> pd.Series:
        return (times >= self.start) & (times < self.end)

    @property
    def last_hour(self) -> pd.Timestamp:
        return self.end - pd.Timedelta(hours=1)

    def describe(self) -> str:
        return f"{self.start:%d.%m.%Y %H:%M} … {self.last_hour:%d.%m.%Y %H:%M} UTC (последний час включительно)"

    def as_dict(self) -> dict:
        p = self.params
        return {
            "start_inclusive": str(self.start), "end_exclusive": str(self.end),
            "first_available": {k: str(v) for k, v in self.starts.items()},
            "end_candidates": {k: str(v) for k, v in self.ends.items()},
            "coverage": {name: table.reset_index().astype(str).to_dict(orient="records")
                         for name, table in self.coverage.items()},
            "params": p,
            "rule": ("начало — первые полные сутки UTC, когда доступны все источники; конец — самая "
                     "ранняя из границ: час после последней свечи цен; начало месяца, в котором "
                     "прекращается регулярный сбор основного источника новостей (доля не меньше "
                     f"{p.get('min_share', MIN_SOURCE_SHARE):.0%} за последние "
                     f"{p.get('recent_months', RECENT_MONTHS)} мес. набора; регулярный — пока среднее "
                     f"число новостей в день в окне {p.get('window_days', WINDOW_DAYS)} суток не ниже "
                     f"{p.get('min_ratio', MIN_RATIO):.0%} медианы за последние "
                     f"{p.get('reference_days', REFERENCE_DAYS)} дней)"),
        }


def sample_period(prices: dict[str, pd.DataFrame], news: dict[str, pd.DataFrame],
                  forced_start=None, forced_end=None, **params) -> SamplePeriod:
    """Границы выборки по ценам (индекс — время открытия свечи, колонка close) и новостям.

    params — параметры правила полного покрытия (см. coverage_params).
    forced_start / forced_end — даты из конфигурации, если правило нужно переопределить
    (такое решение записывается в раздел 12 протокола)."""
    starts: dict[str, pd.Timestamp] = {}
    ends: dict[str, pd.Timestamp] = {}
    coverage: dict[str, pd.DataFrame] = {}
    for symbol, frame in prices.items():
        valid = frame.index[frame["close"].notna()]
        starts[f"Цены {symbol}"] = valid.min()
        ends[f"Цены {symbol}"] = valid.max() + pd.Timedelta(hours=1)
    for name, frame in news.items():
        starts[f"Новости {name}"] = frame["published_utc"].min()
        ends[f"Новости {name}"], coverage[name] = coverage_end(frame, **params)
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
    return SamplePeriod(start=start, end=end, starts=starts, ends=ends, params=params, coverage=coverage)


def from_config(cfg: dict, prices: dict[str, pd.DataFrame], news: dict[str, pd.DataFrame]) -> SamplePeriod:
    """sample_period с параметрами из раздела time конфигурации."""
    return sample_period(prices, news, forced_start=cfg["time"].get("sample_start"),
                         forced_end=cfg["time"].get("sample_end"), **coverage_params(cfg))
