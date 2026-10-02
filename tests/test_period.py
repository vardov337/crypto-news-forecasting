"""Границы выборки и общие правила очистки заголовков. Сеть не нужна."""
import numpy as np
import pandas as pd
import pytest

from cryptonews import period
from cryptonews.data import news as news_rules


def make_news(spec: dict[str, tuple[str, str, int]]) -> pd.DataFrame:
    """Новости источников: {источник: (первая дата, последняя дата, число новостей)}."""
    frames = []
    for source, (first, last, count) in spec.items():
        times = pd.date_range(first, last, periods=count, tz="UTC")
        frames.append(pd.DataFrame({"published_utc": times, "source": source,
                                    "title": [f"{source} {i}" for i in range(count)],
                                    "url": [f"https://{source}/{i}" for i in range(count)]}))
    return pd.concat(frames, ignore_index=True)


def make_prices(first: str, last: str) -> pd.DataFrame:
    index = pd.date_range(first, last, freq="h", tz="UTC", name="open_time")
    return pd.DataFrame({"close": np.ones(len(index))}, index=index)


def test_month_start_and_ceil_day():
    ts = pd.Timestamp("2025-08-27 17:47:34", tz="UTC")
    assert period.month_start(ts) == pd.Timestamp("2025-08-01", tz="UTC")
    assert period.ceil_day(ts) == pd.Timestamp("2025-08-28", tz="UTC")
    assert period.ceil_day(pd.Timestamp("2025-08-28", tz="UTC")) == pd.Timestamp("2025-08-28", tz="UTC")


def test_coverage_end_is_month_where_a_major_source_stops():
    news = make_news({
        "a.com": ("2020-01-01", "2025-08-27 17:00", 3000),
        "b.com": ("2020-01-01", "2025-07-15 12:00", 2000),   # обрывается в июле
        "tiny.com": ("2020-01-01", "2024-03-01", 10),         # мелкий источник не в счёт
    })
    end, major = period.coverage_end(news)
    assert end == pd.Timestamp("2025-07-01", tz="UTC")
    assert set(major.index) == {"a.com", "b.com"}


def collapsing_source(source: str, normal_until: str, tail: list[str], per_day: int = 25) -> pd.DataFrame:
    """Источник с регулярным сбором до normal_until и единичными записями после."""
    normal = pd.date_range("2024-01-01", normal_until, freq=pd.Timedelta(hours=24 / per_day), tz="UTC")
    times = normal.append(pd.DatetimeIndex(pd.to_datetime(tail, utc=True)))
    return pd.DataFrame({"published_utc": times, "source": source, "title": "t",
                         "url": [f"https://{source}/{i}" for i in range(len(times))]})


def test_collapse_with_stray_records_is_detected():
    """Как Cointelegraph в CryptoVision: сбор оборвался, но остались единичные записи."""
    tail = ["2025-07-02 10:00", "2025-07-09 11:00", "2025-07-15 12:00", "2025-07-20 13:00",
            "2025-07-24 14:00", "2025-07-28 15:00", "2025-07-30 16:00", "2025-08-08 09:00", "2025-08-14 14:09"]
    healthy = collapsing_source("a.com", "2025-08-27 17:47", [])
    for cut, expected in (("2025-06-23 12:00", "2025-06-01"), ("2025-07-01 18:00", "2025-07-01")):
        news = pd.concat([healthy, collapsing_source("b.com", cut, tail)], ignore_index=True)
        end, table = period.coverage_end(news)
        assert end == pd.Timestamp(expected, tz="UTC"), (cut, end)
        regular_until = table.loc["b.com", "Регулярный сбор до (не включая)"]
        assert abs((regular_until - pd.Timestamp(cut, tz="UTC").normalize()).days) <= 2
        assert table.loc["a.com", "Граница полного покрытия"] == pd.Timestamp("2025-08-01", tz="UTC")
        assert "Регулярный сбор до" in period.coverage_report(table)


def test_last_month_is_kept_only_when_complete():
    news = collapsing_source("a.com", "2025-08-31 22:00", [])
    assert period.coverage_end(news)[0] == pd.Timestamp("2025-09-01", tz="UTC")   # август собран целиком
    news = collapsing_source("a.com", "2025-09-01 03:00", [])
    assert period.coverage_end(news)[0] == pd.Timestamp("2025-09-01", tz="UTC")   # три часа сентября не в счёт
    news = collapsing_source("a.com", "2025-08-20 12:00", [])
    assert period.coverage_end(news)[0] == pd.Timestamp("2025-08-01", tz="UTC")   # август неполный


def test_coverage_end_ignores_source_that_ended_long_ago():
    news = make_news({
        "a.com": ("2018-01-01", "2025-08-27", 3000),
        "old.com": ("2018-01-01", "2021-01-01", 3000),        # большой, но за последний год его нет
    })
    end, _ = period.coverage_end(news)
    assert end == pd.Timestamp("2025-08-01", tz="UTC")


def test_sample_period_combines_prices_and_news():
    prices = {"BTCUSDT": make_prices("2017-08-17 04:00", "2026-08-31 23:00"),
              "ETHUSDT": make_prices("2017-08-17 04:00", "2026-08-31 23:00")}
    news = {"EN": make_news({"a.com": ("2017-08-17 15:00", "2025-08-27 17:47", 5000)}),
            "RU": make_news({"forklog.com": ("2014-03-10", "2026-10-01 09:00", 5000)})}
    span = period.sample_period(prices, news)
    assert span.start == pd.Timestamp("2017-08-18", tz="UTC")
    assert span.end == pd.Timestamp("2025-08-01", tz="UTC")
    assert "31.07.2025 23:00" in span.describe()
    inside = span.mask(news["EN"]["published_utc"])
    assert news["EN"].loc[inside, "published_utc"].max() < span.end
    forced = period.sample_period(prices, news, forced_end="2025-07-01")
    assert forced.end == pd.Timestamp("2025-07-01", tz="UTC")
    assert "Задано в конфигурации" in forced.as_dict()["end_candidates"]


def test_sample_period_rejects_disjoint_sources():
    prices = {"BTCUSDT": make_prices("2024-01-01", "2024-02-01")}
    news = {"EN": make_news({"a.com": ("2018-01-01", "2020-01-01", 100)})}
    with pytest.raises(ValueError):
        period.sample_period(prices, news)


def test_unescape_title_and_cleaning_use_it():
    assert news_rules.unescape_title("S&amp;P 500 &#8217;s  rally") == "S&P 500 ’s rally"
    raw = pd.DataFrame({
        "published_utc": pd.to_datetime(["2024-01-01 10:15:00", "2024-01-01 11:20:00"], utc=True),
        "title": ["Bitcoin &amp; Ether rally", None],
        "url": ["https://a.com/1", "https://a.com/2"],
    })
    clean, _ = news_rules.clean_utc(raw)
    assert clean["title"].tolist() == ["Bitcoin & Ether rally"]
    assert bool(clean["mentions_btc"].iloc[0]) and bool(clean["mentions_eth"].iloc[0])


def test_sources_span_and_month_table():
    news = make_news({"a.com": ("2025-01-01", "2025-08-27", 240), "b.com": ("2025-03-01", "2025-07-31", 60)})
    span = news_rules.sources_span(news)
    assert span.loc["b.com", "Последняя"] == pd.Timestamp("2025-07-31", tz="UTC")
    assert abs(span["Доля за последние 12 мес."].sum() - 1) < 1e-3
    table = news_rules.by_source_month(news)
    assert table.loc["2025-08", "b.com"] == 0 and table["a.com"].sum() == 240
