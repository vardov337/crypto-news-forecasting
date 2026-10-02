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
