"""Выравнивание, признаки и схема проверки (задачи 6.1–6.4, 7.1). Сеть не нужна.

Главная проверка — отсутствие утечки: новость, вышедшая в 10:59, попадает в признаки
строки 10:00 и используется только для прогноза доходности 11:00–12:00, а новость,
вышедшая позже, не меняет признаков более ранних строк.
"""
import numpy as np
import pandas as pd
import pytest

from cryptonews import align, diagnostics, features, validation

CFG = {
    "features": {
        "price": {"return_lags": [1, 2, 3, 4, 5, 6], "realized_vol_windows": [6, 24], "volume_change": True},
        "news": {"languages": ["en", "ru"], "rolling_window": 6, "trend_window": 3, "sentiment_lag": 1,
                 "no_news_fill": 0.0, "intensity_window_hours": 168, "asset_rule": "coin_or_market_wide"},
        "feature_sets": {"P": ["price"], "P_EN": ["price", "en"], "P_RU": ["price", "ru"],
                         "P_EN_RU": ["price", "en", "ru"]},
    },
}


def grid(start="2024-01-01 00:00", hours=48):
    return pd.date_range(start, periods=hours, freq="h", tz="UTC", name="open_time")


def news_frame(rows):
    """rows: (время, score, label, mentions_btc, mentions_eth)."""
    return pd.DataFrame([{"published_utc": pd.Timestamp(t, tz="UTC"), "score": s, "label": l,
                          "mentions_btc": b, "mentions_eth": e, "url": f"u{i}"}
                         for i, (t, s, l, b, e) in enumerate(rows)])


def test_news_at_10_59_goes_to_10_00_row_and_predicts_11_00_return():
    index = grid(hours=24)
    prices = pd.DataFrame({"close": np.exp(np.arange(24) * 0.01), "volume": 1.0}, index=index)
    news = news_frame([("2024-01-01 10:59:59", 0.8, "positive", True, False),
                       ("2024-01-01 11:00:00", -0.5, "negative", True, False)])
    hourly = align.hourly_news(news, index)
    assert hourly.loc["2024-01-01 10:00", "count"] == 1 and hourly.loc["2024-01-01 10:00", "sent_mean"] == 0.8
    assert hourly.loc["2024-01-01 11:00", "count"] == 1
    y = features.target(prices)
    row = pd.Timestamp("2024-01-01 10:00", tz="UTC")
    assert y[row] == pytest.approx(np.log(prices.loc["2024-01-01 11:00", "close"] / prices.loc[row, "close"]))
    diagnostics.check_news_alignment(align.hour_of(news["published_utc"]), news["published_utc"])


def test_later_news_does_not_change_earlier_rows():
    index = grid(hours=72)
    base = news_frame([(f"2024-01-02 {h:02d}:30", 0.1 * (h % 5) - 0.2, "neutral", True, False) for h in range(24)])
    later = pd.concat([base, news_frame([("2024-01-02 12:00:00", -0.9, "negative", True, False)])], ignore_index=True)
    params = features.news_params(CFG)
    a = features.news_features(align.hourly_news(base, index), "en", **params)
    b = features.news_features(align.hourly_news(later, index), "en", **params)
    cutoff = pd.Timestamp("2024-01-02 12:00", tz="UTC")
    pd.testing.assert_frame_equal(a[a.index < cutoff], b[b.index < cutoff])
    assert not a.loc[cutoff:].equals(b.loc[cutoff:])


def test_news_for_asset_rule():
    news = news_frame([("2024-01-01 01:00", 0, "neutral", True, False),     # только BTC
                       ("2024-01-01 02:00", 0, "neutral", False, True),     # только ETH
                       ("2024-01-01 03:00", 0, "neutral", True, True),      # обе
                       ("2024-01-01 04:00", 0, "neutral", False, False)])   # общерыночная
    assert align.news_for_asset(news, "BTCUSDT")["url"].tolist() == ["u0", "u2", "u3"]
    assert align.news_for_asset(news, "ETHUSDT")["url"].tolist() == ["u1", "u2", "u3"]
    assert align.news_for_asset(news, "BTCUSDT", "coin_only")["url"].tolist() == ["u0", "u2"]
    assert len(align.news_for_asset(news, "ETHUSDT", "all")) == 4
    with pytest.raises(ValueError):
        align.news_for_asset(news, "BTCUSDT", "bad")


def test_hourly_aggregates_and_no_news_hours():
    index = grid(hours=4)
    news = news_frame([("2024-01-01 01:10", 0.6, "positive", True, False),
                       ("2024-01-01 01:50", -0.4, "negative", True, False),
                       ("2024-01-01 03:00", -0.2, "negative", True, False)])
    hourly = align.hourly_news(news, index)
    assert hourly["count"].tolist() == [0, 2, 0, 1]
    assert hourly.loc["2024-01-01 01:00", "sent_mean"] == pytest.approx(0.1)
    assert hourly.loc["2024-01-01 01:00", "neg_share"] == pytest.approx(0.5)
    assert np.isnan(hourly.loc["2024-01-01 00:00", "sent_mean"])


def test_news_feature_formulas():
    index = grid(hours=12)
    values = [0.0, 0.5, -0.5, 1.0, 0.2, -0.2, 0.4, 0.0, 0.0, 0.3, -0.1, 0.6]
    hourly = pd.DataFrame({"count": 1, "sent_mean": values, "neg_share": 0.0}, index=index)
    hourly.loc[index[7], ["count", "sent_mean", "neg_share"]] = [0, np.nan, np.nan]   # час без новостей
    out = features.news_features(hourly, "en", **features.news_params(CFG))
    s = pd.Series(values, index=index)
    s.iloc[7] = 0.0                                                                   # заполнение нулём
    t = index[11]
    window = s.iloc[6:12]
    assert out.loc[t, "en_sent_mean"] == pytest.approx(0.6)
    assert out.loc[t, "en_sent_lag1"] == pytest.approx(-0.1)
    assert out.loc[t, "en_sent_max6"] == pytest.approx(window.max())
    assert out.loc[t, "en_sent_min6"] == pytest.approx(window.min())
    mu = window.mean()
    assert out.loc[t, "en_sent_std6"] == pytest.approx(np.sqrt(((window - mu) ** 2).sum() / 5))   # формула (5)
    assert out.loc[t, "en_sent_trend3"] == pytest.approx(s.iloc[9:12].mean() - s.iloc[6:9].mean())  # формула (6)
    assert np.isnan(out.loc[index[4], "en_sent_max6"])                                 # окно ещё не набрано
    assert out.loc[index[7], "en_news_count"] == 0


def test_news_intensity_is_zero_for_steady_flow_and_reacts_to_burst():
    index = grid(hours=400)
    count = pd.Series(3, index=index)
    count.iloc[300] = 30
    hourly = pd.DataFrame({"count": count, "sent_mean": 0.1, "neg_share": 0.0}, index=index)
    out = features.news_features(hourly, "ru", **features.news_params(CFG))["ru_news_intensity"]
    assert abs(out.iloc[250]) < 1e-12
    assert out.iloc[300] == pytest.approx(np.log1p(30) - np.log1p(3))


def test_price_features():
    index = grid(hours=30)
    close = pd.Series(np.exp(np.cumsum(np.linspace(-0.01, 0.02, 30))), index=index)
    prices = pd.DataFrame({"close": close, "volume": np.arange(30) + 1.0}, index=index)
    out = features.price_features(prices)
    r = np.log(close).diff()
    t = index[29]
    assert out.loc[t, "ret_lag1"] == pytest.approx(r.iloc[29])
    assert out.loc[t, "ret_lag6"] == pytest.approx(r.iloc[24])
    assert out.loc[t, "rv_6"] == pytest.approx(np.sqrt((r.iloc[24:30] ** 2).sum()))
    assert np.isnan(out.loc[index[23], "rv_24"]) and not np.isnan(out.loc[index[24], "rv_24"])
    assert out.loc[t, "volume_change"] == pytest.approx(np.log1p(30) - np.log1p(29))


def test_gap_in_prices_invalidates_neighbouring_returns():
    index = grid(hours=10)
    prices = pd.DataFrame({"close": np.linspace(100, 110, 10), "volume": 1.0}, index=index)
    prices.loc[index[5], "close"] = np.nan
    out = features.price_features(prices)
    assert np.isnan(out.loc[index[5], "ret_lag1"]) and np.isnan(out.loc[index[6], "ret_lag1"])
    assert np.isnan(features.target(prices).loc[index[4]])


def test_feature_sets():
    assert len(features.feature_set_columns(CFG, "P")) == 9
    assert len(features.feature_set_columns(CFG, "P_EN")) == 17
    full = features.feature_set_columns(CFG, "P_EN_RU")
    assert len(full) == 25 and "ru_news_intensity" in full and "en_news_count" not in full
    assert features.warmup_hours(CFG) == 169


def test_walk_forward_splits():
    start, end = pd.Timestamp("2017-08-18", tz="UTC"), pd.Timestamp("2025-06-01", tz="UTC")
    splits = validation.walk_forward(start, end, test_months=24, fold_months=3, tuning_months=3, embargo_hours=1)
    folds = splits["folds"]
    assert len(folds) == 8
    assert folds[0].test_start == pd.Timestamp("2023-06-01", tz="UTC")
    assert folds[-1].test_end == end
    assert all(a.test_end == b.test_start for a, b in zip(folds, folds[1:]))
    assert folds[0].train_end == pd.Timestamp("2023-05-31 23:00", tz="UTC")
    tuning = splits["tuning"]
    assert tuning.test_start == pd.Timestamp("2023-03-01", tz="UTC") and tuning.test_end == folds[0].test_start
    index = pd.date_range("2023-05-31 20:00", "2023-06-01 02:00", freq="h", tz="UTC")
    assert folds[0].train_mask(index).sum() == 3 and folds[0].test_mask(index).sum() == 3
    js = validation.as_json(splits)
    assert js["folds"][0]["name"] == "fold_1" and len(js["folds"]) == 8
    with pytest.raises(ValueError):
        validation.walk_forward(start, end, test_months=25, fold_months=3)


def test_diagnostics_catch_bad_data():
    index = grid(hours=3)
    with pytest.raises(diagnostics.PipelineError):
        diagnostics.check_target(pd.Series([0.01, np.log(0.6), -0.02], index=index), 0.2, "BTC")   # −40%
    diagnostics.check_target(pd.Series([0.01, np.nan, -0.201], index=index), 0.2, "BTC")    # −18,2%: 12.03.2020
    with pytest.raises(diagnostics.PipelineError):
        diagnostics.check_target(pd.Series([np.log(0.03)], index=index[:1]), 0.5, "BTC")         # ряд ETH вместо BTC
    moves = diagnostics.large_moves({"BTC": pd.Series([0.01, -0.201, 0.0], index=index),
                                     "ETH": pd.Series([0.0, -0.25, 0.3], index=index)}, 0.15)
    assert len(moves) == 2 and moves.iloc[0]["BTC"] == pytest.approx(np.expm1(-0.201), abs=1e-4)
    with pytest.raises(diagnostics.PipelineError):
        diagnostics.check_monotonic(index.append(index[:1]), "BTC")
    published = pd.Series(pd.to_datetime(["2024-01-01 10:30"], utc=True))
    with pytest.raises(diagnostics.PipelineError):
        diagnostics.check_news_alignment(pd.Series(pd.to_datetime(["2024-01-01 11:00"], utc=True)), published)


def test_batch_uploads_are_detected_per_source():
    from cryptonews.data import news as news_rules

    t0 = pd.Timestamp("2023-10-23 08:00:00", tz="UTC")
    spread = [t0 + pd.Timedelta(seconds=15 * i) for i in range(30)]           # загрузка архива: 30 записей через 15 с
    same = [pd.Timestamp("2019-04-17 13:00:00", tz="UTC")] * 12                # 12 записей с одной секундой
    short = [pd.Timestamp("2020-01-01 10:00", tz="UTC") + pd.Timedelta(seconds=20 * i) for i in range(9)]  # 9 < 10
    split = ([pd.Timestamp("2021-01-01", tz="UTC") + pd.Timedelta(seconds=30 * i) for i in range(9)]
             + [pd.Timestamp("2021-01-01 00:10", tz="UTC") + pd.Timedelta(seconds=30 * i) for i in range(11)])
    hourly = list(pd.date_range("2019-05-01", periods=24, freq="h", tz="UTC"))
    other = [t0 + pd.Timedelta(seconds=15 * i + 5) for i in range(5)]          # другой источник в те же минуты
    frame = pd.DataFrame({
        "published_utc": spread + same + short + split + hourly + other,
        "source": ["a.com"] * 51 + ["c.com"] * 20 + ["a.com"] * 24 + ["b.com"] * 5,
    })
    frame["title"] = [f"t{i}" for i in range(len(frame))]
    frame = frame.sample(frac=1, random_state=0)                               # порядок строк не важен
    mask, series = news_rules.batch_uploads(frame, 10, 60)
    expected = {f"t{i}" for i in range(42)} | {f"t{i}" for i in range(60, 71)}  # 30 + 12 у a.com, 11 у c.com
    assert set(frame.loc[mask, "title"]) == expected
    assert series["Записей"].tolist() == [30, 12, 11]
    assert (series["Последняя метка"] >= series["Первая метка"]).all()


def test_drop_sources():
    from cryptonews.data import news as news_rules

    frame = pd.DataFrame({"source": ["cryptopanic.com", "decrypt.co", "cryptopanic.com"], "title": ["a", "b", "c"]})
    kept, counts = news_rules.drop_sources(frame, ["cryptopanic.com", "nowhere.org"])
    assert kept["title"].tolist() == ["b"] and counts.to_dict() == {"cryptopanic.com": 2, "nowhere.org": 0}
    same, none = news_rules.drop_sources(frame, None)
    assert len(same) == 3 and none.empty
