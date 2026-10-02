"""Проверки подготовки CryptoVision (задачи 3.3–3.6). Сеть не нужна.

Главная проверка — определение часового пояса: строим искусственные свечи Binance,
к ним новости с метками в заранее известном поясе и проверяем, что сверка цен
находит именно этот пояс и правило выбора свечи.
"""
import numpy as np
import pandas as pd

from cryptonews.data import cryptovision as cv
from cryptonews.data import news as news_rules


def make_candles(months=("2024-01", "2024-07"), seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for month in months:
        start = pd.Timestamp(month + "-01", tz="UTC")
        index = pd.date_range(start, start + pd.offsets.MonthBegin(1), freq="15min",
                              tz="UTC", inclusive="left")
        close = 40000 + np.cumsum(rng.normal(0, 25, len(index)))
        open_ = np.r_[close[0], close[:-1]]
        high = np.maximum(open_, close) + np.abs(rng.normal(0, 8, len(index)))
        low = np.minimum(open_, close) - np.abs(rng.normal(0, 8, len(index)))
        frames.append(pd.DataFrame({"open": open_, "high": high, "low": low, "close": close},
                                   index=index).round(2))
    frame = pd.concat(frames)
    frame.index.name = "open_time"
    return frame


def make_news(candles, offset_hours_by_month, rule="containing", n=300, seed=1,
              source="coindesk.com") -> pd.DataFrame:
    """Новости с метками в поясе UTC+offset и ценами свечи по заданному правилу."""
    rng = np.random.default_rng(seed)
    rows = []
    for month, offset in offset_hours_by_month.items():
        start = pd.Timestamp(month + "-03", tz="UTC")
        for _ in range(n):
            utc = start + pd.Timedelta(seconds=int(rng.integers(0, 22 * 24 * 3600)))
            candle_time = utc.floor("15min")
            if rule == "next":
                candle_time += pd.Timedelta(minutes=15)
            elif rule == "nearest":
                candle_time = utc.round("15min")
            candle = candles.loc[candle_time]
            local = (utc + pd.Timedelta(hours=offset)).tz_localize(None)
            rows.append({
                "URL": f"https://{source}/news/{seed}-{len(rows)}",
                "Title": f"Новость номер {len(rows)}",
                "Date Time": local.strftime("%Y-%m-%d %H:%M:%S"),
                "Coin Type": "Bitcoin",
                "Open": candle["open"], "High": candle["high"],
                "Low": candle["low"], "Close": candle["close"],
            })
    return pd.DataFrame(rows)


def prepared(raw: pd.DataFrame) -> pd.DataFrame:
    frame, _ = cv.normalize_columns(raw)
    frame["date_time"], _ = cv.parse_times(frame["date_time"])
    return frame


def test_detects_dhaka_offset_with_containing_candle():
    candles = make_candles()
    frame = prepared(make_news(candles, {"2024-01": 6, "2024-07": 6}))
    report = cv.detect_timezone(frame, {"BTCUSDT": candles}, ["2024-01", "2024-07"])
    assert report.best_offset_minutes == 360
    assert report.best_rule == "свеча, содержащая новость"
    assert report.best_share > 0.99 and report.runner_up_share < 0.1
    assert cv.is_reliable(report) and report.offset_label == "UTC+06:00"


def test_detects_utc_with_next_candle():
    candles = make_candles()
    frame = prepared(make_news(candles, {"2024-01": 0, "2024-07": 0}, rule="next"))
    report = cv.detect_timezone(frame, {"BTCUSDT": candles}, ["2024-01", "2024-07"])
    assert report.best_offset_minutes == 0
    assert report.best_rule == "следующая свеча"
    assert cv.is_reliable(report)


def test_daylight_saving_is_flagged_not_guessed():
    """Зимой +2, летом +3: постоянного сдвига нет, и скрипт должен это сказать, а не угадать."""
    candles = make_candles()
    frame = prepared(make_news(candles, {"2024-01": 2, "2024-07": 3}))
    report = cv.detect_timezone(frame, {"BTCUSDT": candles}, ["2024-01", "2024-07"])
    assert not cv.is_reliable(report)
    assert "летнее время" in report.verdict


def test_no_matches_is_reported():
    candles = make_candles()
    raw = make_news(candles, {"2024-01": 6})
    raw["Close"] = raw["Close"] + 0.37  # цены не совпадают ни с одной свечой
    report = cv.detect_timezone(prepared(raw), {"BTCUSDT": candles}, ["2024-01"])
    assert report.records_matched == 0 and not cv.is_reliable(report)


def test_normalize_columns_handles_both_versions():
    v1 = pd.DataFrame({"URL": ["u"], "Title": ["t"], "Date Time": ["2024-01-01 10:00:00"],
                       "Coin Type": ["Bitcoin"], "Close": ["100.5"]})
    v2 = pd.DataFrame({"URL": ["u"], "Title": ["t"], "Date_Time": ["2024-01-01 10:00:00"],
                       "Coin_Type": ["Bitcoin"], "Close": ["100.5"]})
    for raw in (v1, v2):
        frame, mapping = cv.normalize_columns(raw)
        assert {"url", "title", "date_time", "coin_type", "close"} <= set(frame.columns)
        assert frame["close"].iloc[0] == 100.5


def test_parse_times_converts_explicit_zone_to_utc():
    times, share = cv.parse_times(pd.Series(["2024-01-01T12:00:00+03:00", "2024-01-01T12:00:00Z"]))
    assert share == 1.0
    assert list(times) == [pd.Timestamp("2024-01-01 09:00"), pd.Timestamp("2024-01-01 12:00")]


def test_dedup_titles_keeps_earliest_within_window_only():
    frame = pd.DataFrame({
        "title": ["Bitcoin hits $50K!", "bitcoin hits 50k", "Bitcoin hits $50K", "Другая новость"],
        "published_utc": pd.to_datetime(["2024-01-01 10:00", "2024-01-01 15:00",
                                         "2024-01-05 10:00", "2024-01-01 11:00"], utc=True),
    })
    kept = news_rules.dedup_titles(frame, window_hours=24)
    # повтор через 5 часов убран, тот же заголовок через 4 дня — отдельная новость
    assert len(kept) == 3
    assert pd.Timestamp("2024-01-01 15:00", tz="UTC") not in set(kept["published_utc"])


def test_clean_converts_to_utc_and_counts_stages():
    raw = pd.DataFrame({
        "URL": ["https://www.coindesk.com/a", "https://www.coindesk.com/a", "https://cryptopanic.com/b", ""],
        "Title": ["Bitcoin rallies", "Bitcoin rallies", "Bitcoin rallies!", "Без адреса"],
        "Date Time": ["2024-01-01 16:00:00", "2024-01-01 16:00:00", "2024-01-01 17:30:00", "2024-01-01 18:00:00"],
        "Coin Type": ["Bitcoin"] * 4,
    })
    frame = prepared(raw)
    news, stages = cv.clean(frame, offset_minutes=360)
    # записей, с меткой, с заголовком и адресом, с точным временем, без повторов адреса, без повторов заголовка
    assert list(stages["Записей"]) == [4, 4, 3, 3, 2, 1]
    assert news.loc[0, "published_utc"] == pd.Timestamp("2024-01-01 10:00", tz="UTC")
    assert news.loc[0, "source"] == "coindesk.com"
    assert "close" not in news.columns  # цены набора в признаки не попадают
    assert bool(news.loc[0, "mentions_btc"]) and not bool(news.loc[0, "mentions_eth"])


def test_pick_sample_months_spreads_over_period():
    times = pd.Series(pd.date_range("2017-01-01", "2025-08-01", freq="7D"))
    months = cv.pick_sample_months(pd.DataFrame({"date_time": times}), count=16)
    assert len(months) == 16
    assert months[0] >= "2017-09" and months[-1] >= "2025-07"


def test_matching_does_not_need_coin_labels():
    """Монета новости для сверки не нужна: свечи обоих активов ищутся вместе."""
    candles = make_candles()
    raw = make_news(candles, {"2024-01": 6, "2024-07": 6})
    raw["Coin Type"] = "BTC"  # другая запись монеты, чем «Bitcoin»
    eth = make_candles(seed=5) / 20  # «эфир»: другие цены
    report = cv.detect_timezone(prepared(raw), {"BTCUSDT": candles, "ETHUSDT": eth},
                                ["2024-01", "2024-07"])
    assert report.best_offset_minutes == 360 and cv.is_reliable(report)
    assert report.top_deltas[0]["delta_min"] <= -360 + 15


def test_sample_months_ignore_missing_dates():
    times = pd.Series(list(pd.date_range("2020-01-01", "2020-06-01", freq="7D")) + [pd.NaT])
    months = cv.pick_sample_months(pd.DataFrame({"date_time": times}), count=16)
    assert "NaT" not in months and months[0] == "2020-01"


# ----------------------------------------------------------- архив, скачанный вручную

import io
import zipfile

import pytest

CSV_TEXT = ("URL,Title,Date_Time,Coin_Type,Open,High,Low,Close\n"
            "https://coindesk.com/a,Bitcoin rallies,2024-01-03 10:00:00,Bitcoin,1,2,0.5,1.5\n"
            "https://coindesk.com/b,Ether dips,2024-01-03 11:00:00,Ethereum,1,2,0.5,1.5\n")


def zip_bytes(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def test_obtain_extracts_download_all_archive(tmp_path):
    """«Download All» кладёт таблицу в папку внутри архива, рядом бывает мусор macOS."""
    (tmp_path / "3c3xtxtfb6-2.zip").write_bytes(zip_bytes({
        "CryptoVision/CryptoDataSet.csv": CSV_TEXT.encode(),
        "__MACOSX/CryptoVision/._CryptoDataSet.csv": b"junk",
        "CryptoVision/readme.txt": b"about",
    }))
    files, origin = cv.obtain(tmp_path, version=2)
    assert [f.name for f in files] == ["CryptoDataSet.csv"]
    assert origin == "распакованы из архива в папке"
    frame, info = cv.load_raw(files)
    assert len(frame) == 2 and info[0]["used"]


def test_obtain_handles_nested_archive(tmp_path):
    inner = zip_bytes({"data/CryptoDataSet.csv": CSV_TEXT.encode()})
    (tmp_path / "download.zip").write_bytes(zip_bytes({"CryptoVision.zip": inner}))
    files, _ = cv.obtain(tmp_path, version=2)
    assert [f.name for f in files] == ["CryptoDataSet.csv"]


def test_obtain_explains_manual_download_on_403(tmp_path, monkeypatch):
    class Forbidden:
        status_code = 403
        content = b""
    monkeypatch.setattr(cv.requests, "get", lambda url, timeout=300: Forbidden())
    with pytest.raises(SystemExit, match="Download All"):
        cv.obtain(tmp_path, version=2)


def test_load_raw_skips_files_without_needed_columns(tmp_path):
    (tmp_path / "news.csv").write_text(CSV_TEXT, encoding="utf-8")
    (tmp_path / "fields.csv").write_text("field,description\nURL,link\n", encoding="utf-8")
    frame, info = cv.load_raw(cv.data_files(tmp_path))
    assert len(frame) == 2
    assert {i["file"]: i["used"] for i in info} == {"fields.csv": False, "news.csv": True}



def test_nearest_candle_rule_in_utc_is_reliable():
    """Так устроен настоящий CryptoVision: метки в UTC, к новости приложена ближайшая свеча,
    поэтому разность «свеча минус метка» лежит по обе стороны от нуля."""
    candles = make_candles()
    frame = prepared(make_news(candles, {"2024-01": 0, "2024-07": 0}, rule="nearest"))
    report = cv.detect_timezone(frame, {"BTCUSDT": candles}, ["2024-01", "2024-07"])
    assert report.best_offset_minutes == 0 and report.best_share > 0.99
    assert report.best_rule == "ближайшая свеча"
    assert cv.is_reliable(report)


def test_source_with_its_own_timezone_is_flagged():
    candles = make_candles()
    good = make_news(candles, {"2024-01": 0, "2024-07": 0}, n=300, seed=1, source="coindesk.com")
    odd = make_news(candles, {"2024-01": 3, "2024-07": 3}, n=120, seed=2, source="cryptonews.com")
    frame = prepared(pd.concat([good, odd], ignore_index=True))
    report = cv.detect_timezone(frame, {"BTCUSDT": candles}, ["2024-01", "2024-07"], min_share=0.7)
    sources = {s["source"]: s for s in report.per_source}
    assert sources["coindesk.com"]["agrees"] and not sources["cryptonews.com"]["agrees"]
    assert sources["cryptonews.com"]["best_offset_minutes"] == 180
    assert not cv.is_reliable(report) and "cryptonews.com" in report.verdict


def test_imprecise_times_are_dropped():
    """Метки ровно в полночь и время-заглушка у источника исключаются: иначе новость
    с одной датой «вышла» бы на много часов раньше настоящей публикации."""
    rng = np.random.default_rng(0)
    precise = pd.date_range("2024-01-01 00:07:13", periods=200, freq="97min", tz="UTC")
    placeholder = pd.to_datetime([f"2024-02-{d:02d} 05:00:00" for d in range(1, 29)] * 3, utc=True)
    frame = pd.DataFrame({
        "published_utc": list(precise) + list(placeholder) + [pd.Timestamp("2024-03-01 00:00:00", tz="UTC")],
        "source": ["a.com"] * 200 + ["b.com"] * 84 + ["a.com"],
    })
    mask = news_rules.imprecise_mask(frame)
    assert mask.iloc[-1]                      # полночь у обычного источника
    assert mask.iloc[200:284].all()           # заглушка 05:00:00 у b.com
    assert not mask.iloc[:200].any()          # обычные метки остаются
    table = news_rules.time_precision(frame).set_index("Источник")
    assert table.loc["b.com", "Его доля"] == 1.0
