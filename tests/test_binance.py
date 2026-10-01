"""Проверки загрузчика цен на искусственных данных (задача 3.1).

Сеть здесь не нужна: архив Binance собирается прямо в памяти, поэтому тесты
быстрые и не зависят от доступности сайта.
"""
import hashlib
import io
import re
import zipfile

import pandas as pd
import pytest

from cryptonews.data import binance


def make_zip(rows: list[list], with_header: bool = False) -> bytes:
    """Собирает такой же zip-архив, какой отдаёт data.binance.vision."""
    lines = []
    if with_header:
        lines.append(",".join(binance.RAW_COLUMNS))
    for row in rows:
        lines.append(",".join(str(v) for v in row))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("BTCUSDT-1h-2024-01.csv", "\n".join(lines) + "\n")
    return buffer.getvalue()


def candle(open_time: int, close: float = 100.0) -> list:
    return [open_time, 99.0, 101.0, 98.0, close, 10.0, open_time + 3_599_999,
            1000.0, 50, 5.0, 500.0, 0]


MS_HOUR = 3_600_000
START_MS = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC


def test_month_range():
    assert binance.month_range("2024-01-01", "2024-03-15") == ["2024-01", "2024-02", "2024-03"]


def test_parse_month_without_header():
    content = make_zip([candle(START_MS), candle(START_MS + MS_HOUR)])
    frame = binance._parse_month(content)
    assert list(frame.index) == [
        pd.Timestamp("2024-01-01 00:00", tz="UTC"),
        pd.Timestamp("2024-01-01 01:00", tz="UTC"),
    ]
    assert frame["close"].iloc[0] == 100.0


def test_parse_month_with_header():
    content = make_zip([candle(START_MS)], with_header=True)
    assert len(binance._parse_month(content)) == 1


def test_parse_month_handles_microseconds():
    """В архивах с 2025 года время записано в микросекундах, а не в миллисекундах."""
    content = make_zip([candle(START_MS * 1000)])
    frame = binance._parse_month(content)
    assert frame.index[0] == pd.Timestamp("2024-01-01 00:00", tz="UTC")


def test_verify_checksum():
    content = b"abc"
    good = hashlib.sha256(content).hexdigest() + "  BTCUSDT-1h-2024-01.zip"
    assert binance._verify_checksum(content, good)
    assert not binance._verify_checksum(content, "0" * 64 + "  file.zip")


def build_prices(hours: int, drop: list[int] | None = None) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=hours, freq="h", tz="UTC")
    frame = pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
        "volume": 1.0, "quote_volume": 100.0, "trades": 1.0,
    }, index=index)
    frame.index.name = "open_time"
    if drop:
        frame.iloc[drop] = float("nan")
    return frame


def test_find_gaps_counts_consecutive_hours():
    gaps = binance.find_gaps(build_prices(10, drop=[3, 4, 5, 8]))
    assert [g["hours"] for g in gaps] == [3, 1]
    assert gaps[0]["start"].startswith("2024-01-01 03:00")


def test_find_gaps_empty_when_full():
    assert binance.find_gaps(build_prices(10)) == []


def test_validate_passes_on_clean_data():
    binance.validate(build_prices(500), "BTCUSDT")


def test_validate_rejects_unsorted_index():
    prices = build_prices(10)
    prices = prices.iloc[::-1]
    with pytest.raises(ValueError, match="не упорядочены"):
        binance.validate(prices, "BTCUSDT")


def test_validate_rejects_broken_candle():
    prices = build_prices(10)
    prices.iloc[2, prices.columns.get_loc("high")] = 98.0  # максимум ниже минимума
    with pytest.raises(ValueError, match="максимум"):
        binance.validate(prices, "BTCUSDT")


def test_validate_rejects_too_many_gaps():
    with pytest.raises(ValueError, match="слишком много"):
        binance.validate(build_prices(100, drop=list(range(10))), "BTCUSDT")


def test_validate_rejects_non_hourly_grid():
    index = pd.date_range("2024-01-01", periods=5, freq="2h", tz="UTC")
    prices = build_prices(5).set_index(index)
    with pytest.raises(ValueError, match="шаг сетки"):
        binance.validate(prices, "BTCUSDT")


def test_coverage_table_columns():
    report = binance.DownloadReport(symbol="BTCUSDT", rows_on_grid=100, rows_raw=98,
                                    gaps=[{"start": "a", "end": "b", "hours": 2}])
    table = binance.coverage_table([report])
    assert table.loc[0, "Пропущено часов"] == 2
    assert table.loc[0, "Доля пропусков"] == 0.02


def test_download_symbol_end_to_end(tmp_path, monkeypatch):
    """Полный проход загрузки без сети: подменяем скачивание на сборку архива в памяти."""
    months = {"2024-01": 744, "2024-02": 696}

    def fake_download(url, timeout=60, retries=3):
        month = re.search(r"(\d{4}-\d{2})\.zip", url).group(1)
        if month not in months:
            return None
        start = int(pd.Timestamp(month + "-01", tz="UTC").timestamp() * 1000)
        rows = [candle(start + i * MS_HOUR) for i in range(months[month])]
        if month == "2024-01":
            del rows[100]  # имитируем пропуск биржи на один час
        content = make_zip(rows)
        if url.endswith(".CHECKSUM"):
            return (hashlib.sha256(content).hexdigest() + "  f.zip").encode()
        return content

    monkeypatch.setattr(binance.requests, "get", lambda *a, **k: None)
    monkeypatch.setattr(binance, "_download", fake_download)
    prices, report = binance.download_symbol(
        "BTCUSDT", "2024-01-01", "2024-03-31", cache_dir=tmp_path, pause_sec=0,
    )
    binance.validate(prices, "BTCUSDT")

    assert report.rows_on_grid == 744 + 696
    assert report.rows_raw == 744 + 696 - 1
    assert report.missing_hours == 1
    assert report.months_missing == ["2024-03"]
    assert report.checksum_ok == 2
    assert prices.index[0] == pd.Timestamp("2024-01-01 00:00", tz="UTC")
    # повторный запуск берёт архивы из кэша и ничего не качает
    _, again = binance.download_symbol("BTCUSDT", "2024-01-01", "2024-02-29",
                                       cache_dir=tmp_path, pause_sec=0)
    assert again.months_from_cache == 2 and again.months_downloaded == 0
