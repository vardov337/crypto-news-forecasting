"""Часовые свечи Binance (задачи 3.1–3.2).

Данные берутся из официального архива data.binance.vision: помесячные zip-файлы
с часовыми свечами спотового рынка. Так сделано по двум причинам.

1. Воспроизводимость. Файл за конкретный месяц больше не меняется, и к каждому
   приложена контрольная сумма SHA-256. Через год загрузка даст те же данные.
2. Доступность. REST API Binance отказывает в обслуживании части адресов
   (в том числе американским, с которых обычно работает Google Colab),
   а архив отдаётся обычным CDN без ограничений.

Время открытия свечи приводится к UTC. Ряд переиндексируется на полную часовую
сетку: пропуски биржи остаются пустыми строками и попадают в отчёт, а не
исчезают молча (именно из-за молча выпавших часов в первой версии работы
лаги указывали на соседнюю строку, а не на предыдущий час).
"""
from __future__ import annotations

import hashlib
import io
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"

# Порядок колонок в архивах Binance. Последняя колонка служебная и не используется.
RAW_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]
KEEP_COLUMNS = ["open", "high", "low", "close", "volume", "quote_volume", "trades"]

# Метки времени в архивах до 2025 года записаны в миллисекундах, позже — в микросекундах.
# Значение больше этого порога означает микросекунды (миллисекунды такого порядка
# соответствовали бы 5138 году).
MICROSECOND_THRESHOLD = 1e14


@dataclass
class DownloadReport:
    """Что получилось при загрузке одного актива."""

    symbol: str
    months_requested: int = 0
    months_downloaded: int = 0
    months_from_cache: int = 0
    months_missing: list[str] = field(default_factory=list)
    checksum_ok: int = 0
    checksum_failed: list[str] = field(default_factory=list)
    rows_raw: int = 0
    rows_on_grid: int = 0
    gaps: list[dict] = field(default_factory=list)
    first_timestamp: str | None = None
    last_timestamp: str | None = None

    @property
    def missing_hours(self) -> int:
        return sum(gap["hours"] for gap in self.gaps)

    def as_dict(self) -> dict:
        return {**self.__dict__, "missing_hours": self.missing_hours}


def month_range(start: str, end: str) -> list[str]:
    """Список месяцев вида YYYY-MM от start до end включительно."""
    months = pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq="M")
    return [str(m) for m in months]


def _download(url: str, timeout: int = 60, retries: int = 3) -> bytes | None:
    """Скачивает файл. Возвращает None, если файла нет (код 404)."""
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=timeout)
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
            continue
        if response.status_code == 404:
            return None
        if response.status_code == 200:
            return response.content
        # 429 — слишком частые запросы, 5xx — временная ошибка сервера: ждём и повторяем
        if attempt == retries - 1:
            response.raise_for_status()
        time.sleep(2 ** attempt)
    return None


def _verify_checksum(content: bytes, checksum_text: str) -> bool:
    expected = checksum_text.split()[0].strip().lower()
    return hashlib.sha256(content).hexdigest() == expected


def _parse_month(content: bytes) -> pd.DataFrame:
    """Разбирает один помесячный архив в таблицу с индексом по времени открытия свечи."""
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        name = archive.namelist()[0]
        with archive.open(name) as f:
            head = f.readline().decode("utf-8", "ignore")
    # В файлах с 2025 года есть строка заголовка, в более ранних её нет
    has_header = head.lower().startswith("open_time")
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        name = archive.namelist()[0]
        with archive.open(name) as f:
            frame = pd.read_csv(
                f,
                header=0 if has_header else None,
                names=RAW_COLUMNS,
                usecols=range(len(RAW_COLUMNS)),
            )

    unit = "us" if frame["open_time"].max() > MICROSECOND_THRESHOLD else "ms"
    frame["open_time"] = pd.to_datetime(frame["open_time"], unit=unit, utc=True)
    frame = frame.set_index("open_time").sort_index()
    return frame[KEEP_COLUMNS].astype("float64")


def download_symbol(
    symbol: str,
    start: str,
    end: str,
    cache_dir: str | Path,
    interval: str = "1h",
    verify_checksums: bool = True,
    pause_sec: float = 0.2,
) -> tuple[pd.DataFrame, DownloadReport]:
    """Скачивает часовые свечи одного актива и приводит их к полной часовой сетке UTC.

    Скачанные архивы складываются в cache_dir: повторный запуск их не перекачивает.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    report = DownloadReport(symbol=symbol)
    frames: list[pd.DataFrame] = []

    for month in month_range(start, end):
        report.months_requested += 1
        filename = f"{symbol}-{interval}-{month}.zip"
        local_path = cache_dir / filename

        if local_path.exists():
            content = local_path.read_bytes()
            report.months_from_cache += 1
        else:
            url = f"{BASE_URL}/{symbol}/{interval}/{filename}"
            content = _download(url)
            if content is None:
                # Месяц до запуска пары или ещё не закрытый — это не ошибка
                report.months_missing.append(month)
                continue
            if verify_checksums:
                checksum = _download(url + ".CHECKSUM")
                if checksum is not None:
                    if _verify_checksum(content, checksum.decode("utf-8", "ignore")):
                        report.checksum_ok += 1
                    else:
                        report.checksum_failed.append(month)
                        raise ValueError(
                            f"Контрольная сумма не совпала: {filename}. "
                            "Файл скачался с ошибкой, удалите его и повторите запуск."
                        )
            local_path.write_bytes(content)
            report.months_downloaded += 1
            time.sleep(pause_sec)

        frames.append(_parse_month(content))

    if not frames:
        raise RuntimeError(f"Для {symbol} не скачалось ни одного месяца: проверьте даты и доступ к сети")

    prices = pd.concat(frames).sort_index()
    prices = prices[~prices.index.duplicated(keep="first")]
    report.rows_raw = len(prices)

    grid = pd.date_range(prices.index[0], prices.index[-1], freq="h", tz="UTC")
    prices = prices.reindex(grid)
    prices.index.name = "open_time"
    report.rows_on_grid = len(prices)
    report.first_timestamp = str(prices.index[0])
    report.last_timestamp = str(prices.index[-1])
    report.gaps = find_gaps(prices)
    return prices, report


def find_gaps(prices: pd.DataFrame) -> list[dict]:
    """Находит пропуски биржи: подряд идущие часы без свечи."""
    missing = prices["close"].isna()
    if not missing.any():
        return []
    gaps = []
    block = (missing != missing.shift()).cumsum()
    for _, group in prices[missing].groupby(block[missing]):
        gaps.append({
            "start": str(group.index[0]),
            "end": str(group.index[-1]),
            "hours": len(group),
        })
    return gaps


def validate(prices: pd.DataFrame, symbol: str) -> None:
    """Проверки, после которых данным можно доверять. При нарушении — остановка."""
    if not prices.index.is_monotonic_increasing:
        raise ValueError(f"{symbol}: метки времени не упорядочены по возрастанию")
    if prices.index.has_duplicates:
        raise ValueError(f"{symbol}: во времени есть дубли")
    if str(prices.index.tz) != "UTC":
        raise ValueError(f"{symbol}: индекс не в UTC, а в {prices.index.tz}")
    deltas = prices.index.to_series().diff().dropna().unique()
    if len(deltas) > 1 or (len(deltas) == 1 and deltas[0] != pd.Timedelta("1h")):
        raise ValueError(f"{symbol}: шаг сетки не равен одному часу")

    filled = prices.dropna(subset=["close"])
    if (filled[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError(f"{symbol}: есть неположительные цены")
    if (filled["high"] < filled["low"]).any():
        raise ValueError(f"{symbol}: максимум свечи меньше минимума")
    beyond = (filled["high"] < filled[["open", "close"]].max(axis=1)) | \
             (filled["low"] > filled[["open", "close"]].min(axis=1))
    if beyond.any():
        raise ValueError(f"{symbol}: цены открытия или закрытия выходят за границы свечи")

    share_missing = prices["close"].isna().mean()
    if share_missing > 0.01:
        raise ValueError(
            f"{symbol}: пропущено {share_missing:.1%} часов — это слишком много, "
            "проверьте, все ли месяцы скачались"
        )


def coverage_table(reports: list[DownloadReport]) -> pd.DataFrame:
    """Сводка по активам для таблицы этапов очистки в статье."""
    rows = []
    for report in reports:
        rows.append({
            "Актив": report.symbol,
            "Начало (UTC)": report.first_timestamp,
            "Конец (UTC)": report.last_timestamp,
            "Часов в сетке": report.rows_on_grid,
            "Свечей получено": report.rows_raw,
            "Пропущено часов": report.missing_hours,
            "Доля пропусков": round(report.missing_hours / max(report.rows_on_grid, 1), 5),
            "Месяцев скачано": report.months_downloaded + report.months_from_cache,
            "Контрольных сумм сверено": report.checksum_ok,
        })
    return pd.DataFrame(rows)


def utc_now_month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")
