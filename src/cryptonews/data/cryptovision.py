"""Англоязычные новости CryptoVision (задачи 3.3–3.6).

Что здесь происходит:
  1. загрузка архива фиксированной версии с Mendeley Data;
  2. приведение названий колонок к единому виду — в разных версиях набора они
     записаны по-разному («Date Time» и «Date_Time», «Coin Type» и «Coin_Type»);
  3. определение часового пояса меток времени;
  4. перевод времени в UTC, удаление дублей, таблица этапов очистки.

Почему часовой пояс определяется, а не принимается на веру. В описании набора
пояс не указан, а ошибка в один час сдвигает новость на соседнюю свечу — ровно
та утечка будущего, которую мы чиним. Зато к каждой новости в наборе приложена
15-минутная свеча Binance. Если найти в архиве Binance свечу с теми же ценами
открытия, максимума, минимума и закрытия, станет известно её точное время в UTC.
Разница между этим временем и меткой новости и даёт сдвиг пояса.

Цены из набора используются только для этой сверки. В признаки модели они не
попадают: они привязаны ко времени публикации и могут содержать информацию
из будущего относительно него.
"""
from __future__ import annotations

import io
import re
import unicodedata
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests

from cryptonews.data import binance

DATASET_ID = "3c3xtxtfb6"
ZIP_URL = "https://data.mendeley.com/public-api/zip/{dataset_id}/download/{version}"

# Канонические имена колонок и варианты, которые встречаются в версиях набора
COLUMN_ALIASES = {
    "url": ("url", "link"),
    "title": ("title", "headline"),
    "date_time": ("date_time", "datetime", "date", "published", "published_at"),
    "coin_type": ("coin_type", "coin", "currency"),
    "open": ("open",),
    "high": ("high",),
    "low": ("low",),
    "close": ("close",),
}
COIN_TO_SYMBOL = {"bitcoin": "BTCUSDT", "ethereum": "ETHUSDT"}

# Кандидаты сдвига от UTC в минутах: целые и получасовые пояса от −12 до +14
OFFSET_CANDIDATES = list(range(-12 * 60, 14 * 60 + 1, 30))
CANDLE_MINUTES = 15
MATCH_WINDOW_HOURS = 18


# --------------------------------------------------------------------------- загрузка

def find_csv(raw_dir: Path) -> Path | None:
    """Самый большой CSV в папке — сам набор (рядом могут лежать служебные файлы)."""
    csvs = sorted(raw_dir.glob("*.csv"), key=lambda p: p.stat().st_size, reverse=True)
    return csvs[0] if csvs else None


def download(raw_dir: str | Path, version: int, timeout: int = 300) -> tuple[Path, str]:
    """Возвращает путь к CSV набора и откуда он взят.

    Если CSV уже лежит в папке (скачан раньше или положен вручную), сеть не нужна.
    Из архива извлекаются только CSV-файлы: архив чужой, распаковывать всё подряд нельзя.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    existing = find_csv(raw_dir)
    if existing is not None:
        return existing, "уже в папке"

    url = ZIP_URL.format(dataset_id=DATASET_ID, version=version)
    response = requests.get(url, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(
            f"Mendeley ответил кодом {response.status_code} на {url}. "
            f"Скачайте набор вручную со страницы https://data.mendeley.com/datasets/{DATASET_ID}/{version} "
            f"и положите CSV в папку {raw_dir}"
        )
    zip_path = raw_dir / f"cryptovision_v{version}.zip"
    zip_path.write_bytes(response.content)

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        for member in archive.namelist():
            if member.lower().endswith(".csv") and not member.startswith("__MACOSX"):
                target = raw_dir / Path(member).name
                target.write_bytes(archive.read(member))
    found = find_csv(raw_dir)
    if found is None:
        raise RuntimeError(f"В архиве {zip_path} нет CSV-файлов")
    return found, f"скачан с {url}"


# --------------------------------------------------------------------------- чтение

def normalize_name(name: str) -> str:
    return re.sub(r"[\s\-]+", "_", str(name).strip().lower())


def normalize_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Приводит колонки к каноническим именам. Возвращает таблицу и найденное соответствие."""
    normalized = {normalize_name(c): c for c in frame.columns}
    mapping: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in normalized:
                mapping[canonical] = normalized[alias]
                break
    missing = [c for c in ("url", "title", "date_time") if c not in mapping]
    if missing:
        raise KeyError(f"В наборе не найдены колонки: {missing}. Есть: {list(frame.columns)}")
    out = pd.DataFrame({canonical: frame[original] for canonical, original in mapping.items()})
    for column in ("open", "high", "low", "close"):
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out, mapping


TZ_SUFFIX = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$")


def parse_times(values: pd.Series) -> tuple[pd.Series, float]:
    """Разбирает метки времени. Возвращает время «как записано» и долю меток с поясом.

    Метки с указанным поясом переводятся в UTC; метки без пояса остаются как есть —
    их пояс определяется дальше сверкой с Binance.
    """
    text = values.astype("string").str.strip()
    share_with_tz = float(text.str.contains(TZ_SUFFIX, na=False).mean())
    parsed = pd.to_datetime(text, errors="coerce", utc=True, format="mixed")
    return parsed.dt.tz_localize(None), share_with_tz


# --------------------------------------------------------------------------- часовой пояс

@dataclass
class TimezoneReport:
    sample_months: list[str] = field(default_factory=list)
    records_checked: int = 0
    records_matched: int = 0
    best_offset_minutes: int | None = None
    best_rule: str | None = None
    best_share: float = 0.0
    runner_up_share: float = 0.0
    candidates: list[dict] = field(default_factory=list)
    per_month: list[dict] = field(default_factory=list)
    months_agreeing: float = 0.0
    verdict: str = ""

    @property
    def offset_label(self) -> str:
        if self.best_offset_minutes is None:
            return "не определён"
        sign = "+" if self.best_offset_minutes >= 0 else "−"
        hours, minutes = divmod(abs(self.best_offset_minutes), 60)
        return f"UTC{sign}{hours:02d}:{minutes:02d}"

    def as_dict(self) -> dict:
        return {**self.__dict__, "offset_label": self.offset_label}


def pick_sample_months(news: pd.DataFrame, count: int = 16) -> list[str]:
    """Месяцы для сверки, равномерно по всему периоду — так видны и лето, и зима."""
    months = sorted(news["date_time"].dt.to_period("M").astype(str).unique())
    months = [m for m in months if m >= "2017-09"]  # раньше у Binance нет пар
    if len(months) <= count:
        return months
    positions = np.linspace(0, len(months) - 1, count).round().astype(int)
    return [months[i] for i in sorted(set(positions))]


def match_candles(news: pd.DataFrame, candles: pd.DataFrame) -> pd.DataFrame:
    """Находит для каждой новости свечу Binance с теми же четырьмя ценами.

    Возвращает разницу в минутах между временем свечи (UTC) и меткой новости.
    Берутся только однозначные совпадения внутри окна ±18 часов.
    """
    def keyed(frame):
        out = frame.copy()
        for column in ("open", "high", "low", "close"):
            out[f"k_{column}"] = frame[column].round(6)
        return out

    left = keyed(news.dropna(subset=["open", "high", "low", "close"]))
    left = left.reset_index(drop=True).reset_index(names="row")
    right = keyed(candles.reset_index(names="candle_time"))
    right["candle_time"] = right["candle_time"].dt.tz_localize(None)
    keys = ["k_open", "k_high", "k_low", "k_close"]
    merged = left[["row", "date_time"] + keys].merge(right[["candle_time"] + keys], on=keys)
    merged["delta_min"] = (merged["candle_time"] - merged["date_time"]).dt.total_seconds() / 60
    merged = merged[merged["delta_min"].abs() <= MATCH_WINDOW_HOURS * 60]
    unique = merged[merged.groupby("row")["row"].transform("size") == 1]
    return unique[["row", "date_time", "candle_time", "delta_min"]]


def score_offsets(delta_min: pd.Series) -> pd.DataFrame:
    """Доля совпадений, которую объясняет каждая пара «сдвиг пояса, правило выбора свечи».

    Если метка записана в поясе UTC+X, то время свечи минус метка равно d, а время
    свечи минус настоящее время UTC равно d + X. Набор мог брать свечу, в которую
    попадает новость (−15 < d + X ≤ 0), или следующую за ней (0 ≤ d + X < 15).
    """
    d = delta_min.to_numpy()
    rows = []
    for offset in OFFSET_CANDIDATES:
        shifted = d + offset
        rows.append({"offset_minutes": offset, "rule": "свеча, содержащая новость",
                     "share": float(((shifted > -CANDLE_MINUTES) & (shifted <= 0)).mean())})
        rows.append({"offset_minutes": offset, "rule": "следующая свеча",
                     "share": float(((shifted >= 0) & (shifted < CANDLE_MINUTES)).mean())})
    return pd.DataFrame(rows).sort_values("share", ascending=False).reset_index(drop=True)


def detect_timezone(news: pd.DataFrame, candles_by_symbol: dict[str, pd.DataFrame],
                    sample_months: list[str], min_share: float = 0.9,
                    min_month_agreement: float = 0.9) -> TimezoneReport:
    report = TimezoneReport(sample_months=sample_months)
    matches = []
    for symbol, candles in candles_by_symbol.items():
        coins = [c for c, s in COIN_TO_SYMBOL.items() if s == symbol]
        subset = news[news["coin_type"].astype(str).str.strip().str.lower().isin(coins)]
        month = subset["date_time"].dt.to_period("M").astype(str)
        day = subset["date_time"].dt.day
        # края месяца пропускаем: со сдвигом пояса новость могла уйти в соседний месяц,
        # свечей которого мы не скачивали
        subset = subset[month.isin(sample_months) & day.between(2, 27)]
        report.records_checked += len(subset)
        if len(subset) and len(candles):
            matches.append(match_candles(subset, candles))

    if not matches or sum(len(m) for m in matches) == 0:
        report.verdict = "Совпадений с ценами Binance не найдено — пояс по ценам не определить"
        return report
    matched = pd.concat(matches, ignore_index=True)
    report.records_matched = len(matched)

    scores = score_offsets(matched["delta_min"])
    report.candidates = scores.head(6).to_dict("records")
    best = scores.iloc[0]
    report.best_offset_minutes = int(best["offset_minutes"])
    report.best_rule = str(best["rule"])
    report.best_share = float(best["share"])
    report.runner_up_share = float(scores.iloc[1]["share"])

    matched["month"] = matched["date_time"].dt.to_period("M").astype(str)
    agree = []
    for month, group in matched.groupby("month"):
        month_scores = score_offsets(group["delta_min"])
        top = month_scores.iloc[0]
        same = int(top["offset_minutes"]) == report.best_offset_minutes
        agree.append(same)
        report.per_month.append({"month": month, "matched": len(group),
                                 "best_offset_minutes": int(top["offset_minutes"]),
                                 "share": round(float(top["share"]), 3), "agrees": same})
    report.months_agreeing = float(np.mean(agree)) if agree else 0.0

    months_clear = all(m["share"] >= min_share for m in report.per_month)
    if report.months_agreeing < min_month_agreement and months_clear:
        # внутри каждого месяца сдвиг однозначен, но между месяцами разный
        offsets = sorted({m["best_offset_minutes"] for m in report.per_month})
        report.verdict = (f"Сдвиг меняется от месяца к месяцу ({offsets} мин) — похоже на летнее "
                          "время; нужен пояс с переходами, а не постоянный сдвиг")
    elif report.best_share < min_share:
        report.verdict = (f"Ненадёжно: лучший вариант объясняет только {report.best_share:.1%} совпадений")
    elif report.months_agreeing < min_month_agreement:
        report.verdict = (f"Ненадёжно: с общим сдвигом согласны только {report.months_agreeing:.0%} месяцев")
    else:
        report.verdict = (f"Надёжно: {report.offset_label}, {report.best_rule}; "
                          f"объясняет {report.best_share:.1%} совпадений, "
                          f"согласны {report.months_agreeing:.0%} месяцев")
    return report


def is_reliable(report: TimezoneReport) -> bool:
    return report.verdict.startswith("Надёжно")


# --------------------------------------------------------------------------- очистка

def normalize_title(title: str) -> str:
    """Заголовок для поиска повторов: нижний регистр, без знаков препинания и лишних пробелов."""
    text = unicodedata.normalize("NFKC", str(title)).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def dedup_titles(frame: pd.DataFrame, window_hours: int = 24) -> pd.DataFrame:
    """Убирает повторы одного заголовка в пределах окна, оставляя самую раннюю публикацию.

    Тот же заголовок через неделю повтором не считается: регулярные рубрики вроде
    «Bitcoin price analysis» выходят с одним и тем же названием.
    """
    ordered = frame.assign(_norm=frame["title"].map(normalize_title))
    ordered = ordered.sort_values(["_norm", "published_utc"])
    keep = np.ones(len(ordered), dtype=bool)
    norms = ordered["_norm"].to_numpy()
    times = ordered["published_utc"].to_numpy()
    window = np.timedelta64(window_hours, "h")
    cluster_start = None
    for i in range(len(ordered)):
        if i > 0 and norms[i] == norms[i - 1] and times[i] - cluster_start <= window:
            keep[i] = False
        else:
            cluster_start = times[i]
    return ordered[keep].drop(columns="_norm").sort_values("published_utc")


def source_of(url: str) -> str:
    host = urlparse(str(url)).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def clean(frame: pd.DataFrame, offset_minutes: int, dedup_window_hours: int = 24
          ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Перевод в UTC и очистка. Возвращает чистую таблицу и таблицу этапов очистки."""
    stages = [("Записей в наборе", len(frame))]

    work = frame[frame["date_time"].notna()].copy()
    stages.append(("С корректной меткой времени", len(work)))

    work["title"] = work["title"].astype("string").str.strip()
    work["url"] = work["url"].astype("string").str.strip()
    work = work[(work["title"].str.len() > 0) & (work["url"].str.len() > 0)]
    stages.append(("С заголовком и адресом", len(work)))

    work["published_utc"] = (work["date_time"] - pd.Timedelta(minutes=offset_minutes)).dt.tz_localize("UTC")
    work = work.sort_values("published_utc").drop_duplicates("url", keep="first")
    stages.append(("Без повторов адреса", len(work)))

    work = dedup_titles(work, window_hours=dedup_window_hours)
    stages.append((f"Без повторов заголовка в пределах {dedup_window_hours} ч", len(work)))

    work["source"] = work["url"].map(source_of)
    work["coin_type"] = work.get("coin_type", pd.Series(index=work.index, dtype="string"))
    columns = ["published_utc", "title", "url", "source", "coin_type"]
    result = work[columns].reset_index(drop=True)

    table = pd.DataFrame(stages, columns=["Этап", "Записей"])
    table["Убрано на этапе"] = (-table["Записей"].diff()).fillna(0).astype(int)
    return result, table
