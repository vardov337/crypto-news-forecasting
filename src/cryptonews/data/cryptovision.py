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
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from cryptonews.data import news as news_rules

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
# Кандидаты сдвига от UTC в минутах: целые и получасовые пояса от −12 до +14
OFFSET_CANDIDATES = list(range(-12 * 60, 14 * 60 + 1, 30))
CANDLE_MINUTES = 15
MATCH_WINDOW_HOURS = 18
PRICE_DECIMALS = 2          # шаг цены у BTCUSDT и ETHUSDT — один цент
MIN_MATCHES_PER_MONTH = 20  # месяцы с меньшим числом совпадений не участвуют в проверке согласия


# --------------------------------------------------------------------------- загрузка

DATA_SUFFIXES = (".csv", ".xlsx")


def data_files(raw_dir: Path) -> list[Path]:
    """Таблицы набора в папке: CSV или Excel, без служебных файлов macOS."""
    return sorted(p for p in raw_dir.iterdir()
                  if p.suffix.lower() in DATA_SUFFIXES and not p.name.startswith("._"))


def _extract_tables(archive: zipfile.ZipFile, raw_dir: Path, depth: int = 0) -> list[Path]:
    """Достаёт из архива только таблицы. Архив чужой, поэтому распаковывать всё подряд нельзя:
    берутся файлы .csv и .xlsx, по имени без папок; вложенный архив раскрывается на один уровень."""
    extracted = []
    for member in archive.namelist():
        name = Path(member).name
        if not name or member.startswith("__MACOSX") or name.startswith("._"):
            continue
        if name.lower().endswith(DATA_SUFFIXES):
            target = raw_dir / name
            if not target.exists():
                target.write_bytes(archive.read(member))
                extracted.append(target)
        elif name.lower().endswith(".zip") and depth == 0:
            with zipfile.ZipFile(io.BytesIO(archive.read(member))) as inner:
                extracted += _extract_tables(inner, raw_dir, depth=1)
    return extracted


def extract_archives(raw_dir: Path) -> list[Path]:
    """Распаковывает архивы, положенные в папку вручную (например, «Download All» с Mendeley)."""
    extracted = []
    for zip_path in sorted(raw_dir.glob("*.zip")):
        with zipfile.ZipFile(zip_path) as archive:
            extracted += _extract_tables(archive, raw_dir)
    return extracted


def manual_download_hint(raw_dir: Path, version: int) -> str:
    return (
        "Скачайте набор вручную: откройте в браузере "
        f"https://data.mendeley.com/datasets/{DATASET_ID}/{version}, нажмите «Download All» "
        f"и положите скачанный архив (распаковывать не нужно) в папку Google Диска {raw_dir}"
    )


def obtain(raw_dir: str | Path, version: int, timeout: int = 300) -> tuple[list[Path], str]:
    """Находит таблицы набора в папке, при необходимости распаковав или скачав их.

    Порядок: таблицы уже лежат в папке → архив, положенный вручную → скачивание с Mendeley.
    Mendeley с осени 2026 года отвечает отказом (403) на автоматическое скачивание,
    поэтому основной путь — архив, скачанный в браузере.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    extracted = extract_archives(raw_dir)
    files = data_files(raw_dir)
    if files:
        return files, ("распакованы из архива в папке" if extracted else "уже в папке")

    url = ZIP_URL.format(dataset_id=DATASET_ID, version=version)
    try:
        response = requests.get(url, timeout=timeout)
    except requests.RequestException as error:
        raise SystemExit(f"Не удалось связаться с Mendeley ({error}). {manual_download_hint(raw_dir, version)}")
    if response.status_code != 200:
        raise SystemExit(f"Mendeley ответил кодом {response.status_code}. {manual_download_hint(raw_dir, version)}")
    (raw_dir / f"cryptovision_v{version}.zip").write_bytes(response.content)
    extract_archives(raw_dir)
    files = data_files(raw_dir)
    if not files:
        raise SystemExit(f"В архиве с Mendeley нет таблиц .csv или .xlsx. {manual_download_hint(raw_dir, version)}")
    return files, f"скачаны с {url}"


def load_raw(files: list[Path]) -> tuple[pd.DataFrame, list[dict]]:
    """Читает все таблицы набора с нужными колонками и объединяет их.

    Возвращает таблицу с каноническими колонками и сведения о каждом файле:
    сколько строк и использован ли он (файлы без нужных колонок, например описание
    полей, пропускаются).
    """
    frames, info = [], []
    for path in files:
        raw = (pd.read_csv(path, low_memory=False) if path.suffix.lower() == ".csv"
               else pd.read_excel(path))
        try:
            frame, mapping = normalize_columns(raw)
        except KeyError:
            info.append({"file": path.name, "rows": len(raw), "used": False})
            continue
        frames.append(frame)
        info.append({"file": path.name, "rows": len(raw), "used": True, "columns": mapping})
    if not frames:
        names = ", ".join(i["file"] for i in info)
        raise SystemExit(f"Ни в одном файле ({names}) нет колонок с адресом, заголовком и временем")
    return pd.concat(frames, ignore_index=True), info


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
    records_imprecise: int = 0      # исключены из сверки: время без часа и минут
    records_checked: int = 0
    records_matched: int = 0
    best_offset_minutes: int | None = None
    best_share: float = 0.0          # доля совпадений в пределах одной свечи от метки
    runner_up_share: float = 0.0     # то же для следующего по силе сдвига
    best_rule: str | None = None     # как набор выбирал свечу — для описания, на вывод не влияет
    rule_shares: dict = field(default_factory=dict)
    outside_before: float = 0.0      # свеча раньше метки больше чем на 15 минут
    outside_after: float = 0.0       # свеча позже метки больше чем на 15 минут
    candidates: list[dict] = field(default_factory=list)
    top_deltas: list[dict] = field(default_factory=list)
    per_month: list[dict] = field(default_factory=list)
    per_source: list[dict] = field(default_factory=list)
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
    months = sorted(news["date_time"].dropna().dt.to_period("M").astype(str).unique())
    months = [m for m in months if m >= "2017-09"]  # раньше у Binance нет пар
    if len(months) <= count:
        return months
    positions = np.linspace(0, len(months) - 1, count).round().astype(int)
    return [months[i] for i in sorted(set(positions))]


def match_candles(news: pd.DataFrame, candles: pd.DataFrame) -> pd.DataFrame:
    """Находит для каждой новости свечу Binance с теми же четырьмя ценами.

    Возвращает разницу в минутах между временем свечи (UTC) и меткой новости.
    Берутся только однозначные совпадения внутри окна ±18 часов. Монету новости
    знать не нужно: цены биткоина и эфира не совпадают никогда, поэтому свечи
    обоих активов можно искать вместе.
    """
    def keyed(frame):
        out = frame.copy()
        for column in ("open", "high", "low", "close"):
            out[f"k_{column}"] = frame[column].round(PRICE_DECIMALS)
        return out

    left = keyed(news.dropna(subset=["open", "high", "low", "close"]))
    left = left.reset_index(drop=True).reset_index(names="row")
    if "url" not in left:
        left["url"] = ""
    right = keyed(candles.reset_index(names="candle_time"))
    right["candle_time"] = right["candle_time"].dt.tz_localize(None)
    keys = ["k_open", "k_high", "k_low", "k_close"]
    merged = left[["row", "date_time", "url"] + keys].merge(right[["candle_time"] + keys], on=keys)
    merged["delta_min"] = (merged["candle_time"] - merged["date_time"]).dt.total_seconds() / 60
    merged = merged[merged["delta_min"].abs() <= MATCH_WINDOW_HOURS * 60]
    unique = merged[merged.groupby("row")["row"].transform("size") == 1]
    return unique[["row", "date_time", "url", "candle_time", "delta_min"]]


def window_scores(delta_min: pd.Series) -> pd.DataFrame:
    """Для каждого сдвига X — доля совпадений, где свеча лежит в пределах одной свечи
    (±15 минут) от метки, переведённой в UTC: |d + X| < 15.

    Это и есть проверка пояса. Правило, по которому набор выбирал свечу (та, в которую
    попадает новость, следующая или ближайшая), на ответ не влияет: при любом из них
    свеча оказывается в пределах 15 минут. Окна разных сдвигов не пересекаются,
    потому что кандидаты отличаются минимум на 30 минут.
    """
    d = delta_min.to_numpy()
    rows = [{"offset_minutes": x, "share": float((np.abs(d + x) < CANDLE_MINUTES).mean())}
            for x in OFFSET_CANDIDATES]
    return pd.DataFrame(rows).sort_values("share", ascending=False).reset_index(drop=True)


def rule_shares(delta_min: pd.Series, offset: int) -> dict[str, float]:
    """Какое правило выбора свечи объясняет совпадения при найденном сдвиге — для описания набора."""
    shifted = delta_min.to_numpy() + offset
    half = CANDLE_MINUTES / 2
    return {
        "свеча, содержащая новость": float(((shifted > -CANDLE_MINUTES) & (shifted <= 0)).mean()),
        "следующая свеча": float(((shifted >= 0) & (shifted < CANDLE_MINUTES)).mean()),
        "ближайшая свеча": float(((shifted >= -half) & (shifted <= half)).mean()),
    }


def detect_timezone(news: pd.DataFrame, candles_by_symbol: dict[str, pd.DataFrame],
                    sample_months: list[str], min_share: float = 0.9,
                    min_month_agreement: float = 0.9, min_source_share: float = 0.8,
                    min_matches: int = MIN_MATCHES_PER_MONTH) -> TimezoneReport:
    report = TimezoneReport(sample_months=sample_months)
    dated = news[news["date_time"].notna()].copy()
    # записи без точного времени (одна дата) в сверке не участвуют: их всё равно
    # исключит очистка, а здесь они только мешали бы
    dated["source"] = dated["url"].map(news_rules.source_of)
    imprecise = news_rules.imprecise_mask(dated, column="date_time")
    report.records_imprecise = int(imprecise.sum())
    dated = dated[~imprecise]
    month = dated["date_time"].dt.to_period("M").astype(str)
    day = dated["date_time"].dt.day
    # края месяца пропускаем: со сдвигом пояса новость могла уйти в соседний месяц,
    # свечей которого мы не скачивали
    subset = dated[month.isin(sample_months) & day.between(2, 27)]
    report.records_checked = len(subset)

    frames = [c for c in candles_by_symbol.values() if len(c)]
    matched = match_candles(subset, pd.concat(frames)) if frames and len(subset) else pd.DataFrame()
    if matched.empty:
        report.verdict = "Совпадений с ценами Binance не найдено — пояс по ценам не определить"
        return report
    report.records_matched = len(matched)

    counts = matched["delta_min"].round().value_counts().head(10)
    report.top_deltas = [{"delta_min": int(k), "count": int(v)} for k, v in counts.items()]

    scores = window_scores(matched["delta_min"])
    report.candidates = scores.head(5).to_dict("records")
    report.best_offset_minutes = int(scores.iloc[0]["offset_minutes"])
    report.best_share = float(scores.iloc[0]["share"])
    report.runner_up_share = float(scores.iloc[1]["share"])
    report.rule_shares = rule_shares(matched["delta_min"], report.best_offset_minutes)
    report.best_rule = max(report.rule_shares, key=report.rule_shares.get)
    shifted = matched["delta_min"] + report.best_offset_minutes
    report.outside_before = float((shifted <= -CANDLE_MINUTES).mean())
    report.outside_after = float((shifted >= CANDLE_MINUTES).mean())

    def best_for(group: pd.DataFrame) -> tuple[int, float, float]:
        top = window_scores(group["delta_min"]).iloc[0]
        at_global = float((np.abs(group["delta_min"] + report.best_offset_minutes) < CANDLE_MINUTES).mean())
        return int(top["offset_minutes"]), float(top["share"]), at_global

    agree = []
    matched = matched.assign(month=matched["date_time"].dt.to_period("M").astype(str))
    for month_name, group in matched.groupby("month"):
        if len(group) < min_matches:
            continue
        offset, share, _ = best_for(group)
        agree.append(offset == report.best_offset_minutes)
        report.per_month.append({"month": month_name, "matched": len(group), "best_offset_minutes": offset,
                                 "share": round(share, 3), "agrees": agree[-1]})
    report.months_agreeing = float(np.mean(agree)) if agree else 0.0

    # По источникам: если какой-то сайт пишет время в своём поясе, это видно здесь
    matched = matched.assign(source=matched["url"].map(news_rules.source_of))
    for source, group in matched.groupby("source"):
        if len(group) < 5 * min_matches:
            continue
        offset, share, at_global = best_for(group)
        report.per_source.append({"source": source, "matched": len(group), "best_offset_minutes": offset,
                                  "share": round(share, 3), "share_at_global": round(at_global, 3),
                                  "agrees": offset == report.best_offset_minutes and at_global >= min_source_share})

    months_clear = bool(report.per_month) and all(m["share"] >= min_share for m in report.per_month)
    bad_sources = [s["source"] for s in report.per_source if not s["agrees"]]
    if report.months_agreeing < min_month_agreement and months_clear:
        # внутри каждого месяца сдвиг однозначен, но между месяцами разный
        offsets = sorted({m["best_offset_minutes"] for m in report.per_month})
        report.verdict = (f"Сдвиг меняется от месяца к месяцу ({offsets} мин) — похоже на летнее "
                          "время; нужен пояс с переходами, а не постоянный сдвиг")
    elif report.best_share < min_share:
        report.verdict = (f"Ненадёжно: в пределах одной свечи от метки только {report.best_share:.1%} "
                          "совпадений")
    elif report.months_agreeing < min_month_agreement:
        report.verdict = f"Ненадёжно: с общим сдвигом согласны только {report.months_agreeing:.0%} месяцев"
    elif bad_sources:
        report.verdict = f"Ненадёжно: источники {bad_sources} расходятся с общим сдвигом"
    else:
        report.verdict = (f"Надёжно: {report.offset_label}; в пределах одной свечи от метки "
                          f"{report.best_share:.1%} совпадений, согласны {report.months_agreeing:.0%} месяцев "
                          f"и все крупные источники; набор брал «{report.best_rule}»")
    return report


def is_reliable(report: TimezoneReport) -> bool:
    return report.verdict.startswith("Надёжно")


# --------------------------------------------------------------------------- очистка

def clean(frame: pd.DataFrame, offset_minutes: int, dedup_window_hours: int = 24
          ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Перевод в UTC и очистка по общим правилам для обоих языков (модуль news).

    Метка записана в поясе UTC + offset, значит время в UTC — это метка минус offset.
    """
    work = frame.copy()
    work["published_utc"] = (work["date_time"] - pd.Timedelta(minutes=offset_minutes)).dt.tz_localize("UTC")
    return news_rules.clean_utc(work, dedup_window_hours=dedup_window_hours, precision_column="date_time")
