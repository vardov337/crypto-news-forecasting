"""Общие правила очистки новостей для обоих языков (задачи 3.5 и 4.2).

Одни и те же правила для англо- и русскоязычных новостей — условие сопоставимости
признаков: иначе разница между языками могла бы оказаться разницей в очистке.
"""
from __future__ import annotations

import html
import re
import unicodedata
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from cryptonews.data import coins

OUTPUT_COLUMNS = ["published_utc", "title", "url", "source", "mentions_btc", "mentions_eth"]


ENTITY = re.compile(r"&(?:#\d+|#x[0-9a-fA-F]+|[a-zA-Z]{2,8});")


def unescape_title(title) -> str:
    """Заголовок без мнемоник HTML (&amp;, &#8217; …) и лишних пробелов.

    Модель тональности должна видеть тот же текст, что и читатель: «S&amp;P 500»
    вместо «S&P 500» — шум для токенизатора. Повтор операции ничего не меняет
    для заголовков, где мнемоник нет.
    """
    return re.sub(r"\s+", " ", html.unescape(str(title))).strip()


def normalize_title(title: str) -> str:
    """Заголовок для поиска повторов: нижний регистр, без знаков препинания и лишних пробелов."""
    text = unicodedata.normalize("NFKC", str(title)).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def dedup_titles(frame: pd.DataFrame, window_hours: int = 24) -> pd.DataFrame:
    """Убирает повторы одного заголовка в пределах окна, оставляя самую раннюю публикацию.

    Тот же заголовок через неделю повтором не считается: регулярные рубрики вроде
    «Bitcoin price analysis» или «Главное за неделю» выходят с одним и тем же названием.
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


# Если одно и то же время суток стоит у трети записей источника, это не время
# публикации, а заглушка: у источника есть только дата.
PLACEHOLDER_SHARE = 0.3
MIN_SOURCE_RECORDS = 50


def time_precision(frame: pd.DataFrame, column: str = "published_utc") -> pd.DataFrame:
    """Насколько точно указано время публикации у каждого источника.

    Показывает долю меток ровно в 00:00:00, долю меток на целом часе и на целой
    минуте и самое частое время суток. Нужна для раздела о данных и для проверки
    правила `imprecise_mask`.
    """
    times = frame[column]
    clock = times.dt.strftime("%H:%M:%S")
    rows = []
    for source, index in frame.groupby("source").groups.items():
        t, c = times.loc[index], clock.loc[index]
        top = c.value_counts()
        rows.append({
            "Источник": source, "Записей": len(index),
            "Время 00:00:00": round(float((c == "00:00:00").mean()), 4),
            "На целом часе": round(float((t.dt.minute.eq(0) & t.dt.second.eq(0)).mean()), 4),
            "На целой минуте": round(float(t.dt.second.eq(0).mean()), 4),
            "Самое частое время": top.index[0],
            "Его доля": round(float(top.iloc[0] / len(index)), 4),
        })
    return pd.DataFrame(rows).sort_values("Записей", ascending=False).reset_index(drop=True)


def imprecise_mask(frame: pd.DataFrame, column: str = "published_utc") -> pd.Series:
    """Записи без настоящего времени публикации.

    Такие метки опасны: новость с одной датой попадает на полночь, то есть на
    много часов раньше настоящей публикации, — модель «увидела» бы её до того,
    как она вышла. Поэтому исключаются:
      * метки ровно 00:00:00;
      * у источника, где одно время суток стоит у трети записей и больше, —
        все записи с этим временем (заглушка вместо времени).

    Проверять нужно время в том виде, в каком оно записано в источнике (column):
    дата без времени выглядит как полночь именно в его записи, а не после
    перевода в UTC.
    """
    clock = frame[column].dt.strftime("%H:%M:%S")
    mask = clock.eq("00:00:00")
    for source, index in frame.groupby("source").groups.items():
        if len(index) < MIN_SOURCE_RECORDS:
            continue
        top = clock.loc[index].value_counts()
        if top.iloc[0] / len(index) >= PLACEHOLDER_SHARE:
            mask |= frame["source"].eq(source) & clock.eq(top.index[0])
    return mask


def batch_uploads(frame: pd.DataFrame, min_records: int = 10, max_gap_seconds: float = 60.0,
                  column: str = "published_utc") -> tuple[pd.Series, pd.DataFrame]:
    """Пакетные загрузки: серии из min_records и более записей одного источника, в которых
    каждая следующая запись отстоит от предыдущей меньше чем на max_gap_seconds.

    Редакция не выпускает десять материалов подряд с промежутками меньше минуты (обычный темп
    источника в наших данных — около новости в час). Так выглядят загрузка архива и массовое
    пересохранение материалов на сайте: время у таких записей — момент загрузки, настоящее
    время публикации неизвестно. Частный случай — десять и больше записей с одной секундой.

    Возвращает маску записей (по индексу frame) и таблицу серий."""
    times = pd.to_datetime(frame[column], utc=True)
    seconds = ((times - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)).to_numpy(float)
    titles = frame["title"].to_numpy() if "title" in frame else np.full(len(frame), None)
    flags = np.zeros(len(frame), dtype=bool)
    rows = []
    for source, positions in frame.groupby("source", sort=True).indices.items():
        positions = positions[~np.isnan(seconds[positions])]
        if len(positions) < min_records:
            continue
        order = positions[np.argsort(seconds[positions], kind="stable")]
        run = np.concatenate([[0], np.cumsum(np.diff(seconds[order]) >= max_gap_seconds)])
        sizes = np.bincount(run)
        for run_id in np.flatnonzero(sizes >= min_records):
            members = order[run == run_id]
            flags[members] = True
            rows.append({"Источник": source, "Первая метка": times.iloc[members[0]],
                         "Последняя метка": times.iloc[members[-1]], "Записей": int(len(members)),
                         "Пример заголовка": titles[members[0]]})
    table = pd.DataFrame(rows, columns=["Источник", "Первая метка", "Последняя метка", "Записей", "Пример заголовка"])
    table = table.sort_values(["Записей", "Первая метка"], ascending=[False, True]).reset_index(drop=True)
    return pd.Series(flags, index=frame.index, name="batch_upload"), table


def drop_sources(frame: pd.DataFrame, sources) -> tuple[pd.DataFrame, pd.Series]:
    """Убирает записи источников из списка; возвращает таблицу и число убранных по источникам."""
    sources = list(sources or [])
    mask = frame["source"].isin(sources).to_numpy()
    counts = frame.loc[mask, "source"].value_counts().reindex(sources, fill_value=0)
    return frame[~mask].reset_index(drop=True), counts


def clean_utc(frame: pd.DataFrame, dedup_window_hours: int = 24,
              first_stage: str = "Записей в наборе",
              precision_column: str = "published_utc") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Очистка таблицы, в которой время уже в UTC (колонка published_utc).

    precision_column — колонка со временем в записи источника, по которой ищутся
    метки без точного времени (см. imprecise_mask).

    Возвращает чистую таблицу со стандартными колонками и таблицу этапов очистки.
    """
    stages = [(first_stage, len(frame))]

    work = frame[frame["published_utc"].notna()].copy()
    stages.append(("С корректной меткой времени", len(work)))

    titles = work["title"].astype("string")
    work["title"] = titles.where(titles.isna(), titles.fillna("").map(unescape_title)).astype("string")
    work["url"] = work["url"].astype("string").str.strip()
    work = work[(work["title"].fillna("").str.len() > 0) & (work["url"].fillna("").str.len() > 0)]
    stages.append(("С заголовком и адресом", len(work)))

    work["source"] = work["url"].map(source_of)
    work = work[~imprecise_mask(work, column=precision_column)]
    stages.append(("С точным временем публикации (без меток 00:00:00 и заглушек)", len(work)))

    work = work.sort_values("published_utc").drop_duplicates("url", keep="first")
    stages.append(("Без повторов адреса", len(work)))

    work = dedup_titles(work, window_hours=dedup_window_hours)
    stages.append((f"Без повторов заголовка в пределах {dedup_window_hours} ч", len(work)))

    work = coins.tag(work)
    extra = [c for c in ("coin_type",) if c in work.columns]
    result = work[OUTPUT_COLUMNS + extra].reset_index(drop=True)

    table = pd.DataFrame(stages, columns=["Этап", "Записей"])
    table["Убрано на этапе"] = (-table["Записей"].diff()).fillna(0).astype(int)
    return result, table


def month_label(times: pd.Series) -> pd.Series:
    """Календарный месяц метки UTC в виде «2025-07»."""
    return times.dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period("M").astype(str)


def sources_span(news: pd.DataFrame, recent_months: int = 12) -> pd.DataFrame:
    """Когда у каждого источника первая и последняя новость.

    Нужна для выбора границ выборки: если источник обрывается раньше других,
    последний месяц набора собран не полностью. Доля за последние recent_months
    месяцев показывает, какие источники активны в конце набора.
    """
    last = news["published_utc"].max()
    recent = news["published_utc"] > last - pd.DateOffset(months=recent_months)
    rows = []
    for source, group in news.groupby("source"):
        rows.append({
            "Источник": source, "Новостей": len(group),
            "Первая": group["published_utc"].min(), "Последняя": group["published_utc"].max(),
            f"За последние {recent_months} мес.": int(recent.loc[group.index].sum()),
        })
    table = pd.DataFrame(rows).sort_values("Новостей", ascending=False).set_index("Источник")
    recent_col = f"За последние {recent_months} мес."
    table[f"Доля за последние {recent_months} мес."] = (
        table[recent_col] / max(int(recent.sum()), 1)).round(4)
    return table


def by_source_month(news: pd.DataFrame) -> pd.DataFrame:
    """Число новостей каждого источника по месяцам — видно, когда источник
    появляется в наборе и когда обрывается."""
    table = pd.crosstab(month_label(news["published_utc"]), news["source"])
    table.index.name = "Месяц"
    table.columns.name = None
    return table


def summary_tables(news: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Разбивки для раздела о данных: по годам, по месяцам, по источникам, по монетам,
    точность времени публикации и период покрытия по источникам."""
    by_year = news.groupby(news["published_utc"].dt.year).size().rename("Новостей").to_frame()
    by_year.index.name = "Год"
    by_month = news.groupby(month_label(news["published_utc"])).size().rename("Новостей").to_frame()
    by_month.index.name = "Месяц"
    by_source = news["source"].value_counts().rename("Новостей").to_frame()
    by_source.index.name = "Источник"
    return {"by_year": by_year, "by_month": by_month, "by_source": by_source,
            "by_coin": coins.coverage_table(news), "time_precision": time_precision(news),
            "sources_span": sources_span(news), "by_source_month": by_source_month(news)}
