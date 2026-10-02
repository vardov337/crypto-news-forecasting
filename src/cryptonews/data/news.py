"""Общие правила очистки новостей для обоих языков (задачи 3.5 и 4.2).

Одни и те же правила для англо- и русскоязычных новостей — условие сопоставимости
признаков: иначе разница между языками могла бы оказаться разницей в очистке.
"""
from __future__ import annotations

import re
import unicodedata
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from cryptonews.data import coins

OUTPUT_COLUMNS = ["published_utc", "title", "url", "source", "mentions_btc", "mentions_eth"]


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

    work["title"] = work["title"].astype("string").str.strip()
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


def summary_tables(news: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Разбивки для раздела о данных: по годам, по месяцам, по источникам, по монетам,
    а также точность времени публикации по источникам."""
    by_year = news.groupby(news["published_utc"].dt.year).size().rename("Новостей").to_frame()
    by_year.index.name = "Год"
    month = news["published_utc"].dt.tz_localize(None).dt.to_period("M").astype(str)
    by_month = news.groupby(month).size().rename("Новостей").to_frame()
    by_month.index.name = "Месяц"
    by_source = news["source"].value_counts().rename("Новостей").to_frame()
    by_source.index.name = "Источник"
    return {"by_year": by_year, "by_month": by_month, "by_source": by_source,
            "by_coin": coins.coverage_table(news), "time_precision": time_precision(news)}
