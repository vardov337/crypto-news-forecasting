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


def clean_utc(frame: pd.DataFrame, dedup_window_hours: int = 24,
              first_stage: str = "Записей в наборе") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Очистка таблицы, в которой время уже в UTC (колонка published_utc).

    Возвращает чистую таблицу со стандартными колонками и таблицу этапов очистки.
    """
    stages = [(first_stage, len(frame))]

    work = frame[frame["published_utc"].notna()].copy()
    stages.append(("С корректной меткой времени", len(work)))

    work["title"] = work["title"].astype("string").str.strip()
    work["url"] = work["url"].astype("string").str.strip()
    work = work[(work["title"].fillna("").str.len() > 0) & (work["url"].fillna("").str.len() > 0)]
    stages.append(("С заголовком и адресом", len(work)))

    work = work.sort_values("published_utc").drop_duplicates("url", keep="first")
    stages.append(("Без повторов адреса", len(work)))

    work = dedup_titles(work, window_hours=dedup_window_hours)
    stages.append((f"Без повторов заголовка в пределах {dedup_window_hours} ч", len(work)))

    work["source"] = work["url"].map(source_of)
    work = coins.tag(work)
    extra = [c for c in ("coin_type",) if c in work.columns]
    result = work[OUTPUT_COLUMNS + extra].reset_index(drop=True)

    table = pd.DataFrame(stages, columns=["Этап", "Записей"])
    table["Убрано на этапе"] = (-table["Записей"].diff()).fillna(0).astype(int)
    return result, table


def summary_tables(news: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Разбивки для раздела о данных: по годам, по источникам, по монетам."""
    by_year = news.groupby(news["published_utc"].dt.year).size().rename("Новостей").to_frame()
    by_year.index.name = "Год"
    by_source = news["source"].value_counts().rename("Новостей").to_frame()
    by_source.index.name = "Источник"
    return {"by_year": by_year, "by_source": by_source, "by_coin": coins.coverage_table(news)}
