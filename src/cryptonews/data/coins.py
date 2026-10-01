"""Привязка новостей к монетам по заголовку (задачи 3.6 и 4.3).

Одно и то же правило применяется к англо- и русскоязычным заголовкам: русские
издания пишут названия монет и латиницей (Bitcoin, ETH), и кириллицей (биткоин,
эфир). Правило намеренно простое и прозрачное, чтобы его можно было описать
в статье одной фразой и проверить на ручной разметке.

Исключения нужны, чтобы не путать монеты с похожими названиями:
  * Bitcoin Cash, Bitcoin SV, Bitcoin Gold — отдельные монеты, не BTC;
  * Ethereum Classic — отдельная монета, не ETH;
  * «в эфире», «прямом эфире» — про трансляции, а не про монету, поэтому
    форма «эфире» не учитывается вовсе;
  * Tether и WBTC не срабатывают, потому что слово должно начинаться с границы.
"""
from __future__ import annotations

import re

import pandas as pd

PATTERNS = {
    "BTCUSDT": re.compile(
        r"\b(?:биткоин\w*|биткойн\w*|bitcoin(?!\s+(?:cash|sv|gold)\b)|btc)\b",
        re.IGNORECASE,
    ),
    "ETHUSDT": re.compile(
        r"\b(?:эфириум\w*|эфир(?:а|у|ом)?|ethereum(?!\s+classic\b)|ether|eth)\b",
        re.IGNORECASE,
    ),
}
FLAG = {"BTCUSDT": "mentions_btc", "ETHUSDT": "mentions_eth"}


def mentions(title: str, symbol: str) -> bool:
    return bool(PATTERNS[symbol].search(str(title)))


def tag(frame: pd.DataFrame, title_column: str = "title") -> pd.DataFrame:
    """Добавляет к таблице колонки mentions_btc и mentions_eth."""
    out = frame.copy()
    titles = out[title_column].astype("string").fillna("")
    for symbol, column in FLAG.items():
        out[column] = titles.str.contains(PATTERNS[symbol], regex=True)
    return out


def coverage_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Сколько новостей упоминает каждую монету — для раздела о данных."""
    total = len(frame)
    btc, eth = frame["mentions_btc"], frame["mentions_eth"]
    rows = [
        ("Упоминают биткоин", int(btc.sum())),
        ("Упоминают эфир", int(eth.sum())),
        ("Упоминают обе монеты", int((btc & eth).sum())),
        ("Не упоминают ни одну из двух", int((~btc & ~eth).sum())),
        ("Всего", total),
    ]
    table = pd.DataFrame(rows, columns=["Группа", "Новостей"])
    table["Доля"] = (table["Новостей"] / max(total, 1)).round(4)
    return table
