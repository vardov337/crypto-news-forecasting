"""Сбор новостей ForkLog (задачи 2.1–2.5).

Источник — программный интерфейс WordPress: /wp-json/wp/v2/posts. Он отдаёт по
двадцать записей за запрос вместе с заголовком и датой публикации в UTC, поэтому
56 тысяч материалов собираются примерно за 2 850 запросов, а не за пятнадцать
часов обхода страниц по одной.

Важное свойство этого источника: поле date_gmt уже приведено к UTC. Это снимает
для ForkLog вопрос часовых поясов — в разметке самих страниц метки киевские,
с переходом на летнее время, и подставлять к ним фиксированный сдвиг было бы
ошибкой (февраль 2015 года помечен +02:00, июнь того же года +03:00).

Сбор устроен так, чтобы его можно было прервать и продолжить: каждая страница
дописывается в файл сразу, а при повторном запуске уже собранные страницы
пропускаются. Это важно в Colab, где сессия может оборваться.

Почему возобновление не теряет записей. Интерфейс отдаёт новости от новых
к старым. Если между запусками вышли свежие материалы, все страницы сдвигаются
вперёд: записи с конца уже собранной страницы переезжают в начало следующей.
Получаются повторы, а не пропуски, а повторы убираются по адресу новости.
"""
from __future__ import annotations

import html as html_module
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

from cryptonews.data.ru_news.probe import HEADERS

API_URL = "https://forklog.com/wp-json/wp/v2/posts"
FIELDS = "id,link,date_gmt,title,categories,tags"
PER_PAGE = 20          # сайт не отдаёт больше двадцати, сколько ни проси
TAGS = re.compile(r"<[^>]+>")


@dataclass
class CollectReport:
    pages_done: int = 0
    pages_failed: list[int] = field(default_factory=list)
    pages_short: list[int] = field(default_factory=list)
    pages_skipped: int = 0
    records_new: int = 0
    records_total: int = 0
    total_pages_declared: int | None = None
    total_records_declared: int | None = None
    end_page: int | None = None
    first_date: str | None = None
    last_date: str | None = None

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def clean_title(raw: str) -> str:
    """Заголовок из интерфейса приходит с разметкой и мнемониками вроде &#8220;."""
    return re.sub(r"\s+", " ", html_module.unescape(TAGS.sub("", raw))).strip()


def parse_items(payload: str) -> list[dict]:
    records = []
    for item in json.loads(payload):
        link = item.get("link")
        date_gmt = item.get("date_gmt")
        if not link or not date_gmt:
            continue
        records.append({
            "id": item.get("id"),
            "url": link,
            # интерфейс отдаёт время без обозначения зоны, но это уже UTC
            "published_utc": date_gmt if date_gmt.endswith("Z") else date_gmt + "Z",
            "title": clean_title((item.get("title") or {}).get("rendered", "")),
            "categories": item.get("categories") or [],
            "tags": item.get("tags") or [],
        })
    return records


def fetch_page(page: int, session: requests.Session, retries: int = 3, timeout: int = 40,
               bust_cache: bool = False) -> tuple[list[dict] | None, dict, int | None]:
    """Одна страница интерфейса: записи, заголовки ответа (ключи в нижнем регистре), код.

    Ответы сайта идут через кэш, который местами ведёт себя странно: при разведке
    первая страница вернула одну запись вместо двадцати. Параметр bust_cache
    добавляет к запросу безвредный уникальный хвост, чтобы получить свежий ответ.

    Код 400 WordPress возвращает на номер страницы за концом архива
    (rest_post_invalid_page_number) — это конец сбора, а не ошибка.
    """
    url = f"{API_URL}?per_page={PER_PAGE}&page={page}&_fields={FIELDS}"
    if bust_cache:
        url += f"&nocache={page}-{int(time.time())}"
    for attempt in range(retries):
        try:
            response = session.get(url, headers=HEADERS, timeout=timeout)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        headers = {k.lower(): v for k, v in dict(response.headers).items()}
        if response.status_code == 200:
            return parse_items(response.text), headers, 200
        if response.status_code == 400:
            return [], headers, 400
        time.sleep(2 ** attempt)
    return None, {}, None


def census(session: requests.Session) -> tuple[int | None, int | None]:
    """Сколько записей и страниц на сайте прямо сейчас.

    Берётся из свежего ответа в обход кэша. Первая версия сборщика брала эти числа
    из первого попавшегося ответа, а он пришёл из устаревшего кэша и заявил
    54 994 записи вместо 56 383 — отсюда были ложные предупреждения в отчёте.
    """
    _, headers, status = fetch_page(1, session, bust_cache=True)
    if status != 200:
        return None, None
    total = int(headers.get("x-wp-total", 0) or 0) or None
    pages = int(headers.get("x-wp-totalpages", 0) or 0) or None
    return total, pages


def load_existing(path: Path) -> tuple[dict[str, dict], set[int]]:
    """Что уже собрано: записи по адресу и номера пройденных страниц."""
    records: dict[str, dict] = {}
    pages: set[int] = set()
    if not path.exists():
        return records, pages
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if "_page" in item:
                pages.add(item["_page"])
            if item.get("url"):
                records[item["url"]] = item
    return records, pages


def collect(out_path: str | Path, pause: float = 1.0, max_pages: int = 3200,
            log=None) -> tuple[list[dict], CollectReport]:
    """Собирает все страницы интерфейса, дописывая результат на диск."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records, done_pages = load_existing(out_path)
    report = CollectReport(records_total=len(records))
    session = requests.Session()
    report.total_records_declared, report.total_pages_declared = census(session)
    if log and report.total_records_declared:
        log.info("на сайте сейчас записей: %d, страниц: %d",
                 report.total_records_declared, report.total_pages_declared or 0)
    time.sleep(pause)

    with open(out_path, "a", encoding="utf-8") as sink:
        for page in range(1, max_pages + 1):
            if page in done_pages:
                report.pages_skipped += 1
                continue

            items, _, status = fetch_page(page, session)
            if items is None:
                report.pages_failed.append(page)
                if log:
                    log.warning("страница %d не ответила после повторов", page)
                continue
            if status == 400:
                report.end_page = page
                if log:
                    log.info("страница %d за концом архива — сбор окончен", page)
                break

            # Последняя страница архива бывает неполной, и это нормально.
            # Неполная страница в середине — подозрение на ответ из кэша:
            # повторяем в обход кэша, а если снова коротко — пишем в отчёт.
            middle = report.total_pages_declared is None or page < report.total_pages_declared
            if len(items) < PER_PAGE and middle:
                retry, _, _ = fetch_page(page, session, bust_cache=True)
                if retry is not None and len(retry) > len(items):
                    items = retry
                if len(items) < PER_PAGE:
                    report.pages_short.append(page)

            if not items:
                beyond = report.total_pages_declared and page > report.total_pages_declared
                if beyond:
                    report.end_page = page
                    break
                # пустой ответ с кодом 200 посреди архива — сбой, а не конец
                report.pages_failed.append(page)
                continue

            for item in items:
                item["_page"] = page
                if item["url"] not in records:
                    records[item["url"]] = item
                    report.records_new += 1
                sink.write(json.dumps(item, ensure_ascii=False) + "\n")
            sink.flush()
            report.pages_done += 1

            if log and page % 50 == 0:
                dates = sorted(r["published_utc"] for r in records.values())
                log.info("страниц %d, записей %d, дошли до %s",
                         page, len(records), dates[0][:10] if dates else "—")
            time.sleep(pause)

    report.records_total = len(records)
    dates = sorted(r["published_utc"] for r in records.values())
    if dates:
        report.first_date, report.last_date = dates[0], dates[-1]
    return list(records.values()), report
