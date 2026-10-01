"""Проверка способов массового сбора (перед задачами 2.1–2.3).

Обойти 118 тысяч статей по одной при вежливой скорости в один запрос в секунду
означает больше тридцати часов работы. Это нереально.

Поэтому сначала выясняем, умеет ли сайт отдавать заголовки и даты пачками:

  * WordPress открывает программный интерфейс /wp-json/wp/v2/posts, который
    возвращает до ста записей за запрос вместе с датой в UTC. Это сокращает
    работу примерно в сто раз.
  * Движки без такого интерфейса обычно имеют постраничные ленты новостей,
    где на одной странице видно два десятка заголовков с датами. Это сокращает
    работу примерно в двадцать раз.

Скрипт ничего не собирает, он только проверяет доступность этих способов.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

import requests

from cryptonews.data.ru_news.probe import HEADERS, TIME_PATTERNS

# Адреса постраничных лент, которые стоит попробовать
LISTING_TEMPLATES = [
    "/news/?PAGEN_1={page}",      # Bitrix
    "/news/page/{page}/",         # WordPress
    "/news/?page={page}",
    "/page/{page}/",
]


@dataclass
class BulkReport:
    name: str
    base: str
    wp_api: dict = field(default_factory=dict)
    listing: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def best_method(self) -> str:
        if self.wp_api.get("works"):
            total = self.wp_api.get("total")
            per_page = self.wp_api.get("per_page", 0)
            requests_needed = (total // per_page + 1) if total and per_page else "?"
            return (f"программный интерфейс WordPress: {per_page} записей за запрос, "
                    f"всего {total or '?'} записей, примерно {requests_needed} запросов")
        if self.listing.get("works"):
            per_page = self.listing.get("articles_on_page", 0)
            depth = self.listing.get("max_page_found")
            return (f"постраничная лента {self.listing.get('template')}: примерно {per_page} "
                    f"материалов на странице, страниц не меньше {depth}")
        return "массовый способ не найден — придётся обходить статьи по одной"


def _get(url: str, timeout: int = 30) -> requests.Response | None:
    try:
        return requests.get(url, headers=HEADERS, timeout=timeout)
    except requests.RequestException:
        return None


def check_wp_api(base: str, report: BulkReport, pause: float) -> None:
    """Пробует программный интерфейс WordPress."""
    url = base.rstrip("/") + "/wp-json/wp/v2/posts?per_page=5&_fields=link,date,date_gmt,title"
    response = _get(url)
    time.sleep(pause)
    if response is None:
        report.wp_api = {"works": False, "reason": "запрос не прошёл"}
        return
    report.wp_api["status"] = response.status_code
    if response.status_code != 200:
        report.wp_api["works"] = False
        report.wp_api["reason"] = f"код {response.status_code}"
        return
    try:
        items = json.loads(response.text)
    except ValueError:
        report.wp_api["works"] = False
        report.wp_api["reason"] = "ответ не разбирается как JSON"
        return
    if not isinstance(items, list) or not items:
        report.wp_api["works"] = False
        report.wp_api["reason"] = "пустой список записей"
        return

    first = items[0]
    report.wp_api.update({
        "works": True,
        "fields": sorted(first.keys()),
        "has_date_gmt": "date_gmt" in first,
        "example_date": first.get("date_gmt") or first.get("date"),
        "example_link": first.get("link"),
        "example_title": (first.get("title") or {}).get("rendered", "")[:80],
        "total": int(response.headers.get("X-WP-Total", 0)) or None,
        "total_pages": int(response.headers.get("X-WP-TotalPages", 0)) or None,
    })

    # Сколько записей отдают за один запрос: обычно не больше ста
    probe = _get(base.rstrip("/") + "/wp-json/wp/v2/posts?per_page=100&_fields=link,date_gmt")
    time.sleep(pause)
    if probe is not None and probe.status_code == 200:
        try:
            report.wp_api["per_page"] = len(json.loads(probe.text))
        except ValueError:
            report.wp_api["per_page"] = len(items)
    else:
        report.wp_api["per_page"] = len(items)


def check_listing(base: str, report: BulkReport, pause: float, deep_page: int = 500) -> None:
    """Пробует постраничные ленты новостей."""
    host = base.rstrip("/")
    for template in LISTING_TEMPLATES:
        url = host + template.format(page=2)
        response = _get(url)
        time.sleep(pause)
        if response is None or response.status_code != 200 or len(response.text) < 3000:
            continue

        html = response.text
        times = sum(len(re.findall(pattern, html, re.I)) for pattern, _ in TIME_PATTERNS)
        links = len(set(re.findall(r'href="([^"]*/[^"/]*-[^"/]*-[^"/]*/?)"', html)))
        if times < 3 or links < 5:
            report.notes.append(f"{template}: страница открылась, но дат {times}, ссылок {links}")
            continue

        report.listing = {
            "works": True, "template": template, "url": url,
            "dates_on_page": times, "articles_on_page": links,
        }
        # Насколько глубоко работает постраничная навигация
        deep = _get(host + template.format(page=deep_page))
        time.sleep(pause)
        report.listing["max_page_found"] = (
            deep_page if deep is not None and deep.status_code == 200 and len(deep.text) > 3000
            else "меньше " + str(deep_page)
        )
        return
    report.listing = {"works": False}


def probe_bulk(name: str, base: str, pause: float = 1.0) -> BulkReport:
    report = BulkReport(name=name, base=base)
    check_wp_api(base, report, pause)
    if not report.wp_api.get("works"):
        check_listing(base, report, pause)
    return report


def format_report(report: BulkReport) -> str:
    lines = [f"=== {report.name} ({report.base})"]
    if report.wp_api:
        if report.wp_api.get("works"):
            lines += [
                f"  Интерфейс WordPress: работает, код {report.wp_api.get('status')}",
                f"      записей за запрос: {report.wp_api.get('per_page')}",
                f"      всего записей:     {report.wp_api.get('total') or 'заголовок не отдан'}",
                f"      дата в UTC:        {'да' if report.wp_api.get('has_date_gmt') else 'нет'}"
                f" ({report.wp_api.get('example_date')})",
                f"      поля:              {', '.join(report.wp_api.get('fields', []))}",
                f"      пример заголовка:  {report.wp_api.get('example_title')}",
                f"      пример адреса:     {report.wp_api.get('example_link')}",
            ]
        else:
            lines.append(f"  Интерфейс WordPress: не работает ({report.wp_api.get('reason')})")
    if report.listing:
        if report.listing.get("works"):
            lines += [
                f"  Постраничная лента:  работает, {report.listing['template']}",
                f"      проверенный адрес: {report.listing['url']}",
                f"      дат на странице:   {report.listing['dates_on_page']}",
                f"      ссылок на статьи:  {report.listing['articles_on_page']}",
                f"      глубина навигации: {report.listing['max_page_found']}",
            ]
        else:
            lines.append("  Постраничная лента:  не найдена")
    for note in report.notes:
        lines.append(f"  Замечание:           {note}")
    lines.append(f"  ВЫВОД: {report.best_method()}")
    return "\n".join(lines)
