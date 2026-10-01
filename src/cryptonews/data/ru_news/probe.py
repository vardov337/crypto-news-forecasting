"""Разведка русскоязычных источников (задачи 1.1–1.4).

Скрипт ничего не скачивает массово: он задаёт каждому сайту несколько вопросов,
от ответов на которые зависит, берём мы этот сайт или нет.

  1. Что разрешает robots.txt и есть ли карта сайта (sitemap).
  2. Отдаёт ли сайт страницы обычному запросу или требует браузер.
  3. Есть ли в разметке время публикации и указано ли оно с точностью до минуты.
  4. До какого года доходит архив.

Защита не обходится: если сайт отвечает отказом, это и есть ответ.
Между запросами выдерживается пауза, запросов делается несколько десятков.
"""
from __future__ import annotations

import re
import time
import urllib.robotparser
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

import requests

USER_AGENT = (
    "Mozilla/5.0 (compatible; academic-research-bot/0.1; "
    "сбор метаданных новостей для научной статьи)"
)
HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "ru,en;q=0.8"}

# Время публикации в разметке новостных сайтов обычно лежит в одном из этих мест
TIME_PATTERNS = [
    (r'<time[^>]+datetime="([^"]+)"', "тег <time datetime>"),
    (r'"datePublished"\s*:\s*"([^"]+)"', "schema.org datePublished"),
    (r'<meta[^>]+property="article:published_time"[^>]+content="([^"]+)"', "og article:published_time"),
    (r'<meta[^>]+name="pubdate"[^>]+content="([^"]+)"', "meta pubdate"),
]
MINUTE_PRECISION = re.compile(r"\d{1,2}:\d{2}")


@dataclass
class SiteReport:
    name: str
    url: str
    reachable: bool = False
    status_code: int | None = None
    robots_allows: bool | None = None
    robots_crawl_delay: float | None = None
    robots_note: str = ""
    sitemaps: list[str] = field(default_factory=list)
    rss_feeds: list[str] = field(default_factory=list)
    article_url: str | None = None
    time_source: str | None = None
    time_value: str | None = None
    has_minutes: bool | None = None
    archive_hint: str = ""
    notes: list[str] = field(default_factory=list)

    def verdict(self) -> str:
        if not self.reachable:
            return "Не берём: сайт не отвечает на обычный запрос"
        if self.robots_allows is False:
            return "Не берём: robots.txt запрещает сбор"
        if self.has_minutes is False:
            return "Не берём: время публикации без минут"
        if self.has_minutes is None:
            return "Проверить вручную: время публикации не найдено в разметке"
        return "Берём: страницы доступны, время публикации с точностью до минуты"


def _get(url: str, timeout: int = 20) -> requests.Response | None:
    try:
        return requests.get(url, headers=HEADERS, timeout=timeout)
    except requests.RequestException:
        return None


def check_robots(base: str, report: SiteReport, pause: float) -> None:
    robots_url = urljoin(base, "/robots.txt")
    response = _get(robots_url)
    time.sleep(pause)
    if response is None or response.status_code != 200:
        report.robots_note = "robots.txt недоступен — считаем, что явного запрета нет"
        report.robots_allows = None
        return

    parser = urllib.robotparser.RobotFileParser()
    parser.parse(response.text.splitlines())
    report.robots_allows = parser.can_fetch(USER_AGENT, base)
    delay = parser.crawl_delay(USER_AGENT)
    report.robots_crawl_delay = float(delay) if delay else None
    report.sitemaps = [
        line.split(":", 1)[1].strip()
        for line in response.text.splitlines()
        if line.lower().startswith("sitemap:")
    ]
    report.robots_note = f"robots.txt получен, строк: {len(response.text.splitlines())}"


def check_homepage(base: str, report: SiteReport, pause: float) -> str | None:
    response = _get(base)
    time.sleep(pause)
    if response is None:
        report.notes.append("Запрос к главной странице не прошёл: сеть или блокировка")
        return None
    report.status_code = response.status_code
    report.reachable = response.status_code == 200
    if not report.reachable:
        report.notes.append(f"Главная страница отвечает кодом {response.status_code}")
        return None

    html = response.text
    report.rss_feeds = sorted(set(re.findall(r'href="([^"]*(?:rss|feed)[^"]*)"', html, re.I)))[:5]

    host = urlparse(base).netloc
    links = re.findall(r'href="([^"]+)"', html)
    for link in links:
        full = urljoin(base, link)
        if urlparse(full).netloc != host:
            continue
        # Ссылка на отдельную новость обычно содержит дату или длинный числовой идентификатор
        if re.search(r"/20\d{2}[/-]\d{2}", full) or re.search(r"/\d{5,}", full):
            return full
    report.notes.append("На главной не нашлась ссылка на отдельную новость — проверьте вручную")
    return None


def check_article(url: str, report: SiteReport, pause: float) -> None:
    response = _get(url)
    time.sleep(pause)
    if response is None or response.status_code != 200:
        report.notes.append(f"Страница новости недоступна: {url}")
        return
    report.article_url = url
    html = response.text
    for pattern, source in TIME_PATTERNS:
        match = re.search(pattern, html, re.I)
        if match:
            report.time_source = source
            report.time_value = match.group(1)[:40]
            report.has_minutes = bool(MINUTE_PRECISION.search(report.time_value))
            return
    report.notes.append("Время публикации не найдено в разметке страницы")


def check_archive(base: str, report: SiteReport, pause: float, probe_years=(2017, 2019)) -> None:
    """Пробует типовые адреса архива за ранние годы, чтобы оценить глубину истории."""
    found = []
    for year in probe_years:
        for template in (f"/news/{year}/01/", f"/{year}/01/", f"/archive/{year}/01/"):
            response = _get(urljoin(base, template))
            time.sleep(pause)
            if response is not None and response.status_code == 200 and len(response.text) > 2000:
                found.append(f"{year}: {template}")
                break
    if found:
        report.archive_hint = "архив по датам отвечает — " + "; ".join(found)
    else:
        report.archive_hint = (
            "типовые адреса архива не ответили; глубину истории определим по карте сайта "
            "или по постраничной навигации"
        )


def probe_site(name: str, url: str, pause: float = 1.0) -> SiteReport:
    report = SiteReport(name=name, url=url)
    check_robots(url, report, pause)
    article_url = check_homepage(url, report, pause)
    if report.robots_allows is False:
        report.notes.append("robots.txt запрещает сбор — дальше не проверяем")
        return report
    if article_url:
        check_article(article_url, report, pause)
    if report.reachable:
        check_archive(url, report, pause)
    return report


def format_report(report: SiteReport) -> str:
    allowed = {True: "да", False: "нет", None: "явного правила нет"}[report.robots_allows]
    minutes = {True: "да", False: "нет", None: "неизвестно"}[report.has_minutes]
    lines = [
        f"=== {report.name} ({report.url})",
        f"  Доступность:      {'да' if report.reachable else 'нет'} (код {report.status_code})",
        f"  robots.txt:       {report.robots_note}",
        f"  Сбор разрешён:    {allowed}",
        f"  Пауза из robots:  {report.robots_crawl_delay or 'не задана'}",
        f"  Карты сайта:      {', '.join(report.sitemaps) if report.sitemaps else 'не указаны'}",
        f"  RSS:              {', '.join(report.rss_feeds) if report.rss_feeds else 'не найдены'}",
        f"  Пример новости:   {report.article_url or 'не найден'}",
        f"  Время публикации: {report.time_value or 'не найдено'} ({report.time_source or '—'})",
        f"  Точность до минут: {minutes}",
        f"  Архив:            {report.archive_hint}",
    ]
    for note in report.notes:
        lines.append(f"  Замечание:        {note}")
    lines.append(f"  ВЕРДИКТ:          {report.verdict()}")
    return "\n".join(lines)
