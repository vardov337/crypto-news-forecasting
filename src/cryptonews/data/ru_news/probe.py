"""Разведка русскоязычных источников (задачи 1.1–1.4), вторая версия.

Первая версия искала ссылку на новость среди ссылок главной страницы и часто
промахивалась: на Profinvestment под шаблон попала картинка, на двух других
сайтах ссылка не нашлась вовсе.

Здесь адреса новостей берутся оттуда, где они публикуются для машин:
из RSS-ленты и из карты сайта. В RSS вдобавок указано время публикации,
поэтому точность до минуты часто видна сразу, без захода на статью.

Что выясняем по каждому сайту:
  1. Что разрешает robots.txt и какая пауза между запросами там указана.
  2. Отдаёт ли сайт страницы обычному запросу.
  3. Есть ли RSS и карта сайта, как карта устроена.
  4. Указано ли время публикации с точностью до минуты и в каком часовом поясе.
  5. До какого года доходит архив.

Защита не обходится: если сайт отвечает отказом, это и есть ответ.
"""
from __future__ import annotations

import re
import time
import urllib.robotparser
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

import requests

# Заголовки HTTP допускают только латиницу, поэтому кириллицы здесь быть не может.
USER_AGENT = (
    "Mozilla/5.0 (compatible; academic-research-bot/0.1; "
    "+https://github.com/vardov337/crypto-news-forecasting)"
)
HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "ru,en;q=0.8"}

# Время публикации в разметке новостных сайтов обычно лежит в одном из этих мест
TIME_PATTERNS = [
    (r'<time[^>]+datetime="([^"]+)"', "тег <time datetime>"),
    (r'"datePublished"\s*:\s*"([^"]+)"', "schema.org datePublished"),
    (r'<meta[^>]+property="article:published_time"[^>]+content="([^"]+)"', "og article:published_time"),
    (r'<meta[^>]+name="pubdate"[^>]+content="([^"]+)"', "meta pubdate"),
    (r'<meta[^>]+itemprop="datePublished"[^>]+content="([^"]+)"', "itemprop datePublished"),
]
MINUTE_PRECISION = re.compile(r"\d{1,2}:\d{2}")
TIMEZONE_HINT = re.compile(r"(Z|[+-]\d{2}:?\d{2}|GMT|MSK|\+0300)")
YEAR_IN_TEXT = re.compile(r"(20[0-2]\d)")

# Карты сайта, в которых лежат новости, и те, которые заведомо не нужны
ARTICLE_SITEMAP = re.compile(r"(news|post|article|iblock|archive)", re.I)
SKIP_SITEMAP = re.compile(
    r"(author|tag|term|categor|page-sitemap|pairs|converter|quiz|calculator|files|image)", re.I)

# Адреса, которые новостями не являются: файлы, разделы, страницы со служебной информацией
NOT_AN_ARTICLE = re.compile(
    r"\.(png|jpe?g|gif|svg|webp|ico|css|js|pdf|xml|zip)(\?|$)"
    r"|/(tag|tags|category|rubrics?|author|page|search|feed|wp-content|wp-json|about|contacts?)/",
    re.I,
)


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
    sitemap_children: list[str] = field(default_factory=list)
    sitemap_details: list[dict] = field(default_factory=list)
    sitemap_urls_total: int = 0
    sitemap_kind: str = ""
    rss_url: str | None = None
    rss_items: int = 0
    rss_time_example: str | None = None
    article_urls: list[str] = field(default_factory=list)
    article_checks: list[dict] = field(default_factory=list)
    time_source: str | None = None
    time_value: str | None = None
    has_minutes: bool | None = None
    has_timezone: bool | None = None
    archive_hint: str = ""
    earliest_year: str | None = None
    earliest_date: str | None = None
    notes: list[str] = field(default_factory=list)

    def verdict(self) -> str:
        if not self.reachable:
            return "Не берём: сайт не отвечает на обычный запрос"
        if self.robots_allows is False:
            return "Не берём: robots.txt запрещает сбор"
        if self.has_minutes is False:
            return "Не берём: время публикации без минут"
        if self.has_minutes is None:
            return "Проверить вручную: время публикации не найдено"
        if not (self.rss_url or self.sitemaps):
            return "Берём с оговоркой: время есть, но нет ни RSS, ни карты сайта"
        return "Берём: страницы доступны, время публикации с точностью до минуты"


def _get(url: str, timeout: int = 25) -> requests.Response | None:
    try:
        return requests.get(url, headers=HEADERS, timeout=timeout)
    except requests.RequestException:
        return None


def check_robots(base: str, report: SiteReport, pause: float) -> None:
    response = _get(urljoin(base, "/robots.txt"))
    time.sleep(pause)
    if response is None or response.status_code != 200:
        report.robots_note = "robots.txt недоступен — явного запрета нет"
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
    report.robots_note = f"получен, строк: {len(response.text.splitlines())}"


def check_homepage(base: str, report: SiteReport, pause: float) -> str | None:
    response = _get(base)
    time.sleep(pause)
    if response is None:
        report.notes.append("Запрос к главной не прошёл: сеть или блокировка")
        return None
    report.status_code = response.status_code
    report.reachable = response.status_code == 200
    if not report.reachable:
        report.notes.append(f"Главная отвечает кодом {response.status_code}")
        return None
    return response.text


def find_rss(base: str, html: str | None, report: SiteReport, pause: float) -> None:
    """Ищет RSS в разметке страницы, иначе пробует типовые адреса."""
    candidates: list[str] = []
    if html:
        candidates += [
            urljoin(base, m) for m in
            re.findall(r'<link[^>]+type="application/rss\+xml"[^>]+href="([^"]+)"', html, re.I)
        ]
    candidates += [urljoin(base, p) for p in
                   ("/feed", "/feed/", "/rss2/", "/rss/", "/rss.xml", "/?feed=rss2")]

    for url in dict.fromkeys(candidates):
        response = _get(url)
        time.sleep(pause)
        if response is None or response.status_code != 200 or "<item" not in response.text[:20000]:
            continue
        report.rss_url = url
        links = re.findall(r"<link>\s*([^<\s]+)\s*</link>", response.text)
        dates = re.findall(r"<pubDate>\s*([^<]+?)\s*</pubDate>", response.text)
        report.rss_items = len(dates)
        if dates:
            report.rss_time_example = dates[0]
        # Первая ссылка в ленте обычно указывает на сам сайт, а не на новость
        report.article_urls = [u for u in links if not NOT_AN_ARTICLE.search(u)][1:4]
        return


def read_sitemaps(report: SiteReport, pause: float, max_children: int = 4) -> None:
    """Разбирает карту сайта: какие в ней разделы, сколько адресов и за какие годы."""
    if not report.sitemaps:
        return
    response = _get(report.sitemaps[0])
    time.sleep(pause)
    if response is None or response.status_code != 200:
        report.notes.append(f"Карта сайта недоступна: {report.sitemaps[0]}")
        return

    text = response.text
    children = re.findall(r"<sitemap>\s*<loc>\s*([^<\s]+)", text)
    if children:
        report.sitemap_kind = f"указатель на {len(children)} карт"
        report.sitemap_children = children[:15]
        # Берём только карты с новостями: карты авторов, рубрик и картинок пропускаем
        picked = [c for c in children if ARTICLE_SITEMAP.search(c) and not SKIP_SITEMAP.search(c)]
        if not picked:
            picked = [c for c in children if not SKIP_SITEMAP.search(c)] or children
        # Первые и последние карты списка: на этих краях обычно самые свежие и самые старые записи
        half = max(1, max_children // 2)
        targets = list(dict.fromkeys(picked[:half] + picked[-half:]))
    else:
        report.sitemap_kind = "одна карта со списком адресов"
        targets = [report.sitemaps[0]]

    dates_seen: list[str] = []
    for target in targets:
        if target != report.sitemaps[0]:
            response = _get(target)
            time.sleep(pause)
            if response is None or response.status_code != 200:
                report.sitemap_details.append({
                    "url": target, "urls": 0,
                    "status": response.status_code if response else "нет ответа",
                })
                continue
            child_text = response.text
        else:
            child_text = text

        locs = re.findall(r"<url>\s*<loc>\s*([^<\s]+)", child_text)
        mods = re.findall(r"<lastmod>\s*(\d{4}-\d{2}-\d{2})", child_text)
        dates_seen += mods
        report.sitemap_urls_total += len(locs)
        report.sitemap_details.append({
            "url": target, "urls": len(locs), "status": 200,
            "first": min(mods) if mods else None,
            "last": max(mods) if mods else None,
        })
        if len(report.article_urls) < 3:
            fresh = [u for u in locs if not NOT_AN_ARTICLE.search(u)]
            report.article_urls += fresh[:3 - len(report.article_urls)]

    if dates_seen:
        report.earliest_date = min(dates_seen)
        report.earliest_year = report.earliest_date[:4]


def check_articles(report: SiteReport, pause: float) -> None:
    """Открывает несколько настоящих новостей и ищет в них время публикации."""
    for url in report.article_urls:
        response = _get(url)
        time.sleep(pause)
        check = {"url": url, "status": None, "source": None, "value": None}
        if response is None or response.status_code != 200:
            check["status"] = response.status_code if response else "нет ответа"
            report.article_checks.append(check)
            continue
        check["status"] = 200
        for pattern, source in TIME_PATTERNS:
            match = re.search(pattern, response.text, re.I)
            if match:
                check["source"], check["value"] = source, match.group(1)[:40]
                break
        report.article_checks.append(check)
        if check["value"] and report.time_value is None:
            report.time_source, report.time_value = check["source"], check["value"]

    # Если на самих страницах время не нашлось, годится время из RSS
    if report.time_value is None and report.rss_time_example:
        report.time_source, report.time_value = "RSS pubDate", report.rss_time_example

    if report.time_value:
        report.has_minutes = bool(MINUTE_PRECISION.search(report.time_value))
        report.has_timezone = bool(TIMEZONE_HINT.search(report.time_value))
    else:
        report.notes.append("Время публикации не найдено ни на страницах, ни в RSS")


def check_archive(base: str, report: SiteReport, pause: float, probe_years=(2017, 2019, 2021)) -> None:
    found = []
    for year in probe_years:
        for template in (f"/{year}/01/", f"/news/{year}/01/", f"/archive/{year}/01/"):
            response = _get(urljoin(base, template))
            time.sleep(pause)
            if response is not None and response.status_code == 200 and len(response.text) > 2000:
                found.append(f"{year}: {template}")
                break
    report.archive_hint = ("архив по датам отвечает — " + "; ".join(found)) if found else \
        "архив по датам не отвечает; глубину истории смотрим по карте сайта"


def probe_site(name: str, url: str, pause: float = 1.0) -> SiteReport:
    report = SiteReport(name=name, url=url)
    check_robots(url, report, pause)
    html = check_homepage(url, report, pause)
    if report.robots_allows is False:
        report.notes.append("robots.txt запрещает сбор — дальше не проверяем")
        return report
    if not report.reachable:
        return report
    if report.robots_crawl_delay:
        pause = max(pause, report.robots_crawl_delay)
    find_rss(url, html, report, pause)
    read_sitemaps(report, pause)
    check_articles(report, pause)
    check_archive(url, report, pause)
    return report


def format_report(report: SiteReport) -> str:
    allowed = {True: "да", False: "нет", None: "явного правила нет"}[report.robots_allows]
    minutes = {True: "да", False: "нет", None: "неизвестно"}[report.has_minutes]
    zone = {True: "да", False: "нет", None: "неизвестно"}[report.has_timezone]
    lines = [
        f"=== {report.name} ({report.url})",
        f"  Доступность:       {'да' if report.reachable else 'нет'} (код {report.status_code})",
        f"  robots.txt:        {report.robots_note}; сбор разрешён: {allowed}; "
        f"пауза: {report.robots_crawl_delay or 'не задана'}",
        f"  RSS:               {report.rss_url or 'не найден'}"
        + (f" — записей {report.rss_items}, пример времени: {report.rss_time_example}"
           if report.rss_url else ""),
        f"  Карта сайта:       {report.sitemaps[0] if report.sitemaps else 'не указана'}"
        + (f" ({report.sitemap_kind})" if report.sitemap_kind else ""),
    ]
    for child in report.sitemap_children:
        lines.append(f"      {child}")
    if report.sitemap_details:
        lines.append(f"  Прочитано карт:    {len(report.sitemap_details)}, "
                     f"адресов в них: {report.sitemap_urls_total}")
        for d in report.sitemap_details:
            lines.append(f"      {d['url']}")
            lines.append(f"            адресов: {d['urls']}, даты: "
                         f"{d.get('first') or '—'} … {d.get('last') or '—'}")
    lines += [
        f"  Самая ранняя дата: {report.earliest_date or 'не определена'}",
        f"  Архив:             {report.archive_hint}",
        "  Проверенные новости:",
    ]
    for check in report.article_checks:
        lines.append(f"      [{check['status']}] {check['url']}")
        lines.append(f"            время: {check['value'] or 'не найдено'} ({check['source'] or '—'})")
    if not report.article_checks:
        lines.append("      адреса новостей не найдены")
    lines += [
        f"  Итоговое время:    {report.time_value or 'не найдено'} ({report.time_source or '—'})",
        f"  Точность до минут: {minutes}; часовой пояс указан: {zone}",
    ]
    for note in report.notes:
        lines.append(f"  Замечание:         {note}")
    lines.append(f"  ВЕРДИКТ:           {report.verdict()}")
    return "\n".join(lines)
