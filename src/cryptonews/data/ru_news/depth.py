"""Замер глубины архива (решение 1.5).

Разведка показывает, какие карты сайта есть, но не отвечает на главный вопрос:
с какого года мы реально можем собрать новости. Даты в картах сайта — это даты
изменения страницы, а не публикации: у Profinvestment, например, все старые
страницы помечены одним днём массового обновления.

Поэтому здесь делается два шага:
  1. читаются все карты сайта с новостями, считается общее число адресов;
  2. у самых старых и самых свежих адресов открывается страница и берётся
     дата публикации из разметки.

Второй шаг и даёт настоящую границу архива.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import requests

from cryptonews.data.ru_news.probe import (
    ARTICLE_SITEMAP, HEADERS, NOT_AN_ARTICLE, SKIP_SITEMAP, TIME_PATTERNS,
)

URL_BLOCK = re.compile(r"<url>(.*?)</url>", re.S)
LOC = re.compile(r"<loc>\s*([^<\s]+)")
LASTMOD = re.compile(r"<lastmod>\s*(\d{4}-\d{2}-\d{2})")


@dataclass
class DepthReport:
    name: str
    sitemaps_read: int = 0
    sitemaps_failed: list[str] = field(default_factory=list)
    total_urls: int = 0
    lastmod_min: str | None = None
    lastmod_max: str | None = None
    per_sitemap: list[dict] = field(default_factory=list)
    samples: list[dict] = field(default_factory=list)

    @property
    def published_dates(self) -> list[str]:
        return sorted(s["published"][:10] for s in self.samples if s.get("published"))

    def as_dict(self) -> dict:
        return {**self.__dict__, "published_dates": self.published_dates}


def _get(url: str, timeout: int = 40) -> requests.Response | None:
    try:
        return requests.get(url, headers=HEADERS, timeout=timeout)
    except requests.RequestException:
        return None


def list_article_sitemaps(index_url: str, pause: float) -> list[str]:
    """Все карты с новостями из указателя; служебные карты пропускаются."""
    response = _get(index_url)
    time.sleep(pause)
    if response is None or response.status_code != 200:
        return []
    children = re.findall(r"<sitemap>\s*<loc>\s*([^<\s]+)", response.text)
    if not children:
        return [index_url]
    picked = [c for c in children if ARTICLE_SITEMAP.search(c) and not SKIP_SITEMAP.search(c)]
    return picked or [c for c in children if not SKIP_SITEMAP.search(c)]


def read_entries(sitemap_url: str, pause: float) -> list[tuple[str, str | None]]:
    """Пары «адрес, дата изменения» из одной карты сайта."""
    response = _get(sitemap_url)
    time.sleep(pause)
    if response is None or response.status_code != 200:
        return []
    entries = []
    for block in URL_BLOCK.findall(response.text):
        loc = LOC.search(block)
        if not loc or NOT_AN_ARTICLE.search(loc.group(1)):
            continue
        mod = LASTMOD.search(block)
        entries.append((loc.group(1), mod.group(1) if mod else None))
    return entries


def published_date(url: str, pause: float) -> tuple[str | None, str | None]:
    """Дата публикации со страницы новости: значение и откуда взято."""
    response = _get(url)
    time.sleep(pause)
    if response is None or response.status_code != 200:
        return None, f"код {response.status_code if response else 'нет ответа'}"
    for pattern, source in TIME_PATTERNS:
        match = re.search(pattern, response.text, re.I)
        if match:
            return match.group(1), source
    return None, "в разметке нет даты"


def measure(name: str, index_url: str, pause: float = 1.0, samples: int = 4) -> DepthReport:
    report = DepthReport(name=name)
    all_entries: list[tuple[str, str | None]] = []

    for sitemap in list_article_sitemaps(index_url, pause):
        entries = read_entries(sitemap, pause)
        if not entries:
            report.sitemaps_failed.append(sitemap)
            continue
        report.sitemaps_read += 1
        report.total_urls += len(entries)
        mods = [m for _, m in entries if m]
        report.per_sitemap.append({
            "url": sitemap, "urls": len(entries),
            "first": min(mods) if mods else None,
            "last": max(mods) if mods else None,
        })
        all_entries += entries

    dated = sorted(((m, u) for u, m in all_entries if m))
    if dated:
        report.lastmod_min, report.lastmod_max = dated[0][0], dated[-1][0]

    # Берём самые старые и самые свежие адреса: по ним видно обе границы архива
    probe_urls = [u for _, u in dated[:samples]] + [u for _, u in dated[-samples:]]
    if not probe_urls:
        probe_urls = [u for u, _ in all_entries[:samples]]
    for url in dict.fromkeys(probe_urls):
        value, source = published_date(url, pause)
        report.samples.append({"url": url, "published": value, "source": source})
    return report


def format_report(report: DepthReport) -> str:
    lines = [
        f"=== {report.name}",
        f"  Карт прочитано:    {report.sitemaps_read}"
        + (f", не ответили: {len(report.sitemaps_failed)}" if report.sitemaps_failed else ""),
        f"  Адресов всего:     {report.total_urls}",
        f"  Даты изменения:    {report.lastmod_min or '—'} … {report.lastmod_max or '—'}",
        "  По картам:",
    ]
    for item in report.per_sitemap:
        lines.append(f"      {item['urls']:>6} адресов, {item['first'] or '—'} … {item['last'] or '—'}"
                     f"   {item['url']}")
    lines.append("  Даты публикации у крайних материалов:")
    for sample in report.samples:
        lines.append(f"      {sample['published'] or 'не найдена'}  ({sample['source']})")
        lines.append(f"            {sample['url']}")
    dates = report.published_dates
    if dates:
        lines.append(f"  ИТОГ: публикации найдены с {dates[0]} по {dates[-1]}")
    else:
        lines.append("  ИТОГ: даты публикации определить не удалось")
    return "\n".join(lines)
