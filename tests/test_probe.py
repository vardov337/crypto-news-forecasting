"""Проверки разведки сайтов (задачи 1.1–1.4). Сеть не нужна."""
from cryptonews.data.ru_news import probe


def test_headers_are_latin1_encodable():
    """Заголовки HTTP допускают только латиницу: кириллица в них роняет запрос."""
    for name, value in probe.HEADERS.items():
        value.encode("latin-1")
        name.encode("latin-1")


def test_verdicts_cover_all_cases():
    cases = [
        (dict(reachable=False), "Не берём"),
        (dict(reachable=True, robots_allows=False), "Не берём"),
        (dict(reachable=True, robots_allows=True, has_minutes=False), "Не берём"),
        (dict(reachable=True, robots_allows=True, has_minutes=None), "Проверить вручную"),
        (dict(reachable=True, robots_allows=True, has_minutes=True), "Берём"),
    ]
    for kwargs, expected in cases:
        report = probe.SiteReport(name="x", url="y", **kwargs)
        assert report.verdict().startswith(expected), (kwargs, report.verdict())


def test_format_report_handles_empty_report():
    """Отчёт должен печататься даже по сайту, который вообще не ответил."""
    text = probe.format_report(probe.SiteReport(name="x", url="https://x/"))
    assert "ВЕРДИКТ" in text and "Не берём" in text


def test_minute_precision_detection():
    assert probe.MINUTE_PRECISION.search("2024-01-15T10:47:00+03:00")
    assert not probe.MINUTE_PRECISION.search("2024-01-15")


def test_time_patterns_match_typical_markup():
    samples = [
        ('<time datetime="2024-01-15T10:47:00+03:00">15 января</time>', "тег <time datetime>"),
        ('{"datePublished": "2024-01-15T10:47:00+03:00"}', "schema.org datePublished"),
        ('<meta property="article:published_time" content="2024-01-15T10:47:00+03:00">',
         "og article:published_time"),
    ]
    import re
    for html, expected_source in samples:
        found = None
        for pattern, source in probe.TIME_PATTERNS:
            if re.search(pattern, html, re.I):
                found = source
                break
        assert found == expected_source, html


class FakeResponse:
    def __init__(self, text="", status_code=200):
        self.text, self.status_code = text, status_code


ROBOTS = "User-agent: *\nAllow: /\nCrawl-delay: 2\nSitemap: https://site.ru/sitemap_index.xml\n"
HOME = '<html><link type="application/rss+xml" href="/feed"></html>'
RSS = """<rss><channel>
<item><link>https://site.ru/</link><pubDate>Wed, 01 Oct 2026 12:00:00 +0300</pubDate></item>
<item><link>https://site.ru/news/bitcoin-rastet</link><pubDate>Wed, 01 Oct 2026 11:47:00 +0300</pubDate></item>
<item><link>https://site.ru/wp-content/uploads/logo.png</link><pubDate>Wed, 01 Oct 2026 11:00:00 +0300</pubDate></item>
<item><link>https://site.ru/news/efir-obnovlenie</link><pubDate>Wed, 01 Oct 2026 10:30:00 +0300</pubDate></item>
</channel></rss>"""
SITEMAP_INDEX = """<sitemapindex>
<sitemap><loc>https://site.ru/post-sitemap-2017.xml</loc></sitemap>
<sitemap><loc>https://site.ru/post-sitemap-2026.xml</loc></sitemap>
</sitemapindex>"""
ARTICLE = '<html><time datetime="2026-10-01T11:47:00+03:00">1 октября</time></html>'


def fake_site(url, headers=None, timeout=25):
    if url.endswith("robots.txt"):
        return FakeResponse(ROBOTS)
    if url.rstrip("/") == "https://site.ru":
        return FakeResponse(HOME)
    if url.endswith("/feed"):
        return FakeResponse(RSS)
    if url.endswith("sitemap_index.xml"):
        return FakeResponse(SITEMAP_INDEX)
    if "/news/" in url:
        return FakeResponse(ARTICLE)
    return FakeResponse("", 404)


def test_probe_site_uses_rss_and_sitemap(monkeypatch):
    monkeypatch.setattr(probe.requests, "get", fake_site)
    report = probe.probe_site("Тест", "https://site.ru/", pause=0)

    assert report.reachable and report.robots_allows is True
    assert report.robots_crawl_delay == 2.0
    assert report.rss_url == "https://site.ru/feed" and report.rss_items == 4
    # картинка и первая ссылка на сам сайт в список новостей не попадают
    assert report.article_urls == ["https://site.ru/news/bitcoin-rastet",
                                   "https://site.ru/news/efir-obnovlenie"]
    assert report.sitemap_kind.startswith("указатель на 2")
    assert report.earliest_year == "2017"
    assert report.time_source == "тег <time datetime>"
    assert report.has_minutes is True and report.has_timezone is True
    assert report.verdict().startswith("Берём")
    assert "ВЕРДИКТ" in probe.format_report(report)


def test_probe_site_falls_back_to_rss_time(monkeypatch):
    """Если на странице новости времени нет, годится время из RSS."""
    def no_time_on_page(url, headers=None, timeout=25):
        return FakeResponse("<html>без времени</html>") if "/news/" in url else fake_site(url)
    monkeypatch.setattr(probe.requests, "get", no_time_on_page)
    report = probe.probe_site("Тест", "https://site.ru/", pause=0)
    assert report.time_source == "RSS pubDate"
    assert report.has_minutes is True and report.has_timezone is True


def test_probe_site_stops_on_blocked_site(monkeypatch):
    def blocked(url, headers=None, timeout=25):
        return FakeResponse(ROBOTS) if url.endswith("robots.txt") else FakeResponse("", 401)
    monkeypatch.setattr(probe.requests, "get", blocked)
    report = probe.probe_site("Закрытый", "https://site.ru/", pause=0)
    assert report.status_code == 401
    assert report.verdict().startswith("Не берём")
    assert report.article_checks == []
