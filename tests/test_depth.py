"""Проверки замера глубины архива (решение 1.5). Сеть не нужна."""
from cryptonews.data.ru_news import depth


class FakeResponse:
    def __init__(self, text="", status_code=200):
        self.text, self.status_code = text, status_code


INDEX = """<sitemapindex>
<sitemap><loc>https://s.ru/sitemap-posts-1.xml</loc></sitemap>
<sitemap><loc>https://s.ru/author-sitemap.xml</loc></sitemap>
<sitemap><loc>https://s.ru/sitemap-posts-2.xml</loc></sitemap>
</sitemapindex>"""
POSTS_1 = """<urlset>
<url><loc>https://s.ru/news/samaya-staraya</loc><lastmod>2016-01-10</lastmod></url>
<url><loc>https://s.ru/news/vtoraya</loc><lastmod>2018-05-20</lastmod></url>
<url><loc>https://s.ru/wp-content/uploads/pic.png</loc><lastmod>2016-01-01</lastmod></url>
</urlset>"""
POSTS_2 = """<urlset>
<url><loc>https://s.ru/news/svezhaya</loc><lastmod>2026-09-30</lastmod></url>
</urlset>"""
ARTICLES = {
    "https://s.ru/news/samaya-staraya": '<time datetime="2016-01-10T09:30:00+03:00">',
    "https://s.ru/news/vtoraya": '<time datetime="2018-05-20T12:00:00+03:00">',
    "https://s.ru/news/svezhaya": '<time datetime="2026-09-30T20:15:00+03:00">',
}


def fake(url, headers=None, timeout=40):
    if url.endswith("sitemap.xml"):
        return FakeResponse(INDEX)
    if url.endswith("sitemap-posts-1.xml"):
        return FakeResponse(POSTS_1)
    if url.endswith("sitemap-posts-2.xml"):
        return FakeResponse(POSTS_2)
    if url in ARTICLES:
        return FakeResponse(ARTICLES[url])
    return FakeResponse("", 404)


def test_measure_finds_archive_boundaries(monkeypatch):
    monkeypatch.setattr(depth.requests, "get", fake)
    report = depth.measure("Тест", "https://s.ru/sitemap.xml", pause=0, samples=2)

    # карта авторов пропущена, картинка в список адресов не попала
    assert report.sitemaps_read == 2
    assert report.total_urls == 3
    assert report.lastmod_min == "2016-01-10" and report.lastmod_max == "2026-09-30"
    assert report.published_dates == ["2016-01-10", "2018-05-20", "2026-09-30"]
    assert "ИТОГ: публикации найдены с 2016-01-10 по 2026-09-30" in depth.format_report(report)


def test_measure_survives_dead_sitemap(monkeypatch):
    def half_dead(url, headers=None, timeout=40):
        return FakeResponse("", 500) if "posts-2" in url else fake(url)
    monkeypatch.setattr(depth.requests, "get", half_dead)
    report = depth.measure("Тест", "https://s.ru/sitemap.xml", pause=0, samples=2)
    assert report.sitemaps_read == 1
    assert len(report.sitemaps_failed) == 1
    assert report.published_dates == ["2016-01-10", "2018-05-20"]


def test_published_date_reports_missing_markup(monkeypatch):
    monkeypatch.setattr(depth.requests, "get",
                        lambda url, headers=None, timeout=40: FakeResponse("<html>нет даты</html>"))
    value, source = depth.published_date("https://s.ru/news/x", pause=0)
    assert value is None and source == "в разметке нет даты"
