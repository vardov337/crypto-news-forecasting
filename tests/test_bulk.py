"""Проверки способов массового сбора. Сеть не нужна."""
import json

from cryptonews.data.ru_news import bulk


class FakeResponse:
    def __init__(self, text="", status_code=200, headers=None):
        self.text, self.status_code = text, status_code
        self.headers = headers or {}


WP_ITEMS = json.dumps([
    {"link": "https://f.com/news/pervaya-novost-dnya", "date": "2026-10-01T18:00:00",
     "date_gmt": "2026-10-01T15:00:00", "title": {"rendered": "Первая новость"}},
    {"link": "https://f.com/news/vtoraya-novost-dnya", "date": "2026-10-01T17:00:00",
     "date_gmt": "2026-10-01T14:00:00", "title": {"rendered": "Вторая новость"}},
])
LISTING = ('<html>' + ''.join(
    f'<a href="https://b.ru/novost-nomer-{i}/">Заголовок</a>'
    f'<time datetime="2026-10-0{i % 9 + 1}T10:00:00+03:00">1 октября</time>'
    for i in range(12)) + '</html>' + 'x' * 4000)


def test_wp_api_detected(monkeypatch):
    def fake(url, headers=None, timeout=30):
        if "wp-json" in url:
            return FakeResponse(WP_ITEMS, headers={"X-WP-Total": "56431", "X-WP-TotalPages": "565"})
        return FakeResponse("", 404)
    monkeypatch.setattr(bulk.requests, "get", fake)
    report = bulk.probe_bulk("ForkLog", "https://f.com/", pause=0)

    assert report.wp_api["works"] is True
    assert report.wp_api["has_date_gmt"] is True
    assert report.wp_api["total"] == 56431
    assert report.wp_api["example_title"] == "Первая новость"
    assert "программный интерфейс WordPress" in report.best_method()
    # когда интерфейс есть, постраничную ленту не проверяем
    assert report.listing == {}


def test_listing_used_when_no_wp_api(monkeypatch):
    def fake(url, headers=None, timeout=30):
        if "wp-json" in url:
            return FakeResponse("<html>страница не найдена</html>", 404)
        if "PAGEN_1" in url:
            return FakeResponse(LISTING)
        return FakeResponse("", 404)
    monkeypatch.setattr(bulk.requests, "get", fake)
    report = bulk.probe_bulk("Bits.Media", "https://b.ru/", pause=0)

    assert report.wp_api["works"] is False
    assert report.listing["works"] is True
    assert report.listing["template"] == "/news/?PAGEN_1={page}"
    assert report.listing["articles_on_page"] >= 5
    assert "постраничная лента" in report.best_method()


def test_no_bulk_method(monkeypatch):
    monkeypatch.setattr(bulk.requests, "get",
                        lambda url, headers=None, timeout=30: FakeResponse("", 404))
    report = bulk.probe_bulk("Тест", "https://x.ru/", pause=0)
    assert "по одной" in report.best_method()
    assert "ВЫВОД" in bulk.format_report(report)
