"""Проверки сборщика ForkLog. Сеть не нужна."""
import json
import re

from cryptonews.data.ru_news import forklog


class FakeResponse:
    def __init__(self, text="", status_code=200, headers=None):
        self.text, self.status_code = text, status_code
        self.headers = headers or {}


def make_page(page: int, count: int = 20) -> str:
    return json.dumps([{
        "id": page * 100 + i,
        "link": f"https://forklog.com/news/novost-{page}-{i}",
        "date_gmt": f"2026-10-{page:02d}T1{i % 10}:00:00",
        "title": {"rendered": f"Заголовок &#171;{page}-{i}&#187; <b>жирный</b>"},
        "categories": [3], "tags": [7],
    } for i in range(count)])


def fake_api(pages_available: int = 3):
    def get(self_or_url, *args, **kwargs):
        url = args[0] if args else self_or_url
        page = int(re.search(r"[?&]page=(\d+)", url).group(1))
        headers = {"X-WP-Total": "56383", "X-WP-TotalPages": str(pages_available)}
        if page > pages_available:
            return FakeResponse("[]", 200, headers)
        return FakeResponse(make_page(page), 200, headers)
    return get


def test_clean_title_removes_markup_and_entities():
    assert forklog.clean_title("Заголовок &#171;тест&#187; <b>жирный</b>") == "Заголовок «тест» жирный"


def test_parse_items_marks_time_as_utc():
    records = forklog.parse_items(make_page(1, count=1))
    assert records[0]["published_utc"].endswith("Z")
    assert records[0]["url"].startswith("https://forklog.com/news/")


def test_parse_items_skips_records_without_date():
    payload = json.dumps([{"link": "https://f.com/a", "title": {"rendered": "Без даты"}}])
    assert forklog.parse_items(payload) == []


def test_collect_walks_pages_and_writes_file(tmp_path, monkeypatch):
    monkeypatch.setattr(forklog.requests.Session, "get",
                        lambda self, url, **kw: fake_api(3)(url))
    path = tmp_path / "forklog.jsonl"
    records, report = forklog.collect(path, pause=0, max_pages=10)

    assert report.pages_done == 3 and report.records_total == 60
    assert report.total_records_declared == 56383
    assert len(records) == 60
    assert path.exists() and len(path.read_text(encoding="utf-8").strip().splitlines()) == 60


def test_collect_resumes_without_refetching(tmp_path, monkeypatch):
    monkeypatch.setattr(forklog.requests.Session, "get",
                        lambda self, url, **kw: fake_api(3)(url))
    path = tmp_path / "forklog.jsonl"
    forklog.collect(path, pause=0, max_pages=10)

    calls = []

    def counting_get(self, url, **kw):
        calls.append(url)
        return fake_api(3)(url)

    monkeypatch.setattr(forklog.requests.Session, "get", counting_get)
    records, report = forklog.collect(path, pause=0, max_pages=10)
    # все три страницы уже собраны: повторно они не запрашиваются,
    # нужен только один запрос, чтобы убедиться, что архив закончился
    assert report.pages_skipped == 3 and report.records_new == 0
    assert len(records) == 60
    assert len(calls) == 1 and "&page=4&" in calls[0]


def test_collect_records_failed_page(tmp_path, monkeypatch):
    def flaky(self, url, **kw):
        if "&page=2&" in url:
            return FakeResponse("", 500)
        return fake_api(3)(url)
    monkeypatch.setattr(forklog.requests.Session, "get", flaky)
    monkeypatch.setattr(forklog.time, "sleep", lambda s: None)
    _, report = forklog.collect(tmp_path / "f.jsonl", pause=0, max_pages=3)
    assert report.pages_failed == [2]
    assert report.pages_done == 2


def test_short_page_is_retried_bypassing_cache(tmp_path, monkeypatch):
    """При разведке кэш сайта вернул одну запись вместо двадцати — повторяем в обход кэша."""
    def cached_badly(self, url, **kw):
        if "&page=1&" in url and "nocache" not in url:
            headers = {"X-WP-Total": "60", "X-WP-TotalPages": "3"}
            return FakeResponse(make_page(1, count=1), 200, headers)
        return fake_api(3)(url)
    monkeypatch.setattr(forklog.requests.Session, "get", cached_badly)
    records, report = forklog.collect(tmp_path / "f.jsonl", pause=0, max_pages=10)
    assert report.records_total == 60
    assert report.pages_short == []


def test_empty_page_inside_archive_is_not_the_end(tmp_path, monkeypatch):
    """Пустой ответ посреди архива — сбой, а не конец: сбор идёт дальше."""
    def hole(self, url, **kw):
        if "&page=2&" in url:
            return FakeResponse("[]", 200, {"X-WP-Total": "60", "X-WP-TotalPages": "3"})
        return fake_api(3)(url)
    monkeypatch.setattr(forklog.requests.Session, "get", hole)
    records, report = forklog.collect(tmp_path / "f.jsonl", pause=0, max_pages=10)
    assert report.pages_failed == [2]
    assert report.records_total == 40
