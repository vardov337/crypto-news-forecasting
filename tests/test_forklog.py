"""Проверки сборщика ForkLog. Сеть не нужна.

Имитация повторяет поведение настоящего сайта: по двадцать записей на странице,
код 400 на номер страницы за концом архива, заголовки X-WP-Total и X-WP-TotalPages.
"""
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


def page_number(url: str) -> int:
    return int(re.search(r"[?&]page=(\d+)", url).group(1))


def fake_api(pages: int = 3, last_page_count: int = 20):
    total = (pages - 1) * 20 + last_page_count
    headers = {"X-WP-Total": str(total), "X-WP-TotalPages": str(pages)}

    def get(url):
        page = page_number(url)
        if page > pages:
            return FakeResponse('{"code":"rest_post_invalid_page_number"}', 400)
        count = last_page_count if page == pages else 20
        return FakeResponse(make_page(page, count), 200, headers)
    return get


def patch_session(monkeypatch, getter, calls=None):
    def get(self, url, **kwargs):
        if calls is not None:
            calls.append(url)
        return getter(url)
    monkeypatch.setattr(forklog.requests.Session, "get", get)
    monkeypatch.setattr(forklog.time, "sleep", lambda seconds: None)


def test_clean_title_removes_markup_and_entities():
    assert forklog.clean_title("Заголовок &#171;тест&#187; <b>жирный</b>") == "Заголовок «тест» жирный"


def test_parse_items_marks_time_as_utc():
    records = forklog.parse_items(make_page(1, count=1))
    assert records[0]["published_utc"].endswith("Z")
    assert records[0]["url"].startswith("https://forklog.com/news/")


def test_parse_items_skips_records_without_date():
    payload = json.dumps([{"link": "https://f.com/a", "title": {"rendered": "Без даты"}}])
    assert forklog.parse_items(payload) == []


def test_collect_walks_pages_and_stops_on_400(tmp_path, monkeypatch):
    patch_session(monkeypatch, fake_api(3))
    path = tmp_path / "forklog.jsonl"
    records, report = forklog.collect(path, pause=0, max_pages=10)

    assert report.pages_done == 3 and report.records_total == 60
    assert report.total_records_declared == 60 and report.total_pages_declared == 3
    assert report.end_page == 4
    assert report.pages_failed == [] and report.pages_short == []
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 60


def test_last_short_page_is_not_an_anomaly(tmp_path, monkeypatch):
    """Последняя страница архива неполная (у настоящего сайта — 3 записи из 20), это норма."""
    patch_session(monkeypatch, fake_api(3, last_page_count=3))
    records, report = forklog.collect(tmp_path / "f.jsonl", pause=0, max_pages=10)
    assert report.records_total == 43
    assert report.pages_short == [] and report.pages_failed == []


def test_collect_resumes_without_refetching(tmp_path, monkeypatch):
    patch_session(monkeypatch, fake_api(3))
    path = tmp_path / "forklog.jsonl"
    forklog.collect(path, pause=0, max_pages=10)

    calls = []
    patch_session(monkeypatch, fake_api(3), calls)
    records, report = forklog.collect(path, pause=0, max_pages=10)
    # уже собранные страницы не запрашиваются: только проверка числа записей
    # на сайте и одна страница, чтобы увидеть конец архива
    assert report.pages_skipped == 3 and report.records_new == 0
    assert len(records) == 60
    assert len(calls) == 2
    assert "nocache" in calls[0] and page_number(calls[1]) == 4


def test_failed_page_is_reported(tmp_path, monkeypatch):
    good = fake_api(3)
    patch_session(monkeypatch, lambda url: FakeResponse("", 500) if page_number(url) == 2
                  and "nocache" not in url else good(url))
    _, report = forklog.collect(tmp_path / "f.jsonl", pause=0, max_pages=10)
    assert report.pages_failed == [2]
    assert report.pages_done == 2 and report.end_page == 4


def test_short_page_is_retried_bypassing_cache(tmp_path, monkeypatch):
    """При разведке кэш сайта вернул одну запись вместо двадцати — повторяем в обход кэша."""
    good = fake_api(3)

    def cached_badly(url):
        if page_number(url) == 1 and "nocache" not in url:
            return FakeResponse(make_page(1, count=1), 200, {"X-WP-Total": "54994",
                                                             "X-WP-TotalPages": "54994"})
        return good(url)
    patch_session(monkeypatch, cached_badly)
    records, report = forklog.collect(tmp_path / "f.jsonl", pause=0, max_pages=10)
    assert report.records_total == 60
    assert report.pages_short == []
    # число записей берётся из свежего ответа, а не из устаревшего кэша
    assert report.total_records_declared == 60


def test_empty_page_inside_archive_is_not_the_end(tmp_path, monkeypatch):
    """Пустой ответ с кодом 200 посреди архива — сбой, а не конец: сбор идёт дальше."""
    good = fake_api(3)
    patch_session(monkeypatch, lambda url: FakeResponse("[]", 200) if page_number(url) == 2
                  else good(url))
    records, report = forklog.collect(tmp_path / "f.jsonl", pause=0, max_pages=10)
    assert report.pages_failed == [2]
    assert report.records_total == 40
    assert report.end_page == 4
