"""Шаг 1д: работает ли постраничный обход у обоих источников.

Два вопроса, от которых зависит, возможен ли сбор вообще.

  ForkLog:    интерфейс проигнорировал отбор по датам и сортировку. Если он так же
              игнорирует номер страницы, то отдаёт только последние двадцать новостей
              и для архива не годится — придётся обходить статьи по адресам из карты сайта.
  Bits.Media: в ленте есть даты, но поиск в прошлый раз зацепился за меню. Теперь
              ищем сами даты и печатаем текст вокруг них — так видно карточку новости.
              Отдельно смотрим глубокую страницу: там должен появиться год.

Запуск:  python scripts/01e_inspect_pagination.py
"""
import json
import re

import requests

from cryptonews.cli import parse_args
from cryptonews.config import load_config
from cryptonews.data.ru_news.probe import HEADERS

MONTHS = "января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря"
RU_DATE_SHORT = re.compile(rf"\d{{1,2}}\s+(?:{MONTHS})")
RU_DATE_YEAR = re.compile(rf"\d{{1,2}}\s+(?:{MONTHS})\s+20\d{{2}}")
TAGS = re.compile(r"<[^>]+>")


def get(url, timeout=30):
    try:
        return requests.get(url, headers=HEADERS, timeout=timeout)
    except requests.RequestException as error:
        print(f"    запрос не прошёл: {error}")
        return None


def clean(chunk: str) -> str:
    return re.sub(r"\s+", " ", TAGS.sub(" ", chunk)).strip()


def forklog():
    base = "https://forklog.com/wp-json/wp/v2/posts?_fields=link,date_gmt&per_page=20"
    print("\n=== ForkLog: меняется ли ответ при смене страницы")
    seen = {}
    for query in ("&page=1", "&page=2", "&page=3", "&offset=40", "&page=2000"):
        r = get(base + query)
        if r is None:
            continue
        if r.status_code != 200:
            print(f"  {query:<12} код {r.status_code}: {clean(r.text)[:160]}")
            continue
        items = json.loads(r.text)
        first = items[0] if items else {}
        seen[query] = first.get("link")
        print(f"  {query:<12} код 200, записей {len(items)}, "
              f"первая дата {first.get('date_gmt')}")
        print(f"               {first.get('link')}")
    unique = len({v for v in seen.values() if v})
    print(f"  РАЗНЫХ первых записей: {unique} из {len(seen)} — "
          f"{'постраничный обход работает' if unique > 1 else 'номер страницы игнорируется'}")


def bits(page: int, label: str):
    url = f"https://bits.media/news/?PAGEN_1={page}"
    r = get(url)
    print(f"\n=== Bits.Media: {label} ({url})")
    if r is None or r.status_code != 200:
        print(f"  код {r.status_code if r else 'нет ответа'}")
        return
    html = r.text
    with_year = RU_DATE_YEAR.findall(html)
    print(f"  длина {len(html)}, дат без года {len(RU_DATE_SHORT.findall(html))}, "
          f"дат с годом {len(with_year)}")
    if with_year:
        print(f"  примеры дат с годом: {', '.join(with_year[:5])}")

    print("  текст вокруг первых дат:")
    shown = 0
    for match in RU_DATE_SHORT.finditer(html):
        chunk = clean(html[max(0, match.start() - 400):match.start() + 200])
        if len(chunk) < 40:
            continue
        print(f"      …{chunk[-300:]}")
        # ближайшая ссылка перед датой — скорее всего, адрес этой новости
        before = html[max(0, match.start() - 2000):match.start()]
        links = re.findall(r'href="(/[^"#?]+/)"', before)
        print(f"          ближайшие ссылки: {links[-3:]}")
        shown += 1
        if shown >= 4:
            break


def main():
    args = parse_args(__doc__)
    load_config(args.config)
    forklog()
    bits(2, "вторая страница ленты")
    bits(500, "глубокая страница — здесь должен появиться год")
    print("\nПришлите вывод целиком.")


if __name__ == "__main__":
    main()
