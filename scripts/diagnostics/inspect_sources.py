"""Шаг 1г: разметка ленты Bits.Media и пределы интерфейса ForkLog.

Диагностика перед написанием сборщиков. Ничего не собирает, только показывает,
как устроены ответы сайтов.

  ForkLog:    сколько записей отдаётся за запрос и работает ли отбор по датам.
              Отбор по датам важен: он позволяет идти по месяцам, а не по номерам
              страниц, которые на больших смещениях работают медленно.
  Bits.Media: что на самом деле написано рядом со ссылками в ленте новостей.
              Если даты там есть обычным текстом, сбор ускорится в двадцать раз.

Запуск:  python scripts/diagnostics/inspect_sources.py
"""
import json
import re

import requests

from cryptonews.cli import parse_args
from cryptonews.config import load_config
from cryptonews.data.ru_news.probe import HEADERS
from cryptonews.utils import get_logger

MONTHS = ("января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря")
RU_DATE = re.compile(rf"\d{{1,2}}\s+(?:{MONTHS})\s+20\d{{2}}")
RU_DATE_SHORT = re.compile(rf"\d{{1,2}}\s+(?:{MONTHS})")
CLOCK = re.compile(r"\b\d{1,2}:\d{2}\b")
ISO_DATE = re.compile(r"20\d{2}-\d{2}-\d{2}")
TAGS = re.compile(r"<[^>]+>")


def get(url, timeout=30):
    try:
        return requests.get(url, headers=HEADERS, timeout=timeout)
    except requests.RequestException as error:
        print(f"    запрос не прошёл: {error}")
        return None


def inspect_forklog(log):
    base = "https://forklog.com/wp-json/wp/v2/posts"
    print("\n=== ForkLog: пределы интерфейса")
    for per_page in (20, 50, 100):
        r = get(f"{base}?per_page={per_page}&_fields=link")
        if r is None:
            continue
        got = len(json.loads(r.text)) if r.status_code == 200 else 0
        print(f"  запрошено {per_page:>3}: код {r.status_code}, получено {got}")

    print("\n=== ForkLog: отбор по датам (нужен, чтобы идти по месяцам)")
    r = get(f"{base}?after=2017-08-01T00:00:00&before=2017-09-01T00:00:00"
            f"&per_page=20&orderby=date&order=asc&_fields=link,date_gmt")
    if r is not None and r.status_code == 200:
        items = json.loads(r.text)
        print(f"  август 2017: код 200, записей в ответе {len(items)}, "
              f"всего за месяц {r.headers.get('X-WP-Total')}")
        for item in items[:3]:
            print(f"      {item.get('date_gmt')}  {item.get('link')}")
    else:
        print(f"  отбор по датам не сработал: код {r.status_code if r else 'нет ответа'}")


def inspect_bits(log):
    print("\n=== Bits.Media: что написано в ленте новостей")
    for url in ("https://bits.media/news/?PAGEN_1=2", "https://bits.media/news/"):
        r = get(url)
        if r is None or r.status_code != 200:
            print(f"  {url}: код {r.status_code if r else 'нет ответа'}")
            continue
        html = r.text
        print(f"\n  {url}: код 200, длина {len(html)} символов")
        print(f"      дат «1 октября 2026»: {len(RU_DATE.findall(html))}")
        print(f"      дат «1 октября»:      {len(RU_DATE_SHORT.findall(html))}")
        print(f"      времени «18:23»:      {len(CLOCK.findall(html))}")
        print(f"      дат «2026-10-01»:     {len(ISO_DATE.findall(html))}")

        links = re.findall(r'href="(/[^"]*-[^"]*-[^"]*/)"', html)
        unique = list(dict.fromkeys(links))
        print(f"      ссылок на статьи:     {len(unique)}")
        for link in unique[:3]:
            position = html.find(f'href="{link}"')
            chunk = html[max(0, position - 600):position + 600]
            text = TAGS.sub(" ", chunk)
            text = re.sub(r"\s+", " ", text).strip()
            print(f"\n      --- {link}")
            print(f"          текст рядом: …{text[-320:]}")
        break


def main():
    args = parse_args(__doc__)
    load_config(args.config)
    log = get_logger()
    inspect_forklog(log)
    inspect_bits(log)
    print("\nПришлите этот вывод целиком — по нему я напишу сборщики.")


if __name__ == "__main__":
    main()
