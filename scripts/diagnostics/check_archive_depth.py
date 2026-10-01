"""Шаг 1б (решение 1.5): глубина архива у выбранных русскоязычных сайтов.

Читает все карты сайта с новостями и проверяет дату публикации у самых старых
и самых свежих материалов. По результату определяется период, за который
вообще можно собрать русскоязычные новости.

Запросов около полусотни с паузой, примерно 3–5 минут.
Запуск:  python scripts/diagnostics/check_archive_depth.py
"""
import pandas as pd

from cryptonews.cli import parse_args
from cryptonews.config import load_config
from cryptonews.data.ru_news import depth
from cryptonews.utils import get_logger, run_manifest, save_json

# Profinvestment исключён: 1688 адресов против десятков тысяч у остальных,
# даты изменения сброшены массовым обновлением, а проверенные страницы оказались
# справочными карточками монет, а не новостями.
# ForkLog измерен 01.10.2026: 56 431 материал, публикации с 2014-09-13.
# Повторно его не читаем — 30 карт сайта это лишние несколько минут.
SITES = [
    ("Bits.Media", "https://bits.media/sitemap.php"),
]


def main() -> None:
    args = parse_args(__doc__)
    cfg = load_config(args.config)
    log = get_logger()
    pause = float(cfg["data"]["news_ru"].get("request_interval_sec", 1.0))

    reports = []
    for name, index_url in SITES:
        log.info("Измеряю глубину архива: %s …", name)
        report = depth.measure(name, index_url, pause=pause)
        reports.append(report)
        print()
        print(depth.format_report(report))

    table = pd.DataFrame([{
        "Сайт": r.name,
        "Карт прочитано": r.sitemaps_read,
        "Адресов всего": r.total_urls,
        "Публикации с": r.published_dates[0] if r.published_dates else "—",
        "Публикации по": r.published_dates[-1] if r.published_dates else "—",
    } for r in reports])

    results_dir = cfg["paths"]["results_dir"] / "metrics"
    table.to_csv(results_dir / "ru_archive_depth.csv", index=False, encoding="utf-8")
    save_json(
        {"manifest": run_manifest(cfg), "sites": [r.as_dict() for r in reports]},
        results_dir / "ru_archive_depth.json",
    )

    print()
    log.info("Сводка:\n%s", table.to_string(index=False))
    log.info("Подробный отчёт: %s", results_dir / "ru_archive_depth.json")


if __name__ == "__main__":
    main()
