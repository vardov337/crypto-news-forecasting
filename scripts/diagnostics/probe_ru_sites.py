"""Шаг 1а (задачи 1.1–1.4): разведка русскоязычных источников.

Скрипт обходит три сайта и по каждому выясняет: отвечает ли сайт обычному
запросу, что разрешает robots.txt, есть ли карта сайта и RSS, указано ли время
публикации с точностью до минуты и до какого года доходит архив.

Запросов делается несколько десятков, с паузой — это разведка, а не сбор данных.
Результат печатается в консоль и сохраняется в results/metrics/ru_sites_probe.json.

Запуск:  python scripts/diagnostics/probe_ru_sites.py
"""
import pandas as pd

from cryptonews.cli import parse_args
from cryptonews.config import load_config
from cryptonews.data.ru_news import probe
from cryptonews.utils import get_logger, run_manifest, save_json

# RBC Crypto исключён по итогам первой разведки: сайт отвечает кодом 401 на обычный
# запрос, то есть закрыт для всего, что не похоже на браузер. Защиту не обходим.
SITES = [
    ("Bits.Media", "https://bits.media/"),
    ("Profinvestment", "https://profinvestment.com/"),
    ("ForkLog", "https://forklog.com/"),
]


def main() -> None:
    args = parse_args(__doc__)
    cfg = load_config(args.config)
    log = get_logger()
    pause = float(cfg["data"]["news_ru"].get("request_interval_sec", 1.0))

    reports = []
    for name, url in SITES:
        log.info("Проверяю %s …", name)
        report = probe.probe_site(name, url, pause=pause)
        reports.append(report)
        print()
        print(probe.format_report(report))

    table = pd.DataFrame([{
        "Сайт": r.name,
        "Доступен": "да" if r.reachable else "нет",
        "robots разрешает": {True: "да", False: "нет", None: "нет правила"}[r.robots_allows],
        "RSS": "да" if r.rss_url else "нет",
        "Карта сайта": "да" if r.sitemaps else "нет",
        "Ранняя дата": r.earliest_date or "—",
        "Адресов в картах": r.sitemap_urls_total,
        "Время до минут": {True: "да", False: "нет", None: "неизвестно"}[r.has_minutes],
        "Вердикт": r.verdict(),
    } for r in reports])

    results_dir = cfg["paths"]["results_dir"] / "metrics"
    table.to_csv(results_dir / "ru_sites_probe.csv", index=False, encoding="utf-8")
    save_json(
        {"manifest": run_manifest(cfg), "sites": [r.__dict__ for r in reports]},
        results_dir / "ru_sites_probe.json",
    )

    print()
    log.info("Сводка:\n%s", table.to_string(index=False))
    log.info("Подробный отчёт: %s", results_dir / "ru_sites_probe.json")
    log.info("Пришлите этот вывод — по нему примем решение 1.5, какие сайты берём.")


if __name__ == "__main__":
    main()
