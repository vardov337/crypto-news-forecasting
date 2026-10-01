"""Шаг 1в (перед задачами 2.1–2.3): какой способ массового сбора доступен.

Проверяет у каждого источника программный интерфейс WordPress и постраничные
ленты новостей. От результата зависит, сколько времени займёт сбор: по одной
статье это больше тридцати часов, пачками — часы или даже минуты.

Несколько запросов на сайт, меньше минуты.
Запуск:  python scripts/01c_probe_bulk_access.py
"""
import pandas as pd

from cryptonews.cli import parse_args
from cryptonews.config import load_config
from cryptonews.data.ru_news import bulk
from cryptonews.utils import get_logger, run_manifest, save_json

SITES = [
    ("ForkLog", "https://forklog.com/"),
    ("Bits.Media", "https://bits.media/"),
]


def main() -> None:
    args = parse_args(__doc__)
    cfg = load_config(args.config)
    log = get_logger()
    pause = float(cfg["data"]["news_ru"].get("request_interval_sec", 1.0))

    reports = []
    for name, base in SITES:
        log.info("Проверяю способы массового сбора: %s …", name)
        report = bulk.probe_bulk(name, base, pause=pause)
        reports.append(report)
        print()
        print(bulk.format_report(report))

    table = pd.DataFrame([{
        "Сайт": r.name,
        "Интерфейс WordPress": "да" if r.wp_api.get("works") else "нет",
        "Постраничная лента": "да" if r.listing.get("works") else "нет",
        "Вывод": r.best_method(),
    } for r in reports])

    results_dir = cfg["paths"]["results_dir"] / "metrics"
    table.to_csv(results_dir / "ru_bulk_access.csv", index=False, encoding="utf-8")
    save_json({"manifest": run_manifest(cfg), "sites": [r.__dict__ for r in reports]},
              results_dir / "ru_bulk_access.json")

    print()
    log.info("Сводка:\n%s", table.to_string(index=False))


if __name__ == "__main__":
    main()
