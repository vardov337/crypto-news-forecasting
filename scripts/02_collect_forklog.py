"""Шаг 2 (задачи 2.1–2.5): сбор русскоязычных новостей ForkLog.

Собирает заголовки, адреса и даты публикации через программный интерфейс сайта.
Около 2 850 запросов с паузой, примерно 50–60 минут.

Сбор можно прерывать: результат дописывается на диск после каждой страницы,
а повторный запуск продолжает с того места, где остановился.

Запуск:  python scripts/02_collect_forklog.py
"""
import pandas as pd

from cryptonews.cli import parse_args
from cryptonews.config import load_config
from cryptonews.data.ru_news import forklog
from cryptonews.utils import get_logger, run_manifest, save_json


def main() -> None:
    args = parse_args(__doc__)
    cfg = load_config(args.config)
    log = get_logger()
    pause = float(cfg["data"]["news_ru"].get("request_interval_sec", 1.0))

    raw_dir = cfg["paths"]["data_dir"] / "raw" / "ru_news"
    jsonl_path = raw_dir / "forklog_posts.jsonl"

    log.info("Собираю ForkLog, файл: %s", jsonl_path)
    log.info("Прервать можно в любой момент — повторный запуск продолжит с того же места.")
    records, report = forklog.collect(jsonl_path, pause=pause, log=log)

    frame = pd.DataFrame(records)
    frame["published_utc"] = pd.to_datetime(frame["published_utc"], utc=True, format="ISO8601")
    frame = frame.sort_values("published_utc").drop_duplicates("url")
    out_path = raw_dir / "forklog_posts.parquet"
    frame.drop(columns=["_page"], errors="ignore").to_parquet(out_path, index=False)

    results_dir = cfg["paths"]["results_dir"] / "metrics"
    save_json({"manifest": run_manifest(cfg), "report": report.as_dict()},
              results_dir / "forklog_collect_report.json")

    log.info("Записей: %d (новых за этот запуск: %d)", len(frame), report.records_new)
    log.info("Период: %s … %s", frame["published_utc"].min(), frame["published_utc"].max())
    log.info("Страниц пройдено: %d, пропущено как уже собранные: %d, не ответили: %d",
             report.pages_done, report.pages_skipped, len(report.pages_failed))
    if report.pages_failed:
        log.warning("Не ответили страницы: %s", report.pages_failed[:20])
        log.warning("Запустите скрипт ещё раз — он доберёт только их.")
    if report.pages_short:
        log.warning("Неполных страниц внутри архива: %d (первые: %s)",
                    len(report.pages_short), report.pages_short[:10])
    declared = report.total_records_declared
    if declared:
        coverage = len(frame) / declared
        log.info("Полнота: собрано %d из %d заявленных сайтом (%.2f%%)",
                 len(frame), declared, 100 * coverage)
        if coverage < 0.99:
            log.warning("Собрано меньше 99%% — запустите скрипт ещё раз.")

    by_year = frame.groupby(frame["published_utc"].dt.year).size()
    log.info("Материалов по годам:\n%s", by_year.to_string())
    log.info("Сохранено: %s", out_path)


if __name__ == "__main__":
    main()
