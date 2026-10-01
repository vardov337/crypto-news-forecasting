"""Шаг 1 (задачи 3.1–3.2): часовые свечи BTCUSDT и ETHUSDT на полной сетке UTC.

Что делает скрипт:
  1. качает помесячные архивы с data.binance.vision и сверяет контрольные суммы;
  2. приводит время к UTC и переиндексирует ряд на полную часовую сетку;
  3. проверяет данные (порядок времени, шаг сетки, корректность свечей);
  4. сохраняет по файлу на актив в data/raw/binance/;
  5. пишет отчёт о покрытии и пропусках в results/metrics/.

Запуск:  python scripts/01_download_prices.py
Повторный запуск ничего не перекачивает: архивы берутся из кэша.
"""
import pandas as pd

from cryptonews.cli import parse_args
from cryptonews.config import load_config
from cryptonews.data import binance
from cryptonews.utils import get_logger, run_manifest, save_json


def main() -> None:
    args = parse_args(__doc__)
    cfg = load_config(args.config)
    log = get_logger()

    prices_cfg = cfg["data"]["prices"]
    symbols = prices_cfg["symbols"]
    interval = prices_cfg["interval"]
    start = prices_cfg["history_start"]
    end = cfg["time"]["sample_end"] or binance.utc_now_month()

    raw_dir = cfg["paths"]["data_dir"] / "raw" / "binance"
    cache_dir = raw_dir / "monthly_zip"
    results_dir = cfg["paths"]["results_dir"] / "metrics"
    reports = []

    for symbol in symbols:
        log.info("%s: скачиваю месяцы с %s по %s", symbol, start, end)
        prices, report = binance.download_symbol(
            symbol=symbol, start=start, end=end, cache_dir=cache_dir, interval=interval,
        )
        binance.validate(prices, symbol)

        out_path = raw_dir / f"{symbol}_{interval}.parquet"
        prices.to_parquet(out_path)
        reports.append(report)

        log.info(
            "%s: %s … %s, часов в сетке %d, пропущено %d (%.3f%%), месяцев %d",
            symbol, report.first_timestamp, report.last_timestamp, report.rows_on_grid,
            report.missing_hours, 100 * report.missing_hours / max(report.rows_on_grid, 1),
            report.months_downloaded + report.months_from_cache,
        )
        if report.gaps:
            log.info("%s: самые длинные пропуски биржи:", symbol)
            for gap in sorted(report.gaps, key=lambda g: -g["hours"])[:5]:
                log.info("    %s … %s — %d ч", gap["start"], gap["end"], gap["hours"])
        log.info("%s: сохранено в %s", symbol, out_path)

    table = binance.coverage_table(reports)
    table.to_csv(results_dir / "prices_coverage.csv", index=False, encoding="utf-8")
    save_json(
        {"manifest": run_manifest(cfg), "reports": [r.as_dict() for r in reports]},
        results_dir / "prices_download_report.json",
    )

    log.info("\n%s", table.to_string(index=False))
    log.info("Отчёт: %s", results_dir / "prices_download_report.json")
    common_start = max(r.first_timestamp for r in reports)
    common_end = min(r.last_timestamp for r in reports)
    log.info("Общий период по обоим активам: %s … %s", common_start, common_end)


if __name__ == "__main__":
    main()
