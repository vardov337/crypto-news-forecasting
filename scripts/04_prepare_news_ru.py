"""Шаг 4 (задачи 4.1–4.4): русскоязычные новости ForkLog — очистка и привязка к монетам.

Что делает скрипт:
  1. берёт снимок ForkLog, собранный шагом 2 (data/raw/ru_news/forklog_posts.parquet);
  2. применяет те же правила очистки, что и к англоязычным новостям (шаг 3):
     время уже в UTC, повторы адреса и повторы заголовка в пределах суток убираются;
  3. отмечает упоминания биткоина и эфира по заголовку — тем же правилом, что и в шаге 3;
  4. сохраняет data/interim/news_ru.parquet и таблицы для раздела о данных.

Сеть не нужна, работает несколько секунд.
Запуск:  python scripts/04_prepare_news_ru.py
"""
import pandas as pd

from cryptonews.cli import parse_args
from cryptonews.config import load_config
from cryptonews.data import news as news_rules
from cryptonews.utils import get_logger, run_manifest, save_json, sha256_file


def main() -> None:
    args = parse_args(__doc__)
    cfg = load_config(args.config)
    log = get_logger()
    data_dir, results_dir = cfg["paths"]["data_dir"], cfg["paths"]["results_dir"]

    raw_path = data_dir / "raw" / "ru_news" / "forklog_posts.parquet"
    if not raw_path.exists():
        raise SystemExit(f"Нет файла {raw_path} — сначала выполните шаг 2 (02_collect_forklog.py)")
    raw = pd.read_parquet(raw_path)
    raw["published_utc"] = pd.to_datetime(raw["published_utc"], utc=True)
    log.info("Снимок ForkLog от %s: %d записей, %s … %s", cfg["data"]["news_ru"].get("snapshot_date"),
             len(raw), raw["published_utc"].min(), raw["published_utc"].max())

    news, stages = news_rules.clean_utc(
        raw, dedup_window_hours=int(cfg["data"]["dedup"]["window_hours"]),
        first_stage="Записей в снимке ForkLog",
    )
    out_path = data_dir / "interim" / "news_ru.parquet"
    news.to_parquet(out_path, index=False)

    tables_dir = results_dir / "tables"
    stages.to_csv(tables_dir / "cleaning_news_ru.csv", index=False, encoding="utf-8")
    summaries = news_rules.summary_tables(news)
    for name, table in summaries.items():
        table.to_csv(tables_dir / f"news_ru_{name}.csv", encoding="utf-8")
    save_json({"manifest": run_manifest(cfg), "input": str(raw_path),
               "input_sha256": sha256_file(raw_path), "records_out": len(news)},
              results_dir / "metrics" / "news_ru_prepare.json")

    log.info("Этапы очистки:\n%s", stages.to_string(index=False))
    log.info("По годам:\n%s", summaries["by_year"].to_string())
    log.info("Привязка к монетам по заголовку:\n%s", summaries["by_coin"].to_string(index=False))
    examples = news[news["mentions_eth"]].sample(min(5, int(news["mentions_eth"].sum())), random_state=0)
    log.info("Примеры заголовков с эфиром (для проверки правила):\n%s",
             "\n".join("    " + t for t in examples["title"]))
    log.info("Сохранено: %s", out_path)


if __name__ == "__main__":
    main()
