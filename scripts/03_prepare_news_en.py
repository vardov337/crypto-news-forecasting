"""Шаг 3 (задачи 3.3–3.6): англоязычные новости CryptoVision.

Что делает скрипт:
  1. берёт набор CryptoVision фиксированной версии (скачивает с Mendeley Data,
     если его ещё нет в папке data/raw/cryptovision/);
  2. определяет часовой пояс меток времени сверкой цен приложенных к новостям
     свечей с архивом Binance — для этого докачивает 15-минутные свечи за
     несколько месяцев;
  3. переводит время в UTC, удаляет повторы, считает таблицу этапов очистки;
  4. сохраняет в data/interim/news_en.parquet только заголовок, адрес, источник,
     монету и время — цены из набора дальше не используются.

Если пояс определить надёжно не удалось, скрипт останавливается и печатает,
что именно он увидел. Угадывать пояс нельзя: ошибка в час сдвигает новость на
соседнюю свечу, а это утечка будущего.

Запуск:  python scripts/03_prepare_news_en.py
"""
import pandas as pd

from cryptonews.cli import parse_args
from cryptonews.config import load_config
from cryptonews.data import binance, cryptovision as cv
from cryptonews.data import news as news_rules
from cryptonews.utils import get_logger, run_manifest, save_json, sha256_file


def configured_offset(value) -> int | None:
    """Пояс из конфигурации: null — определить автоматически, число — сдвиг в минутах."""
    if value in (None, "", "auto"):
        return None
    return int(value)


def main() -> None:
    args = parse_args(__doc__)
    cfg = load_config(args.config)
    log = get_logger()
    news_cfg = cfg["data"]["news_en"]
    data_dir, results_dir = cfg["paths"]["data_dir"], cfg["paths"]["results_dir"]
    raw_dir = data_dir / "raw" / "cryptovision"

    # 1. Набор
    csv_path, origin = cv.download(raw_dir, version=int(news_cfg["version"]))
    log.info("CryptoVision: %s (%s), %.1f МБ", csv_path.name, origin, csv_path.stat().st_size / 1e6)
    raw = pd.read_csv(csv_path, low_memory=False)
    frame, mapping = cv.normalize_columns(raw)
    log.info("Записей: %d. Колонки: %s", len(frame), mapping)

    frame["date_time"], share_tz = cv.parse_times(frame["date_time"])
    log.info("Меток времени с указанным поясом: %.1f%%", 100 * share_tz)
    if 0.05 < share_tz < 0.95:
        log.warning("Пояс указан только у части меток — проверьте формат времени в наборе")
    log.info("Метки времени «как записаны»: %s … %s",
             frame["date_time"].min(), frame["date_time"].max())
    if "coin_type" in frame:
        log.info("Монеты в наборе:\n%s", frame["coin_type"].value_counts().head(15).to_string())

    # 2. Часовой пояс
    months = cv.pick_sample_months(frame)
    log.info("Сверяю пояс по 15-минутным свечам Binance за %d месяцев: %s",
             len(months), ", ".join(months))
    cache = data_dir / "raw" / "binance" / "monthly_zip_15m"
    candles = {symbol: binance.load_months(symbol, months, cache, interval="15m")
               for symbol in ("BTCUSDT", "ETHUSDT")}
    tz = cv.detect_timezone(frame, candles, months)

    log.info("Проверено новостей: %d, однозначных совпадений цен: %d",
             tz.records_checked, tz.records_matched)
    log.info("Лучшие варианты (сдвиг от UTC в минутах, правило, доля совпадений):")
    for c in tz.candidates:
        log.info("    %+5d  %-28s %.1f%%", c["offset_minutes"], c["rule"], 100 * c["share"])
    log.info("Самые частые разности «время свечи минус метка», мин: %s",
             ", ".join(f"{d['delta_min']} ({d['count']})" for d in tz.top_deltas))
    log.info("По месяцам:")
    for m in tz.per_month:
        log.info("    %s  совпадений %4d, лучший сдвиг %+5d, доля %.1f%%%s", m["month"], m["matched"],
                 m["best_offset_minutes"], 100 * m["share"], "" if m["agrees"] else "  ← не совпадает")
    log.info("ВЫВОД: %s", tz.verdict)

    save_json({"manifest": run_manifest(cfg), "csv": str(csv_path), "csv_sha256": sha256_file(csv_path),
               "column_mapping": mapping, "share_with_tz_suffix": share_tz, "timezone": tz.as_dict()},
              results_dir / "metrics" / "cryptovision_timezone.json")

    expected = configured_offset(news_cfg.get("source_timezone"))
    if not cv.is_reliable(tz):
        raise SystemExit("Пояс не определён надёжно — дальше не иду. Пришлите вывод выше.")
    if expected is not None and expected != tz.best_offset_minutes:
        raise SystemExit(f"В конфигурации указан сдвиг {expected} мин, а сверка показала "
                         f"{tz.best_offset_minutes} мин. Разберитесь, прежде чем продолжать.")

    # 3. Очистка
    news, stages = cv.clean(frame, offset_minutes=tz.best_offset_minutes,
                            dedup_window_hours=int(cfg["data"]["dedup"]["window_hours"]))
    out_path = data_dir / "interim" / "news_en.parquet"
    news.to_parquet(out_path, index=False)

    tables_dir = results_dir / "tables"
    stages.to_csv(tables_dir / "cleaning_news_en.csv", index=False, encoding="utf-8")
    summaries = news_rules.summary_tables(news)
    for name, table in summaries.items():
        table.to_csv(tables_dir / f"news_en_{name}.csv", encoding="utf-8")

    log.info("Этапы очистки:\n%s", stages.to_string(index=False))
    log.info("По источникам:\n%s", summaries["by_source"].head(10).to_string())
    log.info("По годам:\n%s", summaries["by_year"].to_string())
    log.info("Привязка к монетам по заголовку:\n%s", summaries["by_coin"].to_string(index=False))
    if "coin_type" in news:
        agree = news.assign(label=news["coin_type"].astype("string").str.lower())
        check = agree.groupby("label")[["mentions_btc", "mentions_eth"]].mean().round(3)
        log.info("Сверка с разметкой монет авторами набора (доля заголовков с упоминанием):\n%s",
                 check.loc[check.index.isin(["bitcoin", "ethereum", "btc", "eth"])].to_string())
    log.info("Период в UTC: %s … %s", news["published_utc"].min(), news["published_utc"].max())
    log.info("Сохранено: %s", out_path)


if __name__ == "__main__":
    main()
