"""Шаг 3 (задачи 3.3–3.6): англоязычные новости CryptoVision.

Что делает скрипт:
  1. берёт набор CryptoVision фиксированной версии из папки data/raw/cryptovision/:
     таблицы или архив, скачанный в браузере кнопкой «Download All» на Mendeley
     (автоматическое скачивание Mendeley с осени 2026 года отклоняет);
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

from cryptonews import period
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
    files, origin = cv.obtain(raw_dir, version=int(news_cfg["version"]))
    log.info("CryptoVision, файлы %s:", origin)
    for path in files:
        log.info("    %s, %.1f МБ", path.name, path.stat().st_size / 1e6)
    frame, files_info = cv.load_raw(files)
    for item in files_info:
        log.info("    %s: строк %d%s", item["file"], item["rows"],
                 "" if item["used"] else " — пропущен, нет нужных колонок")
    used = [i for i in files_info if i["used"]]
    log.info("Записей: %d. Колонки: %s", len(frame), used[0]["columns"])
    declared = news_cfg.get("declared_records")
    if declared and abs(len(frame) - declared) > 0.01 * declared:
        log.warning("Авторы набора заявляют %d записей, а прочитано %d — проверьте, "
                    "та ли версия набора скачана и нет ли лишних файлов в папке", declared, len(frame))

    raw_text = frame["date_time"].astype("string")
    log.info("Примеры меток времени как в файле: %s", ", ".join(raw_text.dropna().head(3)))
    frame["date_time"], share_tz = cv.parse_times(frame["date_time"])
    log.info("Меток времени с указанным поясом: %.1f%%, не разобрано: %d",
             100 * share_tz, int(frame["date_time"].isna().sum()))
    if 0.05 < share_tz < 0.95:
        log.warning("Пояс указан только у части меток — проверьте формат времени в наборе")
    log.info("Метки времени «как записаны»: %s … %s",
             frame["date_time"].min(), frame["date_time"].max())
    if "coin_type" in frame:
        log.info("Монеты в наборе:\n%s", frame["coin_type"].value_counts().head(15).to_string())
    frame["source"] = frame["url"].map(news_rules.source_of)
    precision = news_rules.time_precision(frame.dropna(subset=["date_time"]), column="date_time")
    log.info("Точность времени публикации по источникам (как записано в файле):\n%s",
             precision.to_string(index=False))

    # 2. Часовой пояс
    months = cv.pick_sample_months(frame)
    log.info("Сверяю пояс по 15-минутным свечам Binance за %d месяцев: %s",
             len(months), ", ".join(months))
    cache = data_dir / "raw" / "binance" / "monthly_zip_15m"
    candles = {symbol: binance.load_months(symbol, months, cache, interval="15m")
               for symbol in ("BTCUSDT", "ETHUSDT")}
    tz = cv.detect_timezone(frame, candles, months)

    log.info("Исключено из сверки как записи без точного времени: %d", tz.records_imprecise)
    log.info("Проверено новостей: %d, однозначных совпадений цен: %d",
             tz.records_checked, tz.records_matched)
    log.info("Сдвиги от UTC, при которых свеча лежит в пределах 15 минут от метки (доля совпадений):")
    for c in tz.candidates:
        log.info("    %+5d мин  %.1f%%", c["offset_minutes"], 100 * c["share"])
    log.info("Вне окна: свеча раньше метки — %.1f%%, позже — %.1f%%",
             100 * tz.outside_before, 100 * tz.outside_after)
    log.info("Как набор выбирал свечу при найденном сдвиге: %s",
             ", ".join(f"{k} — {100 * v:.1f}%" for k, v in tz.rule_shares.items()))
    log.info("Самые частые разности «время свечи минус метка», мин: %s",
             ", ".join(f"{d['delta_min']} ({d['count']})" for d in tz.top_deltas))
    log.info("По месяцам:")
    for m in tz.per_month:
        log.info("    %s  совпадений %4d, лучший сдвиг %+5d, в окне %.1f%%%s", m["month"], m["matched"],
                 m["best_offset_minutes"], 100 * m["share"], "" if m["agrees"] else "  ← не совпадает")
    log.info("По источникам:")
    for m in tz.per_source:
        log.info("    %-22s совпадений %5d, лучший сдвиг %+5d, в окне при общем сдвиге %.1f%%%s",
                 m["source"], m["matched"], m["best_offset_minutes"], 100 * m["share_at_global"],
                 "" if m["agrees"] else "  ← расходится")
    if share_tz > 0.95 and tz.best_offset_minutes == 0:
        log.info("Метки в файле явно помечены как UTC (+00:00), и сверка с ценами это подтверждает.")
    log.info("ВЫВОД: %s", tz.verdict)

    save_json({"manifest": run_manifest(cfg),
               "files": [{**i, "sha256": sha256_file(raw_dir / i["file"])} for i in files_info],
               "share_with_tz_suffix": share_tz, "timezone": tz.as_dict()},
              results_dir / "metrics" / "cryptovision_timezone.json")

    expected = configured_offset(news_cfg.get("source_timezone"))
    if not cv.is_reliable(tz):
        raise SystemExit("Пояс не определён надёжно — дальше не иду. Пришлите вывод выше.")
    if expected is not None and expected != tz.best_offset_minutes:
        raise SystemExit(f"В конфигурации указан сдвиг {expected} мин, а сверка показала "
                         f"{tz.best_offset_minutes} мин. Разберитесь, прежде чем продолжать.")

    # 3. Очистка
    entities = int(frame["title"].astype("string").str.contains(news_rules.ENTITY, na=False).sum())
    log.info("Заголовков с мнемониками HTML (&amp;, &#8217; и т. п.): %d — раскрываются при очистке", entities)
    news, stages = cv.clean(frame, offset_minutes=tz.best_offset_minutes,
                            dedup_window_hours=int(cfg["data"]["dedup"]["window_hours"]))
    out_path = data_dir / "interim" / "news_en.parquet"
    news.to_parquet(out_path, index=False)

    tables_dir = results_dir / "tables"
    stages.to_csv(tables_dir / "cleaning_news_en.csv", index=False, encoding="utf-8")
    precision.to_csv(tables_dir / "news_en_time_precision_raw.csv", index=False, encoding="utf-8")
    summaries = news_rules.summary_tables(news)
    for name, table in summaries.items():
        table.to_csv(tables_dir / f"news_en_{name}.csv", encoding="utf-8")

    log.info("Этапы очистки:\n%s", stages.to_string(index=False))
    log.info("По источникам:\n%s", summaries["by_source"].head(10).to_string())
    log.info("По годам:\n%s", summaries["by_year"].to_string())
    log.info("Период по источникам:\n%s", summaries["sources_span"].to_string())
    log.info("Новостей по источникам за последние месяцы:\n%s",
             summaries["by_source_month"].tail(8).to_string())
    end, _ = period.coverage_end(news, **period.coverage_params(cfg))
    log.info("Граница полного покрытия набора — начало месяца, в котором обрывается основной "
             "источник: %s. Новости с этой даты в выборку не войдут.", f"{end:%d.%m.%Y}")
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
