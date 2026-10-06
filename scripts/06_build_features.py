"""Шаг 6 (задачи 6.1–6.4): выравнивание новостей по часам, признаки и схема проверки.

Что делает скрипт:
  1. берёт модели тональности, выбранные шагом 5b (results/metrics/sentiment_choice.json),
     и их оценки для каждой новости;
  2. для каждого актива отбирает новости: упоминающие актив и общерыночные (без упоминания
     биткоина и эфира), отдельно для английских и русских новостей;
  3. относит новость к часу t, если t ≤ τ < t + 1 ч, и считает почасовые величины;
  4. строит ценовые и новостные признаки (PROTOCOL.md, раздел 4) и целевую переменную —
     доходность следующего часа; строки с пропусками из-за остановок биржи помечаются;
  5. выполняет встроенные проверки: возрастание времени, скачки цели больше 20% за час,
     правильность отнесения новостей к часам;
  6. сохраняет data/processed/features_<актив>.parquet, схему проверки results/splits.json
     и сводки в results/.

Видеокарта не нужна, работает около минуты.
Запуск:  python scripts/06_build_features.py
"""
import json

import numpy as np
import pandas as pd

from cryptonews import align, diagnostics, features, period, validation
from cryptonews.cli import parse_args
from cryptonews.config import load_config
from cryptonews.utils import get_logger, run_manifest, save_json, sha256_file

LANGUAGES = ("en", "ru")


def load_scored_news(cfg: dict, log) -> tuple[dict, dict, dict]:
    """Новости каждого языка с оценками модели, выбранной шагом 5b."""
    data_dir, results_dir = cfg["paths"]["data_dir"], cfg["paths"]["results_dir"]
    choice_path = results_dir / "metrics" / "sentiment_choice.json"
    if not choice_path.exists():
        raise SystemExit(f"Нет {choice_path} — сначала выполните шаг 5b (05b_choose_sentiment.py)")
    choice = json.loads(choice_path.read_text(encoding="utf-8"))
    news, inputs, chosen = {}, {}, {}
    for lang in LANGUAGES:
        news_path = data_dir / "interim" / f"news_{lang}.parquet"
        scores_path = data_dir / "interim" / "sentiment" / lang / f"{choice[lang]['slug']}.parquet"
        for path in (news_path, scores_path):
            if not path.exists():
                raise SystemExit(f"Нет файла {path} — выполните шаги 3–5")
        frame = pd.read_parquet(news_path)
        frame["published_utc"] = pd.to_datetime(frame["published_utc"], utc=True)
        scores = pd.read_parquet(scores_path)
        merged = frame.merge(scores, on="url", how="inner", validate="one_to_one")
        if len(merged) != len(frame):
            raise SystemExit(f"[{lang}] оценки тональности есть не для всех новостей: {len(merged)} из {len(frame)}")
        news[lang] = merged
        inputs[lang] = {"news": str(news_path), "news_sha256": sha256_file(news_path),
                        "scores": str(scores_path), "scores_sha256": sha256_file(scores_path)}
        chosen[lang] = {"model": choice[lang]["model"], "revision": choice[lang]["revision"]}
        log.info("[%s] модель тональности: %s (ревизия %s), новостей с оценками: %d",
                 lang, choice[lang]["model"], choice[lang]["revision"][:8], len(merged))
    return news, inputs, chosen


def build_asset(symbol: str, prices: pd.DataFrame, news: dict, span: period.SamplePeriod, cfg: dict,
                log) -> tuple[pd.DataFrame, dict]:
    """Таблица признаков одного актива на часовой сетке выборки."""
    p_cfg, n_cfg = cfg["features"]["price"], cfg["features"]["news"]
    diagnostics.check_monotonic(prices.index, symbol)
    table = features.price_features(prices, p_cfg["return_lags"], p_cfg["realized_vol_windows"],
                                    bool(p_cfg.get("volume_change", True)))
    table[features.TARGET] = features.target(prices)

    params = features.news_params(cfg)
    grid = pd.date_range(span.start - pd.Timedelta(hours=features.warmup_hours(cfg)), span.end,
                         freq="h", inclusive="left", name=prices.index.name)
    news_counts = {}
    for lang in LANGUAGES:
        selected = align.news_for_asset(news[lang], symbol, n_cfg.get("asset_rule", "coin_or_market_wide"))
        diagnostics.check_news_alignment(align.hour_of(selected["published_utc"]), selected["published_utc"])
        hourly = align.hourly_news(selected, grid)
        table = table.join(features.news_features(hourly, lang, **params), how="left")
        news_counts[lang] = int(span.mask(selected["published_utc"]).sum())

    table = table.loc[(table.index >= span.start) & (table.index < span.end)].copy()
    diagnostics.check_monotonic(table.index, symbol)
    model_columns = sorted({c for name in cfg["features"]["feature_sets"]
                            for c in features.feature_set_columns(cfg, name)}, key=list(table.columns).index)
    table["valid"] = table[model_columns + [features.TARGET]].notna().all(axis=1)
    diagnostics.check_target(table.loc[table["valid"], features.TARGET],
                             float(cfg["diagnostics"]["max_abs_hourly_return"]), symbol)
    report = {
        "rows": len(table), "valid_rows": int(table["valid"].sum()),
        "invalid_rows": int((~table["valid"]).sum()),
        "missing_close_hours": int(prices.loc[(prices.index >= span.start) & (prices.index < span.end), "close"].isna().sum()),
        "news_used": news_counts,
        "hours_with_news_share": {lang: round(float((table[f"{lang}_news_count"] > 0).mean()), 4)
                                  for lang in LANGUAGES},
    }
    log.info("%s: строк %d, пригодных %d, исключено %d (часы без свечей и окна после них; пропущенных свечей %d)",
             symbol, report["rows"], report["valid_rows"], report["invalid_rows"], report["missing_close_hours"])
    log.info("%s: новостей в признаках — EN %d, RU %d; часов хотя бы с одной новостью — EN %.1f%%, RU %.1f%%",
             symbol, news_counts["en"], news_counts["ru"], 100 * report["hours_with_news_share"]["en"],
             100 * report["hours_with_news_share"]["ru"])
    return table, report


def summary(table: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Описательная статистика признаков на пригодных строках — для приложения к статье."""
    valid = table.loc[table["valid"]].drop(columns=["valid"])
    stats = valid.describe().T[["mean", "std", "min", "max"]]
    stats["share_zero"] = (valid == 0).mean()
    stats.insert(0, "asset", symbol)
    stats.index.name = "feature"
    return stats.reset_index()


def coverage_by_year(table: pd.DataFrame) -> pd.DataFrame:
    """Доля часов хотя бы с одной новостью по годам — видно, как меняется покрытие."""
    years = table.index.year
    return pd.DataFrame({f"{lang}_hours_with_news": (table[f"{lang}_news_count"] > 0).groupby(years).mean().round(3)
                         for lang in LANGUAGES})


def main() -> None:
    args = parse_args(__doc__)
    cfg = load_config(args.config)
    log = get_logger()
    data_dir, results_dir = cfg["paths"]["data_dir"], cfg["paths"]["results_dir"]

    news, inputs, chosen = load_scored_news(cfg, log)
    prices = {}
    interval = cfg["data"]["prices"]["interval"]
    for symbol in cfg["data"]["prices"]["symbols"]:
        path = data_dir / "raw" / "binance" / f"{symbol}_{interval}.parquet"
        if not path.exists():
            raise SystemExit(f"Нет файла {path} — сначала выполните шаг 1")
        prices[symbol] = pd.read_parquet(path)
        inputs[symbol] = {"prices": str(path), "sha256": sha256_file(path)}

    span = period.from_config(cfg, prices, {"EN": news["en"], "RU": news["ru"]})
    log.info("Границы выборки: %s", span.describe())
    saved = results_dir / "metrics" / "sample_period.json"
    if saved.exists():
        before = json.loads(saved.read_text(encoding="utf-8"))
        if before.get("end_exclusive") != str(span.end) or before.get("start_inclusive") != str(span.start):
            log.warning("Границы выборки отличаются от тех, по которым делалась разметка: %s … %s",
                        before.get("start_inclusive"), before.get("end_exclusive"))

    splits = validation.from_config(cfg, span.start, span.end)
    save_json(validation.as_json(splits), results_dir / "splits.json")
    t = splits["tuning"]
    log.info("Окно настройки: %s … %s; тест: %s … %s, фолдов %d (обучение до начала фолда, зазор %d ч)",
             f"{t.test_start:%d.%m.%Y}", f"{t.test_end - pd.Timedelta(hours=1):%d.%m.%Y}",
             f"{splits['folds'][0].test_start:%d.%m.%Y}",
             f"{splits['folds'][-1].test_end - pd.Timedelta(hours=1):%d.%m.%Y}",
             len(splits["folds"]), splits["embargo_hours"])

    out_dir = data_dir / "processed"
    reports, summaries, targets = {}, [], {}
    for symbol, frame in prices.items():
        table, reports[symbol] = build_asset(symbol, frame, news, span, cfg, log)
        targets[symbol] = table.loc[table["valid"], features.TARGET]
        out_path = out_dir / f"features_{symbol}.parquet"
        table.to_parquet(out_path)
        reports[symbol]["file"] = str(out_path)
        summaries.append(summary(table, symbol))
        coverage = coverage_by_year(table)
        coverage.to_csv(results_dir / "tables" / f"news_coverage_by_year_{symbol}.csv", encoding="utf-8")
        log.info("%s: доля часов с новостями по годам:\n%s", symbol, coverage.to_string())

    threshold = float(cfg["diagnostics"].get("report_abs_hourly_return", 0.15))
    moves = diagnostics.large_moves(targets, threshold)
    moves.to_csv(results_dir / "tables" / "large_hourly_moves.csv", encoding="utf-8")
    log.info("Часы с изменением цены больше %.0f%% хотя бы у одного актива (%d шт.; рядом — второй актив):\n%s",
             100 * threshold, len(moves), moves.to_string() if len(moves) else "нет")

    stats = pd.concat(summaries, ignore_index=True)
    stats.round(6).to_csv(results_dir / "tables" / "features_summary.csv", index=False, encoding="utf-8")
    sets = {name: features.feature_set_columns(cfg, name) for name in cfg["features"]["feature_sets"]}
    save_json({"manifest": run_manifest(cfg), "inputs": inputs, "sentiment_models": chosen,
               "period": span.as_dict(), "feature_sets": sets, "assets": reports},
              results_dir / "metrics" / "features_report.json")
    show = stats[stats["feature"].isin(["target", "en_sent_mean", "ru_sent_mean", "en_news_intensity",
                                        "ru_news_intensity", "rv_24"])]
    log.info("Сводка по ключевым признакам:\n%s", show.round(4).to_string(index=False))
    for name, columns in sets.items():
        log.info("Набор %s: %d признаков", name, len(columns))
    log.info("Готово: %s", out_dir)


if __name__ == "__main__":
    main()
