"""Шаг 8г (описательный, раздел 12 протокола): насколько обоснован вывод о новостных признаках.

Шаг 8 отвечает на вопросы В1 и В2 тестом Диболда–Мариано. Незначимый тест сам по себе не
доказывает отсутствие эффекта, поэтому здесь проверяется, какой эффект данные ещё допускают и не
связан ли результат с тем, как устроены признаки:
  1. Границы эффекта. Для каждого сравнения В1 и В2 — изменение MSE с 95%-м интервалом (HAC
     Ньюи–Уэста, как в тесте Диболда–Мариано) и минимальное уменьшение MSE, которое односторонний
     тест на уровне 5% обнаружил бы с вероятностью 80%.
  2. Устойчивость по кварталам. В скольких парах «сравнение × фолд» новостные признаки уменьшают MSE
     и отличается ли эта доля от половины (знаковый тест).
  3. Часы с новостями. Те же сравнения только на часах, в которые вышла хотя бы одна новость
     нужного языка, и на часах всплеска новостного потока (интенсивность в верхнем дециле теста).
  4. Содержательность тональности. Связь средней тональности новостей часа с доходностью того же
     часа и следующего часа (МНК с HAC-ошибками) на часах с новостями.
Новостные ряды строятся теми же правилами, что в шаге 6 (исключённые источники, пакетные загрузки,
привязка к активу, выравнивание по часам). На выбор модели шаг не влияет.

Нужны результаты шагов 5b, 6 и 8; работает и на архиве релиза (папки data/ и results/ из
release_package.zip). Видеокарта не нужна, около минуты.
Запуск:  python scripts/08d_news_evidence.py
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

from cryptonews import align, features, validation
from cryptonews.cli import parse_args
from cryptonews.config import load_config
from cryptonews.data import news as news_rules
from cryptonews.evaluation import tests as stat_tests
from cryptonews.evaluation.predictions import combined_path
from cryptonews.utils import get_logger, run_manifest, save_json

LANGUAGES = ("en", "ru")
Z_95, Z_POWER = stats.norm.ppf(0.975), stats.norm.ppf(0.95) + stats.norm.ppf(0.80)
BURST_QUANTILE = 0.9


def scored_news(cfg: dict, data_dir, results_dir) -> dict:
    """Новости с оценками выбранной модели тональности после исключений шага 6."""
    choice = json.loads((results_dir / "metrics" / "sentiment_choice.json").read_text(encoding="utf-8"))
    b_cfg = cfg["data"].get("batch_uploads", {})
    out = {}
    for lang in LANGUAGES:
        news = pd.read_parquet(data_dir / "interim" / f"news_{lang}.parquet")
        news["published_utc"] = pd.to_datetime(news["published_utc"], utc=True)
        scores = pd.read_parquet(data_dir / "interim" / "sentiment" / lang / f"{choice[lang]['slug']}.parquet",
                                 columns=["url", "score", "label"])
        merged = news.merge(scores, on="url", how="inner", validate="one_to_one")
        merged, _ = news_rules.drop_sources(merged, cfg["data"].get("feature_excluded_sources"))
        batch, _ = news_rules.batch_uploads(merged, int(b_cfg.get("min_records", 10)),
                                            float(b_cfg.get("max_gap_seconds", 60)))
        out[lang] = merged[~batch.to_numpy()].reset_index(drop=True)
    return out


def hourly_series(news: dict, symbol: str, cfg: dict, splits: dict) -> pd.DataFrame:
    """Число новостей, средняя тональность (NaN без новостей) и интенсивность по часам — как в шаге 6."""
    params = features.news_params(cfg)
    grid = pd.date_range(splits["sample_start"] - pd.Timedelta(hours=features.warmup_hours(cfg)), splits["sample_end"],
                         freq="h", inclusive="left", name="open_time")
    rule = cfg["features"]["news"].get("asset_rule", "coin_or_market_wide")
    columns = {}
    for lang in LANGUAGES:
        hourly = align.hourly_news(align.news_for_asset(news[lang], symbol, rule), grid)
        built = features.news_features(hourly, lang, **params)
        columns[f"{lang}_count"] = hourly["count"]
        columns[f"{lang}_sent"] = hourly["sent_mean"]
        columns[f"{lang}_intensity"] = built[f"{lang}_news_intensity"]
    return pd.DataFrame(columns)


def relevant_language(base: str, alternative: str) -> str:
    """Язык новостей, вклад которых проверяет сравнение: P_EN — en, P_RU — ru, P_EN_RU против P — оба,
    P_EN_RU против P_EN (вопрос В2) — ru."""
    alt_set, base_set = alternative.split(":")[1], base.split(":")[1]
    if alt_set == "P_EN_RU":
        return "ru" if base_set == "P_EN" else "both"
    return "en" if alt_set == "P_EN" else "ru"


def effect(y: np.ndarray, base: np.ndarray, alternative: np.ndarray) -> dict:
    """Изменение MSE (%) при переходе от base к alternative, его HAC-ошибка, интервал и минимальный
    обнаружимый эффект; отрицательное изменение — новости уменьшают ошибку."""
    loss_base, loss_alt = (y - base) ** 2, (y - alternative) ** 2
    d = loss_base - loss_alt
    n = len(d)
    mse_base = float(loss_base.mean())
    lags = stat_tests.newey_west_lags(n)
    variance = stat_tests.long_run_variance(d, lags)
    se = float(np.sqrt(variance / n)) if variance > 0 else float("nan")
    change = -100.0 * float(d.mean()) / mse_base
    se_pct = 100.0 * se / mse_base
    dm = float(d.mean()) / se if se > 0 else float("nan")
    return {"rows": n, "mse_change_pct": change, "se_pct": se_pct, "ci_low_pct": change - Z_95 * se_pct,
            "ci_high_pct": change + Z_95 * se_pct, "mde_pct": Z_POWER * se_pct, "dm_stat": dm,
            "p_value": float(stats.norm.sf(dm)) if np.isfinite(dm) else float("nan"), "hac_lags": lags}


def bounds_table(combined: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    y = combined["y_true"].to_numpy(float)
    rows = []
    for pair in pairs.itertuples():
        rows.append({"question": pair.question, "base": pair.base, "alternative": pair.alternative,
                     "comparison": pair.comparison,
                     **effect(y, combined[pair.base].to_numpy(float), combined[pair.alternative].to_numpy(float))})
    return pd.DataFrame(rows)


def folds_table(combined: pd.DataFrame, pairs: pd.DataFrame, splits: dict) -> pd.DataFrame:
    y = combined["y_true"]
    rows = []
    for pair in pairs.itertuples():
        for fold in splits["folds"]:
            mask = (combined.index >= fold.test_start) & (combined.index < fold.test_end)
            if not mask.any():
                continue
            mse_base = float(((y[mask] - combined.loc[mask, pair.base]) ** 2).mean())
            mse_alt = float(((y[mask] - combined.loc[mask, pair.alternative]) ** 2).mean())
            rows.append({"question": pair.question, "comparison": pair.comparison, "fold": fold.name,
                         "mse_change_pct": 100.0 * (mse_alt / mse_base - 1.0)})
    return pd.DataFrame(rows)


def folds_summary(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for question, part in folds.groupby("question", sort=True):
        improved, total = int((part["mse_change_pct"] < 0).sum()), len(part)
        rows.append({"question": question, "pairs": total, "improved": improved,
                     "share_improved": improved / total if total else float("nan"),
                     "sign_test_p": float(stats.binomtest(improved, total, 0.5, alternative="greater").pvalue)})
    return pd.DataFrame(rows)


def subset_masks(series: pd.DataFrame, index: pd.Index) -> dict:
    """Маски часов теста: хотя бы одна новость языка и всплеск потока (верхний дециль интенсивности)."""
    s = series.reindex(index)
    masks = {}
    for lang in LANGUAGES:
        masks[("news", lang)] = (s[f"{lang}_count"] > 0).to_numpy()
        intensity = s[f"{lang}_intensity"]
        masks[("burst", lang)] = (intensity >= intensity.quantile(BURST_QUANTILE)).to_numpy()
    for kind in ("news", "burst"):
        masks[(kind, "both")] = masks[(kind, "en")] | masks[(kind, "ru")]
    return masks


def subsets_table(combined: pd.DataFrame, pairs: pd.DataFrame, masks: dict) -> pd.DataFrame:
    y = combined["y_true"].to_numpy(float)
    rows = []
    for pair in pairs.itertuples():
        lang = relevant_language(pair.base, pair.alternative)
        for kind in ("news", "burst"):
            mask = masks[(kind, lang)]
            rows.append({"question": pair.question, "comparison": pair.comparison, "subset": kind, "language": lang,
                         **effect(y[mask], combined[pair.base].to_numpy(float)[mask],
                                  combined[pair.alternative].to_numpy(float)[mask])})
    return pd.DataFrame(rows)


def validity_table(combined: pd.DataFrame, series: pd.DataFrame) -> pd.DataFrame:
    """Доходность часа t + 1 (y_true строки t) против средней тональности того же часа (строка t + 1)
    и предыдущего часа (строка t) — на часах, где эта тональность определена (были новости)."""
    y = combined["y_true"]
    rows = []
    for lang in LANGUAGES:
        sent = series[f"{lang}_sent"]
        for relation, shift in (("тот же час", -1), ("следующий час", 0)):
            x = sent.shift(shift).reindex(y.index)               # shift −1: тональность строки t + 1
            mask = x.notna().to_numpy()
            target, regressor = y.to_numpy(float)[mask], x.to_numpy(float)[mask]
            X = np.column_stack([np.ones(mask.sum()), regressor])
            test = stat_tests.hac_wald(target, X, [1])
            beta = np.linalg.lstsq(X, target, rcond=None)[0][1]
            rows.append({"language": lang, "relation": relation, "rows": int(mask.sum()),
                         "corr": float(np.corrcoef(regressor, target)[0, 1]), "slope_pct": 100.0 * float(beta),
                         "wald": test["wald"], "p_value": test["p_value"]})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args(__doc__)
    cfg = load_config(args.config)
    log = get_logger()
    manifest = run_manifest(cfg)                # до записи результатов: состояние кода при запуске
    data_dir, results_dir = cfg["paths"]["data_dir"], cfg["paths"]["results_dir"]
    tables_dir = results_dir / "tables"
    splits = validation.from_json(json.loads((results_dir / "splits.json").read_text(encoding="utf-8")))
    news = scored_news(cfg, data_dir, results_dir)
    report = {}
    for symbol in cfg["data"]["prices"]["symbols"]:
        combined = pd.read_parquet(combined_path(results_dir, symbol))
        pairs = pd.read_csv(tables_dir / f"eval_questions_{symbol}.csv")
        series = hourly_series(news, symbol, cfg, splits)
        bounds = bounds_table(combined, pairs)
        check = bounds.merge(pairs[["base", "alternative", "mse_change_pct", "dm_stat"]],
                             on=["base", "alternative"], suffixes=("", "_step8"))
        mismatch = float(np.max(np.abs(check["mse_change_pct"] - check["mse_change_pct_step8"])))
        if mismatch > 1e-9:
            raise SystemExit(f"[{symbol}] изменение MSE расходится с шагом 8 на {mismatch:g} п. п.")
        folds = folds_table(combined, pairs, splits)
        summary = folds_summary(folds)
        subsets = subsets_table(combined, pairs, subset_masks(series, combined.index))
        validity = validity_table(combined, series)
        for name, frame in (("bounds", bounds), ("folds", folds), ("subsets", subsets), ("validity", validity)):
            frame.to_csv(tables_dir / f"news_evidence_{name}_{symbol}.csv", index=False, encoding="utf-8")
        report[symbol] = {
            "largest_possible_improvement_pct": float(-bounds["ci_low_pct"].min()),
            "mde_pct_range": [float(bounds["mde_pct"].min()), float(bounds["mde_pct"].max())],
            "folds": summary.to_dict(orient="records"),
            "news_hours_min_p": float(subsets.loc[subsets["subset"] == "news", "p_value"].min()),
            "burst_hours_min_p": float(subsets.loc[subsets["subset"] == "burst", "p_value"].min()),
        }
        log.info("[%s] границы эффекта (95%%-й интервал изменения MSE, %%; МОЭ — минимальный обнаружимый эффект):\n%s",
                 symbol, bounds[["question", "comparison", "mse_change_pct", "ci_low_pct", "ci_high_pct", "mde_pct"]]
                 .round(3).to_string(index=False))
        log.info("[%s] по кварталам: %s", symbol, "; ".join(
            f"{r['question']} — новости уменьшают MSE в {r['improved']} из {r['pairs']} пар (знаковый тест p = "
            f"{r['sign_test_p']:.3f})" for r in summary.to_dict(orient="records")))
        log.info("[%s] на часах с новостями и всплесках потока:\n%s", symbol,
                 subsets[["question", "comparison", "subset", "rows", "mse_change_pct", "ci_low_pct", "p_value"]]
                 .round(3).to_string(index=False))
        log.info("[%s] тональность и доходность (часы с новостями):\n%s", symbol,
                 validity.round(4).to_string(index=False))
    save_json({"manifest": manifest, "burst_quantile": BURST_QUANTILE, "assets": report},
              results_dir / "metrics" / "news_evidence.json")
    log.info("Готово")


if __name__ == "__main__":
    main()
