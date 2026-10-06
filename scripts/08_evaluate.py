"""Шаг 8 (задачи 8.1–8.4): метрики, тесты, бэктест с издержками и выбор модели.

Что делает скрипт (PROTOCOL.md, разделы 8–10):
  1. собирает прогнозы шага 7 (у стохастических моделей — среднее по зёрнам) и оставляет
     строки теста, общие для всех моделей актива;
  2. считает RMSE, MAE, точность направления и R²_OOS; тесты Диболда–Мариано против
     нулевого прогноза и для вопросов В1 и В2, тест Песарана–Тиммерманна, Model Confidence Set;
  3. проверяет стратегии long/flat и long/short при издержках 0, 5, 10 и 20 б. п. против
     buy-and-hold; интервалы для разницы коэффициентов Шарпа — стационарный бутстреп;
  4. применяет заранее зафиксированный критерий выбора модели (раздел 9).
Тест Грейнджера — отдельный шаг 8в (scripts/08c_granger.py).

Видеокарта не нужна, работает несколько минут.
Запуск:  python scripts/08_evaluate.py
"""
import json

import numpy as np
import pandas as pd

from cryptonews import validation
from cryptonews.cli import parse_args
from cryptonews.config import load_config
from cryptonews.evaluation import backtest, metrics
from cryptonews.evaluation import tests as stat_tests
from cryptonews.evaluation.predictions import (BASELINES, MEAN, ZERO, collect_all, combine, combined_path,
                                                key_of, label)
from cryptonews.models import run
from cryptonews.utils import get_logger, run_manifest, save_json, set_seed


def add_arguments(parser) -> None:
    parser.add_argument("--assets", nargs="*", default=None, help="активы, например BTCUSDT (по умолчанию все)")


def set_roles(cfg: dict) -> dict:
    """Наборы признаков по составу блоков: только цены, цены + EN, все языки."""
    sets = cfg["features"]["feature_sets"]
    find = lambda blocks: next(s for s, b in sets.items() if set(b) == blocks)  # noqa: E731
    price = find({"price"})
    return {"price": price, "en": find({"price", "en"}), "all": find({"price", "en", "ru"}),
            "news": [s for s in sets if s != price]}


def check_dates(splits: dict, out_dir, symbol: str, log) -> None:
    """Раздел 10: подбор и обучение не используют тестовые даты."""
    tuning, folds, embargo = splits["tuning"], splits["folds"], pd.Timedelta(hours=int(splits["embargo_hours"]))
    problems = []
    if tuning.test_end > folds[0].test_start:
        problems.append("окно настройки заходит в тест")
    for fold in folds:
        if fold.train_end > fold.test_start - embargo:
            problems.append(f"{fold.name}: обучение заходит в зазор перед тестом")
    for previous, current in zip(folds, folds[1:]):
        if previous.test_end != current.test_start:
            problems.append(f"{current.name}: фолды не стыкуются")
    expected = tuning.as_dict()
    for path in sorted((out_dir / symbol).glob("*/tuning.json")):
        if json.loads(path.read_text(encoding="utf-8"))["window"] != expected:
            problems.append(f"{path.parent.name}: подбор шёл не на окне настройки из splits.json")
    if problems:
        raise SystemExit(f"[{symbol}] проверка дат не пройдена: " + "; ".join(problems))
    log.info("[%s] проверка дат пройдена: подбор только на окне настройки, обучение заканчивается за %d ч до каждого фолда",
             symbol, int(splits["embargo_hours"]))


def accuracy_table(y: np.ndarray, preds: dict, keys: list, alpha: float) -> pd.DataFrame:
    zero_loss, mean_pred = (y - preds[ZERO]) ** 2, preds[MEAN]
    rows = []
    for key in keys:
        f = preds[key]
        accuracy, n_dir = metrics.directional_accuracy(y, f)
        row = {"key": key, "label": label(key), "rows": len(y), "rmse": metrics.rmse(y, f), "mae": metrics.mae(y, f),
               "rmse_ratio_zero": metrics.rmse(y, f) / metrics.rmse(y, preds[ZERO]),
               "r2_oos": metrics.r2_oos(y, f, mean_pred), "dir_accuracy": accuracy, "dir_rows": n_dir}
        if key != ZERO:
            dm = stat_tests.diebold_mariano(zero_loss, (y - f) ** 2)
            pt = stat_tests.pesaran_timmermann(y, f)
            row.update({"dm_vs_zero": dm["dm_stat"], "dm_vs_zero_p": dm["p_value"], "hac_lags": dm["hac_lags"],
                        "pt_stat": pt["pt_stat"], "pt_p": pt["p_value"]})
        rows.append(row)
    table = pd.DataFrame(rows)
    family = ~table["key"].isin([key_of(m, run.NO_FEATURES) for m in BASELINES])
    for column in ("dm_vs_zero_p", "pt_p"):
        table[column + "_holm"] = np.nan
        table.loc[family, column + "_holm"] = stat_tests.holm(table.loc[family, column])
    table["accurate"] = (table["dm_vs_zero_p_holm"] < alpha) | (table["pt_p_holm"] < alpha)
    return table


def question_pairs(cfg: dict, keys: set) -> list:
    roles = set_roles(cfg)
    pairs = []
    for model in ("random_forest", "xgboost", "lstm"):
        for feature_set in roles["news"]:
            pairs.append(("В1", key_of(model, roles["price"]), key_of(model, feature_set)))
    for feature_set in cfg["models"]["arimax"]["exog_feature_sets"]:
        pairs.append(("В1", key_of("arima", roles["price"]), key_of("arimax", feature_set)))
    for model in ("random_forest", "xgboost", "lstm", "arimax"):
        pairs.append(("В2", key_of(model, roles["en"]), key_of(model, roles["all"])))
    return [p for p in pairs if p[1] in keys and p[2] in keys]


def questions_table(cfg: dict, y: np.ndarray, preds: dict) -> pd.DataFrame:
    rows = []
    for question, base, alternative in question_pairs(cfg, set(preds)):
        dm = stat_tests.diebold_mariano((y - preds[base]) ** 2, (y - preds[alternative]) ** 2)
        rows.append({"question": question, "base": base, "alternative": alternative,
                     "comparison": f"{label(alternative)} против {label(base)}",
                     "mse_change_pct": 100.0 * (np.mean((y - preds[alternative]) ** 2) / np.mean((y - preds[base]) ** 2) - 1.0),
                     "dm_stat": dm["dm_stat"], "p_value": dm["p_value"], "hac_lags": dm["hac_lags"]})
    table = pd.DataFrame(rows, columns=["question", "base", "alternative", "comparison", "mse_change_pct", "dm_stat",
                                        "p_value", "hac_lags"])
    table["p_holm"] = np.nan
    for question in table["question"].unique():
        mask = table["question"] == question
        table.loc[mask, "p_holm"] = stat_tests.holm(table.loc[mask, "p_value"])
    return table


def trading_tables(cfg: dict, y: np.ndarray, preds: dict, keys: list, seed: int, log, symbol: str):
    s_cfg, b_cfg = cfg["evaluation"]["strategy"], cfg["evaluation"]["bootstrap"]
    periods, primary = float(s_cfg["annualization_hours"]), float(s_cfg["primary_cost_bp"])
    rows, intervals = [], []
    for rule in backtest.STRATEGIES:
        for cost in [float(c) for c in s_cfg["costs_bp"]]:
            hold, hold_net = backtest.evaluate(np.ones(len(y)), y, "buy_and_hold", cost, periods)
            hold["breakeven_cost_bp"] = np.nan                  # один вход за весь тест: показатель не имеет смысла
            rows.append({"strategy": rule, "cost_bp": cost, "key": "buy_and_hold", "label": "Buy-and-hold", **hold})
            series = {}
            for key in keys:
                summary, net = backtest.evaluate(preds[key], y, rule, cost, periods)
                rows.append({"strategy": rule, "cost_bp": cost, "key": key, "label": label(key), **summary})
                if np.std(net) > 0:
                    series[key] = net
            if rule == "long_flat" or cost == primary:      # long/short — только при основных издержках (приложение)
                ci = stat_tests.sharpe_difference_ci(series, hold_net, reps=int(b_cfg["reps"]),
                                                     mean_block=float(b_cfg["mean_block_hours"]), seed=seed,
                                                     periods_per_year=periods)
                ci.insert(0, "cost_bp", cost)
                ci.insert(0, "strategy", rule)
                intervals.append(ci.rename(columns={"name": "key"}))
            log.info("[%s] бэктест %s, издержки %g б. п.: готово", symbol, rule, cost)
    table = pd.DataFrame(rows)
    ci = pd.concat(intervals, ignore_index=True)
    return table.merge(ci, on=["strategy", "cost_bp", "key"], how="left")


def selection_table(accuracy: pd.DataFrame, trading: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    sel = cfg["evaluation"]["selection"]
    primary = trading[(trading["strategy"] == cfg["evaluation"]["strategy"]["primary"])
                      & (trading["cost_bp"] == float(sel["cost_bp"]))]
    table = accuracy[["key", "label", "dm_vs_zero_p_holm", "pt_p_holm", "accurate"]].merge(
        primary[["key", "sharpe", "sharpe_diff", "ci_low", "ci_high"]], on="key", how="left")
    table = table[~table["key"].isin([key_of(m, run.NO_FEATURES) for m in BASELINES])].copy()
    table["beats_hold"] = table["ci_low"] > 0
    table["passes"] = table["accurate"] & table["beats_hold"]
    table["selected"] = False
    passing = table[table["passes"]]
    if len(passing):
        table.loc[passing["sharpe"].idxmax(), "selected"] = True
    return table.sort_values("sharpe", ascending=False).reset_index(drop=True)


def fold_table(y: pd.Series, preds: dict, keys: list, splits: dict) -> pd.DataFrame:
    rows = []
    for fold in splits["folds"]:
        mask = (y.index >= fold.test_start) & (y.index < fold.test_end)
        if not mask.any():
            continue
        zero = metrics.rmse(y[mask], preds[ZERO][mask])
        for key in keys:
            value = metrics.rmse(y[mask], preds[key][mask])
            rows.append({"fold": fold.name, "test_start": str(fold.test_start), "key": key, "label": label(key),
                         "rows": int(mask.sum()), "rmse": value, "rmse_ratio_zero": value / zero})
    return pd.DataFrame(rows)


def show(frame: pd.DataFrame, columns: dict, digits: dict | None = None) -> str:
    view = frame[list(columns)].rename(columns=columns).copy()
    for column, places in (digits or {}).items():
        view[columns[column]] = view[columns[column]].round(places)
    return view.to_string(index=False)


def main() -> None:
    args = parse_args(__doc__, add_arguments)
    cfg = load_config(args.config)
    log = get_logger()
    seed = int(cfg["seeds"]["default"][0])
    set_seed(seed)
    results_dir = cfg["paths"]["results_dir"]
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    alpha = float(cfg["evaluation"]["dm_test"]["alpha"])
    splits_path = results_dir / "splits.json"
    if not splits_path.exists():
        raise SystemExit(f"Нет {splits_path} — сначала выполните шаги 6 и 7")
    splits = validation.from_json(json.loads(splits_path.read_text(encoding="utf-8")))
    out_dir = results_dir / "predictions"
    symbols = args.assets or list(cfg["data"]["prices"]["symbols"])
    report = {}

    for symbol in symbols:
        check_dates(splits, out_dir, symbol, log)
        found, missing = collect_all(cfg, out_dir, symbol, len(splits["folds"]))
        if missing:
            log.warning("[%s] не посчитаны до конца (в оценку не идут): %s", symbol, ", ".join(label(k) for k in missing))
        combined = combine(found)
        combined.to_parquet(combined_path(results_dir, symbol))      # одна таблица прогнозов для шага 9 и архива
        rows, truth = combined.index, combined["y_true"]
        keys = list(found)
        y = truth.to_numpy(float)
        preds = {key: combined[key].to_numpy(float) for key in keys}
        log.info("[%s] моделей в оценке: %d; общих строк теста: %d (у отдельных моделей до %d)", symbol, len(keys),
                 len(rows), max(len(f) for f in found.values()))

        accuracy = accuracy_table(y, preds, keys, alpha)
        questions = questions_table(cfg, y, preds)
        m_cfg = cfg["evaluation"]["mcs"]
        losses = pd.DataFrame({label(k): (y - preds[k]) ** 2 for k in keys}, index=rows)
        mcs = stat_tests.model_confidence_set(losses, size=float(m_cfg["alpha"]), reps=int(m_cfg["bootstrap_reps"]),
                                              block_size=int(cfg["evaluation"]["bootstrap"]["mean_block_hours"]), seed=seed)
        accuracy = accuracy.merge(mcs.rename_axis("label").reset_index(), on="label", how="left")
        trading = trading_tables(cfg, y, preds, keys, seed, log, symbol)
        selection = selection_table(accuracy, trading, cfg)
        folds = fold_table(truth, {k: pd.Series(v, index=rows) for k, v in preds.items()}, keys, splits)

        for name, frame in (("accuracy", accuracy), ("questions", questions), ("trading", trading),
                            ("selection", selection), ("folds", folds)):
            frame.to_csv(tables_dir / f"eval_{name}_{symbol}.csv", index=False, encoding="utf-8")

        family_size = int(accuracy["dm_vs_zero_p_holm"].notna().sum())
        log.info("[%s] точность прогнозов на %d общих часах (p — с поправкой Холма по %d моделям):\n%s", symbol,
                 len(rows), family_size,
                 show(accuracy, {"label": "Модель", "rmse_ratio_zero": "RMSE/нулевой", "r2_oos": "R²_OOS",
                                 "dir_accuracy": "Направление", "dm_vs_zero_p_holm": "p ДМ", "pt_p_holm": "p ПТ",
                                 "p_mcs": "p MCS", "in_mcs": "В MCS"},
                      {"rmse_ratio_zero": 4, "r2_oos": 4, "dir_accuracy": 4, "dm_vs_zero_p_holm": 3, "pt_p_holm": 3,
                       "p_mcs": 3}))
        log.info("[%s] вопросы В1 и В2 — тест Диболда–Мариано (H1: вариант с новостями точнее; p — с поправкой Холма "
                 "внутри вопроса):\n%s", symbol,
                 show(questions, {"question": "Вопрос", "comparison": "Сравнение", "mse_change_pct": "MSE, % изм.",
                                  "dm_stat": "DM", "p_value": "p", "p_holm": "p Холма"},
                      {"mse_change_pct": 3, "dm_stat": 2, "p_value": 3, "p_holm": 3}))
        s_cfg = cfg["evaluation"]["strategy"]
        primary = trading[(trading["strategy"] == s_cfg["primary"]) & (trading["cost_bp"] == float(s_cfg["primary_cost_bp"]))]
        log.info("[%s] стратегия %s, издержки %g б. п. (Шарп — годовой; ΔШарп — разница с buy-and-hold и 95%% интервал):\n%s",
                 symbol, s_cfg["primary"], float(s_cfg["primary_cost_bp"]),
                 show(primary, {"label": "Модель", "sharpe": "Шарп", "sharpe_diff": "ΔШарп", "ci_low": "от", "ci_high": "до",
                                "annual_return": "Год. доходн.", "max_drawdown": "Просадка",
                                "turnover_per_year": "Сделок в год", "breakeven_cost_bp": "Безуб. изд., б. п."},
                      {"sharpe": 2, "sharpe_diff": 2, "ci_low": 2, "ci_high": 2, "annual_return": 3, "max_drawdown": 3,
                       "turnover_per_year": 1, "breakeven_cost_bp": 1}))
        best = selection[selection["selected"]]
        if len(best):
            verdict = f"лучшая модель по критерию раздела 9 — {best.iloc[0]['label']} (Шарп {best.iloc[0]['sharpe']:.2f})"
        else:
            verdict = ("ни одна модель не прошла критерий раздела 9 (значимая точность и Шарп выше buy-and-hold); "
                       "сравнение точности — по Model Confidence Set")
        log.info("[%s] ВЫВОД: %s. В Model Confidence Set (α = %.2f): %s", symbol, verdict, float(m_cfg["alpha"]),
                 ", ".join(accuracy.loc[accuracy["in_mcs"] == True, "label"]))  # noqa: E712
        report[symbol] = {"rows": len(rows), "missing_models": missing, "selected": best["key"].tolist(),
                          "in_mcs": accuracy.loc[accuracy["in_mcs"] == True, "key"].tolist(),  # noqa: E712
                          "verdict": verdict}

    save_json({"manifest": run_manifest(cfg), "assets": report}, results_dir / "metrics" / "evaluation.json")
    log.info("Готово: таблицы eval_*.csv в %s", tables_dir)


if __name__ == "__main__":
    main()
