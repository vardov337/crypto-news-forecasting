"""Шаг 8б (PROTOCOL.md, раздел 10): тест сдвига новостей и плацебо-тест.

Тест сдвига: новостные признаки сдвигаются в прошлое на 1, 2, 3 и 24 ч, и модель обучается
заново с гиперпараметрами, выбранными на шаге 7. Настоящий сигнал от сдвига затухает плавно.
Если точность резко падает уже после сдвига на 1 ч, а дальше почти не меняется, — это признак
того, что в несдвинутые признаки попала информация из будущего.

Плацебо-тест: целевая переменная перемешивается (20 раз) и модель обучается на ней заново.
Точность направления на тесте должна быть около 50% — точнее, около того, что даёт угадывание
по одной доле растущих часов.

Модель — XGBoost (быстрая), одно зерно. На видеокарте несколько минут.
Запуск:  python scripts/08b_leakage_checks.py   (после шага 7)
"""
import json

import numpy as np
import pandas as pd

from cryptonews import features, validation
from cryptonews.cli import parse_args
from cryptonews.config import load_config
from cryptonews.evaluation import metrics
from cryptonews.features import TARGET
from cryptonews.models import run
from cryptonews.utils import get_logger, run_manifest, save_json, set_seed


def add_arguments(parser) -> None:
    parser.add_argument("--assets", nargs="*", default=None, help="активы, например BTCUSDT (по умолчанию все)")


def chosen_params(results_dir, symbol: str, model: str, feature_set: str) -> dict:
    path = results_dir / "predictions" / symbol / f"{model}__{feature_set}" / "tuning.json"
    if not path.exists():
        raise SystemExit(f"Нет {path} — сначала выполните шаг 7 для модели {model}")
    return json.loads(path.read_text(encoding="utf-8"))["chosen"]


def predict_folds(adapter, table: pd.DataFrame, columns: list, splits: dict, params: dict, seed: int) -> pd.Series:
    parts = [adapter.fit_predict(table, columns, fold, params, seed)[0] for fold in splits["folds"]]
    return pd.concat(parts).sort_index()


def shifted(table: pd.DataFrame, columns: list, hours: int) -> pd.DataFrame:
    """Новостные столбцы берутся из строки на hours часов раньше; строки без них — непригодны."""
    out = table.copy()
    if hours:
        out[columns] = table[columns].shift(hours)
    out["valid"] = table["valid"] & out[columns].notna().all(axis=1)
    return out


def shift_test(cfg, table, splits, adapter, model, seed, results_dir, symbol, log) -> pd.DataFrame:
    sets = cfg["features"]["feature_sets"]
    price = next(s for s, blocks in sets.items() if set(blocks) == {"price"})
    base_columns = features.feature_set_columns(cfg, price)
    base = predict_folds(adapter, table, base_columns, splits, chosen_params(results_dir, symbol, model, price), seed)
    hours_list = [0] + [int(h) for h in cfg["diagnostics"]["shift_test_hours"]]
    preds = {}
    for feature_set in (s for s in sets if s != price):
        columns = features.feature_set_columns(cfg, feature_set)
        news = [c for c in columns if c not in base_columns]
        params = chosen_params(results_dir, symbol, model, feature_set)
        for hours in hours_list:
            preds[(feature_set, hours)] = predict_folds(adapter, shifted(table, news, hours), columns, splits, params, seed)
        log.info("[%s] тест сдвига: %s — готово", symbol, feature_set)
    rows = base.index
    for series in preds.values():
        rows = rows.intersection(series.index)
    y = table.loc[rows, TARGET].to_numpy(float)
    base_mse = float(np.mean((y - base.loc[rows].to_numpy()) ** 2))
    out = []
    for (feature_set, hours), series in preds.items():
        f = series.loc[rows].to_numpy()
        accuracy, _ = metrics.directional_accuracy(y, f)
        out.append({"feature_set": feature_set, "shift_hours": hours, "rows": len(rows), "rmse": metrics.rmse(y, f),
                    "mse_change_vs_price_pct": 100.0 * (float(np.mean((y - f) ** 2)) / base_mse - 1.0),
                    "dir_accuracy": accuracy})
    return pd.DataFrame(out)


def placebo_test(cfg, table, splits, adapter, model, seed, results_dir, symbol, log) -> pd.DataFrame:
    feature_set = cfg["diagnostics"].get("placebo_feature_set", "P_EN_RU")
    columns = features.feature_set_columns(cfg, feature_set)
    params = chosen_params(results_dir, symbol, model, feature_set)
    valid = table.index[table["valid"]]
    out = []
    for i in range(int(cfg["diagnostics"]["placebo_permutations"])):
        rng = np.random.default_rng(seed + 1000 + i)
        permuted = table.copy()
        permuted.loc[valid, TARGET] = rng.permutation(table.loc[valid, TARGET].to_numpy())
        pred = predict_folds(adapter, permuted, columns, splits, params, seed)
        y = permuted.loc[pred.index, TARGET].to_numpy(float)
        f = pred.to_numpy()
        accuracy, n = metrics.directional_accuracy(y, f)
        mask = metrics.direction_mask(y, f)
        share_up, pred_up = float(np.mean(y[mask] > 0)), float(np.mean(f[mask] > 0))
        out.append({"permutation": i, "feature_set": feature_set, "rows": n, "dir_accuracy": accuracy,
                    "share_up": share_up, "pred_up_share": pred_up,
                    "chance_accuracy": share_up * pred_up + (1 - share_up) * (1 - pred_up)})
    log.info("[%s] плацебо: перестановок — %d, готово", symbol, len(out))
    return pd.DataFrame(out)


def main() -> None:
    args = parse_args(__doc__, add_arguments)
    cfg = load_config(args.config)
    log = get_logger()
    data_dir, results_dir = cfg["paths"]["data_dir"], cfg["paths"]["results_dir"]
    splits = validation.from_json(json.loads((results_dir / "splits.json").read_text(encoding="utf-8")))
    model = cfg["diagnostics"].get("leakage_model", "xgboost")
    seed = run.seeds_for(cfg, model)[0]
    set_seed(seed)
    adapter = run.build_model(cfg, model)
    report = {}
    for symbol in args.assets or list(cfg["data"]["prices"]["symbols"]):
        table = pd.read_parquet(data_dir / "processed" / f"features_{symbol}.parquet")
        shift = shift_test(cfg, table, splits, adapter, model, seed, results_dir, symbol, log)
        placebo = placebo_test(cfg, table, splits, adapter, model, seed, results_dir, symbol, log)
        shift.to_csv(results_dir / "tables" / f"leakage_shift_{symbol}.csv", index=False, encoding="utf-8")
        placebo.to_csv(results_dir / "tables" / f"leakage_placebo_{symbol}.csv", index=False, encoding="utf-8")
        view = shift.pivot(index="feature_set", columns="shift_hours", values="mse_change_vs_price_pct").round(3)
        accuracy = shift.pivot(index="feature_set", columns="shift_hours", values="dir_accuracy").round(4)
        log.info("[%s] тест сдвига (%s): изменение MSE относительно набора только с ценами, %%; столбцы — сдвиг "
                 "новостей в часах:\n%s\nточность направления:\n%s", symbol, model, view.to_string(), accuracy.to_string())
        log.info("[%s] плацебо (перестановок цели — %d): точность направления %.4f (от %.4f до %.4f); "
                 "угадывание по долям растущих часов дало бы %.4f", symbol, len(placebo), placebo["dir_accuracy"].mean(),
                 placebo["dir_accuracy"].min(), placebo["dir_accuracy"].max(), placebo["chance_accuracy"].mean())
        report[symbol] = {"shift": shift.to_dict(orient="records"),
                          "placebo_mean_accuracy": float(placebo["dir_accuracy"].mean()),
                          "placebo_chance_accuracy": float(placebo["chance_accuracy"].mean())}
    save_json({"manifest": run_manifest(cfg), "model": model, "seed": seed, "assets": report},
              results_dir / "metrics" / "leakage_checks.json")
    log.info("Готово")


if __name__ == "__main__":
    main()
