"""Шаг 7 (задачи 7.1–7.5): модели по схеме walk-forward, прогнозы по фолдам на диск.

Модели разбиты на группы, каждую можно запускать отдельно:
  baselines — нулевой прогноз и историческое среднее (секунды);
  arima     — ARIMA с выбором порядка по AIC и ARIMAX с новостными признаками (около 20 минут);
  trees     — Random Forest и XGBoost: подбор на окне настройки и 8 фолдов × 5 зёрен (нужна видеокарта);
  lstm      — LSTM: подбор и 8 фолдов × 3 зерна (нужна видеокарта, около часа).
Прогноз каждого фолда сразу пишется на Диск; если сессия оборвалась, повторный запуск
той же группы продолжит с места остановки.

Ключ --smoke — пробный прогон всех групп на маленьком куске данных (одна-две минуты):
проверить, что всё работает, прежде чем запускать долгий расчёт. Его результаты пишутся
отдельно и в статью не идут.

Ключ --restart начинает выбранные модели заново: их прежние прогнозы не удаляются, а
переносятся в results/predictions_old/<время запуска>/ (нужно, если после исправления кода
модель надо пересчитать). Ключ --models выбирает отдельные модели вместо группы.

Запуск:  python scripts/07_run_models.py --smoke
         python scripts/07_run_models.py --family baselines   (затем arima, trees, lstm)
         python scripts/07_run_models.py --models xgboost --restart
"""
import json
import shutil
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from cryptonews import validation
from cryptonews.cli import parse_args
from cryptonews.config import load_config
from cryptonews.models import run
from cryptonews.utils import get_logger, package_versions, run_manifest, save_json, set_seed


def add_arguments(parser) -> None:
    parser.add_argument("--family", default="all", choices=["all", *run.FAMILIES],
                        help="какую группу моделей считать (по умолчанию все)")
    parser.add_argument("--assets", nargs="*", default=None, help="активы, например BTCUSDT (по умолчанию все)")
    parser.add_argument("--models", nargs="*", default=None, choices=[m for f in run.FAMILIES.values() for m in f],
                        help="отдельные модели вместо группы, например xgboost lstm")
    parser.add_argument("--smoke", action="store_true", help="пробный прогон на маленьком куске данных")
    parser.add_argument("--restart", action="store_true",
                        help="посчитать выбранные модели заново; прежние прогнозы переносятся в predictions_old")


def environment(log) -> dict:
    info = {"packages": package_versions(("xgboost", "statsmodels", "torch", "scikit-learn"))}
    try:
        import torch
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        info["gpu"] = None
    log.info("Видеокарта: %s; версии: %s", info["gpu"] or "нет",
             ", ".join(f"{k} {v}" for k, v in info["packages"].items()))
    return info


def smoke_splits(splits: dict, table_index: pd.DatetimeIndex) -> tuple[dict, pd.Timestamp]:
    """Один фолд и 6000 часов истории перед ним."""
    fold = splits["folds"][0]
    return {**splits, "folds": [fold]}, fold.test_start - pd.Timedelta(hours=6000)


def move_aside(out_dir: Path, symbols, cfg, models, log) -> None:
    """Прежние прогнозы выбранных моделей — в predictions_old/<время>/ (ничего не удаляется)."""
    target = out_dir.parent / "predictions_old" / time.strftime("%Y%m%d_%H%M%S")
    for symbol in symbols:
        for model in models:
            for feature_set in run.feature_sets(cfg, model):
                folder = out_dir / symbol / f"{model}__{feature_set}"
                if folder.exists():
                    destination = target / symbol / folder.name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(folder), str(destination))
                    log.info("Прежние прогнозы %s %s %s перенесены в %s", symbol, model, feature_set, destination)


def quick_summary(out_dir, symbols, cfg, n_folds: int) -> pd.DataFrame:
    """RMSE по всем фолдам (прогноз усреднён по зёрнам) — грубая проверка, полная оценка — шаг 8."""
    rows = []
    for symbol in symbols:
        base = run.collect(out_dir, symbol, "naive_zero", run.NO_FEATURES, n_folds)
        for family in run.FAMILIES.values():
            for model in family:
                for feature_set in run.feature_sets(cfg, model):
                    pred = run.collect(out_dir, symbol, model, feature_set, n_folds)
                    if pred is None:
                        continue
                    row = {"Актив": symbol, "Модель": model, "Набор": feature_set, "Строк": len(pred),
                           "RMSE": run.rmse(pred)}
                    if base is not None:
                        common = pred.index.intersection(base.index)
                        row["RMSE / нулевой прогноз"] = run.rmse(pred.loc[common]) / run.rmse(base.loc[common])
                        signs = pred.loc[common]
                        signs = signs[(signs["y_pred"] != 0) & (signs["y_true"] != 0)]
                        row["Угадано направлений"] = (float(np.mean(np.sign(signs["y_pred"]) == np.sign(signs["y_true"])))
                                                      if len(signs) else np.nan)
                    rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args(__doc__, add_arguments)
    cfg = load_config(args.config)
    log = get_logger()
    set_seed(int(cfg["seeds"]["default"][0]))
    data_dir, results_dir = cfg["paths"]["data_dir"], cfg["paths"]["results_dir"]
    env = environment(log)

    splits_path = results_dir / "splits.json"
    if not splits_path.exists():
        raise SystemExit(f"Нет {splits_path} — сначала выполните шаг 6")
    splits = validation.from_json(json.loads(splits_path.read_text(encoding="utf-8")))
    symbols = args.assets or list(cfg["data"]["prices"]["symbols"])
    families = list(run.FAMILIES) if args.family == "all" else [args.family]
    models = [m for f in families for m in run.FAMILIES[f]]
    if args.models:
        models = [m for f in run.FAMILIES.values() for m in f if m in args.models]   # порядок групп: ARIMA до ARIMAX
        families = [f for f, members in run.FAMILIES.items() if set(members) & set(models)]

    if args.smoke:
        cfg = run.smoke_config(cfg)
        out_dir = Path(tempfile.mkdtemp(prefix="cryptonews_smoke_"))   # каждый раз новая папка вне Диска
    else:
        out_dir = results_dir / "predictions"
        if args.restart:
            move_aside(out_dir, symbols, cfg, models, log)
    runner = run.Runner(cfg, out_dir, log, smoke=args.smoke)

    started, failures = time.time(), []
    for symbol in symbols:
        path = data_dir / "processed" / f"features_{symbol}.parquet"
        if not path.exists():
            raise SystemExit(f"Нет {path} — сначала выполните шаг 6")
        table = pd.read_parquet(path)
        use_splits = splits
        if args.smoke:
            use_splits, history_start = smoke_splits(splits, table.index)
            table = table.loc[(table.index >= history_start) & (table.index < use_splits["folds"][0].test_end)]
        log.info("[%s] строк %d, пригодных %d; модели: %s", symbol, len(table), int(table["valid"].sum()),
                 ", ".join(models))
        for model in models:
            t0 = time.time()
            try:
                runner.run(symbol, table, use_splits, [model])
            except Exception as error:                     # одна сломавшаяся модель не останавливает остальные
                failures.append(f"{symbol} {model}: {type(error).__name__}: {error}")
                log.error("[%s] %s — ошибка, модель пропущена:\n%s", symbol, model, traceback.format_exc())
                continue
            log.info("[%s] %s: %.0f с", symbol, model, time.time() - t0)

    n_folds = 1 if args.smoke else len(splits["folds"])
    summary = quick_summary(out_dir, symbols, cfg, n_folds)
    if len(summary):
        name = "model_rmse_smoke.csv" if args.smoke else "model_rmse_quick.csv"
        summary.round(6).to_csv(results_dir / "tables" / name, index=False, encoding="utf-8")
        log.info("Сводка (RMSE по посчитанным фолдам; полная оценка — шаг 8):\n%s",
                 summary.round(4).to_string(index=False))
    label = "smoke" if args.smoke else ("_".join(models) if args.models else args.family)
    save_json({"manifest": run_manifest(cfg), "environment": env, "families": families, "models": models,
               "smoke": args.smoke, "restart": args.restart, "failures": failures,
               "seconds": round(time.time() - started, 1)},
              results_dir / "metrics" / f"models_{label}.json")
    if failures:
        raise SystemExit("Не посчитались (подробности — выше в выводе):\n  " + "\n  ".join(failures))
    log.info("Готово за %.1f мин", (time.time() - started) / 60)


if __name__ == "__main__":
    main()
