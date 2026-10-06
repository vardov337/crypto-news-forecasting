"""Запуск моделей по схеме walk-forward (задачи 7.1–7.5, PROTOCOL.md, разделы 6–7).

Для каждой модели и набора признаков:
  1. гиперпараметры выбираются один раз — обучение на строках до окна настройки, выбор по
     MSE на окне настройки (первое зерно); выбор записывается в tuning.json и дальше не меняется;
  2. перед каждым фолдом модель обучается заново на всех строках до него и прогнозирует фолд;
     стохастические модели — с несколькими зёрнами;
  3. прогноз каждого фолда и зерна сразу пишется на диск, поэтому после обрыва сессии
     повторный запуск продолжает с первого непосчитанного фолда.

Раскладка: <out_dir>/<актив>/<модель>__<набор>/<фолд>__seed<зерно>.parquet (y_true, y_pred)
и рядом .json со служебными сведениями (порядок ARIMA, число деревьев, эпохи, время).
"""
from __future__ import annotations

import copy
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cryptonews import features
from cryptonews.features import TARGET
from cryptonews.models import arima, baselines, lstm, trees
from cryptonews.utils import git_commit, save_json

FAMILIES = {
    "baselines": ["naive_zero", "naive_mean"],
    "arima": ["arima", "arimax"],
    "trees": ["random_forest", "xgboost"],
    "lstm": ["lstm"],
}
NO_FEATURES = "none"


def expand_grid(spec: dict) -> list[dict]:
    """Все сочетания значений: списки перебираются, одиночные значения фиксированы."""
    keys = list(spec)
    values = [v if isinstance(v, list) else [v] for v in spec.values()]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def feature_sets(cfg: dict, model: str) -> list[str]:
    if model in FAMILIES["baselines"]:
        return [NO_FEATURES]
    if model == "arima":
        return ["P"]                                        # только ряд доходности
    if model == "arimax":
        return list(cfg["models"]["arimax"]["exog_feature_sets"])
    return list(cfg["features"]["feature_sets"])


def columns_for(cfg: dict, model: str, feature_set: str) -> list[str]:
    if feature_set == NO_FEATURES or model == "arima":
        return []
    if model == "arimax":                                   # экзогенные — только новостные признаки
        return [c for block in cfg["features"]["feature_sets"][feature_set] if block != "price"
                for c in features.news_columns(cfg, block)]
    return features.feature_set_columns(cfg, feature_set)


def grid_for(cfg: dict, model: str) -> list[dict]:
    m = cfg["models"]
    if model == "random_forest":
        return expand_grid(m["random_forest"])
    if model == "xgboost":
        return expand_grid({k: v for k, v in m["xgboost"].items()
                            if k not in ("n_estimators", "early_stopping_rounds")})
    if model == "lstm":
        return expand_grid({k: m["lstm"][k] for k in ("lookback", "hidden_size")})
    return [{}]


def seeds_for(cfg: dict, model: str) -> list[int]:
    if model in ("random_forest", "xgboost"):
        return [int(s) for s in cfg["seeds"]["default"]]
    if model == "lstm":
        return [int(s) for s in cfg["seeds"]["lstm"]]
    return [0]


def build_model(cfg: dict, model: str):
    m, frac = cfg["models"], float(cfg["validation"]["early_stopping_fraction"])
    if model == "naive_zero":
        return baselines.NaiveZero()
    if model == "naive_mean":
        return baselines.NaiveMean()
    if model == "arima":
        return arima.Arima(m["arima"]["p"], m["arima"]["q"], m["arima"]["estimation_window_hours"])
    if model == "arimax":
        return arima.Arimax(m["arima"]["estimation_window_hours"])
    if model == "random_forest":
        return trees.RandomForest()
    if model == "xgboost":
        x = m["xgboost"]
        return trees.XGBoost(x["n_estimators"], x["early_stopping_rounds"], frac)
    if model == "lstm":
        s = m["lstm"]
        return lstm.LSTMModel(s["dropout"], s["learning_rate"], s["batch_size"], s["max_epochs"], s["patience"],
                              s["scale_target"], frac)
    raise ValueError(f"Неизвестная модель: {model}")


def smoke_config(cfg: dict) -> dict:
    """Облегчённая конфигурация для пробного прогона: проверить, что все модели работают."""
    cfg = copy.deepcopy(cfg)
    m = cfg["models"]
    m["arima"].update({"p": [0, 1], "q": [0, 1], "estimation_window_hours": 1000})
    m["random_forest"].update({"n_estimators": [20], "max_depth": [None], "min_samples_leaf": [50],
                               "max_features": ["sqrt"]})
    m["xgboost"].update({"n_estimators": 30, "learning_rate": [0.05], "max_depth": [3], "min_child_weight": [10]})
    m["lstm"].update({"lookback": [12], "hidden_size": [32], "max_epochs": 2})
    cfg["seeds"] = {"default": [0], "lstm": [0]}
    return cfg


def rmse(frame: pd.DataFrame) -> float:
    return float(np.sqrt(np.mean((frame["y_true"] - frame["y_pred"]) ** 2)))


class Runner:
    def __init__(self, cfg: dict, out_dir: Path, log, smoke: bool = False):
        self.cfg, self.out_dir, self.log, self.smoke = cfg, Path(out_dir), log, smoke
        self.commit = git_commit()                          # каким кодом посчитан каждый файл

    def folder(self, symbol: str, model: str, feature_set: str) -> Path:
        return self.out_dir / symbol / f"{model}__{feature_set}"

    def run(self, symbol: str, table: pd.DataFrame, splits: dict, models: list[str]) -> None:
        for model in models:
            for feature_set in feature_sets(self.cfg, model):
                self.run_model(symbol, table, splits, model, feature_set)

    def tune(self, symbol: str, table: pd.DataFrame, splits: dict, model: str, feature_set: str,
             adapter, columns: list[str]) -> dict:
        grid = grid_for(self.cfg, model)
        if not adapter.needs_grid or len(grid) == 1 and self.smoke:
            return grid[0]
        path = self.folder(symbol, model, feature_set) / "tuning.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["chosen"]
        window, seed = splits["tuning"], seeds_for(self.cfg, model)[0]
        started, results = time.time(), []
        for params in grid:
            t0 = time.time()
            pred, info = adapter.fit_predict(table, columns, window, params, seed)
            truth = table.loc[pred.index, TARGET]
            results.append({"params": params, "mse": float(np.mean((truth - pred) ** 2)), "rows": int(len(pred)),
                            "seconds": round(time.time() - t0, 1), "info": info})
        best = min(range(len(results)), key=lambda i: (results[i]["mse"], i))
        save_json({"window": window.as_dict(), "seed": seed, "results": results, "chosen": results[best]["params"],
                   "git_commit": self.commit}, path)
        self.log.info("[%s] %s %s: подбор на окне настройки — %d вариантов за %.0f с, выбрано %s",
                      symbol, model, feature_set, len(grid), time.time() - started, results[best]["params"])
        return results[best]["params"]

    def arima_order(self, symbol: str, fold_name: str) -> list[int]:
        path = self.folder(symbol, "arima", "P") / f"{fold_name}__seed0.json"
        if not path.exists():
            raise RuntimeError(f"Для ARIMAX нужен порядок ARIMA того же фолда: нет {path}. Сначала ARIMA.")
        return json.loads(path.read_text(encoding="utf-8"))["info"]["order"]

    def run_model(self, symbol: str, table: pd.DataFrame, splits: dict, model: str, feature_set: str) -> None:
        adapter = build_model(self.cfg, model)
        columns = columns_for(self.cfg, model, feature_set)
        folder = self.folder(symbol, model, feature_set)
        folder.mkdir(parents=True, exist_ok=True)
        seeds = seeds_for(self.cfg, model)
        todo = [(f, s) for f in splits["folds"] for s in seeds
                if not (folder / f"{f.name}__seed{s}.parquet").exists()]
        if not todo:
            self.log.info("[%s] %s %s: все фолды уже посчитаны", symbol, model, feature_set)
            return
        params = self.tune(symbol, table, splits, model, feature_set, adapter, columns)
        for fold in splits["folds"]:
            fold_seeds = [s for f, s in todo if f.name == fold.name]
            if not fold_seeds:
                continue
            fold_params = {"order": self.arima_order(symbol, fold.name)} if model == "arimax" else params
            started = time.time()
            for seed in fold_seeds:
                t0 = time.time()
                pred, info = adapter.fit_predict(table, columns, fold, fold_params, seed)
                frame = pd.DataFrame({"y_true": table.loc[pred.index, TARGET].to_numpy(), "y_pred": pred.to_numpy()},
                                     index=pred.index)
                frame.index.name = "open_time"
                frame.to_parquet(folder / f"{fold.name}__seed{seed}.parquet")
                save_json({"model": model, "feature_set": feature_set, "fold": fold.as_dict(), "seed": seed,
                           "params": fold_params, "columns": columns, "rows": int(len(frame)),
                           "seconds": round(time.time() - t0, 1), "info": info, "git_commit": self.commit},
                          folder / f"{fold.name}__seed{seed}.json")
            done = [pd.read_parquet(folder / f"{fold.name}__seed{s}.parquet") for s in seeds
                    if (folder / f"{fold.name}__seed{s}.parquet").exists()]
            mean = pd.concat(done).groupby(level=0).mean()
            self.log.info("[%s] %s %s %s: %d зерн., %.0f с, RMSE %.5f", symbol, model, feature_set, fold.name,
                          len(fold_seeds), time.time() - started, rmse(mean))


def collect(out_dir: Path, symbol: str, model: str, feature_set: str, n_folds: int) -> pd.DataFrame | None:
    """Прогнозы модели по всем фолдам, усреднённые по зёрнам; None, если посчитано не всё."""
    folder = Path(out_dir) / symbol / f"{model}__{feature_set}"
    if not folder.exists():
        return None
    frames = []
    for i in range(1, n_folds + 1):
        files = sorted(folder.glob(f"fold_{i}__seed*.parquet"))
        if not files:
            return None
        frames.append(pd.concat([pd.read_parquet(f) for f in files]).groupby(level=0).mean())
    return pd.concat(frames).sort_index()
