"""Шаг 9: описательные статистики доходности, диагностика окна настройки, окружение, схема подхода."""
import importlib.util
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cryptonews import report, validation
from cryptonews.config import load_config
from cryptonews.models.lstm import complete_windows

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_report", ROOT / "scripts" / "09_make_report.py")
step = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step)
LOG = logging.getLogger("test")


def setup(tmp_path):
    cfg = load_config(ROOT / "configs" / "config.yaml")
    cfg["paths"]["data_dir"], cfg["paths"]["results_dir"] = tmp_path / "data", tmp_path / "results"
    start, end = pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2025-06-01", tz="UTC")
    splits = validation.from_config(cfg, start, end)
    index = pd.date_range(start, end - pd.Timedelta(hours=1), freq="h", name="open_time")
    rng = np.random.default_rng(0)
    valid = np.ones(len(index), dtype=bool)
    valid[[100, 10500, 10505]] = False                       # пропуски биржи; 10500 и 10505 — в окне настройки (март 2023)
    table = pd.DataFrame({"valid": valid, "target": rng.normal(0, 0.005, len(index))}, index=index)
    path = cfg["paths"]["data_dir"] / "processed" / "features_BTCUSDT.parquet"
    path.parent.mkdir(parents=True)
    table.to_parquet(path)
    return cfg, splits, table


def test_returns_table(tmp_path):
    cfg, splits, table = setup(tmp_path)
    out = step.returns_table(cfg, splits, ["BTCUSDT"], LOG)
    full = out[out["Выборка"] == "вся"].iloc[0]
    y = table.loc[table["valid"], "target"]
    assert full["Часов"] == len(y)
    assert full["Ст. откл., %"] == pytest.approx(round(100 * y.std(), 3))
    test = out[out["Выборка"] == "тест"].iloc[0]
    assert test["Часов"] == int(((table.index >= splits["folds"][0].test_start) & table["valid"]).sum())


def test_tuning_tables_ratios_and_rows(tmp_path, caplog):
    cfg, splits, table = setup(tmp_path)
    window = splits["tuning"]
    inside = (table.index >= window.test_start) & (table.index < window.test_end)
    valid, y = table["valid"].to_numpy(bool), table["target"].to_numpy()
    assert not valid[inside].all()                            # в окне есть пропуск — проверяем отбор строк
    base = cfg["paths"]["results_dir"] / "predictions" / "BTCUSDT"
    forest = []
    for depth, factor in ((5, 1.002), (10, 0.999), (None, 1.004)):
        rows = valid & inside
        forest.append({"params": {"n_estimators": 300, "max_depth": depth, "min_samples_leaf": 50,
                                  "max_features": "sqrt"},
                       "mse": float(np.mean(y[rows] ** 2)) * factor ** 2, "rows": int(rows.sum())})
    lstm = []
    for lookback, factor in ((12, 1.001), (24, 1.003)):
        rows = complete_windows(valid, lookback) & inside
        lstm.append({"params": {"lookback": lookback, "hidden_size": 32},
                     "mse": float(np.mean(y[rows] ** 2)) * factor ** 2, "rows": int(rows.sum())})
    for folder, results, chosen in (("random_forest__P", forest, 1), ("lstm__P", lstm, 0)):
        (base / folder).mkdir(parents=True)
        (base / folder / "tuning.json").write_text(json.dumps(
            {"window": window.as_dict(), "results": results, "chosen": results[chosen]["params"]}), encoding="utf-8")
    accuracy = {"BTCUSDT": pd.DataFrame({"key": ["random_forest:P", "lstm:P"], "rmse_ratio_zero": [1.0007, 0.9998]})}
    with caplog.at_level(logging.WARNING):
        summary, depth = step.tuning_tables(cfg, accuracy, ["BTCUSDT"], LOG)
    assert not caplog.records                                 # строки окна совпали с записанными
    forest_row = summary[summary["Модель"] == "Random Forest P"].iloc[0]
    assert forest_row["Окно настройки: выбранный вариант"] == pytest.approx(0.999)
    assert forest_row["Окно настройки: худший вариант"] == pytest.approx(1.004)
    assert forest_row["Тест (8 фолдов)"] == pytest.approx(1.0007)
    assert summary[summary["Модель"] == "LSTM P"].iloc[0]["Окно настройки: выбранный вариант"] == pytest.approx(1.001)
    assert list(depth.columns[2:]) == ["глубина 5", "глубина 10", "глубина без ограничения"]
    assert depth.iloc[0]["глубина без ограничения"] == pytest.approx(1.004)


def test_environment_blocks(tmp_path):
    results = tmp_path / "results"
    (results / "env").mkdir(parents=True)
    (results / "metrics").mkdir()
    (results / "env" / "manifest.json").write_text(json.dumps(
        {"python": "3.12.11", "git_commit": "abc", "packages": {"pandas": "2.2.2"}}), encoding="utf-8")
    (results / "metrics" / "evaluation.json").write_text(json.dumps(
        {"manifest": {"git_commit": "0123456789abcdef", "git_dirty": False, "packages": {"xgboost": "3.0.5"}}}),
        encoding="utf-8")
    (results / "metrics" / "plain.json").write_text("[1, 2]", encoding="utf-8")     # без манифеста — пропускается
    folder = results / "predictions" / "BTCUSDT" / "xgboost__P"
    folder.mkdir(parents=True)
    for i in (1, 2):
        (folder / f"fold_{i}__seed0.json").write_text(json.dumps({"git_commit": "0123456789abcdef"}), encoding="utf-8")
    blocks = dict(step.environment_blocks(results, ["BTCUSDT"]))
    assert len(blocks) == 3
    runs = next(frame for title, frame in blocks.items() if title.startswith("Запуски"))
    assert runs.iloc[0]["Коммит кода"] == "0123456789" and runs.iloc[0]["xgboost"] == "3.0.5"
    commits = next(frame for title, frame in blocks.items() if title.startswith("Коммиты"))
    assert commits.iloc[0]["Файлов прогнозов (фолд × зерно)"] == 2


def test_scheme_figure(tmp_path):
    plt = report.pyplot()
    step.figure_scheme(plt, tmp_path)
    assert (tmp_path / "fig1_scheme.png").stat().st_size > 10_000 and (tmp_path / "fig1_scheme.svg").exists()
    fig = plt.figure(figsize=(4, 1))
    lines = step.wrap_to_width(fig, "один два $x_{t+1} = 1$ три четыре", 8, 0.6)
    plt.close(fig)
    assert any("$x_{t+1}" in line for line in lines) and any("три четыре" in line for line in lines)
