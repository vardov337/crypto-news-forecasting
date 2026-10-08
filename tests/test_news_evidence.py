"""Шаг 8г: границы эффекта новостных признаков, кварталы, часы с новостями, содержательность тональности."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cryptonews import validation
from cryptonews.evaluation import tests as stat_tests

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("news_evidence", ROOT / "scripts" / "08d_news_evidence.py")
step = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step)


def combined_frame(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2023-06-01", periods=n, freq="h", tz="UTC", name="open_time")
    y = rng.normal(0, 0.005, n)
    frame = pd.DataFrame({"y_true": y,
                          "xgboost:P": 0.1 * y + rng.normal(0, 0.002, n),
                          "xgboost:P_EN": 0.1 * y + rng.normal(0, 0.002, n)}, index=index)
    pairs = pd.DataFrame({"question": ["В1"], "base": ["xgboost:P"], "alternative": ["xgboost:P_EN"],
                          "comparison": ["XGBoost P_EN против XGBoost P"]})
    return frame, pairs


def test_effect_matches_diebold_mariano():
    frame, _ = combined_frame()
    y, base, alt = (frame[c].to_numpy() for c in ("y_true", "xgboost:P", "xgboost:P_EN"))
    out = step.effect(y, base, alt)
    dm = stat_tests.diebold_mariano((y - base) ** 2, (y - alt) ** 2)
    assert out["dm_stat"] == pytest.approx(dm["dm_stat"]) and out["p_value"] == pytest.approx(dm["p_value"])
    direct = 100 * (np.mean((y - alt) ** 2) / np.mean((y - base) ** 2) - 1)
    assert out["mse_change_pct"] == pytest.approx(direct)
    assert out["ci_low_pct"] < out["mse_change_pct"] < out["ci_high_pct"]
    assert out["mde_pct"] == pytest.approx(2.4865 * out["se_pct"], rel=1e-3)


def test_relevant_language():
    assert step.relevant_language("lstm:P", "lstm:P_EN") == "en"
    assert step.relevant_language("lstm:P", "lstm:P_RU") == "ru"
    assert step.relevant_language("lstm:P", "lstm:P_EN_RU") == "both"
    assert step.relevant_language("lstm:P_EN", "lstm:P_EN_RU") == "ru"
    assert step.relevant_language("arima:P", "arimax:P_EN") == "en"


def test_folds_and_subsets():
    frame, pairs = combined_frame()
    start = frame.index[0]
    splits = validation.walk_forward(start - pd.DateOffset(months=30), start + pd.DateOffset(months=2),
                                     test_months=2, fold_months=1, tuning_months=1, embargo_hours=1)
    folds = step.folds_table(frame, pairs, splits)
    assert len(folds) == 2 and set(folds["fold"]) == {"fold_1", "fold_2"}
    summary = step.folds_summary(folds)
    assert summary.iloc[0]["pairs"] == 2 and 0 <= summary.iloc[0]["sign_test_p"] <= 1
    series = pd.DataFrame({"en_count": 0, "ru_count": 0, "en_intensity": 0.0, "ru_intensity": 0.0},
                          index=frame.index)
    series.iloc[::4, 0] = 2                                   # новости в каждом четвёртом часе
    series["en_intensity"] = np.arange(len(series), dtype=float)
    masks = step.subset_masks(series, frame.index)
    assert masks[("news", "en")].sum() == len(frame) // 4
    assert masks[("burst", "en")].sum() == pytest.approx(0.1 * len(frame), abs=2)
    subsets = step.subsets_table(frame, pairs, masks)
    assert list(subsets["subset"]) == ["news", "burst"] and subsets.iloc[0]["rows"] == len(frame) // 4


def test_validity_detects_same_hour_relation():
    frame, _ = combined_frame(n=4000, seed=1)
    rng = np.random.default_rng(2)
    index = frame.index
    sent = pd.Series(np.nan, index=index)
    news_hours = rng.random(len(index)) < 0.5
    # тональность часа t + 1 (строка t + 1) следует за доходностью этого часа (y_true строки t)
    same_hour = np.concatenate([[0.0], frame["y_true"].to_numpy()[:-1]]) * 40 + rng.normal(0, 0.1, len(index))
    sent[news_hours] = same_hour[news_hours]
    series = pd.DataFrame({"en_sent": sent, "ru_sent": np.nan * sent}, index=index)
    series["ru_sent"] = rng.normal(0, 0.1, len(index))
    out = step.validity_table(frame, series).set_index(["language", "relation"])
    assert out.loc[("en", "тот же час"), "p_value"] < 0.001 and out.loc[("en", "тот же час"), "corr"] > 0.5
    assert out.loc[("en", "следующий час"), "p_value"] > 0.01
