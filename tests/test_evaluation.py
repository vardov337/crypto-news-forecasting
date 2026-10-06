"""Шаг 8: метрики, тесты, бэктест и критерий выбора на искусственных данных."""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cryptonews.evaluation import backtest, metrics
from cryptonews.evaluation import tests as st

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), ROOT / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_point_metrics():
    y = np.array([0.01, -0.02, 0.0, 0.03])
    f = np.array([0.02, -0.01, 0.01, -0.01])
    assert metrics.rmse(y, f) == pytest.approx(np.sqrt(np.mean((y - f) ** 2)))
    assert metrics.mae(y, f) == pytest.approx(np.mean(np.abs(y - f)))
    accuracy, n = metrics.directional_accuracy(y, f)
    assert n == 3 and accuracy == pytest.approx(2 / 3)                   # нулевой час исключён
    assert np.isnan(metrics.directional_accuracy(y, np.zeros(4))[0])     # нулевой прогноз направления не задаёт
    bench = np.full(4, y.mean())
    assert metrics.r2_oos(y, bench, bench) == pytest.approx(0.0)
    assert metrics.r2_oos(y, y, bench) == pytest.approx(1.0)


def test_holm_and_long_run_variance():
    adjusted = st.holm([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06, 0.02])
    partial = st.holm([0.01, np.nan])
    assert partial[0] == pytest.approx(0.01) and np.isnan(partial[1])
    x = np.random.default_rng(0).normal(size=500)
    assert st.long_run_variance(x, 0) == pytest.approx(np.var(x))
    assert st.newey_west_lags(17544) == 12


def test_diebold_mariano_direction_and_formula():
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, 3000)
    good, bad = y + rng.normal(0, 0.5, 3000), y + rng.normal(0, 1.0, 3000)
    better = st.diebold_mariano((y - bad) ** 2, (y - good) ** 2)
    assert better["dm_stat"] > 5 and better["p_value"] < 1e-6
    worse = st.diebold_mariano((y - good) ** 2, (y - bad) ** 2)
    assert worse["p_value"] > 0.99                                       # односторонний тест
    d = (y - bad) ** 2 - (y - good) ** 2
    manual = d.mean() / np.sqrt(np.var(d) / len(d))
    assert st.diebold_mariano((y - bad) ** 2, (y - good) ** 2, lags=0)["dm_stat"] == pytest.approx(manual)
    assert np.isnan(st.diebold_mariano(d, d)["dm_stat"])                 # одинаковые прогнозы


def test_pesaran_timmermann():
    rng = np.random.default_rng(2)
    y = rng.normal(size=4000)
    perfect = st.pesaran_timmermann(y, y)
    assert perfect["hit_rate"] == 1.0 and perfect["p_value"] < 1e-10
    random = st.pesaran_timmermann(y, rng.normal(size=4000))
    assert abs(random["pt_stat"]) < 4
    assert np.isnan(st.pesaran_timmermann(y, np.ones(4000))["pt_stat"])  # прогноз всегда одного знака


def test_stationary_bootstrap_blocks():
    rng = np.random.default_rng(3)
    index = st.stationary_bootstrap_indices(1000, 50, 24, rng)
    assert index.shape == (50, 1000) and index.min() >= 0 and index.max() < 1000
    continued = np.mean(index[:, 1:] == (index[:, :-1] + 1) % 1000)
    assert 0.94 < continued < 0.98                                       # новый блок с вероятностью 1/24


def test_sharpe_difference_ci():
    rng = np.random.default_rng(4)
    hold = rng.normal(0.0, 0.01, 5000)
    same = st.sharpe_difference_ci({"same": hold}, hold, reps=200, mean_block=24, seed=0, periods_per_year=8760)
    assert same.loc[0, "sharpe_diff"] == pytest.approx(0.0) and same.loc[0, "ci_low"] == pytest.approx(0.0)
    better = st.sharpe_difference_ci({"better": hold + 0.002}, hold, reps=200, mean_block=24, seed=0,
                                     periods_per_year=8760)
    assert better.loc[0, "ci_low"] > 0


def test_backtest_costs_and_metrics():
    pred = np.array([1.0, -1.0, 1.0, 1.0])
    log_returns = np.log1p([0.01, -0.02, 0.03, -0.01])
    summary, net = backtest.evaluate(pred, log_returns, "long_flat", 10, periods_per_year=8760)
    # позиции 1, 0, 1, 1: вход, выход, вход — три смены по 10 б. п.
    np.testing.assert_allclose(net, [0.01 - 0.001, -0.001, 0.03 - 0.001, -0.01])
    assert summary["breakeven_cost_bp"] == pytest.approx(np.mean([0.01, 0, 0.03, -0.01]) / 0.75 * 1e4)
    assert summary["max_drawdown"] == pytest.approx(min(np.cumprod(1 + net) / np.maximum.accumulate(
        np.concatenate([[1], np.cumprod(1 + net)]))[1:] - 1))
    short, net_short = backtest.evaluate(pred, log_returns, "long_short", 10, periods_per_year=8760)
    np.testing.assert_allclose(net_short, [0.01 - 0.001, 0.02 - 0.002, 0.03 - 0.002, -0.01])  # разворот стоит 2c
    assert short["time_in_market"] == 1.0 and summary["time_in_market"] == 0.75
    hold, _ = backtest.evaluate(np.ones(4), log_returns, "buy_and_hold", 10, periods_per_year=8760)
    assert hold["turnover_per_year"] == pytest.approx(8760 / 4)        # один вход за весь тест
    assert np.isnan(hold["breakeven_cost_bp"])


def test_hac_wald_and_granger():
    rng = np.random.default_rng(5)
    n = 3000
    x = rng.normal(size=n)
    noise = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x])
    assert st.hac_wald(0.5 * x + noise, X, [1])["p_value"] < 1e-10
    assert st.hac_wald(noise, X, [1], lags=0)["p_value"] > 0.001
    # White (HC0) при нулевых лагах
    beta = np.linalg.lstsq(X, noise, rcond=None)[0]
    u = noise - X @ beta
    bread = np.linalg.inv(X.T @ X)
    cov = bread @ (X * u[:, None]).T @ (X * u[:, None]) @ bread
    assert st.hac_wald(noise, X, [1], lags=0)["wald"] == pytest.approx(beta[1] ** 2 / cov[1, 1])

    index = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    news = rng.normal(size=n)
    ret = rng.normal(0, 0.01, n)
    table = pd.DataFrame({"ret_lag1": ret, "en_sent_mean": news, "valid": True}, index=index)
    table["target"] = 0.004 * news + rng.normal(0, 0.01, n)        # новость часа t двигает доходность часа t + 1
    assert st.granger_news(table, "target", "ret_lag1", ["en_sent_mean"], 6)["p_value"] < 1e-6
    table["target"] = rng.normal(0, 0.01, n)
    result = st.granger_news(table, "target", "ret_lag1", ["en_sent_mean"], 6, rows=index[1000:])
    assert result["df"] == 6 and result["n"] == n - 1000 and result["p_value"] > 1e-4


def fake_arch():
    class MCS:
        def __init__(self, losses, size, reps, block_size, method, bootstrap, seed=None):
            self.losses = losses

        def compute(self):
            means = self.losses.mean()
            self.included = list(means[means <= means.min() * 1.01].index)
            self.pvalues = pd.DataFrame({"Pvalue": np.where(means.index.isin(self.included), 0.5, 0.01)},
                                        index=pd.Index(means.index, name="Model name"))

    bootstrap = types.ModuleType("arch.bootstrap")
    bootstrap.MCS = MCS
    return {"arch": types.ModuleType("arch"), "arch.bootstrap": bootstrap}


def test_mcs_wrapper_with_stub():
    saved = {k: sys.modules.get(k) for k in ("arch", "arch.bootstrap")}
    sys.modules.update(fake_arch())
    try:
        losses = pd.DataFrame({"A": [1.0, 1.0], "B": [2.0, 2.0]})
        out = st.model_confidence_set(losses, size=0.1, reps=10, block_size=24, seed=0)
        assert out.index.tolist() == ["A", "B"] and out["in_mcs"].tolist() == [True, False]
    finally:
        for k, v in saved.items():
            sys.modules.pop(k) if v is None else sys.modules.__setitem__(k, v)


CFG = {"features": {"feature_sets": {"P": ["price"], "P_EN": ["price", "en"], "P_RU": ["price", "ru"],
                                     "P_EN_RU": ["price", "en", "ru"]}},
       "models": {"arimax": {"exog_feature_sets": ["P_EN", "P_RU", "P_EN_RU"]}},
       "evaluation": {"selection": {"cost_bp": 10}, "strategy": {"primary": "long_flat"}}}


def test_question_pairs_and_selection():
    script = load_script("08_evaluate.py")
    keys = {f"{m}:{s}" for m in ("random_forest", "xgboost", "lstm") for s in ("P", "P_EN", "P_RU", "P_EN_RU")}
    keys |= {"arima:P"} | {f"arimax:{s}" for s in ("P_EN", "P_RU", "P_EN_RU")}
    pairs = script.question_pairs(CFG, keys)
    assert sum(p[0] == "В1" for p in pairs) == 12 and sum(p[0] == "В2" for p in pairs) == 4
    assert ("В2", "arimax:P_EN", "arimax:P_EN_RU") in pairs and ("В1", "arima:P", "arimax:P_RU") in pairs

    accuracy = pd.DataFrame({"key": ["naive_mean:none", "xgboost:P", "lstm:P_EN", "arima:P"],
                             "label": ["mean", "x", "l", "a"],
                             "dm_vs_zero_p_holm": [np.nan, 0.01, 0.20, 0.01], "pt_p_holm": [np.nan, 0.5, 0.01, 0.5]})
    accuracy["accurate"] = (accuracy["dm_vs_zero_p_holm"] < 0.05) | (accuracy["pt_p_holm"] < 0.05)
    trading = pd.DataFrame({"strategy": "long_flat", "cost_bp": 10.0,
                            "key": ["naive_mean:none", "xgboost:P", "lstm:P_EN", "arima:P"],
                            "sharpe": [3.0, 1.0, 2.0, 2.5], "sharpe_diff": [0, 0.5, 0.6, 0.7],
                            "ci_low": [0.0, 0.1, 0.2, -0.1], "ci_high": [1, 1, 1, 1]})
    table = script.selection_table(accuracy, trading, CFG)
    assert "naive_mean:none" not in table["key"].tolist()                # базовые модели не выбираются
    assert table.loc[table["selected"], "key"].tolist() == ["lstm:P_EN"]  # arima: интервал задевает ноль
    trading["ci_low"] = -1.0
    assert not script.selection_table(accuracy, trading, CFG)["selected"].any()
