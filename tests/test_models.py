"""Шаг 7 без тяжёлых библиотек: statsmodels и xgboost подменяются простыми заглушками.

Главное здесь — выравнивание: прогноз для строки t должен опираться только на данные
до закрытия свечи t, а сравниваться с доходностью свечи t + 1.
"""
import sys
import types

import numpy as np
import pandas as pd
import pytest

from cryptonews import validation
from cryptonews.features import TARGET
from cryptonews.models import arima, lstm, run

CFG = {
    "features": {
        "price": {"return_lags": [1, 2, 3, 4, 5, 6], "realized_vol_windows": [6, 24], "volume_change": True},
        "news": {"languages": ["en", "ru"], "rolling_window": 6, "trend_window": 3, "sentiment_lag": 1,
                 "no_news_fill": 0.0, "intensity_window_hours": 168},
        "feature_sets": {"P": ["price"], "P_EN": ["price", "en"], "P_RU": ["price", "ru"],
                         "P_EN_RU": ["price", "en", "ru"]},
    },
    "validation": {"test_months": 6, "fold_months": 3, "tuning_window_months": 3, "embargo_hours": 1,
                   "early_stopping_fraction": 0.1},
    "seeds": {"default": [0, 1], "lstm": [0]},
    "models": {
        "arima": {"p": [0, 1], "q": [0, 1], "estimation_window_hours": 500},
        "arimax": {"exog_feature_sets": ["P_EN", "P_RU", "P_EN_RU"]},
        "random_forest": {"n_estimators": [10], "max_depth": [5, None], "min_samples_leaf": [20],
                          "max_features": ["sqrt", 0.5]},
        "xgboost": {"n_estimators": 20, "early_stopping_rounds": 5, "learning_rate": [0.05, 0.1], "max_depth": [3],
                    "min_child_weight": [10], "subsample": [0.8], "colsample_bytree": [0.8], "reg_lambda": [1.0]},
        "lstm": {"lookback": [12, 24], "hidden_size": [32], "dropout": 0.2, "learning_rate": 0.001,
                 "batch_size": 256, "max_epochs": 2, "patience": 1, "scale_target": True},
    },
}


def make_table(hours=24 * 400, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2023-01-01", periods=hours, freq="h", tz="UTC", name="open_time")
    returns = rng.normal(0, 0.01, hours)
    table = pd.DataFrame({"ret_lag1": returns}, index=index)
    for lag in range(2, 7):
        table[f"ret_lag{lag}"] = table["ret_lag1"].shift(lag - 1)
    table["rv_6"], table["rv_24"], table["volume_change"] = 0.01, 0.01, rng.normal(0, 1, hours)
    for lang in ("en", "ru"):
        for name in ("sent_mean", "news_intensity", "neg_share", "sent_lag1", "sent_max6", "sent_min6",
                     "sent_std6", "sent_trend3"):
            table[f"{lang}_{name}"] = rng.normal(0, 1, hours)
    table[TARGET] = table["ret_lag1"].shift(-1)
    table.iloc[100:103, 0] = np.nan                       # остановка биржи
    table["valid"] = table.notna().all(axis=1)
    return table


def splits_for(table):
    return validation.walk_forward(table.index[0], table.index[-1] + pd.Timedelta(hours=1), test_months=6,
                                   fold_months=3, tuning_months=3, embargo_hours=1)


def test_grid_and_sets():
    assert len(run.grid_for(CFG, "random_forest")) == 4
    assert len(run.grid_for(CFG, "xgboost")) == 2 and "n_estimators" not in run.grid_for(CFG, "xgboost")[0]
    assert len(run.grid_for(CFG, "lstm")) == 2
    assert run.feature_sets(CFG, "arima") == ["P"] and run.feature_sets(CFG, "naive_zero") == ["none"]
    assert run.feature_sets(CFG, "arimax") == ["P_EN", "P_RU", "P_EN_RU"]
    assert run.columns_for(CFG, "arimax", "P_EN") == [c for c in run.columns_for(CFG, "xgboost", "P_EN")
                                                      if c.startswith("en_")]
    assert len(run.columns_for(CFG, "arimax", "P_EN_RU")) == 16 and run.columns_for(CFG, "arima", "P") == []
    assert run.seeds_for(CFG, "xgboost") == [0, 1] and run.seeds_for(CFG, "arima") == [0]


def test_baselines_run_resume_and_no_leakage(tmp_path):
    table = make_table()
    splits = splits_for(table)
    runner = run.Runner(CFG, tmp_path, _Log())
    runner.run("BTCUSDT", table, splits, ["naive_zero", "naive_mean"])
    files = sorted((tmp_path / "BTCUSDT" / "naive_mean__none").glob("*.parquet"))
    assert [f.name for f in files] == ["fold_1__seed0.parquet", "fold_2__seed0.parquet"]
    mtime = files[0].stat().st_mtime
    runner.run("BTCUSDT", table, splits, ["naive_mean"])           # повторный запуск ничего не пересчитывает
    assert files[0].stat().st_mtime == mtime
    pred = run.collect(tmp_path, "BTCUSDT", "naive_mean", "none", 2)
    row = pred.index[10]
    assert pred.loc[row, "y_pred"] == pytest.approx(table.loc[:row, "ret_lag1"].mean())
    assert pred.loc[row, "y_true"] == pytest.approx(table.loc[row + pd.Timedelta(hours=1), "ret_lag1"])
    zero = run.collect(tmp_path, "BTCUSDT", "naive_zero", "none", 2)
    assert (zero["y_pred"] == 0).all() and len(zero) == len(pred)


class _Log:
    def __init__(self):
        self.lines = []

    def info(self, msg, *args):
        self.lines.append(msg % args if args else msg)

    warning = info


def fake_statsmodels():
    """ARIMA-заглушка: одношаговый прогноз — предыдущее наблюдение (или первый столбец exog)."""
    class Result:
        def __init__(self, endog, exog, order):
            self.endog, self.exog, self.order = np.asarray(endog, float), exog, order
            self.aic = float(sum(order)) + 0.1 * order[0]
            self.param_names, self.params = ["const"], np.array([0.0])

        def apply(self, endog, exog=None):
            return Result(endog, exog, self.order)

        def predict(self):
            if self.exog is not None:
                return np.asarray(self.exog, float)[:, 0]
            prev = np.concatenate([[0.0], self.endog[:-1]])
            return np.nan_to_num(prev)

    class ARIMA:
        def __init__(self, endog, exog=None, order=(0, 0, 0), trend="c"):
            self.args = (endog, exog, order)

        def fit(self):
            return Result(*self.args)

    module = types.ModuleType("statsmodels.tsa.arima.model")
    module.ARIMA = ARIMA
    return module


def with_fake(names_to_modules, fn):
    saved = {name: sys.modules.get(name) for name in names_to_modules}
    sys.modules.update(names_to_modules)
    try:
        return fn()
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_arima_forecast_for_row_t_uses_return_of_candle_t():
    table = make_table()
    fold = splits_for(table)["folds"][0]

    def check():
        model = arima.Arima([0, 1], [0, 1], 500)
        pred, info = model.fit_predict(table, [], fold, {}, 0)
        assert info["order"] == [0, 0, 0]                         # наименьший AIC у заглушки
        # заглушка прогнозирует r_{t+1} = r_t: прогноз строки t обязан совпасть с ret_lag1 строки t
        np.testing.assert_allclose(pred.to_numpy(), table.loc[pred.index, "ret_lag1"].fillna(0).to_numpy())
        assert pred.index.min() >= fold.test_start and pred.index.max() < fold.test_end
        columns = run.columns_for(CFG, "arimax", "P_EN")
        pred_x, _ = arima.Arimax(500).fit_predict(table, columns, fold, {"order": [1, 0, 0]}, 0)
        # заглушка ARIMAX прогнозирует первым столбцом exog: для строки t это признак строки t
        np.testing.assert_allclose(pred_x.to_numpy(), table.loc[pred_x.index, columns[0]].to_numpy())
        # таблица кончается вместе с фолдом (последний фолд выборки): последняя строка — тоже по своему признаку
        cut = table.loc[table.index < fold.test_end]
        pred_cut, _ = arima.Arimax(500).fit_predict(cut, columns, fold, {"order": [1, 0, 0]}, 0)
        assert pred_cut.index[-1] == cut.index[-1]
        np.testing.assert_allclose(pred_cut.to_numpy(), cut.loc[pred_cut.index, columns[0]].to_numpy())
        pred_a, _ = model.fit_predict(cut, [], fold, {}, 0)
        assert pred_a.index[-1] == cut.index[-1] and pred_a.iloc[-1] == cut["ret_lag1"].iloc[-1]

    with_fake({"statsmodels": types.ModuleType("statsmodels"), "statsmodels.tsa": types.ModuleType("statsmodels.tsa"),
               "statsmodels.tsa.arima": types.ModuleType("statsmodels.tsa.arima"),
               "statsmodels.tsa.arima.model": fake_statsmodels()}, check)


def fake_xgboost():
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

    seen = {}

    class XGBRegressor:
        """Бустинг или — при num_parallel_tree — случайный лес (один шаг из многих деревьев)."""
        def __init__(self, **kw):
            self.kw, self.forest = kw, "num_parallel_tree" in kw
            seen["rf" if self.forest else "xgb"] = kw

        def fit(self, X, y, eval_set=None, verbose=False):
            if self.forest:
                assert eval_set is None and self.kw["n_estimators"] == 1
                self.model = RandomForestRegressor(n_estimators=5, max_depth=self.kw["max_depth"] or None,
                                                   min_samples_leaf=int(self.kw["min_child_weight"]),
                                                   max_features=self.kw["colsample_bynode"],
                                                   random_state=self.kw["random_state"]).fit(X, y)
                return self
            assert eval_set is not None and len(eval_set[0][0]) > 0
            self.model = GradientBoostingRegressor(n_estimators=10, max_depth=self.kw["max_depth"],
                                                   random_state=self.kw["random_state"]).fit(X, y)
            self.best_iteration = 9
            return self

        def predict(self, X):
            return self.model.predict(X)

    module = types.ModuleType("xgboost")
    module.XGBRegressor, module.seen = XGBRegressor, seen
    return module


def test_trees_tuning_and_folds_with_stub(tmp_path):
    table = make_table(hours=24 * 300)
    splits = splits_for(table)
    xgb = fake_xgboost()

    def check():
        runner = run.Runner(CFG, tmp_path, _Log())
        runner.run("ETHUSDT", table, splits, ["random_forest", "xgboost"])
        for model in ("random_forest", "xgboost"):
            folder = tmp_path / "ETHUSDT" / f"{model}__P_EN_RU"
            assert (folder / "tuning.json").exists()
            assert len(list(folder.glob("fold_*__seed*.parquet"))) == 4        # 2 фолда × 2 зерна
        rf = xgb.seen["rf"]
        assert rf["subsample"] == 0.632 and rf["learning_rate"] == 1.0 and rf["reg_lambda"] == 0.0
        assert rf["n_estimators"] == 1 and rf["num_parallel_tree"] == 10
        xg = xgb.seen["xgb"]
        assert xg["early_stopping_rounds"] == 5 and xg["tree_method"] == "hist"
        pred = run.collect(tmp_path, "ETHUSDT", "xgboost", "P", 2)
        fold = splits["folds"][0]
        train_last = table.index[table["valid"] & (table.index < fold.train_end)].max()
        assert train_last == fold.test_start - pd.Timedelta(hours=2)            # зазор: последняя строка не в обучении
        assert len(pred) == int((table["valid"] & (table.index >= fold.test_start)).sum())

    with_fake({"xgboost": xgb}, check)


def test_column_fraction_matches_sklearn_count():
    from cryptonews.models import trees
    for k in (9, 17, 25):
        for setting, expected in (("sqrt", int(np.sqrt(k))), (0.5, int(0.5 * k))):
            fraction = np.float32(trees.column_fraction(setting, k))    # XGBoost хранит долю как float32
            assert int(fraction * np.float32(k)) == expected


def test_complete_windows():
    valid = np.array([True, True, False, True, True, True, True])
    ok = lstm.complete_windows(valid, 3)
    assert ok.tolist() == [False, False, False, False, False, True, True]
    assert lstm.complete_windows(np.ones(2, bool), 3).tolist() == [False, False]


def test_smoke_config_is_light():
    cfg = run.smoke_config({**CFG, "models": {**CFG["models"], "arima": {"p": [0, 1, 2, 3], "q": [0, 1, 2, 3],
                                                                           "estimation_window_hours": 8760}}})
    assert len(run.grid_for(cfg, "random_forest")) == 1 and cfg["seeds"]["default"] == [0]
    assert cfg["models"]["arima"]["p"] == [0, 1] and cfg["models"]["lstm"]["max_epochs"] == 2
