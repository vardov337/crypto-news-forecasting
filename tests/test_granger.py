"""Шаг 8в: выбор числа лагов по BIC и направления теста Грейнджера на искусственных рядах."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("granger_step", ROOT / "scripts" / "08c_granger.py")
step = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step)


def make_table(n=6000, effect=0.0, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    news = rng.normal(size=n)
    ret = rng.normal(0, 0.01, n)
    ret[3:] += effect * news[:-3]                      # новость часа t − 3 сдвигает доходность часа t
    table = pd.DataFrame({"ret_lag1": ret, "en_sent_mean": news, "en_news_intensity": rng.normal(size=n),
                          "valid": True}, index=index)
    table["target"] = table["ret_lag1"].shift(-1)
    table["valid"] = table["target"].notna()
    return table


def test_bic_picks_the_true_lag_and_direction():
    table = make_table(effect=0.004)
    rows = table.index[table["valid"]]
    y, own = table["target"], table["ret_lag1"]
    other = {"sent_mean": table["en_sent_mean"]}
    assert step.bic_lags(y, own, other, rows, 8) == 3          # нужен лаг t − 2 в строке t, то есть три лага
    tests = step.granger(table, "en", {"full": rows}, [1, 6], 8)
    forward = tests[(tests["direction"] == "новости → доходность") & (tests["lags"] == 6)]
    assert forward["p_value"].iloc[0] < 1e-6
    backward = tests[(tests["direction"] == "доходность → новости") & (tests["lag_choice"] == "фиксированное")]
    assert (backward["p_value"] > 1e-3).all()                   # обратной связи в данных нет


def test_no_effect_no_rejection():
    table = make_table(effect=0.0, seed=1)
    rows = table.index[table["valid"]]
    tests = step.granger(table, "en", {"full": rows}, [1], 4)
    assert (tests["p_value"] > 1e-3).all() and set(tests["lag_choice"]) == {"фиксированное", "по BIC"}
