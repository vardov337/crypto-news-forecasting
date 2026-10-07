"""Шаг 8в (PROTOCOL.md, раздел 8): проверка прогностической связи по Грейнджеру.

Тест Грейнджера проверяет, помогают ли прошлые значения одного ряда предсказать другой сверх
его собственных прошлых значений. Это прогностическое предшествование, а не причинность.
Для каждого актива и языка новостей:
  1) стационарность рядов — расширенный тест Дики–Фуллера (с константой, число лагов
     по AIC, не больше 24);
  2) число лагов — 1, 6 и 24 ч (час, четверть суток, сутки) и выбранное по BIC из 1–24;
  3) оба направления: новости → доходность и доходность → новости;
  4) регрессия МНК с HAC-ковариацией (Newey–West), тест Вальда для лагов второго ряда.
Выборки — вся (2017–2025 гг.) и тестовый период. На выбор модели тест не влияет.

Видеокарта не нужна, 1–3 минуты.
Запуск:  python scripts/08c_granger.py
"""
import json
import warnings

import numpy as np
import pandas as pd

from cryptonews import validation
from cryptonews.cli import parse_args
from cryptonews.config import load_config
from cryptonews.evaluation import tests as stat_tests
from cryptonews.features import TARGET
from cryptonews.utils import get_logger, run_manifest, save_json

RETURN = "ret_lag1"            # доходность часа t в строке t


def lagged(series: pd.Series, lags: int, name: str) -> pd.DataFrame:
    return pd.DataFrame({f"{name}_{k}": series.shift(k) for k in range(lags)})


def design(y: pd.Series, own: pd.Series, other: dict, lags: int, rows: pd.Index) -> tuple:
    """y_t на константу, lags лагов own и lags лагов каждого ряда из other (строки t, …, t − lags + 1)."""
    parts = [y.rename("y"), lagged(own, lags, "own")]
    parts += [lagged(series, lags, name) for name, series in other.items()]
    frame = pd.concat(parts, axis=1).loc[rows].dropna()
    names = list(frame.columns[1:])
    X = np.column_stack([np.ones(len(frame)), frame[names].to_numpy(float)])
    restricted = [1 + i for i, n in enumerate(names) if not n.startswith("own_")]
    return frame["y"].to_numpy(float), X, restricted, frame.index


def bic_lags(y, own, other, rows, max_lags: int) -> int:
    """Число лагов с наименьшим BIC на общих строках (у всех вариантов одни и те же наблюдения)."""
    _, _, _, common = design(y, own, other, max_lags, rows)
    best, best_bic = 1, np.inf
    for lags in range(1, max_lags + 1):
        target, X, _, _ = design(y, own, other, lags, common)
        beta, *_ = np.linalg.lstsq(X, target, rcond=None)
        rss = float(np.sum((target - X @ beta) ** 2))
        n, k = len(target), X.shape[1]
        bic = n * np.log(rss / n) + k * np.log(n)
        if bic < best_bic:
            best, best_bic = lags, bic
    return best


def stationarity(table: pd.DataFrame, columns: list, samples: dict, max_lags: int) -> pd.DataFrame:
    from statsmodels.tsa.stattools import adfuller

    rows = []
    for column in columns:
        for sample, index in samples.items():
            series = table.loc[index, column].dropna()
            with warnings.catch_warnings():          # statsmodels 0.15 предупреждает о будущем формате результата
                warnings.simplefilter("ignore", FutureWarning)
                stat, p, used, n, *_ = adfuller(series.to_numpy(float), maxlag=max_lags, regression="c", autolag="AIC")
            rows.append({"variable": column, "sample": sample, "adf_stat": float(stat), "p_value": float(p),
                         "lags_used": int(used), "n": int(n)})
    return pd.DataFrame(rows)


def granger(table: pd.DataFrame, lang: str, samples: dict, fixed: list, max_lags: int) -> pd.DataFrame:
    news = {name: table[f"{lang}_{name}"] for name in ("sent_mean", "news_intensity")}
    returns = table[RETURN]
    equations = [("новости → доходность", "return", table[TARGET].where(table["valid"]), returns, news)]
    for name, series in news.items():                            # обратное направление — для каждого новостного ряда
        equations.append(("доходность → новости", name, series.shift(-1), series, {"return": returns}))
    rows = []
    for direction, dependent, y, own, other in equations:
        for sample, index in samples.items():
            chosen = bic_lags(y, own, other, index, max_lags)
            for lags, how in [(int(l), "фиксированное") for l in fixed] + [(chosen, "по BIC")]:
                target, X, restricted, used = design(y, own, other, lags, index)
                result = stat_tests.hac_wald(target, X, restricted)
                rows.append({"language": lang, "direction": direction, "dependent": dependent, "sample": sample,
                             "lags": lags, "lag_choice": how, **result})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args(__doc__)
    cfg = load_config(args.config)
    log = get_logger()
    data_dir, results_dir = cfg["paths"]["data_dir"], cfg["paths"]["results_dir"]
    g_cfg = cfg["evaluation"].get("granger", {})
    fixed, max_lags = list(g_cfg.get("lags", [1, 6, 24])), int(g_cfg.get("max_lags", 24))
    splits = validation.from_json(json.loads((results_dir / "splits.json").read_text(encoding="utf-8")))
    test_start, test_end = splits["folds"][0].test_start, splits["folds"][-1].test_end
    report = {}
    for symbol in cfg["data"]["prices"]["symbols"]:
        table = pd.read_parquet(data_dir / "processed" / f"features_{symbol}.parquet")
        full = table.index[table["valid"]]
        samples = {"full": full, "test": full[(full >= test_start) & (full < test_end)]}
        languages = cfg["features"]["news"]["languages"]
        adf = stationarity(table, [RETURN] + [f"{lang}_{name}" for lang in languages
                                              for name in ("sent_mean", "news_intensity")], samples, max_lags)
        tests = pd.concat([granger(table, lang, samples, fixed, max_lags) for lang in languages], ignore_index=True)
        adf.to_csv(results_dir / "tables" / f"stationarity_{symbol}.csv", index=False, encoding="utf-8")
        tests.to_csv(results_dir / "tables" / f"granger_{symbol}.csv", index=False, encoding="utf-8")
        log.info("[%s] стационарность (расширенный тест Дики–Фуллера; p < 0,05 — ряд стационарен):\n%s", symbol,
                 adf.round({"adf_stat": 2, "p_value": 4}).to_string(index=False))
        view = tests.assign(lags=tests["lags"].astype(str) + np.where(tests["lag_choice"] == "по BIC", " (BIC)", ""))
        log.info("[%s] тест Грейнджера (HAC; p — для лагов второго ряда):\n%s", symbol,
                 view[["language", "direction", "dependent", "sample", "lags", "df", "wald", "p_value"]]
                 .round({"wald": 2, "p_value": 3}).to_string(index=False))
        report[symbol] = {"significant_at_5pct": int((tests["p_value"] < 0.05).sum()), "tests": len(tests),
                          "nonstationary_at_5pct": adf.loc[adf["p_value"] >= 0.05, "variable"].tolist()}
        log.info("[%s] значимых при 5%%: %d из %d тестов", symbol, report[symbol]["significant_at_5pct"], len(tests))
    save_json({"manifest": run_manifest(cfg), "fixed_lags": fixed, "max_lags": max_lags, "assets": report},
              results_dir / "metrics" / "granger.json")
    log.info("Готово")


if __name__ == "__main__":
    main()
