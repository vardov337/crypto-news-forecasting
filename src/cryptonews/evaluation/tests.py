"""Статистические тесты (задачи 8.1–8.4, PROTOCOL.md, раздел 8).

- Диболда–Мариано с HAC-оценкой дисперсии (Newey–West, ядро Бартлетта), односторонний:
  H0 — модель не точнее сравниваемой по квадратичной ошибке.
- Песарана–Тиммерманна для точности направления, односторонний.
- Поправка Холма на множественные сравнения.
- Стационарный бутстреп (Politis, Romano, 1994) для разницы коэффициентов Шарпа.
- Model Confidence Set (Hansen, Lunde, Nason, 2011) — реализация пакета arch.
- Тест Грейнджера: регрессия доходности следующего часа на лаги доходности и новостных
  признаков, проверка Вальда с HAC-ковариацией (описательный, на выбор модели не влияет).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from cryptonews.evaluation.metrics import direction_mask


def newey_west_lags(n: int) -> int:
    """Правило Newey, West (1994): floor(4 · (n / 100)^(2/9))."""
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def long_run_variance(x, lags: int) -> float:
    """Долгосрочная дисперсия ряда с весами Бартлетта."""
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    value = float(x @ x) / n
    for k in range(1, int(lags) + 1):
        value += 2.0 * (1.0 - k / (lags + 1.0)) * float(x[k:] @ x[:-k]) / n
    return value


def diebold_mariano(loss_base, loss_model, lags: int | None = None) -> dict:
    """Тест Диболда–Мариано. d = потери базовой модели − потери проверяемой;
    H0: E[d] ≤ 0 (проверяемая модель не точнее), H1: E[d] > 0. p-значение одностороннее."""
    d = np.asarray(loss_base, float) - np.asarray(loss_model, float)
    n = len(d)
    lags = newey_west_lags(n) if lags is None else int(lags)
    variance = long_run_variance(d, lags) if n > 1 else float("nan")
    if not variance > 0:
        return {"dm_stat": float("nan"), "p_value": float("nan"), "mean_loss_diff": float(d.mean()) if n else float("nan"),
                "hac_lags": lags, "n": n}
    stat = float(d.mean() / np.sqrt(variance / n))
    return {"dm_stat": stat, "p_value": float(stats.norm.sf(stat)), "mean_loss_diff": float(d.mean()),
            "hac_lags": lags, "n": n}


def pesaran_timmermann(y, f) -> dict:
    """Тест Песарана–Тиммерманна (1992): угадывает ли прогноз знак лучше случайного при тех же
    долях роста в факте и в прогнозе. H1: угадывает лучше; p-значение одностороннее.
    Если прогноз всегда одного знака, статистика не определена (NaN)."""
    y, f = np.asarray(y, float), np.asarray(f, float)
    mask = direction_mask(y, f)
    y, f = y[mask], f[mask]
    n = len(y)
    if n == 0:
        return {"pt_stat": float("nan"), "p_value": float("nan"), "hit_rate": float("nan"), "n": 0}
    up_y, up_f = float(np.mean(y > 0)), float(np.mean(f > 0))
    hit = float(np.mean((y > 0) == (f > 0)))
    expected = up_y * up_f + (1 - up_y) * (1 - up_f)
    var_hit = expected * (1 - expected) / n
    var_expected = ((2 * up_y - 1) ** 2 * up_f * (1 - up_f) / n + (2 * up_f - 1) ** 2 * up_y * (1 - up_y) / n
                    + 4 * up_y * up_f * (1 - up_y) * (1 - up_f) / n ** 2)
    denominator = var_hit - var_expected
    if not denominator > 0:
        return {"pt_stat": float("nan"), "p_value": float("nan"), "hit_rate": hit, "n": n}
    stat = float((hit - expected) / np.sqrt(denominator))
    return {"pt_stat": stat, "p_value": float(stats.norm.sf(stat)), "hit_rate": hit, "n": n}


def holm(p_values) -> np.ndarray:
    """Поправка Холма; пропуски (NaN) в поправке не участвуют и остаются NaN."""
    p = np.asarray(p_values, float)
    out = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    m = int(ok.sum())
    if m == 0:
        return out
    order = np.argsort(p[ok], kind="stable")
    adjusted = np.minimum(np.maximum.accumulate((m - np.arange(m)) * p[ok][order]), 1.0)
    result = np.empty(m)
    result[order] = adjusted
    out[ok] = result
    return out


def stationary_bootstrap_indices(n: int, reps: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """Индексы стационарного бутстрепа: блоки случайной длины (геометрическое распределение со
    средним mean_block), ряд замыкается в кольцо. Форма — (reps, n)."""
    restart = rng.random((reps, n)) < 1.0 / float(mean_block)
    jumps = rng.integers(0, n, size=(reps, n))
    index = np.empty((reps, n), dtype=np.int64)
    index[:, 0] = jumps[:, 0]
    for t in range(1, n):
        index[:, t] = np.where(restart[:, t], jumps[:, t], (index[:, t - 1] + 1) % n)
    return index


def sharpe_ratio(returns, periods_per_year: float, axis: int = -1):
    """Коэффициент Шарпа при нулевой безрисковой ставке: среднее / ст. откл. · √(часов в году)."""
    returns = np.asarray(returns, float)
    mean = returns.mean(axis=axis)
    sd = returns.std(axis=axis, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, mean / sd * np.sqrt(periods_per_year), np.nan)


def sharpe_difference_ci(strategies: dict[str, np.ndarray], benchmark: np.ndarray, *, reps: int,
                         mean_block: float, seed: int, periods_per_year: float, level: float = 0.95,
                         chunk: int = 250) -> pd.DataFrame:
    """Разница коэффициентов Шарпа стратегии и бенчмарка: точечная оценка и перцентильный
    интервал стационарного бутстрепа. Все стратегии пересэмплируются одними и теми же
    индексами вместе с бенчмарком (парный бутстреп)."""
    benchmark = np.asarray(benchmark, float)
    n = len(benchmark)
    rng = np.random.default_rng(seed)
    draws = {name: [] for name in strategies}
    done = 0
    while done < reps:
        size = min(chunk, reps - done)
        index = stationary_bootstrap_indices(n, size, mean_block, rng)
        base = sharpe_ratio(benchmark[index], periods_per_year, axis=1)
        for name, series in strategies.items():
            draws[name].append(sharpe_ratio(np.asarray(series, float)[index], periods_per_year, axis=1) - base)
        done += size
    alpha = (1.0 - level) / 2.0
    base_point = float(sharpe_ratio(benchmark, periods_per_year))
    rows = []
    for name, series in strategies.items():
        values = np.concatenate(draws[name])
        values = values[~np.isnan(values)]
        point = float(sharpe_ratio(series, periods_per_year)) - base_point
        rows.append({"name": name, "sharpe_diff": point,
                     "ci_low": float(np.quantile(values, alpha)) if len(values) else float("nan"),
                     "ci_high": float(np.quantile(values, 1 - alpha)) if len(values) else float("nan"),
                     "share_not_above_zero": float(np.mean(values <= 0)) if len(values) else float("nan"),
                     "valid_reps": int(len(values))})
    return pd.DataFrame(rows)


def model_confidence_set(losses: pd.DataFrame, *, size: float, reps: int, block_size: int, seed: int) -> pd.DataFrame:
    """Model Confidence Set по таблице потерь (строки — часы, столбцы — модели), пакет arch:
    статистика R, стационарный бутстреп. Возвращает p-значение MCS и признак попадания в набор."""
    from arch.bootstrap import MCS

    kwargs = {"size": size, "reps": reps, "block_size": block_size, "method": "R", "bootstrap": "stationary"}
    try:
        mcs = MCS(losses, seed=seed, **kwargs)
    except TypeError:                       # старые версии arch не принимают seed
        np.random.seed(seed)
        mcs = MCS(losses, **kwargs)
    mcs.compute()
    p = mcs.pvalues.iloc[:, 0]
    out = pd.DataFrame({"p_mcs": p.astype(float)})
    out.index = out.index.astype(str)
    out["in_mcs"] = out.index.isin([str(name) for name in mcs.included])
    return out.reindex([str(c) for c in losses.columns])


def hac_wald(y, X, restricted: list[int], lags: int | None = None) -> dict:
    """МНК y на X с HAC-ковариацией (Newey–West, Бартлетт) и тест Вальда:
    коэффициенты при столбцах restricted равны нулю. χ² с len(restricted) степенями свободы."""
    y, X = np.asarray(y, float), np.asarray(X, float)
    n = len(y)
    lags = newey_west_lags(n) if lags is None else int(lags)
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    scores = X * (y - X @ beta)[:, None]
    meat = scores.T @ scores
    for k in range(1, lags + 1):
        gamma = scores[k:].T @ scores[:-k]
        meat += (1.0 - k / (lags + 1.0)) * (gamma + gamma.T)
    cov = xtx_inv @ meat @ xtx_inv
    b = beta[restricted]
    wald = float(b @ np.linalg.solve(cov[np.ix_(restricted, restricted)], b))
    return {"wald": wald, "df": len(restricted), "p_value": float(stats.chi2.sf(wald, len(restricted))),
            "n": n, "hac_lags": lags}


def granger_news(table: pd.DataFrame, target: str, own: str, news: list[str], lags: int,
                 rows: pd.Index | None = None) -> dict:
    """Предсказывают ли лаги новостных признаков доходность следующего часа сверх её собственных лагов.

    y_t = доходность часа t + 1 (колонка target строки t); регрессоры — константа, own (доходность
    часа t) и news в строках t, t − 1, …, t − lags + 1. Строки с пропусками отбрасываются."""
    parts = {"y": table[target]}
    for k in range(lags):
        parts[f"own_{k}"] = table[own].shift(k)
    for column in news:
        for k in range(lags):
            parts[f"{column}_{k}"] = table[column].shift(k)
    frame = pd.DataFrame(parts)
    if "valid" in table:
        frame = frame[table["valid"]]
    if rows is not None:
        frame = frame.loc[frame.index.intersection(rows)]
    frame = frame.dropna()
    names = list(frame.columns[1:])
    X = np.column_stack([np.ones(len(frame)), frame[names].to_numpy(float)])
    restricted = [1 + i for i, name in enumerate(names) if not name.startswith("own_")]
    result = hac_wald(frame["y"].to_numpy(float), X, restricted)
    result.update({"lags": lags, "first_row": str(frame.index.min()), "last_row": str(frame.index.max())})
    return result
