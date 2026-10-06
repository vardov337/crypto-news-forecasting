"""Бэктест long/flat и long/short с издержками (задача 8.2, PROTOCOL.md, раздел 8).

Позиция на час t + 1 выбирается по прогнозу, сделанному на закрытии свечи t:
  long/flat  — 1, если прогноз доходности больше нуля, иначе 0;
  long/short — знак прогноза (+1 или −1; при нулевом прогнозе 0).
Доходность позиции — простая доходность свечи t + 1 (exp(лог-доходность) − 1). Издержки —
cost_bp базисных пунктов за единицу изменения позиции: вход и выход из рынка стоят по c,
разворот long/short — 2c. В начале теста капитал вне рынка, поэтому первый вход тоже платный.
Часы, которых нет в общих строках теста (остановки биржи), пропускаются.
"""
from __future__ import annotations

import numpy as np

STRATEGIES = ("long_flat", "long_short")


def positions(pred, rule: str) -> np.ndarray:
    pred = np.asarray(pred, float)
    if rule == "long_flat":
        return (pred > 0).astype(float)
    if rule == "long_short":
        return np.sign(pred).astype(float)
    if rule == "buy_and_hold":
        return np.ones(len(pred))
    raise ValueError(f"Неизвестная стратегия: {rule}")


def run_strategy(position, simple_returns, cost_bp: float, start_position: float = 0.0):
    """Чистая и валовая доходность по часам и оборот (|изменение позиции|)."""
    position, simple_returns = np.asarray(position, float), np.asarray(simple_returns, float)
    previous = np.concatenate([[start_position], position[:-1]])
    turnover = np.abs(position - previous)
    gross = position * simple_returns
    net = gross - cost_bp * 1e-4 * turnover
    return net, gross, turnover


def max_drawdown(net) -> float:
    """Максимальная просадка по стоимости портфеля (начальная стоимость 1)."""
    value = np.cumprod(1.0 + np.asarray(net, float))
    peak = np.maximum.accumulate(np.concatenate([[1.0], value]))[1:]
    return float(np.min(value / peak - 1.0)) if len(value) else float("nan")


def summarize(net, gross, turnover, position, periods_per_year: float) -> dict:
    """Метрики стратегии: итоговая и годовая доходность, годовая волатильность, Шарп, Сортино
    (нижнее отклонение по всем часам), максимальная просадка, оборот, доля часов в рынке и
    безубыточные издержки — уровень, при котором средняя чистая доходность равна нулю
    (средняя валовая доходность / средний оборот; отрицательный — убыточна и без издержек;
    у стратегии с единственным входом — NaN)."""
    net, gross, turnover = (np.asarray(v, float) for v in (net, gross, turnover))
    n = len(net)
    years = n / periods_per_year
    value = float(np.prod(1.0 + net))
    sd = float(np.std(net, ddof=1)) if n > 1 else float("nan")
    downside = float(np.sqrt(np.mean(np.minimum(net, 0.0) ** 2)))
    root = np.sqrt(periods_per_year)
    mean_turnover = float(np.mean(turnover))
    return {
        "hours": n,
        "total_return": value - 1.0,
        "annual_return": value ** (1.0 / years) - 1.0 if value > 0 else -1.0,
        "annual_volatility": sd * root,
        "sharpe": float(np.mean(net)) / sd * root if sd > 0 else float("nan"),
        "sortino": float(np.mean(net)) / downside * root if downside > 0 else float("nan"),
        "max_drawdown": max_drawdown(net),
        "turnover_per_year": float(np.sum(turnover)) / years,
        "time_in_market": float(np.mean(np.asarray(position, float) != 0)),
        # при единственном входе в начале теста (buy-and-hold) показатель смысла не имеет
        "breakeven_cost_bp": (float(np.mean(gross)) / mean_turnover * 1e4 if float(np.sum(turnover)) > 1
                              else float("nan")),
    }


def evaluate(pred, log_returns, rule: str, cost_bp: float, periods_per_year: float) -> tuple[dict, np.ndarray]:
    """Метрики стратегии и ряд её чистой доходности."""
    position = positions(pred, rule)
    net, gross, turnover = run_strategy(position, np.expm1(np.asarray(log_returns, float)), cost_bp)
    return summarize(net, gross, turnover, position, periods_per_year), net
