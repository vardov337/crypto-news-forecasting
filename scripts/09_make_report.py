"""Шаг 9 (задача 8.5): таблицы и рисунки для статьи.

Собирает в results/report/ всё, что идёт в статью, прямо из результатов шагов 5б–8в —
вручную ни одно число не переносится (PROTOCOL.md, раздел 11):
  tables.xlsx     — таблицы статьи (Т1–Т5) и приложения (П1–П12); лист «Описание» — что где;
  рисунки в порядке статьи:
  fig1_scheme     — схема подхода;
  fig2_coverage   — доля часов с новостями по годам;
  fig3_accuracy   — изменение RMSE относительно нулевого прогноза, %;
  fig4_strategies — стоимость портфеля: buy-and-hold и лучшая по Шарпу стратегия без издержек и с ними;
  fig5_breakeven  — безубыточные издержки и число смен позиции в год;
  fig6_shift      — тест сдвига новостей;
  requirements-lock.txt — точные версии пакетов окружения, в котором получены результаты (шаг 0).
Рисунки — в PNG (300 точек на дюйм) и SVG. Всё вместе — ещё и в архиве results/report.zip.
Перед записью папка report очищается от файлов прошлых запусков.

Видеокарта не нужна, около минуты. Если шаг 8 запускался до появления общей таблицы
прогнозов, она сначала собирается из файлов шага 7 — это ещё несколько минут.
Запуск:  python scripts/09_make_report.py   (после шагов 8, 8б и 8в)
"""
import json
import re
import shutil
import zipfile

import numpy as np
import pandas as pd

from cryptonews import align, report, validation
from cryptonews.cli import parse_args
from cryptonews.config import load_config
from cryptonews.data import news as news_rules
from cryptonews.evaluation import backtest
from cryptonews.evaluation.predictions import (PRETTY, all_keys, collect_all, combine, combined_path, is_baseline,
                                                label)
from cryptonews.features import TARGET
from cryptonews.models.lstm import complete_windows
from cryptonews.utils import get_logger, run_manifest, save_json

REPORT_SUFFIXES = (".xlsx", ".png", ".svg", ".txt")

ASSET = {"BTCUSDT": "Биткоин (BTCUSDT)", "ETHUSDT": "Эфир (ETHUSDT)"}
LANGUAGE = {"en": "англоязычные", "ru": "русскоязычные"}
SET_NAMES = {"P_EN": "P_EN — цены и англ. новости", "P_RU": "P_RU — цены и рус. новости",
             "P_EN_RU": "P_EN_RU — цены и новости на обоих языках"}


def table_path(results_dir, name: str):
    path = results_dir / "tables" / name
    if not path.exists():
        raise SystemExit(f"Нет {path} — сначала выполните шаги 8, 8б и 8в")
    return path


def yes_no(value) -> str:
    return "да" if bool(value) else "нет"


# ---------- таблицы ----------

def sample_table(features_report: dict, splits: dict, symbols: list) -> pd.DataFrame:
    period = features_report["period"]
    start, end = pd.Timestamp(period["start_inclusive"]), pd.Timestamp(period["end_exclusive"])
    test_start, test_end = splits["folds"][0].test_start, splits["folds"][-1].test_end
    inputs, assets = features_report["inputs"], features_report["assets"]
    rows = [
        ("Выборка (UTC)", [f"{start:%d.%m.%Y} – {end - pd.Timedelta(hours=1):%d.%m.%Y %H:%M}"] * len(symbols)),
        ("Тестовый период (UTC)", [f"{test_start:%d.%m.%Y} – {test_end - pd.Timedelta(hours=1):%d.%m.%Y %H:%M}"] * len(symbols)),
        ("Часов в выборке", [assets[s]["rows"] for s in symbols]),
        ("Пригодных часов (есть все признаки и цель)", [assets[s]["valid_rows"] for s in symbols]),
        ("Часов без свечей Binance", [assets[s]["missing_close_hours"] for s in symbols]),
    ]
    for lang in ("en", "ru"):
        rows += [
            (f"Новостей в признаках: {LANGUAGE[lang]}", [assets[s]["news_used"][lang] for s in symbols]),
            (f"Часов хотя бы с одной новостью, %: {LANGUAGE[lang]}",
             [round(100 * assets[s]["hours_with_news_share"][lang], 1) for s in symbols]),
            (f"Исключено записей пакетных загрузок: {LANGUAGE[lang]}",
             [inputs[lang].get("batch_upload_records_excluded", 0)] * len(symbols)),
        ]
        for source, count in inputs[lang].get("excluded_sources", {}).items():
            rows.append((f"Исключено записей источника {source} (ненадёжное время)", [count] * len(symbols)))
        model = features_report["sentiment_models"][lang]
        rows.append((f"Модель тональности: {LANGUAGE[lang]}", [f"{model['model']} ({model['revision'][:8]})"] * len(symbols)))
    return pd.DataFrame([[name, *values] for name, values in rows], columns=["Показатель", *symbols])


def sentiment_tables(results_dir) -> tuple[pd.DataFrame, pd.DataFrame]:
    choice = json.loads((results_dir / "metrics" / "sentiment_choice.json").read_text(encoding="utf-8"))
    models, agreement = [], []
    for lang in ("en", "ru"):
        table = pd.read_csv(table_path(results_dir, f"sentiment_validation_{lang}.csv"))
        for row in table.itertuples():
            models.append({"Язык": LANGUAGE[lang], "Модель": row.model, "Выбрана": yes_no(row.chosen),
                           "Accuracy": round(row.accuracy, 3), "Macro-F1": round(row.macro_f1, 3),
                           "95% ДИ: от": round(row.macro_f1_ci_low, 3), "до": round(row.macro_f1_ci_high, 3),
                           "F1 негатив": round(row.f1_negative, 3), "F1 нейтрал": round(row.f1_neutral, 3),
                           "F1 позитив": round(row.f1_positive, 3)})
        entry = choice[lang]
        agree = entry.get("agreement", {})
        agreement.append({"Язык": LANGUAGE[lang], "Заголовков у первого разметчика": entry.get("n_labels"),
                          "Общих заголовков двух разметчиков": agree.get("n"),
                          "Совпадений, %": round(100 * agree["percent_agreement"], 1) if agree else None,
                          "Каппа Коэна": round(agree["cohen_kappa"], 3) if agree else None})
    return pd.DataFrame(models), pd.DataFrame(agreement)


def accuracy_table(frames: dict) -> pd.DataFrame:
    rows = []
    for symbol, table in frames.items():
        for r in table.itertuples():
            rows.append({"Актив": symbol, "Модель": r.label, "RMSE / нулевой прогноз": round(r.rmse_ratio_zero, 4),
                         "R²_OOS, %": round(100 * r.r2_oos, 2),
                         "Направление угадано, %": round(100 * r.dir_accuracy, 1) if pd.notna(r.dir_accuracy) else None,
                         "p ДМ (Холм)": round(r.dm_vs_zero_p_holm, 3) if pd.notna(r.dm_vs_zero_p_holm) else None,
                         "p ПТ (Холм)": round(r.pt_p_holm, 3) if pd.notna(r.pt_p_holm) else None,
                         "p MCS": round(r.p_mcs, 3), "В MCS (α = 0,10)": yes_no(r.in_mcs)})
    return pd.DataFrame(rows)


def questions_table(frames: dict) -> pd.DataFrame:
    rows = []
    for symbol, table in frames.items():
        for r in table.itertuples():
            rows.append({"Актив": symbol, "Вопрос": r.question, "Сравнение": r.comparison,
                         "Изменение MSE, %": round(r.mse_change_pct, 3),
                         "Статистика ДМ": round(r.dm_stat, 2) if pd.notna(r.dm_stat) else None,
                         "p": round(r.p_value, 3) if pd.notna(r.p_value) else None,
                         "p Холма": round(r.p_holm, 3) if pd.notna(r.p_holm) else None})
    return pd.DataFrame(rows)


def strategy_table(frames: dict, strategy: str, cost: float) -> pd.DataFrame:
    rows = []
    for symbol, table in frames.items():
        part = table[(table["strategy"] == strategy) & (table["cost_bp"] == cost)]
        for r in part.itertuples():
            rows.append({"Актив": symbol, "Стратегия": r.label, "Шарп": round(r.sharpe, 2) if pd.notna(r.sharpe) else None,
                         "Разница с buy-and-hold": round(r.sharpe_diff, 2) if pd.notna(r.sharpe_diff) else None,
                         "95% ДИ: от": round(r.ci_low, 2) if pd.notna(r.ci_low) else None,
                         "до": round(r.ci_high, 2) if pd.notna(r.ci_high) else None,
                         "Годовая доходность, %": round(100 * r.annual_return, 1),
                         "Макс. просадка, %": round(100 * r.max_drawdown, 1),
                         "Смен позиции в год": round(r.turnover_per_year, 1 if r.turnover_per_year < 10 else 0),
                         "Безубыточные издержки, б. п.": (round(r.breakeven_cost_bp, 1)
                                                          if pd.notna(r.breakeven_cost_bp) and r.turnover_per_year > 1
                                                          else None)})
    return pd.DataFrame(rows)


def costs_table(frames: dict, costs: list, primary: float) -> pd.DataFrame:
    rows = []
    for symbol, table in frames.items():
        flat = table[table["strategy"] == "long_flat"].pivot_table(index="label", columns="cost_bp", values="sharpe",
                                                                     sort=False)
        gross = table[(table["strategy"] == "long_flat") & (table["cost_bp"] == 0)].set_index("label")
        short = table[(table["strategy"] == "long_short") & (table["cost_bp"] == primary)]
        for name in flat.index:
            row = {"Актив": symbol, "Стратегия": name}
            for cost in costs:
                value = flat.loc[name, cost] if cost in flat.columns else np.nan
                row[f"Шарп long/flat, {cost:g} б. п."] = round(value, 2) if pd.notna(value) else None
            for column, title in (("sharpe_diff", "Разница с buy-and-hold, 0 б. п."), ("ci_low", "95% ДИ: от"),
                                  ("ci_high", "до")):
                value = gross.loc[name, column] if name in gross.index and column in gross.columns else np.nan
                row[title] = round(float(value), 2) if pd.notna(value) else None
            match = short[short["label"] == name]
            row[f"Шарп long/short, {primary:g} б. п."] = round(match["sharpe"].iloc[0], 2) if len(match) and pd.notna(match["sharpe"].iloc[0]) else None
            rows.append(row)
    return pd.DataFrame(rows)


def selection_table(frames: dict) -> pd.DataFrame:
    rows = []
    for symbol, table in frames.items():
        for r in table.itertuples():
            rows.append({"Актив": symbol, "Модель": r.label,
                         "Значимо точнее нуля или угадывает направление (Холм, 5%)": yes_no(r.accurate),
                         "Шарп": round(r.sharpe, 2) if pd.notna(r.sharpe) else None,
                         "Разница с buy-and-hold": round(r.sharpe_diff, 2) if pd.notna(r.sharpe_diff) else None,
                         "95% ДИ: от": round(r.ci_low, 2) if pd.notna(r.ci_low) else None,
                         "до": round(r.ci_high, 2) if pd.notna(r.ci_high) else None,
                         "Нижняя граница ДИ > 0": yes_no(r.beats_hold), "Проходит критерий": yes_no(r.passes),
                         "Названа лучшей": yes_no(r.selected)})
    return pd.DataFrame(rows)


def folds_table(frames: dict) -> pd.DataFrame:
    parts = []
    for symbol, table in frames.items():
        table = table.copy()
        table["Тест с"] = pd.to_datetime(table["test_start"]).dt.strftime("%m.%Y")
        pivot = table.pivot_table(index="label", columns="Тест с", values="rmse_ratio_zero", sort=False).round(4)
        pivot = pivot.reset_index().rename(columns={"label": "Модель"})
        pivot.insert(0, "Актив", symbol)
        parts.append(pivot)
    return pd.concat(parts, ignore_index=True)


DEPENDENT = {"return": "доходность", "sent_mean": "средняя тональность", "news_intensity": "интенсивность"}


def granger_table(frames: dict) -> pd.DataFrame:
    rows = []
    for symbol, table in frames.items():
        for r in table.itertuples():
            rows.append({"Актив": symbol, "Язык": LANGUAGE[r.language], "Направление": r.direction,
                         "Зависимая переменная": DEPENDENT.get(r.dependent, r.dependent),
                         "Выборка": "вся" if r.sample == "full" else "тест",
                         "Лагов, ч": f"{r.lags} (BIC)" if r.lag_choice == "по BIC" else str(r.lags),
                         "Степеней свободы": r.df, "Статистика Вальда": round(r.wald, 2), "p": round(r.p_value, 3),
                         "Наблюдений": r.n})
    return pd.DataFrame(rows)


def stationarity_table(frames: dict) -> pd.DataFrame:
    rows = []
    for symbol, table in frames.items():
        for r in table.itertuples():
            rows.append({"Актив": symbol, "Ряд": r.variable, "Выборка": "вся" if r.sample == "full" else "тест",
                         "Статистика ADF": round(r.adf_stat, 2), "p": round(r.p_value, 4), "Лагов (AIC)": r.lags_used,
                         "Наблюдений": r.n})
    return pd.DataFrame(rows)


def leakage_tables(shift: dict, placebo: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = []
    for symbol, table in shift.items():
        pivot = table.pivot_table(index="feature_set", columns="shift_hours", values="mse_change_vs_price_pct").round(3)
        pivot.columns = [f"сдвиг {int(c)} ч" for c in pivot.columns]
        pivot = pivot.reset_index().rename(columns={"feature_set": "Набор признаков"})
        pivot.insert(0, "Актив", symbol)
        parts.append(pivot)
    rows = [{"Актив": symbol, "Перестановок цели": len(t), "Точность направления, средняя, %": round(100 * t["dir_accuracy"].mean(), 2),
             "мин, %": round(100 * t["dir_accuracy"].min(), 2), "макс, %": round(100 * t["dir_accuracy"].max(), 2),
             "Угадывание по долям растущих часов, %": round(100 * t["chance_accuracy"].mean(), 2)}
            for symbol, t in placebo.items()]
    return pd.concat(parts, ignore_index=True), pd.DataFrame(rows)


def hyperparameter_tables(cfg: dict, results_dir, symbols: list, n_folds: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    tuned, orders = [], []
    for symbol in symbols:
        base = results_dir / "predictions" / symbol
        for key in all_keys(cfg):
            path = base / key.replace(":", "__") / "tuning.json"
            if not path.exists():
                continue
            chosen = json.loads(path.read_text(encoding="utf-8"))["chosen"]
            tuned.append({"Актив": symbol, "Модель": label(key),
                          "Выбранные гиперпараметры": ", ".join(f"{k} = {v}" for k, v in chosen.items())})
        for i in range(1, n_folds + 1):
            path = base / "arima__P" / f"fold_{i}__seed0.json"
            if path.exists():
                order = json.loads(path.read_text(encoding="utf-8"))["info"]["order"]
                orders.append({"Актив": symbol, "Фолд": i, "Порядок ARIMA (p, d, q)": f"({order[0]}, {order[1]}, {order[2]})"})
    return pd.DataFrame(tuned), pd.DataFrame(orders)


def data_flow_tables(cfg: dict, features_report: dict, splits: dict, symbols: list, combined: dict,
                     log) -> list[tuple[str, pd.DataFrame]]:
    """Схема данных для статьи: число записей после каждого этапа очистки и отбора новостей,
    привязка к монетам, часы с новостями (вся выборка и тест), объём обучения и теста по фолдам.

    Новости пересчитываются теми же правилами, что и в шаге 6 (исключённые источники, пакетные
    загрузки, период выборки, отбор новостей актива), — только чтение, файлы не меняются."""
    data_dir, results_dir = cfg["paths"]["data_dir"], cfg["paths"]["results_dir"]
    period = features_report["period"]
    start, end = pd.Timestamp(period["start_inclusive"]), pd.Timestamp(period["end_exclusive"])
    rule = cfg["features"]["news"].get("asset_rule", "coin_or_market_wide")
    b_cfg = cfg["data"].get("batch_uploads", {})
    flow, coins_rows = [], []
    for lang in ("en", "ru"):
        steps = []
        stages_path = results_dir / "tables" / f"cleaning_news_{lang}.csv"
        if stages_path.exists():
            stages = pd.read_csv(stages_path)
            steps += [(str(stage), int(count)) for stage, count in zip(stages.iloc[:, 0], stages.iloc[:, 1])]
        news_path = data_dir / "interim" / f"news_{lang}.parquet"
        if not news_path.exists():
            log.warning("Нет %s — этапы отбора новостей (%s) в таблицу П8 не попадут", news_path, lang)
            flow += [{"Язык": LANGUAGE[lang], "Этап": s, "Записей": n} for s, n in steps]
            continue
        news = pd.read_parquet(news_path)
        news["published_utc"] = pd.to_datetime(news["published_utc"], utc=True)
        if steps and steps[-1][1] != len(news):
            log.warning("[%s] в таблице очистки %d записей, а в %s — %d", lang, steps[-1][1], news_path.name, len(news))
        news, dropped = news_rules.drop_sources(news, cfg["data"].get("feature_excluded_sources"))
        if dropped.sum():
            steps.append((f"Без источников с ненадёжным временем публикации ({', '.join(dropped[dropped > 0].index)})",
                          len(news)))
        batch, _ = news_rules.batch_uploads(news, int(b_cfg.get("min_records", 10)),
                                            float(b_cfg.get("max_gap_seconds", 60)))
        news = news[~batch.to_numpy()]
        steps.append(("Без пакетных загрузок", len(news)))
        inside = news[(news["published_utc"] >= start) & (news["published_utc"] < end)]
        steps.append((f"В периоде выборки ({start:%d.%m.%Y} – {end - pd.Timedelta(hours=1):%d.%m.%Y})", len(inside)))
        rows = [{"Язык": LANGUAGE[lang], "Этап": s, "Записей": n,
                 "Убрано на этапе": (steps[i - 1][1] - n) if i else None} for i, (s, n) in enumerate(steps)]
        for symbol in symbols:
            rows.append({"Язык": LANGUAGE[lang], "Этап": f"Идут в признаки {symbol} (упоминают монету или общерыночные)",
                         "Записей": len(align.news_for_asset(inside, symbol, rule)), "Убрано на этапе": None})
        flow += rows
        btc, eth = inside["mentions_btc"].astype(bool), inside["mentions_eth"].astype(bool)
        coins_rows.append({"Язык": LANGUAGE[lang], "Только биткоин": int((btc & ~eth).sum()),
                           "Только эфир": int((eth & ~btc).sum()), "Обе монеты": int((btc & eth).sum()),
                           "Ни одной (общерыночные)": int((~btc & ~eth).sum()), "Всего": len(inside)})

    test_start, test_end = splits["folds"][0].test_start, splits["folds"][-1].test_end
    hours_rows, valid_index = [], {}
    for symbol in symbols:
        path = data_dir / "processed" / f"features_{symbol}.parquet"
        if not path.exists():
            log.warning("Нет %s — часовые ряды и фолды в таблицу П8 не попадут", path)
            continue
        table = pd.read_parquet(path, columns=["valid", "en_news_count", "ru_news_count"])
        valid = table["valid"].astype(bool)
        valid_index[symbol] = table.index[valid]
        test = (table.index >= test_start) & (table.index < test_end)
        row = {"Актив": symbol, "Часов в выборке": len(table), "Пригодных": int(valid.sum()),
               "Часов в тесте": int(test.sum()), "Пригодных в тесте": int((valid & test).sum()),
               "Строк теста, общих для всех моделей": len(combined[symbol]) if symbol in combined else None}
        for lang in ("en", "ru"):
            has = table[f"{lang}_news_count"] > 0
            row[f"Часов с новостями, %: {LANGUAGE[lang]}, вся выборка"] = round(100 * float(has.mean()), 1)
            row[f"Часов с новостями, %: {LANGUAGE[lang]}, тест"] = round(100 * float(has[test].mean()), 1)
            row[f"Новостей в час, среднее: {LANGUAGE[lang]}, тест"] = round(float(table.loc[test, f"{lang}_news_count"].mean()), 2)
        hours_rows.append(row)

    fold_rows = []
    windows = [("настройка", splits["tuning"])] + [(str(i), f) for i, f in enumerate(splits["folds"], 1)]
    for name, window in windows:
        row = {"Окно": name, "Обучение: строки до (UTC)": f"{window.train_end:%d.%m.%Y %H:%M}",
               "Проверка: с": f"{window.test_start:%d.%m.%Y}",
               "по (UTC)": f"{window.test_end - pd.Timedelta(hours=1):%d.%m.%Y %H:%M}"}
        for symbol, index in valid_index.items():
            row[f"Строк обучения {symbol}"] = int((index < window.train_end).sum())
            row[f"Строк проверки {symbol}"] = int(((index >= window.test_start) & (index < window.test_end)).sum())
        fold_rows.append(row)

    blocks = [("Новости: число записей после каждого этапа очистки и отбора", pd.DataFrame(flow)),
              ("Привязка новостей к монетам по заголовку (в периоде выборки, после исключений)", pd.DataFrame(coins_rows)),
              ("Часовые ряды: пригодные часы и часы с новостями", pd.DataFrame(hours_rows)),
              ("Окна обучения и проверки (расширяющееся окно, зазор 1 ч)", pd.DataFrame(fold_rows))]
    prices_path = results_dir / "metrics" / "prices_coverage.csv"     # пишет шаг 1
    if prices_path.exists():
        blocks.append(("Цены Binance: загруженные свечи (весь период загрузки)", pd.read_csv(prices_path)))
    return [(title, frame) for title, frame in blocks if len(frame)]


def sources_table(results_dir) -> pd.DataFrame:
    parts = []
    for lang in ("en", "ru"):
        path = results_dir / "tables" / f"news_{lang}_sources_span.csv"
        if path.exists():
            table = pd.read_csv(path).iloc[:, :4]
            table.insert(0, "Язык", LANGUAGE[lang])
            parts.append(table)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame({"Нет данных": []})


def returns_table(cfg: dict, splits: dict, symbols: list, log) -> pd.DataFrame:
    """Описательные статистики целевой переменной — часовой лог-доходности — на пригодных часах:
    вся выборка и тестовый период. Автокорреляция — на полной сетке часов (непригодные часы — пропуски)."""
    data_dir = cfg["paths"]["data_dir"]
    periods = float(cfg["evaluation"]["strategy"]["annualization_hours"])
    test_start, test_end = splits["folds"][0].test_start, splits["folds"][-1].test_end
    rows = []
    for symbol in symbols:
        path = data_dir / "processed" / f"features_{symbol}.parquet"
        if not path.exists():
            log.warning("Нет %s — описательные статистики доходности (П10) не посчитаны", path)
            continue
        table = pd.read_parquet(path, columns=["valid", TARGET])
        y = table[TARGET].where(table["valid"].astype(bool))
        for sample, mask in (("вся", np.ones(len(y), dtype=bool)),
                             ("тест", (table.index >= test_start) & (table.index < test_end))):
            series = y[mask]
            values = series.dropna()
            rows.append({"Актив": symbol, "Выборка": sample, "Часов": len(values),
                         "Среднее, %": round(100 * values.mean(), 4), "Ст. откл., %": round(100 * values.std(), 3),
                         "Годовая волатильность, %": round(100 * values.std() * np.sqrt(periods), 1),
                         "Минимум, %": round(100 * values.min(), 2), "Максимум, %": round(100 * values.max(), 2),
                         "Асимметрия": round(float(values.skew()), 2), "Эксцесс": round(float(values.kurt()), 1),
                         "Доля часов роста, %": round(100 * float((values > 0).mean()), 2),
                         "Доля часов без изменения, %": round(100 * float((values == 0).mean()), 2),
                         "Автокорреляция, лаг 1": round(float(series.autocorr(1)), 4)})
    return pd.DataFrame(rows)


def tuning_tables(cfg: dict, accuracy: dict, symbols: list, log) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Окно настройки против теста — диагностика переобучения при выборе гиперпараметров.

    Для каждой настраиваемой модели — RMSE относительно нулевого прогноза на окне настройки:
    у выбранного варианта (он же лучший) и у худшего варианта сетки, и тот же показатель на
    тесте. Нулевой прогноз берётся на тех же строках, что и прогноз модели: у деревьев — все
    пригодные часы окна, у LSTM — часы, для которых есть полное окно наблюдений длины lookback.
    Для леса отдельно — лучший вариант при каждой глубине деревьев (кривая сложности)."""
    data_dir, results_dir = cfg["paths"]["data_dir"], cfg["paths"]["results_dir"]
    summary, depth_rows = [], []
    for symbol in symbols:
        path = data_dir / "processed" / f"features_{symbol}.parquet"
        if not path.exists():
            log.warning("Нет %s — диагностика окна настройки (П11) не посчитана", path)
            continue
        table = pd.read_parquet(path, columns=["valid", TARGET])
        valid = table["valid"].to_numpy(bool)
        y = table[TARGET].to_numpy(float)
        test_ratio = accuracy[symbol].set_index("key")["rmse_ratio_zero"] if symbol in accuracy else pd.Series(dtype=float)
        for key in all_keys(cfg):
            tuning_path = results_dir / "predictions" / symbol / key.replace(":", "__") / "tuning.json"
            if not tuning_path.exists():
                continue
            info = json.loads(tuning_path.read_text(encoding="utf-8"))
            window = info["window"]
            inside = ((table.index >= pd.Timestamp(window["test_start"])) & (table.index < pd.Timestamp(window["test_end"])))
            model = key.split(":")[0]
            ratios = []
            for result in info["results"]:
                rows = (complete_windows(valid, int(result["params"]["lookback"])) if model == "lstm" else valid) & inside
                if int(rows.sum()) != int(result["rows"]):
                    log.warning("[%s] %s: в окне настройки %d строк, а в tuning.json — %d", symbol, label(key),
                                int(rows.sum()), int(result["rows"]))
                zero = float(np.mean(y[rows] ** 2))
                ratios.append(float(np.sqrt(result["mse"] / zero)) if zero > 0 else np.nan)
            chosen = next(i for i, r in enumerate(info["results"]) if r["params"] == info["chosen"])
            summary.append({"Актив": symbol, "Модель": label(key), "Вариантов в сетке": len(ratios),
                            "Окно настройки: выбранный вариант": round(ratios[chosen], 4),
                            "Окно настройки: худший вариант": round(float(np.nanmax(ratios)), 4),
                            "Тест (8 фолдов)": round(float(test_ratio[key]), 4) if key in test_ratio.index else None})
            if model == "random_forest":
                row = {"Актив": symbol, "Модель": label(key)}
                for depth in sorted({r["params"].get("max_depth") for r in info["results"]},
                                    key=lambda d: np.inf if d is None else d):
                    best = min(ratio for ratio, r in zip(ratios, info["results"]) if r["params"].get("max_depth") == depth)
                    row["глубина без ограничения" if depth is None else f"глубина {depth}"] = round(best, 4)
                depth_rows.append(row)
    return pd.DataFrame(summary), pd.DataFrame(depth_rows)


def environment_blocks(results_dir, symbols: list) -> list[tuple[str, pd.DataFrame]]:
    """Окружение и версии кода: пакеты из манифеста шага 0, коммит и время запуска каждого шага,
    коммиты кода, которыми посчитаны прогнозы шага 7."""
    blocks = []
    env_path = results_dir / "env" / "manifest.json"
    if env_path.exists():
        env = json.loads(env_path.read_text(encoding="utf-8"))
        rows = [("Версия репозитория (VERSION)", env.get("repo_version")), ("Python", env.get("python")),
                ("Платформа", env.get("platform")), ("Видеокарта", env.get("gpu") or "нет"),
                ("Коммит кода", env.get("git_commit")), ("Время проверки окружения (UTC)", env.get("time_utc"))]
        rows += [(f"Пакет {name}", version) for name, version in env.get("packages", {}).items()]
        blocks.append(("Окружение расчётов (шаг 0, results/env/manifest.json)",
                       pd.DataFrame(rows, columns=["Параметр", "Значение"])))
    steps = []
    for path in sorted((results_dir / "metrics").glob("*.json")):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8")).get("manifest")
        except (json.JSONDecodeError, AttributeError):
            continue
        if not isinstance(manifest, dict):
            continue
        packages = manifest.get("packages", {})
        steps.append({"Файл результатов": path.name, "Время запуска (UTC)": manifest.get("time_utc"),
                      "Коммит кода": (manifest.get("git_commit") or "")[:10],
                      "Изменённые файлы в рабочей копии": "да" if manifest.get("git_dirty") else "нет",
                      "Python": manifest.get("python"), "Конфигурация, SHA-256": (manifest.get("config_sha256") or "")[:16],
                      "pandas": packages.get("pandas"), "xgboost": packages.get("xgboost"),
                      "statsmodels": packages.get("statsmodels"), "torch": packages.get("torch")})
    if steps:
        blocks.append(("Запуски шагов: коммит кода, время, версии ключевых пакетов", pd.DataFrame(steps)))
    commits, timing = [], {}
    for symbol in symbols:
        counts: dict[str, int] = {}
        for path in (results_dir / "predictions" / symbol).glob("*/fold_*__seed*.json"):
            info = json.loads(path.read_text(encoding="utf-8"))
            commit = (info.get("git_commit") or "нет")[:10]
            counts[commit] = counts.get(commit, 0) + 1
            entry = timing.setdefault((symbol, path.parent.name.split("__")[0]), [0.0, 0.0])
            entry[1] += float(info.get("seconds") or 0)
        for path in (results_dir / "predictions" / symbol).glob("*/tuning.json"):
            entry = timing.setdefault((symbol, path.parent.name.split("__")[0]), [0.0, 0.0])
            entry[0] += sum(float(r.get("seconds") or 0) for r in json.loads(path.read_text(encoding="utf-8"))["results"])
        commits += [{"Актив": symbol, "Коммит кода": c, "Файлов прогнозов (фолд × зерно)": n}
                    for c, n in sorted(counts.items(), key=lambda item: -item[1])]
    if commits:
        blocks.append(("Коммиты кода, которыми посчитаны прогнозы шага 7", pd.DataFrame(commits)))
    if timing:
        order = {model: i for i, model in enumerate(PRETTY)}
        rows = [{"Актив": symbol, "Модель": PRETTY.get(model, model), "Подбор гиперпараметров, мин": round(tune / 60, 1),
                 "Обучение и прогноз по фолдам и зёрнам, мин": round(folds / 60, 1),
                 "Всего, мин": round((tune + folds) / 60, 1)}
                for (symbol, model), (tune, folds) in sorted(timing.items(), key=lambda item: (
                    symbols.index(item[0][0]) if item[0][0] in symbols else 99, order.get(item[0][1], 99)))]
        total = sum(tune + folds for tune, folds in timing.values())
        rows.append({"Актив": "все", "Модель": "все", "Подбор гиперпараметров, мин": round(sum(t for t, _ in timing.values()) / 60, 1),
                     "Обучение и прогноз по фолдам и зёрнам, мин": round(sum(f for _, f in timing.values()) / 60, 1),
                     "Всего, мин": round(total / 60, 1)})
        blocks.append(("Время расчёта шага 7 по записям в файлах прогнозов (без чтения данных и пауз), мин",
                       pd.DataFrame(rows)))
    return blocks


# ---------- рисунки ----------

SCHEME_STAGES = [
    ("Данные",
     "Binance: часовые свечи BTCUSDT и ETHUSDT из месячных архивов с проверкой контрольных сумм SHA-256. "
     "CryptoVision: англоязычные заголовки. ForkLog: русскоязычные заголовки (WordPress REST API)"),
    ("Очистка и проверка времени публикации",
     "Все метки времени — в UTC; дубли по заголовку в окне 24\u00a0ч; исключаются записи без точного времени, "
     "пакетные загрузки и источник с ненадёжным временем; время сверено с сайтами источников"),
    ("Тональность заголовков",
     "Шесть моделей-кандидатов; ручная разметка 400 заголовков на каждом языке, 100\u00a0из них — вторым "
     "разметчиком; выбор модели по macro-F1; оценка s\u00a0=\u00a0P(позитив)\u00a0−\u00a0P(негатив)"),
    ("Часовая сетка и признаки",
     "Новость с временем $\\tau$ относится к часу $\\lfloor\\tau\\rfloor$; признаки $x_t$ строятся только "
     "из данных до закрытия свечи $t$; цель $y_{t+1}=\\ln(C_{t+1}/C_t)$; наборы признаков P, P_EN, P_RU, P_EN_RU"),
    ("Модели и схема обучения",
     "Нулевой прогноз, историческое среднее, ARIMA, ARIMAX, Random Forest, XGBoost, LSTM; гиперпараметры "
     "выбираются на окне настройки перед тестом; 8\u00a0квартальных фолдов, расширяющееся окно, зазор 1\u00a0ч"),
    ("Оценка",
     "RMSE, MAE, точность направления, $R^2_{\\mathrm{OOS}}$; тесты Диболда–Мариано с поправкой Холма, "
     "Песарана–Тиммерманна, Model Confidence Set; стратегии с издержками 0–20\u00a0б.\u00a0п. против "
     "buy-and-hold; заранее заданный критерий выбора модели"),
    ("Проверки устойчивости",
     "Результаты по фолдам; тест сдвига новостей и плацебо-тест на утечку информации; тест Грейнджера "
     "в обе стороны"),
]
SCHEME_SIDE = ("Воспроизводимость", [
    "Протокол зафиксирован до расчётов, отклонения — в журнале с датами и причинами",
    "Все параметры — в одном файле конфигурации",
    "Манифест каждого шага: версии пакетов, зёрна, коммит кода, хеши входных файлов",
    "Проверки в коде: порядок меток времени, скачки цены, даты обучения и теста",
    "Все таблицы и рисунки строятся кодом",
    "Код и инструкция запуска — GitHub, архив релиза — Zenodo",
])


def wrap_to_width(fig, text: str, fontsize: float, width_in: float) -> list[str]:
    """Перенос по словам с измерением ширины строки; формулы $…$ и неразрывные пробелы не разрываются."""
    renderer = fig.canvas.get_renderer()
    tokens = re.findall(r"\$[^$]*\$[^ ]*|[^ ]+", text)

    def width(line: str) -> float:
        probe = fig.text(0, 0, line, fontsize=fontsize)
        extent = probe.get_window_extent(renderer)
        probe.remove()
        return extent.width / fig.dpi

    lines, current = [], ""
    for token in tokens:
        trial = f"{current} {token}".strip()
        if current and width(trial) > width_in:
            lines.append(current)
            current = token
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def figure_scheme(plt, out_dir) -> None:
    """Рисунок 1: этапы подхода и меры воспроизводимости (от данных не зависит)."""
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

    width, left_w, gap, side_w = 7.2, 5.0, 0.18, 1.92
    x0, badge = 0.04, 0.30
    title_size, text_size = 8.5, 7.3
    title_h, line_h, pad, step_gap = 0.21, 0.145, 0.08, 0.16
    fig = plt.figure(figsize=(width, 8))
    text_left, text_w = x0 + badge + 0.12, left_w - badge - 0.22
    blocks = []
    for title, text in SCHEME_STAGES:
        lines = wrap_to_width(fig, text, text_size, text_w)
        blocks.append((title, lines, title_h + line_h * len(lines) + 2 * pad))
    side_items = [wrap_to_width(fig, item, text_size, side_w - 0.36) for item in SCHEME_SIDE[1]]
    height = sum(h for *_, h in blocks) + step_gap * (len(blocks) - 1) + 0.08
    fig.set_size_inches(width, height)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    top = y = height - 0.04
    for i, (title, lines, h) in enumerate(blocks, start=1):
        ax.add_patch(FancyBboxPatch((x0 + badge, y - h), left_w - badge, h, boxstyle="round,pad=0,rounding_size=0.05",
                                    linewidth=0.9, edgecolor=report.SERIES[0], facecolor="white"))
        ax.add_patch(Circle((x0 + 0.12, y - h / 2), 0.105, color=report.SERIES[0]))
        ax.text(x0 + 0.12, y - h / 2, str(i), ha="center", va="center_baseline", color="white", fontsize=8,
                fontweight="bold")
        ax.text(text_left, y - pad - 0.15, title, ha="left", va="baseline", fontsize=title_size, fontweight="bold",
                color=report.INK)
        for j, line in enumerate(lines):
            ax.text(text_left, y - pad - title_h - 0.105 - j * line_h, line, ha="left", va="baseline",
                    fontsize=text_size, color=report.INK2)
        if i < len(blocks):
            middle = x0 + badge + (left_w - badge) / 2
            ax.add_patch(FancyArrowPatch((middle, y - h), (middle, y - h - step_gap), arrowstyle="-|>",
                                         mutation_scale=8, linewidth=0.9, color=report.MUTED, shrinkA=0, shrinkB=0))
        y -= h + step_gap
    bottom, side_x = y + step_gap, x0 + left_w + gap
    ax.add_patch(FancyBboxPatch((side_x, bottom), side_w, top - bottom, boxstyle="round,pad=0,rounding_size=0.05",
                                linewidth=0.9, edgecolor=report.AXIS, facecolor="#f5f4ef"))
    ax.text(side_x + 0.12, top - pad - 0.15, SCHEME_SIDE[0], ha="left", va="baseline", fontsize=title_size,
            fontweight="bold", color=report.INK)
    y = top - pad - title_h - 0.155
    for lines in side_items:
        ax.text(side_x + 0.12, y, "•", ha="left", va="baseline", fontsize=text_size, color=report.INK2)
        for j, line in enumerate(lines):
            ax.text(side_x + 0.24, y - j * line_h, line, ha="left", va="baseline", fontsize=text_size, color=report.INK2)
        y -= line_h * len(lines) + 0.13
    report.save_figure(fig, out_dir / "fig1_scheme")
    plt.close(fig)

def figure_strategies(plt, combined: dict, trading: dict, cfg: dict, out_dir) -> dict:
    from matplotlib.dates import MonthLocator
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    s_cfg = cfg["evaluation"]["strategy"]
    cost, periods = float(s_cfg["primary_cost_bp"]), float(s_cfg["annualization_hours"])
    fig, axes = plt.subplots(1, len(combined), figsize=(7.2, 3.4), constrained_layout=True)
    chosen = {}
    for ax, (symbol, table) in zip(np.atleast_1d(axes), combined.items()):
        flat = trading[symbol]
        flat = flat[(flat["strategy"] == "long_flat") & (flat["cost_bp"] == cost)
                    & ~flat["key"].map(lambda k: k == "buy_and_hold" or is_baseline(k))]
        best = flat.loc[flat["sharpe"].idxmax(), "key"]
        chosen[symbol] = best
        y = table["y_true"].to_numpy(float)
        when = table.index + pd.Timedelta(hours=2)             # доход часа t + 1 известен на закрытии его свечи
        _, hold = backtest.evaluate(np.ones(len(y)), y, "buy_and_hold", cost, periods)
        _, gross = backtest.evaluate(table[best].to_numpy(float), y, "long_flat", 0.0, periods)
        _, net = backtest.evaluate(table[best].to_numpy(float), y, "long_flat", cost, periods)
        for series, name, color in ((hold, f"Buy-and-hold, {cost:g} б. п. за вход", report.SERIES[0]),
                                    (gross, f"{label(best)}, без издержек", report.SERIES[1]),
                                    (net, f"{label(best)}, {cost:g} б. п.", report.SERIES[2])):
            ax.plot(when, np.cumprod(1.0 + series), color=color, label=name)
        ax.set_yscale("log")
        low, high = ax.get_ylim()
        decades = np.log10(high / low) if low > 0 else np.inf          # при большом размахе — только степени 10
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0) if decades <= 2.5 else (1.0,)))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_locator(MonthLocator(bymonth=(1, 7)))
        ax.xaxis.set_major_formatter(report.month_formatter())
        ax.set_title(ASSET.get(symbol, symbol), loc="left")
        ax.set_ylabel("Стоимость портфеля (начало = 1)")
        ax.tick_params(axis="x", labelsize=7.5)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=7.5)
    report.save_figure(fig, out_dir / "fig4_strategies")
    plt.close(fig)
    return chosen


def figure_breakeven(plt, trading: dict, cfg: dict, out_dir) -> None:
    cost = float(cfg["evaluation"]["strategy"]["primary_cost_bp"])
    fig, axes = plt.subplots(1, len(trading), figsize=(7.2, 2.7), constrained_layout=True, sharey=True)
    for ax, (symbol, table) in zip(np.atleast_1d(axes), trading.items()):
        part = table[(table["strategy"] == "long_flat") & (table["cost_bp"] == cost)
                     & ~table["key"].map(lambda k: k == "buy_and_hold" or is_baseline(k))]
        ax.scatter(part["turnover_per_year"] / 1000, part["breakeven_cost_bp"], s=26, color=report.SERIES[0],
                   edgecolor="white", linewidth=1.0, zorder=3)
        ax.axhline(cost, color=report.INK2, linewidth=0.9)
        ax.axhline(0, color=report.AXIS, linewidth=0.8)
        ax.text(ax.get_xlim()[0], cost, f" комиссия Binance — {cost:g} б. п.", va="bottom", ha="left",
                fontsize=7.5, color=report.INK2)
        ax.set_ylim(min(0, part["breakeven_cost_bp"].min() * 1.2), cost * 1.25)
        ax.set_title(ASSET.get(symbol, symbol), loc="left")
        ax.set_xlabel("Смен позиции в год, тыс.")
        ax.xaxis.set_major_formatter(report.comma_formatter(1))
        ax.yaxis.set_major_formatter(report.comma_formatter(0))
    np.atleast_1d(axes)[0].set_ylabel("Безубыточные издержки, б. п.")
    report.save_figure(fig, out_dir / "fig5_breakeven")
    plt.close(fig)


def figure_accuracy(plt, accuracy: dict, out_dir) -> None:
    keys = [k for k in next(iter(accuracy.values()))["key"] if k != "naive_zero:none"]
    fig, axes = plt.subplots(1, len(accuracy), figsize=(7.2, 0.22 * len(keys) + 1.2), layout="constrained",
                             sharey=True, sharex=True)
    positions = np.arange(len(keys))[::-1]
    for ax, (symbol, table) in zip(np.atleast_1d(axes), accuracy.items()):
        values = table.set_index("key").loc[keys, "rmse_ratio_zero"].to_numpy(float)
        ax.axvline(0, color=report.AXIS, linewidth=0.9)
        ax.scatter(100 * (values - 1), positions, s=22, color=report.SERIES[0], edgecolor="white", linewidth=1.0,
                   zorder=3)
        ax.set_title(ASSET.get(symbol, symbol), loc="left")
        spread = float(np.nanmax(np.abs(100 * (values - 1)))) if len(values) else 0.0
        ax.xaxis.set_major_formatter(report.comma_formatter(2 if spread < 1 else 1 if spread < 10 else 0))
        ax.grid(axis="y", visible=False)
    first = np.atleast_1d(axes)[0]
    first.set_yticks(positions)
    first.set_yticklabels([label(k) for k in keys])
    fig.supxlabel("Изменение RMSE относительно нулевого прогноза, %", fontsize=8.5, color=report.INK2)
    report.save_figure(fig, out_dir / "fig3_accuracy")
    plt.close(fig)


def figure_shift(plt, shift: dict, out_dir) -> None:
    fig, axes = plt.subplots(1, len(shift), figsize=(7.2, 3.0), layout="constrained", sharey=True)
    for ax, (symbol, table) in zip(np.atleast_1d(axes), shift.items()):
        hours = sorted(table["shift_hours"].unique())
        x = np.arange(len(hours))
        ax.axhline(0, color=report.AXIS, linewidth=0.9)
        for color, (feature_set, part) in zip(report.SERIES, table.groupby("feature_set", sort=False)):
            part = part.set_index("shift_hours").loc[hours]
            ax.plot(x, part["mse_change_vs_price_pct"], color=color, marker="o", markersize=4.5,
                    markeredgecolor="white", markeredgewidth=1.0, label=SET_NAMES.get(feature_set, feature_set))
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(h)) for h in hours])
        ax.set_xlabel("Сдвиг новостных признаков в прошлое, ч")
        ax.set_title(ASSET.get(symbol, symbol), loc="left")
        ax.yaxis.set_major_formatter(report.comma_formatter(2))
    first = np.atleast_1d(axes)[0]
    first.set_ylabel("Изменение MSE к набору P, %")
    handles, labels = first.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels), fontsize=7.5)
    report.save_figure(fig, out_dir / "fig6_shift")
    plt.close(fig)


def figure_coverage(plt, results_dir, symbols: list, out_dir) -> None:
    fig, axes = plt.subplots(1, len(symbols), figsize=(7.2, 2.9), layout="constrained", sharey=True)
    for ax, symbol in zip(np.atleast_1d(axes), symbols):
        table = pd.read_csv(table_path(results_dir, f"news_coverage_by_year_{symbol}.csv"), index_col=0)
        for color, (column, name) in zip(report.SERIES, (("en_hours_with_news", "англоязычные"),
                                                         ("ru_hours_with_news", "русскоязычные"))):
            ax.plot(table.index, 100 * table[column], color=color, marker="o", markersize=4.5,
                    markeredgecolor="white", markeredgewidth=1.0, label=name)
        ax.set_ylim(0, 100)
        ax.set_xticks(table.index)
        ax.set_xticklabels([str(y) for y in table.index], rotation=0, fontsize=7.5)
        ax.set_title(ASSET.get(symbol, symbol), loc="left")
    first = np.atleast_1d(axes)[0]
    first.set_ylabel("Часов хотя бы с одной новостью, %")
    handles, labels = first.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels), fontsize=7.5)
    report.save_figure(fig, out_dir / "fig2_coverage")
    plt.close(fig)


def main() -> None:
    args = parse_args(__doc__)
    cfg = load_config(args.config)
    log = get_logger()
    results_dir = cfg["paths"]["results_dir"]
    out_dir = results_dir / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.iterdir():                     # файлы прошлых запусков (в том числе со старыми именами)
        if old.is_file() and old.suffix in REPORT_SUFFIXES:
            old.unlink()
    splits = validation.from_json(json.loads((results_dir / "splits.json").read_text(encoding="utf-8")))
    symbols = list(cfg["data"]["prices"]["symbols"])
    features_report = json.loads((results_dir / "metrics" / "features_report.json").read_text(encoding="utf-8"))

    read = lambda name: {s: pd.read_csv(table_path(results_dir, name.format(s))) for s in symbols}  # noqa: E731
    accuracy, questions, trading = read("eval_accuracy_{}.csv"), read("eval_questions_{}.csv"), read("eval_trading_{}.csv")
    folds, granger, adf = read("eval_folds_{}.csv"), read("granger_{}.csv"), read("stationarity_{}.csv")
    shift, placebo = read("leakage_shift_{}.csv"), read("leakage_placebo_{}.csv")
    combined = {}
    for symbol in symbols:
        path = combined_path(results_dir, symbol)
        if not path.exists():
            log.info("[%s] общей таблицы прогнозов нет — собираю из файлов шага 7 (несколько минут)", symbol)
            found, _ = collect_all(cfg, results_dir / "predictions", symbol, len(splits["folds"]))
            combine(found).to_parquet(path)
        combined[symbol] = pd.read_parquet(path)

    costs = [float(c) for c in cfg["evaluation"]["strategy"]["costs_bp"]]
    primary = float(cfg["evaluation"]["strategy"]["primary_cost_bp"])
    sentiment_models, agreement = sentiment_tables(results_dir)
    shift_table, placebo_table = leakage_tables(shift, placebo)
    tuned, orders = hyperparameter_tables(cfg, results_dir, symbols, len(splits["folds"]))
    batch_path = results_dir / "tables" / "batch_uploads_en.csv"
    sheets = [
        ("Т1 Выборка", [("Т1. Выборка, новости и исключения", sample_table(features_report, splits, symbols))]),
        ("Т2 Тональность", [("Т2. Модели тональности на ручной разметке (macro-F1 с 95%-м бутстреп-интервалом)",
                             sentiment_models), ("Согласие разметчиков", agreement)]),
        ("Т3 Точность", [("Т3. Точность прогнозов на тестовом периоде; p-значения с поправкой Холма",
                          accuracy_table(accuracy))]),
        ("Т4 В1 и В2", [("Т4. Вопросы В1 и В2: тест Диболда–Мариано (H1 — вариант с новостями точнее)",
                         questions_table(questions))]),
        ("Т5 Стратегии", [(f"Т5. Стратегия long/flat при издержках {primary:g} б. п.: коэффициент Шарпа и разница с buy-and-hold",
                           strategy_table(trading, "long_flat", primary))]),
        ("П1 Издержки", [("П1. Коэффициент Шарпа при разных издержках; разница с buy-and-hold без издержек (95% ДИ); "
                          "long/short — при основных издержках",
                          costs_table(trading, costs, primary))]),
        ("П2 Фолды", [("П2. RMSE относительно нулевого прогноза по фолдам (столбец — начало фолда)", folds_table(folds))]),
        ("П3 Грейнджер", [("П3. Тест Грейнджера (прогностическое предшествование, HAC): оба направления, "
                           "лаги 1, 6, 24 ч и по BIC", granger_table(granger)),
                          ("Стационарность рядов: расширенный тест Дики–Фуллера", stationarity_table(adf))]),
        ("П4 Утечка", [("П4. Тест сдвига (XGBoost): изменение MSE относительно набора P, %", shift_table),
                       ("Плацебо-тест: модель обучена на перемешанной цели", placebo_table)]),
        ("П5 Гиперпараметры", [("П5. Гиперпараметры, выбранные на окне настройки", tuned),
                               ("Порядок ARIMA по фолдам (по AIC на последнем годе перед фолдом)", orders)]),
        ("П6 Источники", [("П6. Источники новостей после очистки (весь набор, до ограничения периодом выборки)",
                           sources_table(results_dir))]),
        ("П7 Пакетные загрузки", [("П7. Серии пакетных загрузок, исключённые из признаков",
                                   pd.read_csv(batch_path) if batch_path.exists() else pd.DataFrame({"Нет данных": []}))]),
    ]
    flow = data_flow_tables(cfg, features_report, splits, symbols, combined, log)
    if flow:
        flow[0] = (f"П8. Схема данных. {flow[0][0]}", flow[0][1])
        sheets.append(("П8 Схема данных", flow))
    selection = {s: pd.read_csv(results_dir / "tables" / f"eval_selection_{s}.csv") for s in symbols
                 if (results_dir / "tables" / f"eval_selection_{s}.csv").exists()}
    if selection:
        sheets.append(("П9 Критерий выбора", [(f"П9. Критерий выбора модели (раздел 9 протокола): стратегия long/flat, "
                                               f"{primary:g} б. п.", selection_table(selection))]))
    returns = returns_table(cfg, splits, symbols, log)
    if len(returns):
        sheets.append(("П10 Доходность", [("П10. Описательные статистики часовой лог-доходности (целевая переменная)",
                                           returns)]))
    tuning_summary, forest_depth = tuning_tables(cfg, accuracy, symbols, log)
    if len(tuning_summary):
        sheets.append(("П11 Окно настройки", [
            ("П11. Диагностика переобучения: RMSE относительно нулевого прогноза на окне настройки и на тесте",
             tuning_summary),
            ("Random Forest: лучший вариант на окне настройки при каждой глубине деревьев (RMSE / нулевой прогноз)",
             forest_depth)]))
    environment = environment_blocks(results_dir, symbols)
    if environment:
        environment[0] = (f"П12. Окружение и версии. {environment[0][0]}", environment[0][1])
        sheets.append(("П12 Окружение", environment))
    contents = pd.DataFrame([(name, blocks[0][0]) for name, blocks in sheets], columns=["Лист", "Содержание"])
    report.write_sheets(out_dir / "tables.xlsx", [("Описание", [("Таблицы статьи и приложения", contents)])] + sheets)
    log.info("Таблицы: %s (листов — %d)", out_dir / "tables.xlsx", len(sheets) + 1)

    lock = results_dir / "env" / "requirements-lock.txt"
    if lock.exists():
        shutil.copyfile(lock, out_dir / "requirements-lock.txt")
    else:
        log.warning("Нет %s — выполните шаг 0, чтобы зафиксировать версии пакетов", lock)

    plt = report.pyplot()
    figure_scheme(plt, out_dir)
    figure_coverage(plt, results_dir, symbols, out_dir)
    figure_accuracy(plt, accuracy, out_dir)
    chosen = figure_strategies(plt, combined, trading, cfg, out_dir)
    figure_breakeven(plt, trading, cfg, out_dir)
    figure_shift(plt, shift, out_dir)
    log.info("Рисунки: fig1–fig6 (PNG и SVG) в %s; на рисунке 4 — лучшая по Шарпу стратегия: %s", out_dir,
             "; ".join(f"{s} — {label(k)}" for s, k in chosen.items()))
    files = sorted(p.name for p in out_dir.iterdir() if p.suffix in REPORT_SUFFIXES)
    with zipfile.ZipFile(results_dir / "report.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            archive.write(out_dir / name, arcname=f"report/{name}")
    save_json({"manifest": run_manifest(cfg), "figure4_strategies": chosen, "files": files},
              results_dir / "metrics" / "report.json")
    log.info("Готово")


if __name__ == "__main__":
    main()
