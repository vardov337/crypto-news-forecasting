"""Шаг 5b (задача 5.7): проверка моделей тональности на ручной разметке и выбор модели.

Что делает скрипт:
  1. читает ручную разметку из annotation/labels/: <язык>_annotator1.csv (400 заголовков)
     и <язык>_annotator2.csv (100 из них, второй разметчик);
  2. считает согласие разметчиков на общих заголовках: долю совпадений и каппу Коэна;
  3. для каждой модели-кандидата считает accuracy, macro-F1 и F1 по классам на разметке
     первого разметчика, 95%-е бутстреп-интервалы и парные разности с лучшей моделью;
  4. выбирает модель с наибольшим macro-F1 (правило зафиксировано в PROTOCOL.md до разметки,
     при равенстве — первая по списку в конфигурации) и записывает выбор
     в results/metrics/sentiment_choice.json — его читает шаг 6.

Нужны результаты шага 5 (оценки моделей в data/interim/sentiment/). Видеокарта не нужна.
Запуск:  python scripts/05b_choose_sentiment.py
"""
import json

import numpy as np
import pandas as pd

from cryptonews import annotation, sentiment
from cryptonews.cli import parse_args
from cryptonews.config import PROJECT_ROOT, load_config
from cryptonews.utils import get_logger, run_manifest, save_json, sha256_file

LABELS_DIR = PROJECT_ROOT / "annotation" / "labels"


def read_labels(path) -> pd.DataFrame:
    labels = pd.read_csv(path, dtype={"id": str, "url": str, "label": str})
    labels = labels.dropna(subset=["label"])
    unknown = sorted(set(labels["label"]) - set(sentiment.CLASSES))
    if unknown:
        raise SystemExit(f"В {path.name} неизвестные метки: {unknown}")
    return labels


def used_revisions(cfg: dict) -> dict[tuple[str, str], str]:
    """Ревизии, с которыми шаг 5 посчитал оценки: из конфигурации или из его отчёта."""
    report_path = cfg["paths"]["results_dir"] / "metrics" / "sentiment_models.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {"models": []}
    revisions = {(m["language"], m["model"]): m["revision"] for m in report["models"]}
    for candidate in sentiment.candidates_from_config(cfg):
        if candidate.revision:
            revisions[(candidate.language, candidate.name)] = candidate.revision
    return revisions


def main() -> None:
    args = parse_args(__doc__)
    cfg = load_config(args.config)
    log = get_logger()
    reps = int(cfg["sentiment"]["selection"]["bootstrap_reps"])
    seed = int(cfg["sentiment"]["annotation"]["seed"])
    revisions = used_revisions(cfg)
    interim = cfg["paths"]["data_dir"] / "interim" / "sentiment"
    tables_dir = cfg["paths"]["results_dir"] / "tables"
    choice = {"manifest": run_manifest(cfg), "rule": "наибольший macro-F1 на разметке первого разметчика"}

    for lang in ("en", "ru"):
        first_path = LABELS_DIR / f"{lang}_annotator1.csv"
        if not first_path.exists():
            raise SystemExit(f"Нет разметки {first_path} — шаг 5b запускается после ручной разметки")
        first = read_labels(first_path)
        log.info("[%s] Разметка первого разметчика: %d заголовков; классы: %s", lang, len(first),
                 ", ".join(f"{c} {v:.1%}" for c, v in first["label"].value_counts(normalize=True).items()))
        entry = {"labels_file": first_path.name, "labels_sha256": sha256_file(first_path), "n_labels": len(first)}

        second_path = LABELS_DIR / f"{lang}_annotator2.csv"
        if second_path.exists():
            second = read_labels(second_path)
            pair = first.merge(second, on="id", suffixes=("_1", "_2"))
            agree = annotation.agreement(pair["label_1"], pair["label_2"])
            entry["agreement"] = agree
            log.info("[%s] Согласие разметчиков на %d общих заголовках: совпадений %.1f%%, каппа Коэна %.3f",
                     lang, agree["n"], 100 * agree["percent_agreement"], agree["cohen_kappa"])
        else:
            log.warning("[%s] Разметки второго разметчика нет (%s) — согласие не посчитано", lang, second_path.name)

        rows, predictions = [], {}
        for candidate in [c for c in sentiment.candidates_from_config(cfg) if c.language == lang]:
            revision = revisions.get((lang, candidate.name))
            if revision is None:
                raise SystemExit(f"Не знаю, с какой ревизией считалась {candidate.name}: сначала шаг 5")
            slug = sentiment.model_slug(candidate.name, revision)
            path = interim / lang / f"{slug}.parquet"
            if not path.exists():
                raise SystemExit(f"Нет оценок {path} — сначала шаг 5")
            scores = pd.read_parquet(path, columns=["url", "label"])
            merged = first.merge(scores, on="url", how="left", suffixes=("", "_model"))
            if merged["label_model"].isna().any():
                raise SystemExit(f"У {candidate.name} нет оценок для {int(merged['label_model'].isna().sum())} "
                                 "размеченных заголовков — разметка и оценки от разных версий данных")
            predictions[candidate.name] = merged["label_model"].tolist()
            rows.append({"model": candidate.name, "revision": revision, "slug": slug,
                         **annotation.classification_metrics(merged["label"], merged["label_model"])})

        table = pd.DataFrame(rows)
        boot = annotation.bootstrap_macro_f1(first["label"].tolist(), predictions, reps=reps, seed=seed)
        table["macro_f1_ci_low"] = np.percentile(boot, 2.5, axis=0)
        table["macro_f1_ci_high"] = np.percentile(boot, 97.5, axis=0)
        best = int(table["macro_f1"].to_numpy().argmax())     # при равенстве — первая по списку
        diffs = boot[:, [best]] - boot
        table["diff_vs_best"] = table.loc[best, "macro_f1"] - table["macro_f1"]
        table["diff_ci_low"] = np.percentile(diffs, 2.5, axis=0)
        table["diff_ci_high"] = np.percentile(diffs, 97.5, axis=0)
        table["chosen"] = table.index == best
        table.round(4).to_csv(tables_dir / f"sentiment_validation_{lang}.csv", index=False, encoding="utf-8")

        log.info("[%s] Качество моделей на разметке:", lang)
        for r in table.itertuples():
            log.info("    %-48s accuracy %.3f  macro-F1 %.3f [%.3f; %.3f]%s", r.model, r.accuracy, r.macro_f1,
                     r.macro_f1_ci_low, r.macro_f1_ci_high, "  ← выбрана" if r.chosen else "")
        winner = table.loc[best]
        entry.update({"model": winner["model"], "revision": winner["revision"], "slug": winner["slug"],
                      "macro_f1": float(winner["macro_f1"]), "accuracy": float(winner["accuracy"]),
                      "candidates": table.drop(columns=["chosen"]).to_dict(orient="records")})
        choice[lang] = entry

    save_json(choice, cfg["paths"]["results_dir"] / "metrics" / "sentiment_choice.json")
    log.info("Выбор записан: %s", cfg["paths"]["results_dir"] / "metrics" / "sentiment_choice.json")


if __name__ == "__main__":
    main()
