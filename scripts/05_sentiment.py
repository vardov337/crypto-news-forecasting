"""Шаг 5 (задачи 5.2–5.6): выборка для ручной разметки и тональность заголовков.

Что делает скрипт:
  1. определяет границы общей выборки по ценам и новостям (PROTOCOL.md, раздел 2)
     и записывает их в results/metrics/sample_period.json;
  2. готовит файлы для ручной разметки: по 400 заголовков на язык из периода выборки
     (стратификация по годам и источникам) и по 100 из них для второго разметчика.
     Файлы Excel с инструкцией внутри появляются в results/annotation/ и в архиве
     results/annotation/annotation_files.zip. Уже созданные файлы не перезаписываются;
  3. оценивает тональность всех заголовков каждой моделью-кандидатом из конфигурации
     и сохраняет вероятности классов в data/interim/sentiment/<язык>/;
  4. пишет сводку по моделям и их ревизии (хеши коммитов) в results/.

Нужна видеокарта (в Colab — T4): расчёт занимает 5–15 минут, на процессоре — пару часов.
Если сеанс оборвался, повторный запуск продолжит с первой непосчитанной модели.

Запуск:  python scripts/05_sentiment.py          (ключ --cpu — считать без видеокарты)
"""
import json
import os
import time
import traceback
import zipfile
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd  # noqa: E402

from cryptonews import annotation, period, sentiment  # noqa: E402
from cryptonews.cli import parse_args  # noqa: E402
from cryptonews.config import PROJECT_ROOT, load_config  # noqa: E402
from cryptonews.data import news as news_rules  # noqa: E402
from cryptonews.utils import (get_logger, package_versions, run_manifest, save_json,  # noqa: E402
                              set_seed, sha256_file)

LANGUAGE_NAMES = {"en": "англоязычные новости (CryptoVision)", "ru": "русскоязычные новости (ForkLog)"}


def add_arguments(parser) -> None:
    parser.add_argument("--cpu", action="store_true", help="считать без видеокарты (несколько часов)")


def load_inputs(cfg: dict) -> tuple[dict, dict, dict]:
    data_dir = cfg["paths"]["data_dir"]
    news, hashes = {}, {}
    for lang, step in (("en", "3 (03_prepare_news_en.py)"), ("ru", "4 (04_prepare_news_ru.py)")):
        path = data_dir / "interim" / f"news_{lang}.parquet"
        if not path.exists():
            raise SystemExit(f"Нет файла {path} — сначала выполните шаг {step}")
        frame = pd.read_parquet(path)
        frame["published_utc"] = pd.to_datetime(frame["published_utc"], utc=True)
        # шаги 3–4 уже раскрывают мнемоники HTML; повтор на случай файла старой версии
        frame["title"] = frame["title"].astype(str).map(news_rules.unescape_title)
        news[lang], hashes[lang] = frame, sha256_file(path)
    prices = {}
    interval = cfg["data"]["prices"]["interval"]
    for symbol in cfg["data"]["prices"]["symbols"]:
        path = data_dir / "raw" / "binance" / f"{symbol}_{interval}.parquet"
        if not path.exists():
            raise SystemExit(f"Нет файла {path} — сначала выполните шаг 1 (01_download_prices.py)")
        prices[symbol] = pd.read_parquet(path)
    return news, hashes, prices


def prepare_annotation(cfg: dict, news: dict, hashes: dict, span: period.SamplePeriod, log) -> Path:
    """Файлы для разметчиков. Существующие файлы не трогаются: в них может быть разметка."""
    ann_cfg = cfg["sentiment"]["annotation"]
    size, second_size, seed = (int(ann_cfg["sample_size_per_language"]),
                               int(ann_cfg["second_annotator_sample"]), int(ann_cfg["seed"]))
    out_dir = cfg["paths"]["results_dir"] / "annotation"
    out_dir.mkdir(parents=True, exist_ok=True)
    instruction = (PROJECT_ROOT / "annotation" / "instruction.md").read_text(encoding="utf-8")
    period_text = f"{span.start} … {span.end}"
    created = False
    for lang in ("en", "ru"):
        main_path = out_dir / f"annotation_{lang}_main.xlsx"
        second_path = out_dir / f"annotation_{lang}_second.xlsx"
        meta_path = out_dir / f"annotation_{lang}_meta.json"
        if main_path.exists() or second_path.exists():
            log.info("Файлы разметки (%s) уже есть, не пересоздаю: %s", lang, main_path.name)
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("period") != period_text:
                    log.warning("Выборка для разметки (%s) сделана для другого периода: %s", lang, meta.get("period"))
            continue
        frame = news[lang][span.mask(news[lang]["published_utc"])]
        main, second = annotation.make_samples(frame, lang, size, second_size, seed)
        heading = f"Разметка тональности: {LANGUAGE_NAMES[lang]}"
        annotation.write_workbook(main, main_path, instruction, f"{heading}, {len(main)} заголовков")
        annotation.write_workbook(second, second_path, instruction,
                                  f"{heading}, второй разметчик, {len(second)} заголовков")
        main.to_csv(out_dir / f"annotation_{lang}_main_key.csv", index=False, encoding="utf-8")
        second.to_csv(out_dir / f"annotation_{lang}_second_key.csv", index=False, encoding="utf-8")
        strata = annotation.strata_table(main, frame)
        strata.to_csv(cfg["paths"]["results_dir"] / "tables" / f"annotation_{lang}_strata.csv", encoding="utf-8")
        save_json({"manifest": run_manifest(cfg), "news_sha256": hashes[lang], "seed": seed,
                   "period": period_text,
                   "news_in_period": len(frame), "main": len(main), "second": len(second)}, meta_path)
        by_year = main["year"].value_counts().sort_index()
        log.info("Выборка для разметки (%s): %d заголовков из %d в периоде, %d — второму разметчику. "
                 "По годам: %s", lang, len(main), len(frame), len(second),
                 ", ".join(f"{y}: {n}" for y, n in by_year.items()))
        if main["source"].nunique() > 1:
            log.info("    По источникам: %s", ", ".join(f"{s}: {n}" for s, n in main["source"].value_counts().items()))
        created = True
    zip_path = out_dir / "annotation_files.zip"
    if created or not zip_path.exists():
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(out_dir.glob("annotation_*_*.xlsx")):
                archive.write(path, arcname=path.name)
            archive.writestr("instruction.md", instruction)
    log.info("Файлы для разметки: %s", zip_path)
    return zip_path


def progress_reporter(log, name: str):
    marks = [25, 50, 75]

    def report(done: int, total: int) -> None:
        while marks and 100 * done / total >= marks[0]:
            log.info("    %s: %d%%", name, marks.pop(0))
    return report


def score_candidate(candidate, frame: pd.DataFrame, news_hash: str, device: str, cfg: dict, log) -> dict:
    """Оценки одной модели для всех заголовков языка; повторный запуск берёт готовый файл."""
    s_cfg = cfg["sentiment"]
    max_length, batch_size = int(s_cfg["max_length"]), int(s_cfg["batch_size"])
    revision = sentiment.resolve_revision(candidate.name, candidate.revision)
    slug = sentiment.model_slug(candidate.name, revision)
    out_dir = cfg["paths"]["data_dir"] / "interim" / "sentiment" / candidate.language
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path, meta_path = out_dir / f"{slug}.parquet", out_dir / f"{slug}.json"
    if out_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("news_sha256") == news_hash and meta.get("max_length") == max_length:
            log.info("%s: оценки уже посчитаны, беру готовые (%s)", candidate.name, out_path.name)
            return meta

    log.info("%s [%s], ревизия %s — загружаю модель", candidate.name, candidate.language, revision)
    started = time.time()
    model = sentiment.HFClassifier(candidate.name, revision, device, max_length, candidate.labels)
    log.info("    классы модели: %s", ", ".join(f"{model.id2label[i]} → {model.mapping[i]}"
                                             for i in sorted(model.id2label)))
    correct, total, rows = sentiment.sanity_check(candidate.language, model.classify)
    log.info("    проверочные фразы: верно %d из %d", correct, total)
    for row in rows:
        if row["expected"] != row["predicted"]:
            log.info("        «%s»: ожидалось %s, модель — %s", row["text"], row["expected"], row["predicted"])
    if correct <= total // 2:
        log.warning("    Модель ошибается в половине однозначных фраз — проверьте сопоставление классов")

    texts = frame["title"].astype(str).tolist()
    lengths = model.token_lengths(texts)
    truncated = int((lengths > max_length).sum())
    probs = sentiment.predict_proba(texts, model.logits, model.order, batch_size, lengths=lengths,
                                    progress=progress_reporter(log, candidate.name))
    scores = sentiment.scores_frame(frame["url"].astype(str), probs)
    scores.to_parquet(out_path, index=False)
    top, bottom = sentiment.extremes(frame, scores)
    meta = {
        "language": candidate.language, "model": candidate.name, "revision": revision, "slug": slug,
        "file": out_path.name, "news_sha256": news_hash, "records": len(scores),
        "max_length": max_length, "batch_size": batch_size, "device": sentiment.device_name(device),
        "parameters": model.n_params, "id2label": model.id2label,
        "mapping": {model.id2label[i]: c for i, c in model.mapping.items()},
        "truncated": truncated, "token_length_median": float(pd.Series(lengths).median()),
        "label_shares": sentiment.label_shares(scores), "mean_score": float(scores["score"].mean()),
        "sanity_correct": correct, "sanity_total": total, "seconds": round(time.time() - started, 1),
        "most_positive": top, "most_negative": bottom,
        "packages": package_versions(("torch", "transformers", "tokenizers", "huggingface-hub")),
    }
    save_json(meta, meta_path)
    model.close()
    shares = meta["label_shares"]
    log.info("    готово за %.0f с: негатив %.1f%%, нейтрал %.1f%%, позитив %.1f%%, средняя оценка %+.3f, "
             "обрезано заголовков %d", meta["seconds"], 100 * shares["negative"], 100 * shares["neutral"],
             100 * shares["positive"], meta["mean_score"], truncated)
    log.info("    самые позитивные: %s", " | ".join(t[:110] for t in top))
    log.info("    самые негативные: %s", " | ".join(t[:110] for t in bottom))
    return meta


def main() -> None:
    args = parse_args(__doc__, add_arguments)
    cfg = load_config(args.config)
    log = get_logger()
    set_seed(int(cfg["seeds"]["default"][0]))
    results_dir = cfg["paths"]["results_dir"]

    news, hashes, prices = load_inputs(cfg)
    span = period.from_config(cfg, prices, {"EN": news["en"], "RU": news["ru"]})
    log.info("Границы выборки: %s", span.describe())
    for name, ts in span.starts.items():
        log.info("    первая запись — %-22s %s", name + ":", ts)
    for name, ts in span.ends.items():
        log.info("    граница конца — %-22s %s", name + ":", ts)
    save_json({"manifest": run_manifest(cfg), **span.as_dict()}, results_dir / "metrics" / "sample_period.json")
    for lang in ("en", "ru"):
        inside = int(span.mask(news[lang]["published_utc"]).sum())
        log.info("Новостей в периоде выборки (%s): %d из %d", lang, inside, len(news[lang]))

    prepare_annotation(cfg, news, hashes, span, log)

    device = sentiment.pick_device(allow_cpu=args.cpu)
    log.info("Считаю тональность на устройстве: %s", sentiment.device_name(device))
    records, failures = [], []
    for candidate in sentiment.candidates_from_config(cfg):
        try:
            records.append(score_candidate(candidate, news[candidate.language], hashes[candidate.language],
                                           device, cfg, log))
        except Exception as error:  # одна неудачная модель не должна останавливать остальные
            log.error("%s: посчитать не удалось — %s: %s\n%s", candidate.name, type(error).__name__,
                      error, traceback.format_exc(limit=6))
            failures.append({"language": candidate.language, "model": candidate.name,
                             "error": f"{type(error).__name__}: {error}"})
            sentiment.free_memory()

    table = pd.DataFrame([{
        "Язык": r["language"], "Модель": r["model"], "Ревизия": r["revision"][:8],
        "Параметров, млн": round(r["parameters"] / 1e6, 1), "Заголовков": r["records"],
        "Обрезано": r["truncated"], "Негатив": r["label_shares"]["negative"],
        "Нейтрал": r["label_shares"]["neutral"], "Позитив": r["label_shares"]["positive"],
        "Средняя оценка": round(r["mean_score"], 4),
        "Проверочные фразы": f"{r['sanity_correct']}/{r['sanity_total']}", "Секунд": r["seconds"],
    } for r in records])
    table.to_csv(results_dir / "tables" / "sentiment_candidates.csv", index=False, encoding="utf-8")
    save_json({"manifest": run_manifest(cfg), "models": records, "failures": failures},
              results_dir / "metrics" / "sentiment_models.json")
    log.info("Сводка по моделям:\n%s", table.to_string(index=False))
    log.info("Ревизии моделей (будут зафиксированы в configs/config.yaml):")
    for r in records:
        log.info("    %s: %s  revision: %s", r["language"], r["model"], r["revision"])
    if failures:
        raise SystemExit("Не посчитаны модели: " + ", ".join(f["model"] for f in failures)
                         + ". Пришлите вывод целиком; посчитанные модели при повторном запуске пересчитываться не будут.")


if __name__ == "__main__":
    main()
