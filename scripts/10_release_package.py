"""Шаг 10: архив результатов для релиза в Zenodo.

Собирает results/release_package.zip — всё, что нужно, чтобы проверить числа статьи без
повторного расчёта и повторить шаги 5b–9, не собирая новости и не считая тональность заново:
  results/report/        — таблицы статьи и приложения (tables.xlsx), рисунки, версии пакетов (шаг 9);
  results/metrics/, results/tables/, results/splits.json, results/env/ — отчёты и таблицы шагов,
                           схема проверки, окружение;
  results/predictions_combined_<актив>.parquet — прогнозы всех 18 вариантов (среднее по зёрнам)
                           и факт на общих часах теста; по ним считаются все метрики, тесты и стратегии;
  results/predictions/<актив>/<модель>/tuning.json — подбор гиперпараметров на окне настройки;
  data/interim/news_<язык>.parquet — метаданные новостей БЕЗ заголовков: время публикации (UTC),
                           адрес, источник, упоминания монет;
  data/interim/sentiment/<язык>/<модель>.parquet — вероятности классов тональности: у выбранной
                           модели — для всех новостей, у остальных кандидатов — для размеченных
                           заголовков (этого достаточно для шага 5b);
  release_manifest.json  — список файлов с хешами SHA-256 и манифест запуска.
Заголовки новостей в архив не входят: пользовательское соглашение ForkLog запрещает копирование
материалов без согласия редакции, а набор CryptoVision доступен по DOI (data/README.md).
Шаг 2 по адресам собирает заголовки заново.

Запускать после шага 9. Видеокарта не нужна, около минуты.
Запуск:  python scripts/10_release_package.py
"""
import hashlib
import io
import json
import zipfile

import pandas as pd

from cryptonews import sentiment
from cryptonews.cli import parse_args
from cryptonews.config import PROJECT_ROOT, load_config
from cryptonews.utils import get_logger, run_manifest

TEXT_COLUMNS = ("title",)                       # в архив не идут
LANGUAGES = ("en", "ru")


def result_files(results_dir) -> list:
    """Файлы результатов, которые идут в архив как есть (пути относительно results_dir)."""
    patterns = ["report/*.xlsx", "report/*.png", "report/*.svg", "report/*.txt", "metrics/*.json", "metrics/*.csv",
                "tables/*.csv", "splits.json", "env/manifest.json", "env/requirements-lock.txt",
                "predictions_combined_*.parquet", "predictions/*/*/tuning.json"]
    files = []
    for pattern in patterns:
        files += sorted(path for path in results_dir.glob(pattern) if path.is_file())
    return files


def news_without_titles(data_dir, lang: str) -> pd.DataFrame:
    news = pd.read_parquet(data_dir / "interim" / f"news_{lang}.parquet")
    return news.drop(columns=[c for c in TEXT_COLUMNS if c in news.columns])


def annotated_urls(lang: str) -> set:
    urls = set()
    for path in (PROJECT_ROOT / "annotation" / "labels").glob(f"{lang}_annotator*.csv"):
        urls |= set(pd.read_csv(path, dtype={"url": str})["url"].dropna())
    return urls


def sentiment_files(cfg: dict, data_dir, results_dir, log) -> list[tuple[str, pd.DataFrame | None, object]]:
    """(путь в архиве, таблица или None, исходный файл): выбранная модель — целиком, остальные — размеченные заголовки."""
    choice = json.loads((results_dir / "metrics" / "sentiment_choice.json").read_text(encoding="utf-8"))
    models = json.loads((results_dir / "metrics" / "sentiment_models.json").read_text(encoding="utf-8"))
    revisions = {(r["language"], r["model"]): r["revision"] for r in models.get("models", [])
                 if r.get("language") and r.get("model") and r.get("revision")}
    for candidate in sentiment.candidates_from_config(cfg):   # как в шаге 5b: ревизия из конфигурации главнее
        if candidate.revision:
            revisions[(candidate.language, candidate.name)] = candidate.revision
    out = []
    for lang in LANGUAGES:
        chosen = choice[lang]["slug"]
        urls = annotated_urls(lang)
        for candidate in [c for c in sentiment.candidates_from_config(cfg) if c.language == lang]:
            revision = revisions.get((lang, candidate.name))
            if revision is None:
                log.warning("[%s] нет ревизии %s в sentiment_models.json — оценки этой модели не войдут", lang,
                            candidate.name)
                continue
            slug = sentiment.model_slug(candidate.name, revision)
            folder = data_dir / "interim" / "sentiment" / lang
            path = folder / f"{slug}.parquet"
            if not path.exists():
                log.warning("Нет %s — оценки этой модели не войдут", path)
                continue
            scores = pd.read_parquet(path)
            if slug != chosen:
                scores = scores[scores["url"].isin(urls)].reset_index(drop=True)
            out.append((f"data/interim/sentiment/{lang}/{slug}.parquet", scores, path))
            meta = folder / f"{slug}.json"
            if meta.exists():
                out.append((f"data/interim/sentiment/{lang}/{slug}.json", None, meta))
    return out


def parquet_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def main() -> None:
    args = parse_args(__doc__)
    cfg = load_config(args.config)
    log = get_logger()
    data_dir, results_dir = cfg["paths"]["data_dir"], cfg["paths"]["results_dir"]
    if not (results_dir / "report" / "tables.xlsx").exists():
        raise SystemExit("Нет results/report/tables.xlsx — сначала выполните шаг 9")

    entries = []                                    # (путь в архиве, байты)
    for path in result_files(results_dir):
        entries.append(("results/" + path.relative_to(results_dir).as_posix(), path.read_bytes()))
    for lang in LANGUAGES:
        news = news_without_titles(data_dir, lang)
        entries.append((f"data/interim/news_{lang}.parquet", parquet_bytes(news)))
        log.info("[%s] новости без заголовков: %d записей, колонки: %s", lang, len(news), ", ".join(news.columns))
    for name, frame, source in sentiment_files(cfg, data_dir, results_dir, log):
        entries.append((name, parquet_bytes(frame) if frame is not None else source.read_bytes()))

    listing = [{"path": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
               for name, content in entries]
    manifest = {"manifest": run_manifest(cfg), "files": listing,
                "note": "Заголовки новостей не включены (data/README.md); шаг 2 собирает их заново по адресам."}
    out_path = results_dir / "release_package.zip"
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in entries:
            archive.writestr(name, content)
        archive.writestr("release_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    total = sum(item["bytes"] for item in listing)
    log.info("Архив: %s — файлов %d, %.1f МБ до сжатия, %.1f МБ в архиве", out_path, len(listing), total / 2 ** 20,
             out_path.stat().st_size / 2 ** 20)


if __name__ == "__main__":
    main()
