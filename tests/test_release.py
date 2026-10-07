"""Шаг 10: архив для релиза — без заголовков новостей, с оценками тональности для шагов 5b–6."""
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

from cryptonews import sentiment
from cryptonews.config import load_config

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("release_package", ROOT / "scripts" / "10_release_package.py")
step = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_release_package(tmp_path, monkeypatch):
    data, results = tmp_path / "data", tmp_path / "results"
    monkeypatch.setenv("CRYPTONEWS_DATA_DIR", str(data))
    monkeypatch.setenv("CRYPTONEWS_RESULTS_DIR", str(results))
    monkeypatch.setattr(sys, "argv", ["10_release_package.py"])
    cfg = load_config(None)
    write(results / "report" / "tables.xlsx", "xlsx")
    write(results / "report" / "fig1_scheme.png", "png")
    write(results / "tables" / "eval_accuracy_BTCUSDT.csv", "key\n")
    write(results / "splits.json", "{}")
    write(results / "env" / "manifest.json", "{}")
    write(results / "predictions" / "BTCUSDT" / "xgboost__P" / "tuning.json", "{}")
    write(results / "predictions" / "BTCUSDT" / "xgboost__P" / "fold_1__seed0.json", "{}")   # в архив не идёт
    pd.DataFrame({"y_true": [0.1]}).to_parquet(results / "predictions_combined_BTCUSDT.parquet")

    candidates = sentiment.candidates_from_config(cfg)
    models, choice = [], {}
    for lang in ("en", "ru"):
        labelled = pd.read_csv(ROOT / "annotation" / "labels" / f"{lang}_annotator1.csv")["url"].tolist()[:2]
        urls = labelled + [f"https://example.com/{lang}/{i}" for i in range(3)]
        news = pd.DataFrame({"published_utc": pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC"),
                             "title": [f"секретный заголовок {i}" for i in range(5)], "url": urls,
                             "source": "forklog.com", "mentions_btc": True, "mentions_eth": False})
        (data / "interim" / "sentiment" / lang).mkdir(parents=True, exist_ok=True)
        news.to_parquet(data / "interim" / f"news_{lang}.parquet")
        for i, candidate in enumerate(c for c in candidates if c.language == lang):
            revision = candidate.revision or f"{i}" * 40
            models.append({"language": lang, "model": candidate.name, "revision": revision})
            slug = sentiment.model_slug(candidate.name, revision)
            pd.DataFrame({"url": urls, "label": "neutral", "p_negative": 0.2, "p_neutral": 0.6,
                          "p_positive": 0.2}).to_parquet(data / "interim" / "sentiment" / lang / f"{slug}.parquet")
            write(data / "interim" / "sentiment" / lang / f"{slug}.json", "{}")
            if i == 0:
                choice[lang] = {"slug": slug}
    write(results / "metrics" / "sentiment_models.json", json.dumps({"models": models}))
    write(results / "metrics" / "sentiment_choice.json", json.dumps(choice))

    step.main()
    with zipfile.ZipFile(results / "release_package.zip") as archive:
        names = set(archive.namelist())
        assert "results/report/tables.xlsx" in names and "results/predictions_combined_BTCUSDT.parquet" in names
        assert "results/predictions/BTCUSDT/xgboost__P/tuning.json" in names
        assert not any("fold_1__seed0" in name for name in names)
        for lang in ("en", "ru"):
            news = pd.read_parquet(io.BytesIO(archive.read(f"data/interim/news_{lang}.parquet")))
            assert "title" not in news.columns and len(news) == 5
            files = sorted(n for n in names if n.startswith(f"data/interim/sentiment/{lang}/") and n.endswith(".parquet"))
            assert len(files) == len([c for c in candidates if c.language == lang])
            sizes = {name: len(pd.read_parquet(io.BytesIO(archive.read(name)))) for name in files}
            chosen = f"data/interim/sentiment/{lang}/{choice[lang]['slug']}.parquet"
            assert sizes.pop(chosen) == 5 and set(sizes.values()) == {2}   # остальные — только размеченные
        manifest = json.loads(archive.read("release_manifest.json"))
        assert {item["path"] for item in manifest["files"]} == names - {"release_manifest.json"}
        assert b"\xd1\x81\xd0\xb5\xd0\xba\xd1\x80\xd0\xb5\xd1\x82" not in b"".join(archive.read(n) for n in names)
