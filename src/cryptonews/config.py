"""Загрузка конфигурации и путей проекта (задача 0.6).

Все параметры эксперимента лежат в configs/config.yaml. Пути к данным и результатам
можно переопределить переменными окружения CRYPTONEWS_DATA_DIR и CRYPTONEWS_RESULTS_DIR,
например чтобы в Colab держать данные на Google Drive.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(os.environ.get("CRYPTONEWS_ROOT", Path(__file__).resolve().parents[2]))
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"

REQUIRED_SECTIONS = ("paths", "time", "data", "sentiment", "features",
                     "validation", "seeds", "models", "evaluation", "diagnostics")


def _resolve(path: str | os.PathLike) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Читает YAML-конфигурацию, проверяет обязательные разделы и превращает пути в Path."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise KeyError(f"В конфигурации {cfg_path} нет разделов: {', '.join(missing)}")

    paths = cfg["paths"]
    paths["data_dir"] = _resolve(os.environ.get("CRYPTONEWS_DATA_DIR", paths.get("data_dir", "data")))
    paths["results_dir"] = _resolve(os.environ.get("CRYPTONEWS_RESULTS_DIR", paths.get("results_dir", "results")))
    cfg["_config_path"] = str(cfg_path.resolve())
    return cfg
