"""Загрузка конфигурации и путей проекта (задача 0.6).

Все параметры эксперимента лежат в configs/config.yaml. Пути к данным и результатам
можно переопределить переменными окружения CRYPTONEWS_DATA_DIR и CRYPTONEWS_RESULTS_DIR,
например чтобы в Colab держать данные на Google Drive.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(os.environ.get("CRYPTONEWS_ROOT", Path(__file__).resolve().parents[2]))
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"

REQUIRED_SECTIONS = ("paths", "time", "data", "sentiment", "features",
                     "validation", "seeds", "models", "evaluation", "diagnostics")

# Стандартные папки проекта. Создаются при загрузке конфигурации, чтобы любой скрипт
# работал на чистой машине и в Colab, где папки для результатов ещё не существуют.
DATA_SUBDIRS = ("raw/binance", "raw/cryptovision", "raw/ru_news", "interim", "processed")
RESULTS_SUBDIRS = ("env", "predictions", "metrics", "tables", "figures")


def _resolve(path: str | os.PathLike) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def running_in_colab() -> bool:
    return importlib.util.find_spec("google.colab") is not None


def check_colab_paths() -> None:
    """В Colab данные и результаты обязаны лежать на Google Диске.

    Иначе после перезапуска Colab скрипт молча начал бы писать во временную папку,
    которая стирается вместе со средой: например, заново собирал бы новости час
    в никуда. Проверку можно отключить переменной CRYPTONEWS_ALLOW_LOCAL=1.
    """
    if not running_in_colab() or os.environ.get("CRYPTONEWS_ALLOW_LOCAL") == "1":
        return
    missing = [v for v in ("CRYPTONEWS_DATA_DIR", "CRYPTONEWS_RESULTS_DIR") if not os.environ.get(v)]
    if missing:
        raise SystemExit(
            "Не заданы пути к Google Диску (" + ", ".join(missing) + "). "
            "Скорее всего, Colab перезапустился. Выполните ячейки 1–3 ноутбука, потом этот шаг."
        )


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Читает YAML-конфигурацию, проверяет обязательные разделы и превращает пути в Path."""
    check_colab_paths()
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise KeyError(f"В конфигурации {cfg_path} нет разделов: {', '.join(missing)}")

    paths = cfg["paths"]
    paths["data_dir"] = _resolve(os.environ.get("CRYPTONEWS_DATA_DIR") or paths.get("data_dir", "data"))
    paths["results_dir"] = _resolve(os.environ.get("CRYPTONEWS_RESULTS_DIR") or paths.get("results_dir", "results"))
    cfg["_config_path"] = str(cfg_path.resolve())
    ensure_dirs(cfg)
    return cfg


def ensure_dirs(cfg: dict[str, Any]) -> None:
    """Создаёт стандартные папки данных и результатов, если их ещё нет."""
    for sub in DATA_SUBDIRS:
        (cfg["paths"]["data_dir"] / sub).mkdir(parents=True, exist_ok=True)
    for sub in RESULTS_SUBDIRS:
        (cfg["paths"]["results_dir"] / sub).mkdir(parents=True, exist_ok=True)
