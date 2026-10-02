"""Служебные функции: зёрна, логирование, хеши файлов, манифест запуска (задача 0.6)."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

KEY_PACKAGES = ("numpy", "pandas", "pyarrow", "pyyaml", "requests", "beautifulsoup4", "python-binance",
                "scikit-learn", "xgboost", "statsmodels", "arch", "torch", "transformers", "openpyxl", "matplotlib")


def set_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Фиксирует зёрна Python, NumPy и PyTorch (если он установлен).

    PYTHONHASHSEED влияет только на дочерние процессы; для текущего процесса
    его нужно задавать до запуска интерпретатора, если важен порядок в множествах.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_logger(name: str = "cryptonews", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def sha256_file(path: str | os.PathLike, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    """Хеш текущего коммита или None, если код запущен не из git-репозитория."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_is_dirty() -> bool | None:
    try:
        out = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)
        return bool(out.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def package_versions(names=KEY_PACKAGES) -> dict[str, str]:
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "не установлен"
    return versions


def run_manifest(cfg: dict[str, Any], inputs: list[str | os.PathLike] | None = None) -> dict[str, Any]:
    """Сведения о запуске: время, коммит, версии, хеши конфигурации и входных файлов."""
    cfg_path = cfg.get("_config_path")
    return {
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": git_commit(),
        "git_dirty": git_is_dirty(),
        "config_path": cfg_path,
        "config_sha256": sha256_file(cfg_path) if cfg_path else None,
        "protocol_version": cfg.get("project", {}).get("protocol_version"),
        "packages": package_versions(),
        "inputs": {str(p): sha256_file(p) for p in (inputs or [])},
    }


def save_json(obj: Any, path: str | os.PathLike) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    return path
