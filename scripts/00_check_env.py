"""Шаг 0 (задачи 0.6 и 10.1): проверка окружения и фиксация точных версий пакетов.

Пишет results/env/manifest.json и results/env/requirements-lock.txt.
"""
import subprocess
import sys

from cryptonews.cli import parse_args
from cryptonews.config import load_config
from cryptonews.utils import get_logger, run_manifest, save_json


def main() -> None:
    args = parse_args(__doc__)
    cfg = load_config(args.config)
    log = get_logger()

    manifest = run_manifest(cfg)
    try:
        import torch
        manifest["cuda_available"] = torch.cuda.is_available()
        manifest["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        manifest["cuda_available"] = None
        manifest["gpu"] = None

    out_dir = cfg["paths"]["results_dir"] / "env"
    save_json(manifest, out_dir / "manifest.json")
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    (out_dir / "requirements-lock.txt").write_text(freeze, encoding="utf-8")

    log.info("Python %s, коммит %s", manifest["python"], manifest["git_commit"] or "нет (не git-репозиторий)")
    for name, version in manifest["packages"].items():
        log.info("  %-16s %s", name, version)
    log.info("GPU: %s", manifest["gpu"] or "нет")
    log.info("Данные: %s", cfg["paths"]["data_dir"])
    log.info("Результаты: %s", cfg["paths"]["results_dir"])
    missing = [n for n, v in manifest["packages"].items() if v == "не установлен"]
    if missing:
        log.warning("Не установлены: %s — выполните pip install -r requirements.txt", ", ".join(missing))


if __name__ == "__main__":
    main()
