"""Общий запуск шагов пайплайна: разбор аргументов и заглушка для ещё не реализованных шагов."""
from __future__ import annotations

import argparse
import sys

from cryptonews.config import load_config


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=None, help="путь к YAML-конфигурации (по умолчанию configs/config.yaml)")
    return parser.parse_args()


def not_implemented(doc: str) -> None:
    """Сообщает, что шаг ещё не реализован, и останавливает run_all.sh."""
    args = parse_args(doc)
    load_config(args.config)  # конфигурация проверяется даже у пустого шага
    first_line = (doc or "").strip().splitlines()[0]
    print(f"Шаг ещё не реализован: {first_line}")
    sys.exit(2)
