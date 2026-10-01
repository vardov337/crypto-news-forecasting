#!/usr/bin/env bash
# Полный прогон пайплайна. Все таблицы и рисунки статьи получаются отсюда.
# Использование: bash run_all.sh [путь_к_конфигу]
set -euo pipefail
CONFIG="${1:-configs/config.yaml}"
for script in scripts/0*.py; do
  echo ">>> ${script}"
  python "${script}" --config "${CONFIG}"
done
