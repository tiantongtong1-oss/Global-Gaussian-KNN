#!/usr/bin/env bash
set -euo pipefail
if (( $# < 2 )); then
  echo "Usage: bash scripts/run_v7.sh RAF_ROOT FER_ROOT [extra train.py arguments]" >&2
  exit 2
fi
source_path=$(realpath "$1")
target_path=$(realpath "$2")
shift 2
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs
python -u train.py --data1 rafdb --data2 fer \
  --source_path "$source_path" --target_path "$target_path" \
  --backbone mobilenet_v2 --pre_epochs 30 --epochs 30 "$@" \
  2>&1 | tee "logs/v7_$(date +%Y%m%d_%H%M%S).log"
