#!/usr/bin/env bash
# Data pipeline: inspect -> manifest -> split (dedupe, group-aware) -> leakage audit.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/prepare_data.sh [--config configs/train.yaml] [--data-dir DIR]

  --config    project config (default: configs/train.yaml)
  --data-dir  raw dataset folder to inspect; must equal data.raw_dir in the config
              (default: the config's data.raw_dir)

Writes data/processed/{manifest.csv, splits.csv, split_report.json, leakage_audit.json}
and reports/inventory.json. Exits non-zero if the leakage audit fails.
EOF
}

config="configs/train.yaml"
data_dir=""
while [ $# -gt 0 ]; do
  case "$1" in
    --config) config="$2"; shift 2 ;;
    --data-dir) data_dir="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; echo "error: unknown argument '$1'" >&2; exit 2 ;;
  esac
done

source "$(dirname "$0")/_common.sh"
require_venv
[ -f "$config" ] || die "config not found: $config"

raw_dir="$("$PY" -c "import sys; from pathlib import Path; from defect_detection.config import load_config; print(load_config(Path(sys.argv[1])).data.raw_dir.as_posix())" "$config")"
if [ -n "$data_dir" ] && [ "$(cd "$data_dir" && pwd)" != "$(cd "$raw_dir" && pwd)" ]; then
  die "--data-dir '$data_dir' differs from data.raw_dir '$raw_dir' in $config; edit the config"
fi

step "1/4 inspect $raw_dir"
"$PY" -m defect_detection inspect "$raw_dir" --json-out reports/inventory.json
step "2/4 manifest (SHA-256 per image)"
"$PY" -m defect_detection manifest --config "$config"
step "3/4 split (exact + same-part dedupe, group-aware, stratified)"
"$PY" -m defect_detection split --config "$config"
step "4/4 leakage audit"
"$PY" -m defect_detection audit --config "$config"
