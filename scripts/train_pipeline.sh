#!/usr/bin/env bash
# Model pipeline: train -> calibrate (val) -> evaluate (test, ONCE) -> export -> benchmark.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/train_pipeline.sh [--config configs/train.yaml] [--skip-train]

  --config      project config (default: configs/train.yaml)
  --skip-train  reuse runs/LATEST instead of training a new model

Run scripts/prepare_data.sh first. Training takes ~3 h on a 4-core CPU.
`evaluate` refuses to run twice on the same run (the test set is used once).
Outputs: runs/<stamp>_<config-hash>/, models/<version>/, reports/.
EOF
}

config="configs/train.yaml"
skip_train=false
while [ $# -gt 0 ]; do
  case "$1" in
    --config) config="$2"; shift 2 ;;
    --skip-train) skip_train=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage; echo "error: unknown argument '$1'" >&2; exit 2 ;;
  esac
done

source "$(dirname "$0")/_common.sh"
require_venv
[ -f data/processed/leakage_audit.json ] || die "no leakage audit found; run scripts/prepare_data.sh first"

if $skip_train; then
  step "1/5 train: skipped (using runs/LATEST)"
else
  step "1/5 train (two-stage fine-tuning)"
  "$PY" -m defect_detection train --config "$config"
fi
step "2/5 calibrate on validation (temperature, threshold, review band)"
"$PY" -m defect_detection calibrate --config "$config"
step "3/5 evaluate ONCE on test + external test"
"$PY" -m defect_detection evaluate --config "$config"
step "4/5 export to models/<version> (checker + parity + SHA256SUMS)"
"$PY" -m defect_detection export --config "$config"
step "5/5 benchmark (CPU latency/throughput)"
"$PY" -m defect_detection benchmark --config "$config"
step "done: see reports/metrics.json and reports/benchmark.md"
