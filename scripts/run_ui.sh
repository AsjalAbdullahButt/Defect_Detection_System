#!/usr/bin/env bash
# Start the Streamlit demo UI locally (reads .env if present). The API must be running.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/run_ui.sh [--port 8501] [--samples-from-data]

  --port               UI port (default 8501)
  --samples-from-data  copy six demo images from data/raw into ui/assets/samples/
                       (local only: gitignored, the dataset licence forbids redistribution)

Environment (shell or .env; see .env.example):
  DD_UI_API_URL   API base URL (default http://127.0.0.1:8000)
  DD_UI_API_KEY   API key the UI sends (kept server-side, never shown in the browser)
EOF
}

port=8501
samples=false
while [ $# -gt 0 ]; do
  case "$1" in
    --port) port="$2"; shift 2 ;;
    --samples-from-data) samples=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage; echo "error: unknown argument '$1'" >&2; exit 2 ;;
  esac
done

source "$(dirname "$0")/_common.sh"
require_venv
load_dotenv

if $samples; then
  step "copying demo samples into ui/assets/samples (not committed)"
  mkdir -p ui/assets/samples
  raw="data/raw"
  for pair in \
    "normal_01:casting_data/casting_data/test/ok_front/cast_ok_0_1210.jpeg" \
    "defective_01:casting_data/casting_data/test/def_front/cast_def_0_1647.jpeg" \
    "defective_review:casting_data/casting_data/test/def_front/cast_def_0_1591.jpeg" \
    "external_defective:casting_512x512/casting_512x512/def_front/cast_def_0_9435.jpeg" \
    "external_false_alarm:casting_512x512/casting_512x512/ok_front/cast_ok_0_2819.jpeg" \
    "external_missed:casting_512x512/casting_512x512/def_front/cast_def_0_335.jpeg"; do
    name="${pair%%:*}"; src="$raw/${pair#*:}"
    [ -f "$src" ] || die "sample not found: $src (place the dataset in data/raw)"
    cp "$src" "ui/assets/samples/$name.jpeg"
  done
fi

UI_PY="$(cd "$(dirname "$PY")" && pwd)/$(basename "$PY")"
step "UI on http://127.0.0.1:$port  (API: ${DD_UI_API_URL:-http://127.0.0.1:8000})"
cd ui
exec "$UI_PY" -m streamlit run app.py --server.port "$port" --server.address 127.0.0.1
