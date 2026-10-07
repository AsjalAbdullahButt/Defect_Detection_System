#!/usr/bin/env bash
# Start the inference API locally (reads .env if present).
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/run_api.sh [--port 8000] [--host 127.0.0.1]

Environment (from the shell or .env; see .env.example):
  MODEL_VERSION   folder under models/ to serve (required unless DD_MODEL_DIR is set)
  DD_API_KEYS     comma-separated keys (required when DD_ENVIRONMENT=production)
  DD_ENVIRONMENT  production (default) | development (enables /docs)
EOF
}

port=8000
host=127.0.0.1
while [ $# -gt 0 ]; do
  case "$1" in
    --port) port="$2"; shift 2 ;;
    --host) host="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; echo "error: unknown argument '$1'" >&2; exit 2 ;;
  esac
done

source "$(dirname "$0")/_common.sh"
require_venv
load_dotenv
if [ -z "${DD_MODEL_DIR:-}" ]; then
  [ -n "${MODEL_VERSION:-}" ] || die "set MODEL_VERSION (or DD_MODEL_DIR) in .env"
  export DD_MODEL_DIR="models/$MODEL_VERSION"
fi
[ -d "$DD_MODEL_DIR" ] || die "model folder not found: $DD_MODEL_DIR (run scripts/train_pipeline.sh)"

step "serving $DD_MODEL_DIR on http://$host:$port"
exec "$PY" -m uvicorn --factory defect_detection.serving.main:create_app \
  --host "$host" --port "$port" --workers 1 --no-server-header
