#!/usr/bin/env bash
# Smoke test a running API: health, readiness, auth, model info, one prediction.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/smoke_test.sh [--url http://127.0.0.1:8000] [--image PATH]

  --url    API base URL (default: http://127.0.0.1:8000)
  --image  image to predict (default: first .jpeg under data/raw)

The API key is taken from $API_KEY, else the first key in DD_API_KEYS (shell or .env).
Exits non-zero on the first failed check.
EOF
}

url="http://127.0.0.1:8000"
image=""
while [ $# -gt 0 ]; do
  case "$1" in
    --url) url="$2"; shift 2 ;;
    --image) image="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; echo "error: unknown argument '$1'" >&2; exit 2 ;;
  esac
done

source "$(dirname "$0")/_common.sh"
load_dotenv
key="${API_KEY:-${DD_API_KEYS:-}}"
key="${key%%,*}"
[ -n "$key" ] || die "no API key: set API_KEY or DD_API_KEYS"
if [ -z "$image" ]; then
  image="$(find data/raw -name '*.jpeg' -print -quit 2>/dev/null || true)"
fi
[ -f "$image" ] || die "no image to send; pass --image PATH"

check() {  # name, expected status, curl args...
  local name="$1" expected="$2"; shift 2
  local status
  status="$(curl -s -o /dev/null -w '%{http_code}' "$@")"
  if [ "$status" = "$expected" ]; then echo "PASS  $name ($status)"; else echo "FAIL  $name (got $status, want $expected)"; exit 1; fi
}

check "GET /health"               200 "$url/health"
check "GET /ready"                200 "$url/ready"
check "POST /v1/predict no key"   401 -F "file=@$image" "$url/v1/predict"
check "GET /v1/model"             200 -H "X-API-Key: $key" "$url/v1/model"
check "POST /v1/predict"          200 -H "X-API-Key: $key" -F "file=@$image" "$url/v1/predict"
echo
curl -s -H "X-API-Key: $key" -F "file=@$image" "$url/v1/predict"
echo
