#!/usr/bin/env bash
# Create .venv (Python 3.12) and install hash-pinned dependencies for one or more profiles.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/setup.sh [PROFILE...]

Profiles (default: dev):
  serve   inference API (onnxruntime, fastapi) - no torch
  ui      Streamlit demo UI (streamlit, httpx)
  train   data pipeline + training (torch, timm) + serve
  dev     everything above + lint/test/security tools

Every package is installed from requirements/<profile>.txt with --require-hashes.
Uses uv when available (fast), otherwise python3.12 -m venv + pip.
EOF
}

case "${1:-}" in -h|--help) usage; exit 0 ;; esac
source "$(dirname "$0")/_common.sh"

profiles=("$@")
[ ${#profiles[@]} -eq 0 ] && profiles=(dev)

lockfiles=()
install_package=false
for profile in "${profiles[@]}"; do
  case "$profile" in
    serve) lockfiles+=(requirements/serve.txt); install_package=true ;;
    ui)    lockfiles+=(requirements/ui.txt) ;;
    train) lockfiles+=(requirements/train.txt requirements/serve.txt); install_package=true ;;
    dev)   lockfiles+=(requirements/train.txt requirements/serve.txt requirements/ui.txt requirements/dev.txt)
           install_package=true ;;
    *) usage; die "unknown profile '$profile'" ;;
  esac
done
# De-duplicate while keeping order.
mapfile -t lockfiles < <(printf '%s\n' "${lockfiles[@]}" | awk '!seen[$0]++')

if command -v uv >/dev/null 2>&1; then
  step "uv: creating .venv (Python 3.12)"
  uv venv --python 3.12 --allow-existing .venv
  step "uv: installing ${lockfiles[*]} (hashes required)"
  uv pip sync --require-hashes "${lockfiles[@]}"
  if $install_package; then uv pip install --no-deps -e .; fi
else
  command -v python3.12 >/dev/null 2>&1 || die "need uv or python3.12 on PATH"
  step "venv: creating .venv with python3.12"
  python3.12 -m venv .venv
  source "$(dirname "$0")/_common.sh"   # re-detect bin/ vs Scripts/
  step "pip: installing ${lockfiles[*]} (hashes required)"
  args=()
  for f in "${lockfiles[@]}"; do args+=(-r "$f"); done
  "$PY" -m pip install --require-hashes --no-deps "${args[@]}"
  if $install_package; then "$PY" -m pip install --no-deps -e .; fi
fi

step "done. Activate with: source $VENV_BIN/activate   (PowerShell: .venv\\Scripts\\Activate.ps1)"
