#!/usr/bin/env bash
# Shared helpers, sourced by the other scripts (not run directly).

# Run from the repository root no matter where the script was called from.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# The venv keeps executables in bin/ (Linux/macOS) or Scripts/ (Windows).
if [ -d .venv/Scripts ]; then VENV_BIN=".venv/Scripts"; else VENV_BIN=".venv/bin"; fi
PY="$VENV_BIN/python"

die() { echo "error: $*" >&2; exit 1; }
step() { printf '\n==> %s\n' "$*"; }

require_venv() {
  [ -x "$PY" ] || [ -x "$PY.exe" ] || die "no .venv found; run: bash scripts/setup.sh <profile>"
}

# Load KEY=VALUE pairs from .env (if present) without overriding variables already set.
load_dotenv() {
  [ -f .env ] || return 0
  while IFS='=' read -r key value || [ -n "$key" ]; do
    key="${key%$'\r'}"
    value="${value%$'\r'}"
    case "$key" in ''|\#*) continue ;; esac
    if [ -z "${!key:-}" ]; then export "$key=$value"; fi
  done < .env
}
