#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

if [[ ! -x "$VENV_DIR/bin/lightrag-server" ]]; then
  printf 'LightRAG 尚未安装。请先运行 scripts/setup_local_ai.sh。\n' >&2
  exit 1
fi

if [[ -f "$SCRIPT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/.env"
  set +a
fi

export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-9621}"
if [[ -n "${WORKING_DIR:-}" && "$WORKING_DIR" != /* ]]; then
  WORKING_DIR="$SCRIPT_DIR/$WORKING_DIR"
fi
if [[ -n "${INPUT_DIR:-}" && "$INPUT_DIR" != /* ]]; then
  INPUT_DIR="$SCRIPT_DIR/$INPUT_DIR"
fi
export WORKING_DIR="${WORKING_DIR:-$PROJECT_ROOT/runtime/lightrag/active}"
export INPUT_DIR="${INPUT_DIR:-$PROJECT_ROOT/runtime/lightrag/input}"
mkdir -p "$WORKING_DIR" "$INPUT_DIR"

cd "$SCRIPT_DIR"
exec "$VENV_DIR/bin/lightrag-server" --host "$HOST" --port "$PORT"
