#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_DIR="$ROOT/services/lightrag"
VENV_DIR="$SERVICE_DIR/.venv"

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/pip" install --requirement "$SERVICE_DIR/requirements.lock"

if [[ ! -f "$SERVICE_DIR/.env" ]]; then
  cp "$SERVICE_DIR/.env.example" "$SERVICE_DIR/.env"
fi

mkdir -p "$ROOT/runtime/lightrag"/{staging,active,failed,input,reports}
printf '本地 AI 依赖已准备完成。请检查 %s 后启动服务。\n' "$SERVICE_DIR/.env"
