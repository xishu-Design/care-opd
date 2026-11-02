#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if ! curl --silent --fail http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  mkdir -p "$ROOT/runtime/ollama"
  nohup "$ROOT/scripts/start_ollama.sh" >"$ROOT/runtime/ollama/ollama.log" 2>&1 &
fi

for _ in $(seq 1 30); do
  if "$ROOT/scripts/start_ollama_check.sh" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
"$ROOT/scripts/start_ollama_check.sh"

if ! curl --silent --fail http://127.0.0.1:9621/health >/dev/null 2>&1; then
  nohup "$ROOT/scripts/start_lightrag.sh" >"$ROOT/runtime/lightrag/reports/lightrag.log" 2>&1 &
fi

for _ in $(seq 1 30); do
  if "$ROOT/services/lightrag/.venv/bin/python" "$ROOT/services/lightrag/health_check.py" >/dev/null 2>&1; then
    exec "$ROOT/.conda/bin/python" "$ROOT/scripts/check_local_ai_stack.py"
  fi
  sleep 2
done
printf 'LightRAG 未能在 60 秒内通过健康检查，请查看 runtime/lightrag/reports/lightrag.log。\n' >&2
exit 1
