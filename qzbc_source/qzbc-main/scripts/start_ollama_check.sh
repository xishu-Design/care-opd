#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -x "$ROOT/runtime/ollama/bin/ollama" ]]; then
  DEFAULT_OLLAMA_BIN="$ROOT/runtime/ollama/bin/ollama"
else
  DEFAULT_OLLAMA_BIN="ollama"
fi
OLLAMA_BIN="${OLLAMA_BIN:-$DEFAULT_OLLAMA_BIN}"

if ! command -v "$OLLAMA_BIN" >/dev/null 2>&1; then
  printf '未找到 Ollama。\n' >&2
  exit 1
fi
if ! "$OLLAMA_BIN" list >/dev/null 2>&1; then
  printf 'Ollama 服务未运行，请先执行：%s/scripts/start_ollama.sh\n' "$ROOT" >&2
  exit 1
fi
for model in qwen2.5:7b bge-m3; do
  if ! "$OLLAMA_BIN" list | awk 'NR > 1 {print $1}' | grep -qx "${model}\|${model}:latest"; then
    printf '缺少模型：%s\n' "$model" >&2
    exit 1
  fi
done
printf 'Ollama 与两个模型均已就绪。\n'
