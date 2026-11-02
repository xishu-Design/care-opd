#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -x "$ROOT/runtime/ollama/bin/ollama" ]]; then
  OLLAMA_BIN="${OLLAMA_BIN:-$ROOT/runtime/ollama/bin/ollama}"
else
  OLLAMA_BIN="${OLLAMA_BIN:-ollama}"
fi

exec env OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-32768}" OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}" "$OLLAMA_BIN" serve
