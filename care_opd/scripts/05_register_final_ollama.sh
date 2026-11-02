#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

FINAL_MODEL="$(qzbc_final_model_path)"
require_file "${FINAL_MODEL}/config.json"
require_file "${FINAL_MODEL}/export_manifest.json"

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:?set LLAMA_CPP_DIR to a current llama.cpp checkout}"
CONVERTER="${LLAMA_CPP_CONVERT:-${LLAMA_CPP_DIR}/convert_hf_to_gguf.py}"
GGUF_OUTTYPE="${QZBC_GGUF_OUTTYPE:-q8_0}"
BUNDLE_ROOT="${QZBC_RUN_ROOT}/ollama"
GGUF_PATH="${BUNDLE_ROOT}/qwen3.5-4b-care-opd-${GGUF_OUTTYPE}.gguf"
MODELFILE="${BUNDLE_ROOT}/Modelfile"
ENV_OVERLAY="${BUNDLE_ROOT}/qzbc-trained-model.env"
mkdir -p -- "${BUNDLE_ROOT}"

if [[ ! -s "${GGUF_PATH}" ]]; then
  "${PYTHON_BIN}" "${CONVERTER}" "${FINAL_MODEL}" \
    --outfile "${GGUF_PATH}" \
    --outtype "${GGUF_OUTTYPE}"
fi
test -s "${GGUF_PATH}"

{
  printf 'FROM %s\n' "${GGUF_PATH}"
  printf 'PARAMETER num_ctx 32768\n'
  printf 'PARAMETER temperature 0.2\n'
  printf 'PARAMETER seed 42\n'
} >"${MODELFILE}"

ollama create "${TRAINED_OLLAMA_MODEL}" -f "${MODELFILE}"
run_module care_opd.qzbc_bridge check-ollama \
  --ollama-base-url "${OLLAMA_BASE_URL}" \
  --model "${BASE_OLLAMA_MODEL}" \
  --model "${TRAINED_OLLAMA_MODEL}"

{
  printf 'LLM_PROVIDER=ollama\n'
  printf 'OLLAMA_BASE_URL=%s\n' "${OLLAMA_BASE_URL}"
  printf 'OLLAMA_MODEL=%s\n' "${TRAINED_OLLAMA_MODEL}"
  printf 'LLM_ALLOW_FALLBACK=false\n'
} >"${ENV_OVERLAY}"

printf 'Registered model: %s\n' "${TRAINED_OLLAMA_MODEL}"
printf 'qzbc env overlay: %s\n' "${ENV_OVERLAY}"
