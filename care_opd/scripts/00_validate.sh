#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

require_dir "${QZBC_ROOT}/backend/app/llm"
require_file "${CARE_CONFIG}"

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m compileall -q src/care_opd
"${PYTHON_BIN}" - <<'PY'
import json
import os
from care_opd.qzbc_contract import (
    assert_qzbc_message_parity,
    qzbc_contract_manifest,
)

manifest = qzbc_contract_manifest(os.environ["QZBC_ROOT"])
parity = assert_qzbc_message_parity(os.environ["QZBC_ROOT"])
print(json.dumps({
    "status": "ok",
    "model_boundary": manifest["model_boundary"],
    "model_routes_tools": manifest["model_routes_tools"],
    "contract_fingerprint": manifest["contract_fingerprint"],
    "message_parity": parity,
}, ensure_ascii=False))
PY
