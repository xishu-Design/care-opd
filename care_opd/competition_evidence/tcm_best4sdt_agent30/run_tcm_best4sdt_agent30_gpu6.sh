#!/usr/bin/env bash
set -Eeuo pipefail

# GPU-6 entrypoint. The shared implementation remains in the legacy-named
# runner so existing frozen bundles stay compatible.
export PHYSICAL_GPUS="${PHYSICAL_GPUS:-6}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_tcm_best4sdt_agent30_gpu3.sh" "$@"
