#!/usr/bin/env bash
# Deployment helper only. Do not put this file inside the frozen runtime/ folder.
# Run with bash, never source into an interactive terminal.
set -euo pipefail
umask 077

PHASE=${1:-status}
BOUNDARY_BASE=/opt/care-workspace
PKG="$BOUNDARY_BASE/care_opd_multimedia_boundary_v1"
RUNNER="$PKG/run_boundary_gpu1.sh"
CONTROL="$BOUNDARY_BASE/care_opd_boundary_gpu1_control"
CACHE="$CONTROL/cache"

# Deliberately override stale variables from previous experiments.
export WORK="$BOUNDARY_BASE/care_opd_multimedia_boundary_gpu1_v1_diskfix"
export DATA_ROOT="$BOUNDARY_BASE/care_opd_multimedia_boundary_chartqa_val_v1"
export DATA_PY="$BOUNDARY_BASE/venvs/care_boundary_data/bin/python"
export HF_PY=/opt/care-data/venvs/mhg_eval_qwen25vl/bin/python
export PHYSICAL_GPU=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
export TMPDIR="$CACHE/tmp" TMP="$CACHE/tmp" TEMP="$CACHE/tmp"
export XDG_CACHE_HOME="$CACHE/xdg"
export HF_HOME="$CACHE/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets" HF_MODULES_CACHE="$HF_HOME/modules"
export HF_XET_CACHE="$HF_HOME/xet" TRANSFORMERS_CACHE="$HF_HOME/hub"
export TORCH_HOME="$CACHE/torch" TORCH_EXTENSIONS_DIR="$CACHE/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/torchinductor" TRITON_CACHE_DIR="$CACHE/triton"
export CUDA_CACHE_PATH="$CACHE/cuda"
export HF_HUB_DISABLE_XET=1
export IGNORE_PROXY=${IGNORE_PROXY:-1}
MIN_FREE_GIB=${MIN_FREE_GIB:-8}

show_paths() {
  printf 'PHYSICAL_GPU=%s\nPACKAGE=%s\nWORK=%s\nDATA_ROOT=%s\nDATA_PY=%s\nHF_PY=%s\nCACHE=%s\nCONTROL=%s\n' \
    "$PHYSICAL_GPU" "$PKG" "$WORK" "$DATA_ROOT" "$DATA_PY" "$HF_PY" "$CACHE" "$CONTROL"
}

case "$PHASE" in
  print-env) show_paths; exit 0 ;;
  all|setup|run|resume|status|score|preflight|smoke) ;;
  *) echo "Usage: bash $0 {all|setup|run|resume|status|score|preflight|smoke|print-env}" >&2; exit 2 ;;
esac

test -f "$RUNNER" || { echo "Missing original package: $RUNNER" >&2; exit 2; }
test -x "$HF_PY" || { echo "Missing proven GPU Python: $HF_PY" >&2; exit 2; }
[[ "$MIN_FREE_GIB" =~ ^[1-9][0-9]*$ ]] || { echo 'MIN_FREE_GIB must be positive integer' >&2; exit 2; }

# Refuse symlinked writable paths resolving back onto the full /data tree.
for target in "$WORK" "$DATA_ROOT" "$CONTROL" "$BOUNDARY_BASE/venvs/care_boundary_data" \
  "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" \
  "$HF_MODULES_CACHE" "$HF_XET_CACHE" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"; do
  resolved=$(readlink -m -- "$target")
  case "$resolved" in
    /data|/data/*) echo "Writable path resolves to full /data: $target -> $resolved" >&2; exit 2 ;;
  esac
done

check_space() {
  local available free_inodes
  available=$(df -Pk "$BOUNDARY_BASE" | awk 'NR==2 {print $4}')
  free_inodes=$(df -Pi "$BOUNDARY_BASE" | awk 'NR==2 {print $4}')
  [[ "$available" =~ ^[0-9]+$ ]] || { echo 'Cannot read free space' >&2; return 2; }
  [[ "$free_inodes" =~ ^[0-9]+$ ]] || { echo 'Cannot read free inodes' >&2; return 2; }
  echo "ROOT_AVAILABLE_KIB=$available MIN_FREE_GIB=$MIN_FREE_GIB FREE_INODES=$free_inodes"
  if (( available < MIN_FREE_GIB * 1024 * 1024 )); then
    echo 'Root filesystem below free-space reserve; stopped without deleting files.' >&2
    return 2
  fi
  if (( free_inodes < 10000 )); then
    echo 'Root filesystem has fewer than 10000 free inodes; stopped without deleting files.' >&2
    return 2
  fi
}

# status is read-only and remains available even below the free-space reserve.
if [[ "$PHASE" == status ]]; then
  show_paths
  if ! test -s "$WORK/selection_manifest.json"; then
    echo 'DATA_PREPARATION_NOT_COMPLETE: inspect the launcher log; generation has not been validated.'
    exit 0
  fi
  exec bash "$RUNNER" status
fi

check_space
# Never pre-create WORK or DATA_ROOT: prepare/fetch require new empty roots.
mkdir -p "$CONTROL" "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HUB_CACHE" \
  "$HF_DATASETS_CACHE" "$HF_MODULES_CACHE" "$HF_XET_CACHE" "$TORCH_HOME" \
  "$TORCH_EXTENSIONS_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
command -v flock >/dev/null || { echo 'flock not found; cannot protect against duplicate launchers' >&2; exit 2; }
exec 9>"$CONTROL/launcher.lock"
flock -n 9 || { echo 'This root-directory experiment already has an active launcher; not starting a duplicate.' >&2; exit 2; }
show_paths
df -h "$BOUNDARY_BASE"
"$HF_PY" -B "$PKG/check_package.py"

bootstrap_data_environment() {
  local data_env="$BOUNDARY_BASE/venvs/care_boundary_data"
  local base_python=/opt/care-workspace/opdrevi_venv/bin/python
  if ! test -x "$base_python"; then base_python="$HF_PY"; fi
  if ! test -x "$DATA_PY"; then
    mkdir -p "$BOUNDARY_BASE/venvs"
    "$base_python" -B -m venv "$data_env"
  fi
  if ! "$DATA_PY" -B -c 'import PIL, pyarrow, huggingface_hub' >/dev/null 2>&1; then
    local pip_env=(env)
    if [[ "$IGNORE_PROXY" == 1 ]]; then
      pip_env+=(-u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy)
    fi
    "${pip_env[@]}" "$DATA_PY" -B -m pip install --no-cache-dir -r "$PKG/requirements_data.txt"
  fi
  "$DATA_PY" -B -c 'import PIL, pyarrow, huggingface_hub; print("DATA_STACK_OK", PIL.__version__, pyarrow.__version__, huggingface_hub.__version__)'
}

setup_experiment() {
  bootstrap_data_environment
  bash "$RUNNER" selftest
  check_space
  bash "$RUNNER" prepare
  check_space
  bash "$RUNNER" init
  bash "$RUNNER" preflight
  echo 'ROOT_SETUP_AND_PREFLIGHT_OK'
}

case "$PHASE" in
  setup) setup_experiment ;;
  all) setup_experiment; check_space; bash "$RUNNER" run ;;
  run|resume) bash "$RUNNER" resume ;;
  score|preflight|smoke) bash "$RUNNER" "$PHASE" ;;
esac
echo "ROOT_LAUNCHER_PHASE_COMPLETE=$PHASE"
