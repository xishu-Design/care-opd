#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

printf '[pipeline] stage=validate status=start\n'
bash "${SCRIPT_DIR}/00_validate.sh"
printf '[pipeline] stage=validate status=complete\n'
printf '[pipeline] stage=data status=start\n'
bash "${SCRIPT_DIR}/01_build_formal_data.sh"
printf '[pipeline] stage=data status=complete\n'
if [[ "${RUN_BASELINES:-0}" == "1" ]]; then
  printf '[pipeline] stage=baselines status=start\n'
  bash "${SCRIPT_DIR}/02_train_baselines.sh"
  printf '[pipeline] stage=baselines status=complete\n'
fi
printf '[pipeline] stage=care_opd status=start\n'
bash "${SCRIPT_DIR}/03_train_care_opd.sh"
printf '[pipeline] stage=care_opd status=complete\n'
