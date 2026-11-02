#!/usr/bin/env python3
"""Package both locked v5 direct-clinical-gold tracks for private server eval.

The resulting archive intentionally skips the annotation stage.  It contains
only the locked seed/gold rows needed by the evaluator, the exact evaluation
code, the frozen QZBC message-contract sources, and operational scripts.

This is a PRIVATE archive: the seed and gold JSONL files contain deidentified
clinical report content/fields and must never be uploaded to a public service.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import tarfile
from pathlib import Path
from typing import Sequence


BUNDLE_NAME = "care-opd-real-thyroid-direct-gold-eval-v5-dual-track"
ARCHIVE_NAME = "CARE_OPD_REAL_THYROID_DIRECT_GOLD_EVAL_v5_DUAL_TRACK.PRIVATE.tar.gz"
BENCHMARK_SCHEMA_VERSION = "care_opd.real_thyroid_external.locked_benchmark.v5.aggregate_manifest"

TRACKS = {
    "structured": {
        "evaluation_track": "structured_evidence_faithfulness",
        "directory": "benchmark_v5_structured",
        "families": 65,
        "rows": 260,
        "gold_fields": 586,
    },
    "raw": {
        "evaluation_track": "raw_strict_extraction_challenge",
        "directory": "benchmark_v5_raw",
        "families": 54,
        "rows": 216,
        "gold_fields": 163,
    },
}

PROMPTS = (
    "doctor_review_prompt.py",
    "doctor_ai_chat_prompt.py",
    "patient_education_prompt.py",
    "patient_diet_guidance_prompt.py",
    "student_analysis_prompt.py",
    "student_quiz_explanation_prompt.py",
    "student_ai_chat_prompt.py",
    "quiz_generation_prompt.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.replace("\r\n", "\n"), encoding="utf-8", newline="\n")


def _copy(source: Path, destination: Path) -> None:
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(f"Required non-empty file is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


INSTALL_SCRIPT = r'''#!/usr/bin/env bash
set -euo pipefail
umask 077

BUNDLE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CARE_ROOT=${CARE_ROOT:-/opt/care-workspace/care_opd}
QZBC_ROOT=${QZBC_ROOT:-/opt/care-workspace/qzbc_source/qzbc-main}
PY=${PY:-/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python}
BENCH_ROOT="$CARE_ROOT/runs/private_real_thyroid_external"

cd "$BUNDLE_DIR"
sha256sum -c SHA256SUMS.bundle
test -x "$PY" || { echo "BAD PYTHON: $PY" >&2; exit 1; }
test -d "$CARE_ROOT/src/care_opd" || { echo "BAD CARE_ROOT: $CARE_ROOT" >&2; exit 1; }

for track in structured raw; do
  src="private_benchmarks/$track"
  dst="$BENCH_ROOT/benchmark_v5_$track"
  install -d -m 700 "$dst"
  install -m 600 "$src/locked_external_qzbc.private.jsonl" "$dst/"
  install -m 600 "$src/field_gold.private.jsonl" "$dst/"
  install -m 644 "$src/benchmark_manifest.json" "$dst/"
done

CODE_DST="$CARE_ROOT/mechanism_experiments/real_thyroid_external"
install -d -m 755 "$CODE_DST" "$CARE_ROOT/src/care_opd"
install -m 644 \
  code/build_locked_benchmark.py \
  code/thyroid_external_eval.py \
  code/thyroid_verifier.py \
  code/reaggregate_existing_eval.py \
  "$CODE_DST/"
install -m 644 code/qzbc_contract.py code/models.py code/care_opd_init.py "$CARE_ROOT/src/care_opd/"
mv -f "$CARE_ROOT/src/care_opd/care_opd_init.py" "$CARE_ROOT/src/care_opd/__init__.py"

for name in doctor_review_prompt.py doctor_ai_chat_prompt.py patient_education_prompt.py patient_diet_guidance_prompt.py student_analysis_prompt.py student_quiz_explanation_prompt.py student_ai_chat_prompt.py quiz_generation_prompt.py; do
  install -d -m 755 "$QZBC_ROOT/backend/app/llm/prompts"
  install -m 644 "qzbc_contract_source/backend/app/llm/prompts/$name" "$QZBC_ROOT/backend/app/llm/prompts/$name"
done
install -d -m 755 "$QZBC_ROOT/backend/app/llm/providers"
install -m 644 qzbc_contract_source/backend/app/llm/providers/ollama_provider.py "$QZBC_ROOT/backend/app/llm/providers/ollama_provider.py"
install -m 644 qzbc_contract_source/backend/app/llm/schemas.py "$QZBC_ROOT/backend/app/llm/schemas.py"
install -m 644 qzbc_contract_source/backend/app/llm/safety.py "$QZBC_ROOT/backend/app/llm/safety.py"

export PYTHONPATH="$CARE_ROOT/src:$CARE_ROOT"
"$PY" -m py_compile \
  "$CODE_DST/build_locked_benchmark.py" \
  "$CODE_DST/thyroid_external_eval.py" \
  "$CODE_DST/thyroid_verifier.py" \
  "$CODE_DST/reaggregate_existing_eval.py" \
  "$CARE_ROOT/src/care_opd/qzbc_contract.py" \
  "$CARE_ROOT/src/care_opd/models.py"

BENCH_ROOT="$BENCH_ROOT" QZBC="$QZBC_ROOT" "$PY" - <<'PY'
import json, os
from pathlib import Path
from mechanism_experiments.real_thyroid_external.thyroid_external_eval import load_locked_inputs

root = Path(os.environ["BENCH_ROOT"])
expected = {
    "structured": ("structured_evidence_faithfulness", 65, 260, 586),
    "raw": ("raw_strict_extraction_challenge", 54, 216, 163),
}
for short, (evaluation_track, families, cases, gold_fields) in expected.items():
    bench = root / f"benchmark_v5_{short}"
    manifest = json.loads((bench / "benchmark_manifest.json").read_text(encoding="utf-8"))
    counts = manifest["counts"]
    assert manifest["status"] == "READY"
    assert manifest["schema_version"] == "care_opd.real_thyroid_external.locked_benchmark.v5.aggregate_manifest"
    assert manifest["evaluation_track"] == evaluation_track
    assert counts["primary_report_families"] == families
    assert counts["primary_qzbc_seed_rows"] == cases
    assert counts["clinical_gold_fields"] == gold_fields
    rows, contract = load_locked_inputs(
        bench / "locked_external_qzbc.private.jsonl",
        field_gold_path=bench / "field_gold.private.jsonl",
        qzbc_root=Path(os.environ["QZBC"]),
    )
    assert contract["family_mode"] == "four_state"
    assert contract["family_count"] == families
    assert contract["case_count"] == cases
    assert contract["evaluation_track"] == evaluation_track
    assert len(rows) == cases
    print(f"TRACK_PREFLIGHT=PASS track={short} families={families} rows={cases} gold={gold_fields}")
PY

echo "INSTALL=PASS tracks=structured,raw"
echo "QZBC_ROOT=$QZBC_ROOT"
'''


COMMON_MODEL_BLOCK = r'''PY=${PY:-/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python}
CARE_ROOT=${CARE_ROOT:-/opt/care-workspace/care_opd}
QZBC_ROOT=${QZBC_ROOT:-/opt/care-workspace/qzbc_source/qzbc-main}
TRACK=${TRACK:-structured}
case "$TRACK" in
  structured)
    BENCH_DEFAULT="$CARE_ROOT/runs/private_real_thyroid_external/benchmark_v5_structured"
    EXPECTED_TRACK=structured_evidence_faithfulness
    EXPECTED_FAMILIES=65
    EXPECTED_CASES=260
    EXPECTED_GOLD=586
    ;;
  raw)
    BENCH_DEFAULT="$CARE_ROOT/runs/private_real_thyroid_external/benchmark_v5_raw"
    EXPECTED_TRACK=raw_strict_extraction_challenge
    EXPECTED_FAMILIES=54
    EXPECTED_CASES=216
    EXPECTED_GOLD=163
    ;;
  *) echo "TRACK must be structured or raw (got: $TRACK)" >&2; exit 2 ;;
esac
BENCH=${BENCH:-$BENCH_DEFAULT}
FROZEN=${FROZEN:-/opt/care-workspace/models/Qwen3.5-4B}
CLEAN=${CLEAN:-$CARE_ROOT/runs/qzbc_base_model_3gpu/warm_start/merged}
RECOVERY_SFT=${RECOVERY_SFT:-$CARE_ROOT/runs/qzbc_baselines_3gpu/baselines/recovery_sft/merged}
RECOVERY_DPO=${RECOVERY_DPO:-$CARE_ROOT/runs/qzbc_baselines_3gpu/baselines/recovery_dpo/merged}
GKD=${GKD:-$CARE_ROOT/runs/qzbc_baselines_3gpu/baselines/gkd_full/final_model}
CAREMODEL=${CAREMODEL:-$CARE_ROOT/runs/qzbc_base_model_3gpu/final_model}
GPU=${GPU:-4}
export PYTHONPATH="$CARE_ROOT/src:$CARE_ROOT"
'''


SMOKE_SCRIPT = r'''#!/usr/bin/env bash
set -euo pipefail
umask 077
''' + COMMON_MODEL_BLOCK + r'''
SMOKE=${SMOKE:-$CARE_ROOT/runs/private_real_thyroid_external/eval_direct_gold_${TRACK}_smoke_v5}

test -s "$CAREMODEL/config.json" || { echo "BAD CARE MODEL: $CAREMODEL" >&2; exit 1; }
if [[ -e "$SMOKE" ]]; then
  echo "REFUSE: smoke output already exists: $SMOKE (set SMOKE to a new path)" >&2
  exit 2
fi
mkdir -p "$SMOKE"

env CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$PYTHONPATH" \
  "$PY" -u "$CARE_ROOT/mechanism_experiments/real_thyroid_external/thyroid_external_eval.py" \
    --seeds "$BENCH/locked_external_qzbc.private.jsonl" \
    --field-gold "$BENCH/field_gold.private.jsonl" \
    --qzbc-root "$QZBC_ROOT" \
    --model "care_opd=$CAREMODEL" \
    --reference-model care_opd \
    --output-dir "$SMOKE" \
    --device cuda:0 \
    --max-examples 3 \
    --generation-seed 42 \
    --bootstrap-replicates 10000 \
    --bootstrap-seed 20260819 \
    --acknowledge-private-clinical-data \
    --overwrite 2>&1 | tee "$SMOKE/driver.log"

SMOKE="$SMOKE" EXPECTED_TRACK="$EXPECTED_TRACK" "$PY" - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ["SMOKE"])
s = json.loads((p / "public/summary.json").read_text(encoding="utf-8"))
m = json.loads((p / "public/run_manifest.json").read_text(encoding="utf-8"))
primary = s["models"][0]["primary"]
assert m["evaluation_track"] == os.environ["EXPECTED_TRACK"], m
assert primary["family_mode"] == "four_state", primary
assert primary["families"] == 3, primary
assert primary["reports"] == 12, primary
assert primary["complete_four_state_families"] == 3, primary
assert primary["generation_error_count"] == 0, primary
print(f"SMOKE=PASS track={m['evaluation_track']} families=3 rows=12 generation_errors=0")
PY
cat "$SMOKE/public/summary.md"
'''


FORMAL_SCRIPT = r'''#!/usr/bin/env bash
set -euo pipefail
umask 077
''' + COMMON_MODEL_BLOCK + r'''
OUT=${OUT:-$CARE_ROOT/runs/private_real_thyroid_external/eval_direct_gold_${TRACK}_six_models_v5}

if [[ -e "$OUT" ]]; then
  echo "REFUSE: formal output already exists: $OUT (use 31_resume.sh or set OUT to a new path)" >&2
  exit 2
fi
for d in "$FROZEN" "$CLEAN" "$RECOVERY_SFT" "$RECOVERY_DPO" "$GKD" "$CAREMODEL"; do
  test -s "$d/config.json" || { echo "BAD MODEL CONFIG: $d" >&2; exit 1; }
  find "$d" -maxdepth 1 -type f \( -name '*.safetensors' -o -name '*.bin' \) -print -quit | grep -q . || {
    echo "NO MODEL WEIGHTS: $d" >&2; exit 1;
  }
done
mkdir -p "$OUT"

nohup setsid env CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$PYTHONPATH" \
  "$PY" -u "$CARE_ROOT/mechanism_experiments/real_thyroid_external/thyroid_external_eval.py" \
    --seeds "$BENCH/locked_external_qzbc.private.jsonl" \
    --field-gold "$BENCH/field_gold.private.jsonl" \
    --qzbc-root "$QZBC_ROOT" \
    --model "frozen=$FROZEN" \
    --model "clean_sft=$CLEAN" \
    --model "recovery_sft=$RECOVERY_SFT" \
    --model "recovery_dpo=$RECOVERY_DPO" \
    --model "gkd_full=$GKD" \
    --model "care_opd=$CAREMODEL" \
    --reference-model clean_sft \
    --additional-reference-model gkd_full \
    --output-dir "$OUT" \
    --device cuda:0 \
    --generation-seed 42 \
    --bootstrap-replicates 10000 \
    --bootstrap-seed 20260819 \
    --acknowledge-private-clinical-data \
    --overwrite >"$OUT/driver.log" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" > "$OUT/driver.pid"
echo "FORMAL_STARTED track=$TRACK pid=$pid gpu=$GPU"
echo "LOG=$OUT/driver.log"
echo "TAIL: tail -n 120 -F '$OUT/driver.log'"
'''


RESUME_SCRIPT = r'''#!/usr/bin/env bash
set -euo pipefail
umask 077
''' + COMMON_MODEL_BLOCK + r'''
OUT=${OUT:-$CARE_ROOT/runs/private_real_thyroid_external/eval_direct_gold_${TRACK}_six_models_v5}

test -d "$OUT" || { echo "MISSING PARTIAL OUT: $OUT" >&2; exit 1; }
if [[ -s "$OUT/driver.pid" ]] && kill -0 "$(cat "$OUT/driver.pid")" 2>/dev/null; then
  echo "STILL RUNNING pid=$(cat "$OUT/driver.pid")" >&2
  exit 2
fi

nohup setsid env CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$PYTHONPATH" \
  "$PY" -u "$CARE_ROOT/mechanism_experiments/real_thyroid_external/thyroid_external_eval.py" \
    --seeds "$BENCH/locked_external_qzbc.private.jsonl" \
    --field-gold "$BENCH/field_gold.private.jsonl" \
    --qzbc-root "$QZBC_ROOT" \
    --model "frozen=$FROZEN" \
    --model "clean_sft=$CLEAN" \
    --model "recovery_sft=$RECOVERY_SFT" \
    --model "recovery_dpo=$RECOVERY_DPO" \
    --model "gkd_full=$GKD" \
    --model "care_opd=$CAREMODEL" \
    --reference-model clean_sft \
    --additional-reference-model gkd_full \
    --output-dir "$OUT" \
    --device cuda:0 \
    --generation-seed 42 \
    --bootstrap-replicates 10000 \
    --bootstrap-seed 20260819 \
    --acknowledge-private-clinical-data \
    --resume >>"$OUT/driver.log" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" > "$OUT/driver.pid"
echo "RESUME_STARTED track=$TRACK pid=$pid gpu=$GPU log=$OUT/driver.log"
'''


PACKAGE_PUBLIC_SCRIPT = r'''#!/usr/bin/env bash
set -euo pipefail
umask 077
''' + COMMON_MODEL_BLOCK + r'''
OUT=${OUT:-$CARE_ROOT/runs/private_real_thyroid_external/eval_direct_gold_${TRACK}_six_models_v5}
PUBLIC="$OUT/public"

OUT="$OUT" EXPECTED_TRACK="$EXPECTED_TRACK" EXPECTED_FAMILIES="$EXPECTED_FAMILIES" EXPECTED_CASES="$EXPECTED_CASES" "$PY" - <<'PY'
import json, os
from pathlib import Path
out = Path(os.environ["OUT"])
names = ["summary.md", "summary.json", "model_metrics.csv", "secondary_field_metrics.csv", "three_axis_metrics.csv", "paired_comparisons.csv", "run_manifest.json"]
missing = [name for name in names if not (out / "public" / name).is_file() or not (out / "public" / name).stat().st_size]
assert not missing, f"missing public outputs: {missing}"
s = json.loads((out / "public/summary.json").read_text(encoding="utf-8"))
m = json.loads((out / "public/run_manifest.json").read_text(encoding="utf-8"))
expected = ["frozen", "clean_sft", "recovery_sft", "recovery_dpo", "gkd_full", "care_opd"]
assert [x["model"] for x in s["models"]] == expected
assert [x["label"] for x in m["models"]] == expected
assert m["family_mode"] == "four_state"
families = int(os.environ["EXPECTED_FAMILIES"])
cases = int(os.environ["EXPECTED_CASES"])
assert m["evaluation_track"] == os.environ["EXPECTED_TRACK"], m
assert m["family_count"] == families and m["case_count"] == cases, m
assert m["reference_models"] == ["clean_sft", "gkd_full"]
for model in s["models"]:
    p = model["primary"]
    assert p["families"] == families and p["reports"] == cases, (model["model"], p)
    assert p["complete_four_state_families"] == families, (model["model"], p)
    assert p["generation_error_count"] == 0, (model["model"], p)
    assert all(p["by_variant"][v]["reports"] == families for v in ["clean", "bad_history", "source_outage", "evidence_conflict"])
pairs = {(x["reference_model"], x["candidate_model"]) for x in s["paired_family_bootstrap"]}
assert ("clean_sft", "care_opd") in pairs and ("gkd_full", "care_opd") in pairs
print(f"FORMAL_EVAL=PASS track={m['evaluation_track']} models=6 families={families} cases={cases}")
PY

ARCHIVE="$OUT/care_opd_real_thyroid_${TRACK}_public_v5.tar.gz"
rm -f "$ARCHIVE" "$ARCHIVE.sha256"
tar -C "$PUBLIC" -czf "$ARCHIVE" \
  summary.md summary.json model_metrics.csv secondary_field_metrics.csv three_axis_metrics.csv paired_comparisons.csv run_manifest.json
chmod 600 "$ARCHIVE"
(cd "$OUT" && sha256sum "$(basename "$ARCHIVE")" > "$(basename "$ARCHIVE").sha256")
tar -tzf "$ARCHIVE"
cat "$ARCHIVE.sha256"
echo "PUBLIC_PACKAGE=$ARCHIVE"
'''


RUNBOOK = r'''# CARE-OPD 真实甲状腺 v5 双轨私有评测

本包同时包含：structured（65 family / 260 case / 586 gold fields）与 raw（54 / 216 / 163）。两轨均为四状态闭合 family，服务器无需 Agent/Ollama 标注。

v5 把条件式输出契约显式放入模型可见请求：有可引用来源且 slots 非空时，只在 sections 中逐 slot 输出字段，并保持 title/summary/warnings/next_actions/disclaimer 为严格空值；无可引用来源或所有来源均不可用时，禁止医疗字段 section/引用，并要求非空的 summary/warnings/next_actions/disclaimer。conflict 请求已清空 extraction schema/slots；任何隐藏 variant/control 标签都不进入模型消息。现有 strict verifier 目标保持不变。

## 上传、校验、安装

```powershell
$SERVER = "root@gpu-a800-094"
$LOCAL  = "/opt/care-workspace\care_opd\runs\private_real_thyroid_external"
$REMOTE = "/opt/care-workspace/care_opd/runs/private_real_thyroid_external"
scp "$LOCAL\CARE_OPD_REAL_THYROID_DIRECT_GOLD_EVAL_v5_DUAL_TRACK.PRIVATE.tar.gz" `
    "$LOCAL\CARE_OPD_REAL_THYROID_DIRECT_GOLD_EVAL_v5_DUAL_TRACK.PRIVATE.tar.gz.sha256" `
    "${SERVER}:${REMOTE}/"
```

```bash
umask 077
ROOT=/opt/care-workspace/care_opd/runs/private_real_thyroid_external
cd "$ROOT" || exit 1
sha256sum -c CARE_OPD_REAL_THYROID_DIRECT_GOLD_EVAL_v5_DUAL_TRACK.PRIVATE.tar.gz.sha256
tar -xzf CARE_OPD_REAL_THYROID_DIRECT_GOLD_EVAL_v5_DUAL_TRACK.PRIVATE.tar.gz
cd care-opd-real-thyroid-direct-gold-eval-v5-dual-track || exit 1
bash server_scripts/00_install_and_verify.sh
```

## 选择一轨运行

`TRACK` 只能是 `structured` 或 `raw`。先 smoke：

```bash
TRACK=structured GPU=4 bash server_scripts/20_smoke.sh
# 或：TRACK=raw GPU=4 bash server_scripts/20_smoke.sh
```

六模型正式评测使用 Clean-SFT 与 GKD-full 双 reference、10,000 次 family bootstrap：

```bash
TRACK=structured GPU=4 bash server_scripts/30_formal.sh
OUT=/opt/care-workspace/care_opd/runs/private_real_thyroid_external/eval_direct_gold_structured_six_models_v5
tail -n 120 -F "$OUT/driver.log"
```

另一轨在前一轨结束后运行：

```bash
TRACK=raw GPU=4 bash server_scripts/30_formal.sh
OUT=/opt/care-workspace/care_opd/runs/private_real_thyroid_external/eval_direct_gold_raw_six_models_v5
tail -n 120 -F "$OUT/driver.log"
```

意外退出时，确认旧 PID 已结束，再用相同 TRACK：

```bash
TRACK=structured GPU=4 bash server_scripts/31_resume.sh
```

完成后验证并仅打包公开聚合：

```bash
TRACK=structured bash server_scripts/40_validate_and_package_public.sh
TRACK=raw bash server_scripts/40_validate_and_package_public.sh
```

不要公开 bundle、seed/gold JSONL、`OUT/private/` 或 driver.log。只可下载通过验证的 `*_public_v5.tar.gz` 与 `.sha256`。
'''


NOTICE = '''# PRIVATE EVALUATION BUNDLE

This archive contains both v5 evaluation tracks with deidentified clinical
report material and structured clinical-field gold.  The data owner states
that offline authorization exists.
The archive is for an approved private server only.  Do not upload it to
GitHub, Hugging Face, public cloud storage, or a public model/API endpoint.

The archive deliberately excludes the original XLSX, HMAC key, canonical
records, annotation queue, Agent outputs, and human-review queue.  It is still
sensitive because the locked seed/gold JSONL files contain report-level data.
'''


def package(
    repo_root: Path,
    structured_benchmark_dir: Path,
    raw_benchmark_dir: Path,
    output_root: Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    bundle = output_root / BUNDLE_NAME
    archive = output_root / ARCHIVE_NAME
    checksum = output_root / f"{ARCHIVE_NAME}.sha256"
    for target in (bundle, archive, checksum):
        if target.parent.resolve() != output_root.resolve():
            raise ValueError(f"Refusing target outside output root: {target}")
        if not target.exists():
            continue
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing target: {target}")
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()

    benchmark_sources = {
        "structured": structured_benchmark_dir,
        "raw": raw_benchmark_dir,
    }
    validated_tracks: dict[str, dict[str, object]] = {}
    for short, source in benchmark_sources.items():
        contract = TRACKS[short]
        manifest = json.loads(
            (source / "benchmark_manifest.json").read_text(encoding="utf-8")
        )
        counts = manifest.get("counts", {})
        expected = {
            "primary_report_families": contract["families"],
            "primary_qzbc_seed_rows": contract["rows"],
            "clinical_gold_fields": contract["gold_fields"],
        }
        if (
            manifest.get("status") != "READY"
            or manifest.get("schema_version") != BENCHMARK_SCHEMA_VERSION
            or manifest.get("evaluation_track") != contract["evaluation_track"]
            or any(counts.get(key) != value for key, value in expected.items())
        ):
            raise ValueError(
                f"Benchmark readiness/count contract failed for {short}: {counts}"
            )
        validated_tracks[short] = {
            "evaluation_track": contract["evaluation_track"],
            "benchmark_schema_version": manifest["schema_version"],
            "transform_protocol_sha256": manifest["construction_protocol"][
                "transform_protocol_sha256"
            ],
            **expected,
        }
        private_benchmark = bundle / "private_benchmarks" / short
        for name in (
            "locked_external_qzbc.private.jsonl",
            "field_gold.private.jsonl",
            "benchmark_manifest.json",
        ):
            _copy(source / name, private_benchmark / name)

    source_dir = repo_root / "mechanism_experiments" / "real_thyroid_external"
    for name in (
        "build_locked_benchmark.py",
        "thyroid_external_eval.py",
        "thyroid_verifier.py",
        "reaggregate_existing_eval.py",
    ):
        _copy(source_dir / name, bundle / "code" / name)
    _copy(repo_root / "src" / "care_opd" / "qzbc_contract.py", bundle / "code" / "qzbc_contract.py")
    _copy(repo_root / "src" / "care_opd" / "models.py", bundle / "code" / "models.py")
    _copy(repo_root / "src" / "care_opd" / "__init__.py", bundle / "code" / "care_opd_init.py")

    qzbc = repo_root.parent / "qzbc_source" / "qzbc-main" / "backend" / "app" / "llm"
    for name in PROMPTS:
        _copy(qzbc / "prompts" / name, bundle / "qzbc_contract_source" / "backend" / "app" / "llm" / "prompts" / name)
    for relative in (Path("providers/ollama_provider.py"), Path("schemas.py"), Path("safety.py")):
        _copy(qzbc / relative, bundle / "qzbc_contract_source" / "backend" / "app" / "llm" / relative)

    _write(bundle / "server_scripts" / "00_install_and_verify.sh", INSTALL_SCRIPT)
    _write(bundle / "server_scripts" / "20_smoke.sh", SMOKE_SCRIPT)
    _write(bundle / "server_scripts" / "30_formal.sh", FORMAL_SCRIPT)
    _write(bundle / "server_scripts" / "31_resume.sh", RESUME_SCRIPT)
    _write(bundle / "server_scripts" / "40_validate_and_package_public.sh", PACKAGE_PUBLIC_SCRIPT)
    _write(bundle / "SERVER_RUNBOOK_ZH.md", RUNBOOK)
    _write(bundle / "PRIVATE_BUNDLE_NOTICE.md", NOTICE)
    _write(bundle / ".gitignore", "*\n!.gitignore\n")

    forbidden_names = (".xlsx", "hmac_key", "canonical_records", "agent_work_queue", "agent_annotations", "human_review")
    leaked = [str(p.relative_to(bundle)) for p in bundle.rglob("*") if p.is_file() and any(token in p.name.lower() for token in forbidden_names)]
    if leaked:
        raise ValueError(f"Forbidden files entered private eval bundle: {leaked}")

    bundle_manifest = {
        "schema_version": "care_opd.real_thyroid.direct_gold_server_bundle.v5",
        "created_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "privacy": "PRIVATE_DO_NOT_PUBLISH",
        "annotation_stage_required_on_server": False,
        "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
        "model_visible_conditional_output_contract": True,
        "strict_verifier_target_changed": False,
        "tracks": validated_tracks,
        "excluded": ["original_xlsx", "hmac_key", "canonical_records", "agent_queue", "agent_outputs", "human_review_queue"],
    }
    _write(bundle / "BUNDLE_MANIFEST.json", json.dumps(bundle_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    files = sorted(
        p for p in bundle.rglob("*")
        if p.is_file() and p.name != "SHA256SUMS.bundle"
    )
    bundle_manifest["files_in_checksum_manifest"] = len(files)
    _write(
        bundle / "BUNDLE_MANIFEST.json",
        json.dumps(bundle_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    # Re-enumerate because the manifest content changed after adding its count.
    files = sorted(
        p for p in bundle.rglob("*")
        if p.is_file() and p.name != "SHA256SUMS.bundle"
    )
    sha_lines = [
        f"{_sha256(path)}  {path.relative_to(bundle).as_posix()}" for path in files
    ]
    _write(bundle / "SHA256SUMS.bundle", "\n".join(sha_lines) + "\n")

    # Include the top-level bundle directory in the archive.
    def _private_tar_mode(member: tarfile.TarInfo) -> tarfile.TarInfo:
        if member.isdir():
            member.mode = 0o700
        elif "/server_scripts/" in f"/{member.name}":
            member.mode = 0o700
        else:
            member.mode = 0o600
        member.uid = member.gid = 0
        member.uname = member.gname = "root"
        return member

    with tarfile.open(archive, "w:gz") as handle:
        handle.add(bundle, arcname=bundle.name, filter=_private_tar_mode)
    archive_hash = _sha256(archive)
    _write(checksum, f"{archive_hash}  {archive.name}\n")
    return archive, checksum


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--structured-benchmark-dir", type=Path, required=True)
    parser.add_argument("--raw-benchmark-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    archive, checksum = package(
        args.repo_root.resolve(),
        args.structured_benchmark_dir.resolve(),
        args.raw_benchmark_dir.resolve(),
        args.output_root.resolve(),
        overwrite=args.overwrite,
    )
    print(f"DIRECT_GOLD_SERVER_BUNDLE=PASS archive={archive} sha256={_sha256(archive)}")
    print(f"CHECKSUM={checksum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
