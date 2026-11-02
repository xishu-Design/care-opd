# 真实甲状腺外部测试：完整运行手册

本手册把真实 Excel 转换为**报告组级、私有、锁定的四状态外测**，并用甲状腺专用 verifier 评估本地 Hugging Face checkpoint。每个进入主评测的报告固定派生 `clean / bad_history / source_outage / evidence_conflict` 四个匹配状态；四行始终以同一个 report family 为统计簇。

v5 必须分成两个互不混用的评测轨：

- 主轨 `structured_evidence_faithfulness`：原始报告与临床结构化记录是两个独立可引用证据源；冻结口径为 65 families、260 rows、586 fields；
- 次轨 `raw_strict_extraction_challenge`：模型只看到原始报告，仅纳入唯一逐字 span 可定位字段；冻结口径为 54 families、216 rows、163 fields。

主问题与公开主表保持一致：模型能否在可用证据上完成任务，并在错误历史、证据中断和证据冲突下保持或安全降级。主端点是二元 `state_task_success`：报告 Clean TSR、每种扰动的 TSR、Conditional Retention/Regression 和四状态联合 TSR。把三种扰动合并得到的 pooled Perturbed TSR 混合了“精确渲染”和“安全边界”两类目标，**只作描述性统计**，不能作为单独的优越性主张。

structured 主轨的 clean/bad-history 必须同时满足 exact slot cardinality、exact key/heading、exact singleton source 和 exact literal content；缺失、重复、额外、错 key、错 source 或错 content 均失败。V5 通过模型可见的 `constraints.conditional_output_contract.with_citable_slots` 分支把辅助目标显式写成 `title=""`、`summary=""`、`warnings=[]`、`next_actions=[]`、`disclaimer=""`，任何实质辅助叙述均失败且不能替代 section。三轴抽取诊断只统计 `clean+bad_history`；outage/conflict 只进入各自边界指标与 joint TSR，不解释字段/content recall。

边界状态选择 `constraints.conditional_output_contract.without_citable_slots` 分支，不再携带可执行的抽取目标：outage 清空全部证据与 structured context；conflict 保留被标记为 non-citable 的可见证据用于测试状态识别，但同时把 `allowed_extraction_schema=[]` 与 `required_output_slots=[]`。边界回答必须给出 summary、warning、next action 和 disclaimer，且不得产生临床字段或引用来源。`sections=[]` 合法；若模型额外给出 generic boundary section，该 section 必须无 source、不是字段/packet heading、且不作临床结论。generic section 是条件允许项，不是必填项，也不能补偿四个必填边界组件的缺失。

variant 名称、control 标签和 sentinel 只留在私有 verifier 元数据中，不进入 rendered messages；conflict sentinel 也不再作为 retrieved-evidence item。模型必须依据可见 Evidence Packet 条件判断状态，不能依赖评测专用标签捷径。

此前 v4 的三-family smoke 仅是**结果盲的协议校准**：`SMOKE=PASS` 只证明 12 个 quartet 请求完成生成、无运行时错误并成功写出结果，不表示 CARE-OPD 的 TSR/F1 达标。V5 改变了模型可见提示、conflict request 与边界契约，因此 v4 的 smoke、partial formal 和 complete formal 都不能 `--resume`、重标或重汇总成 v5。必须用新建的 v5 benchmark 和全新的 v5 OUT 起跑。

实验边界：structured 主轨测量“可见结构化证据的忠实渲染与受控证据状态恢复”，raw 次轨测量“原始报告上逐字 span 可定位子集的严格抽取与恢复”。两者都不测临床诊断正确性、患者结局、超声预测病理或跨模态病灶匹配。工作簿没有可靠患者/检查/病灶关联键，因此统计单位始终是 report group，而不是患者。

## 0. 目录与安全门槛

在 PowerShell 中设置变量：

```powershell
$ROOT   = "/opt/care-workspace"
$CARE   = "$ROOT\care_opd"
$PY     = "/opt/care-user\AppData\Local\Programs\Python\Python311\python.exe"
$XLSX   = "D:\wxstrore\xwechat_files\wxid_eiqukq8h6ydr22_a1b1\msg\file\2026-08\甲状腺结构化数据_分离式_20251130_173905.xlsx"
$LOCKED = "$CARE\runs\private_real_thyroid_external\locked_v1"
$ANN    = "$LOCKED\agent_annotations"
$GOLD   = "$CARE\runs\private_real_thyroid_external\direct_clinical_gold_v1\clinical_gold.private.jsonl"
$BENCH_STRUCTURED = "$CARE\runs\private_real_thyroid_external\benchmark_v5_structured"
$BENCH_RAW        = "$CARE\runs\private_real_thyroid_external\benchmark_v5_raw"
$SMOKE_STRUCTURED = "$CARE\runs\private_real_thyroid_external\eval_v5_structured_smoke"
$EVAL_STRUCTURED  = "$CARE\runs\private_real_thyroid_external\eval_v5_structured_care"
$EVAL_RAW         = "$CARE\runs\private_real_thyroid_external\eval_v5_raw_care"
$QZBC   = "$ROOT\qzbc_source\qzbc-main"
$MODEL  = "$ROOT\final_model"

Set-Location $CARE
$env:PYTHONPATH = "$CARE\src;$CARE"
```

必须先由有权限的人完成：

1. 数据使用授权/伦理和机构政策核对；
2. 原始自由文本的人工去标识复核；
3. 确认仅在受控本地或机构服务器上处理，不上传 GitHub、Hugging Face 或公共 API。

`reviewed_private_use` 只是复核状态记录，不会自动脱敏。未真正完成上述复核时，保持 `unreviewed` 并停止在审计阶段。

## 1. 复用冻结 HMAC，重新导出经复核的私有记录

当前 `locked_v1` 已有稳定私钥。必须复用它，不能再次使用 `--generate-hmac-key-file`，否则报告 ID 会变化。

```powershell
& $PY mechanism_experiments\real_thyroid_external\prepare_external_test.py `
  $XLSX `
  --output-dir $LOCKED `
  --private-export `
  --acknowledge-private-clinical-data `
  --hmac-key-file "$LOCKED\.thyroid_hmac_key.private" `
  --deidentification-status reviewed_private_use `
  --authorization-reference "<本地授权或伦理审批编号>"
```

只检查安全汇总，不打印私有 JSONL：

```powershell
$audit = Get-Content "$LOCKED\audit_summary.json" -Raw -Encoding UTF8 | ConvertFrom-Json
$audit | ConvertTo-Json -Depth 8
```

预期报告组为 72：超声 23、病理 49。超声续行仍归属于同一报告组，不能按 Excel 行随机切分。

## 2. 可选：启动本地三 Agent 源证据补标

当前 direct-clinical-gold 双轨构建直接使用 `$GOLD`，不依赖本节 Agent
结果。只有要另做 legacy `exact_verified` / source-exact 子集时才执行本
节；不要把 Agent silver 与表格提供的 clinical gold 混入同一个 reference
文件。

当前机器检查时 `127.0.0.1:11434` 未响应，因此这一步开始前需先启动本地 Ollama 或换成机构批准的私有端点。

另开一个 PowerShell：

```powershell
ollama serve
```

回到实验终端检查可用模型：

```powershell
Invoke-RestMethod "http://127.0.0.1:11434/api/tags" |
  Select-Object -ExpandProperty models |
  Select-Object name, size
```

把下面三个值替换为上一步真实列出的 tag。最好使用相互独立的已批准本地模型；若只有一个模型，可以三个角色共用，但最终必须称为“单模型多角色、Agent 仲裁的 source-exact silver labels”，不能称独立专家共识。

```powershell
$LOCATOR    = "<APPROVED_LOCAL_MODEL_A>"
$AUDITOR    = "<APPROVED_LOCAL_MODEL_B>"
$ADJUDICATOR = "<APPROVED_LOCAL_MODEL_C>"

& $PY mechanism_experiments\real_thyroid_external\run_local_agent_annotation.py `
  --queue "$LOCKED\agent_work_queue.private.jsonl" `
  --output-dir $ANN `
  --base-url "http://127.0.0.1:11434" `
  --locator-model $LOCATOR `
  --auditor-model $AUDITOR `
  --adjudicator-model $ADJUDICATOR `
  --acknowledge-private-clinical-data `
  --seed 20260818
```

网络/进程中断后，用完全相同的命令重跑即可按 case 恢复。不要改变模型、seed、queue 或 protocol 后继续写同一个输出目录。

完成检查：

```powershell
$annSummary = Get-Content "$ANN\agent_annotation_summary.json" -Raw -Encoding UTF8 | ConvertFrom-Json
$annSummary | ConvertTo-Json -Depth 8
if ($annSummary.status -ne "COMPLETE") { throw "Agent annotation is incomplete." }
if ($annSummary.exact_verified_fields -le 0) { throw "No strict source-exact fields were accepted." }
```

主要输出：

- `exact_verified_fields.private.jsonl`：三角色一致且原文精确可定位的主评测字段；
- `human_review_queue.private.jsonl`：歧义、不一致、无法精确定位或新发现字段，只进入人工复核层；
- `case_results.private/`：私有逐例审计轨迹。

## 3. 分别构建 structured 主轨与 raw challenge 次轨

两个轨道必须使用同一份 frozen canonical 和 clinical gold，但写入不同
目录。不要依赖 builder 默认值，始终显式写 canonical `--task-mode`。

```powershell
& $PY mechanism_experiments\real_thyroid_external\build_locked_benchmark.py `
  --canonical "$LOCKED\canonical_records.private.jsonl" `
  --clinical-gold $GOLD `
  --task-mode structured_evidence_faithfulness `
  --output-dir $BENCH_STRUCTURED `
  --acknowledge-private-clinical-data

& $PY mechanism_experiments\real_thyroid_external\build_locked_benchmark.py `
  --canonical "$LOCKED\canonical_records.private.jsonl" `
  --clinical-gold $GOLD `
  --task-mode raw_strict_extraction_challenge `
  --output-dir $BENCH_RAW `
  --acknowledge-private-clinical-data
```

只读取 public-safe manifest 做冻结计数检查：

```powershell
$ms = Get-Content "$BENCH_STRUCTURED\benchmark_manifest.json" -Raw -Encoding UTF8 | ConvertFrom-Json
$mr = Get-Content "$BENCH_RAW\benchmark_manifest.json" -Raw -Encoding UTF8 | ConvertFrom-Json

if ($ms.schema_version -ne "care_opd.real_thyroid_external.locked_benchmark.v5.aggregate_manifest" -or $ms.status -ne "READY" -or $ms.evaluation_track -ne "structured_evidence_faithfulness") { throw "Bad structured v5 manifest." }
if ($ms.counts.primary_report_families -ne 65 -or $ms.counts.primary_qzbc_seed_rows -ne 260 -or $ms.counts.primary_gold_fields -ne 586 -or $ms.counts.boundary_only_reports_without_primary_gold -ne 7) { throw "Structured frozen counts changed." }

if ($mr.schema_version -ne "care_opd.real_thyroid_external.locked_benchmark.v5.aggregate_manifest" -or $mr.status -ne "READY" -or $mr.evaluation_track -ne "raw_strict_extraction_challenge") { throw "Bad raw v5 manifest." }
if ($mr.counts.primary_report_families -ne 54 -or $mr.counts.primary_qzbc_seed_rows -ne 216 -or $mr.counts.primary_gold_fields -ne 163 -or $mr.counts.boundary_only_reports_without_primary_gold -ne 18) { throw "Raw frozen counts changed." }

foreach ($m in @($ms, $mr)) {
  if ($m.family_contract.complete_quartet_required -ne $true) { throw "Four-state family contract is missing." }
  if ($m.counts.primary_qzbc_seed_rows -ne 4 * $m.counts.primary_report_families) { throw "Family quartet is incomplete." }
}
```

每个目录都生成：

- `locked_external_qzbc.private.jsonl`：四状态模型请求；
- `field_gold.private.jsonl`：clean/bad-history exact 输出契约及 outage/conflict 边界契约；
- `boundary_only.private.jsonl`：该轨道未纳入主端点的报告/槽位；
- `benchmark_lock.private.json`：私有输入输出 hash 与轨道/eligibility 锁；
- `benchmark_manifest.json`：不含原文、字段值、HMAC、病例 ID、私有路径或私有 hash 的公开安全汇总。

structured 主轨把 586 个非空结构化字段作为独立可引用证据提供给模型，
所以只能主张 **structured-evidence faithfulness TSR**，不能称 hidden-label
抽取。raw 次轨只纳入 163 个唯一逐字 span 可定位字段，只能主张
**raw-report strict extraction challenge TSR**。空单元格始终解释为
unknown/not reported，绝不自动当作阴性。

## 4. 先做 structured 主轨 CARE-OPD 三个 family 协议校准

本地 checkpoint 已位于 `/opt/care-workspace\final_model`。`--max-examples 3` 现在按完整 family 抽样，因此会跑 3 个报告 family、共 12 个状态请求；不会截断 quartet。

```powershell
& $PY mechanism_experiments\real_thyroid_external\thyroid_external_eval.py `
  --seeds "$BENCH_STRUCTURED\locked_external_qzbc.private.jsonl" `
  --field-gold "$BENCH_STRUCTURED\field_gold.private.jsonl" `
  --qzbc-root $QZBC `
  --model "care_opd=$MODEL" `
  --reference-model care_opd `
  --output-dir $SMOKE_STRUCTURED `
  --device cuda:0 `
  --max-examples 3 `
  --acknowledge-private-clinical-data `
  --overwrite
```

无 CUDA 时可把 `--device cuda:0` 改成 `--device cpu`，但 4B 模型会明显更慢。该 smoke 的 PASS 门只检查 evaluation track、3 个完整 family、12 行结果和零 generation error；它不设置任何 TSR/F1 效果阈值。不要把 smoke 数值写进论文，也不要因为终端打印 `SMOKE=PASS` 就宣称模型效果通过。

```powershell
Get-Content "$SMOKE_STRUCTURED\public\summary.md" -Encoding UTF8
```

## 5. 正式 structured 主轨 CARE-OPD 单模型外测（仅用于绝对值检查）

首次运行：

```powershell
& $PY mechanism_experiments\real_thyroid_external\thyroid_external_eval.py `
  --seeds "$BENCH_STRUCTURED\locked_external_qzbc.private.jsonl" `
  --field-gold "$BENCH_STRUCTURED\field_gold.private.jsonl" `
  --qzbc-root $QZBC `
  --model "care_opd=$MODEL" `
  --reference-model care_opd `
  --output-dir $EVAL_STRUCTURED `
  --device cuda:0 `
  --generation-seed 42 `
  --bootstrap-replicates 10000 `
  --bootstrap-seed 20260819 `
  --acknowledge-private-clinical-data `
  --overwrite
```

只有**同一 v5 协议**的运行中断时，才可把最后的 `--overwrite` 改为
`--resume`，且其余输入、代码、模型、生成参数、bootstrap 参数与路径必须
完全不变。任何 v4 输出均禁止 resume；先保留旧目录，再用 v5 benchmark
与新 v5 OUT 从头运行。

raw 次轨必须另起输出目录，不能在 structured 结果上 resume：

```powershell
& $PY mechanism_experiments\real_thyroid_external\thyroid_external_eval.py `
  --seeds "$BENCH_RAW\locked_external_qzbc.private.jsonl" `
  --field-gold "$BENCH_RAW\field_gold.private.jsonl" `
  --qzbc-root $QZBC `
  --model "care_opd=$MODEL" `
  --reference-model care_opd `
  --output-dir $EVAL_RAW `
  --device cuda:0 `
  --generation-seed 42 `
  --bootstrap-replicates 10000 `
  --bootstrap-seed 20260819 `
  --acknowledge-private-clinical-data `
  --overwrite
```

## 6. 论文正式比较：模型行与主表保持一致

只报 CARE 的绝对值不足以支持方法优势。资源允许时，正式表保持公开主表的六模型顺序：Frozen、Clean-SFT、Recovery-SFT、Recovery-DPO、GKD-full、CARE-OPD；最低可先跑 Clean-SFT、GKD-full、CARE-OPD 三者。主比较基线仍是 Clean-SFT 和 GKD-full，不跨 benchmark 求平均。

三个最低必跑 checkpoint：

- Clean-SFT：服务器 `/opt/care-workspace/care_opd/runs/qzbc_base_model_3gpu/warm_start/merged`
- GKD-full：服务器 `/opt/care-workspace/care_opd/runs/qzbc_baselines_3gpu/baselines/gkd_full/final_model`
- CARE-OPD：服务器 `/opt/care-workspace/care_opd/runs/qzbc_base_model_3gpu/final_model`

如果三个完整 checkpoint 都在 Windows 本地，可先执行：

```powershell
$CLEAN = "<本地完整 Clean-SFT checkpoint>"
$GKD   = "<本地完整 GKD-full checkpoint>"
$CAREMODEL = "$ROOT\final_model"
$BENCH = $BENCH_STRUCTURED
$EVAL3 = "$CARE\runs\private_real_thyroid_external\eval_v5_structured_three_models"

& $PY mechanism_experiments\real_thyroid_external\thyroid_external_eval.py `
  --seeds "$BENCH\locked_external_qzbc.private.jsonl" `
  --field-gold "$BENCH\field_gold.private.jsonl" `
  --qzbc-root $QZBC `
  --model "clean_sft=$CLEAN" `
  --model "gkd_full=$GKD" `
  --model "care_opd=$CAREMODEL" `
  --reference-model clean_sft `
  --additional-reference-model gkd_full `
  --output-dir $EVAL3 `
  --device cuda:0 `
  --generation-seed 42 `
  --bootstrap-replicates 10000 `
  --bootstrap-seed 20260819 `
  --acknowledge-private-clinical-data `
  --overwrite
```

三个模型会在同一逻辑设备上依次加载，避免同时占用三份显存。
上述命令是 structured 主轨。要跑 raw 次轨，必须把 `$BENCH` 改为
`$BENCH_RAW`，把 `$EVAL3` 改为独立目录
`eval_v5_raw_three_models`，然后完整重跑；不能混合两个轨道的结果行。

## 7. 推荐的 Linux/A800 六模型正式运行

先把新实验代码和**仅用于受控服务器的两个私有 benchmark**传到有
授权的服务器。不要传到公共仓库、公共对象存储或 Hugging Face。原始
XLSX、HMAC key、canonical、clinical-gold 源文件和 Agent case outputs 均
不需要上传。

在 Windows PowerShell 中分别生成传输校验并同步。下列命令是操作说明；
本次文档更新本身不会复制任何私有文件。

```powershell
$SERVER = "root@gpu-a800-094"
$TRACKS = @(
  @{ Local=$BENCH_STRUCTURED; Remote="/opt/care-workspace/care_opd/runs/private_real_thyroid_external/benchmark_v5_structured" },
  @{ Local=$BENCH_RAW;        Remote="/opt/care-workspace/care_opd/runs/private_real_thyroid_external/benchmark_v5_raw" }
)

foreach ($t in $TRACKS) {
  $h1 = (Get-FileHash "$($t.Local)\locked_external_qzbc.private.jsonl" -Algorithm SHA256).Hash.ToLower()
  $h2 = (Get-FileHash "$($t.Local)\field_gold.private.jsonl" -Algorithm SHA256).Hash.ToLower()
  @(
    "$h1  locked_external_qzbc.private.jsonl",
    "$h2  field_gold.private.jsonl"
  ) | Set-Content "$($t.Local)\SHA256SUMS.private" -Encoding ascii

  ssh $SERVER "install -d -m 700 '$($t.Remote)'"
  scp "$($t.Local)\locked_external_qzbc.private.jsonl" `
      "$($t.Local)\field_gold.private.jsonl" `
      "$($t.Local)\benchmark_manifest.json" `
      "$($t.Local)\SHA256SUMS.private" `
      "${SERVER}:$($t.Remote)/"
}

ssh $SERVER "install -d -m 755 '/opt/care-workspace/care_opd/mechanism_experiments/real_thyroid_external'"
scp "$CARE\mechanism_experiments\real_thyroid_external\thyroid_external_eval.py" `
    "$CARE\mechanism_experiments\real_thyroid_external\thyroid_verifier.py" `
    "$CARE\mechanism_experiments\real_thyroid_external\build_locked_benchmark.py" `
    "${SERVER}:/opt/care-workspace/care_opd/mechanism_experiments/real_thyroid_external/"
```

服务器上定义一次模型和双轨路径：

```bash
cd /opt/care-workspace/care_opd

PY=/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python
QZBC=/opt/care-workspace/qzbc_source/qzbc-main
BENCH_STRUCTURED=/opt/care-workspace/care_opd/runs/private_real_thyroid_external/benchmark_v5_structured
BENCH_RAW=/opt/care-workspace/care_opd/runs/private_real_thyroid_external/benchmark_v5_raw
OUT_STRUCTURED=/opt/care-workspace/care_opd/runs/private_real_thyroid_external/eval_direct_gold_structured_six_models_v5
OUT_RAW=/opt/care-workspace/care_opd/runs/private_real_thyroid_external/eval_direct_gold_raw_six_models_v5

FROZEN=/opt/care-workspace/models/Qwen3.5-4B
CLEAN=/opt/care-workspace/care_opd/runs/qzbc_base_model_3gpu/warm_start/merged
RECOVERY_SFT=/opt/care-workspace/care_opd/runs/qzbc_baselines_3gpu/baselines/recovery_sft/merged
RECOVERY_DPO=/opt/care-workspace/care_opd/runs/qzbc_baselines_3gpu/baselines/recovery_dpo/merged
GKD=/opt/care-workspace/care_opd/runs/qzbc_baselines_3gpu/baselines/gkd_full/final_model
CAREMODEL=/opt/care-workspace/care_opd/runs/qzbc_base_model_3gpu/final_model

for bench in "$BENCH_STRUCTURED" "$BENCH_RAW"; do
  for f in "$bench/locked_external_qzbc.private.jsonl" \
           "$bench/field_gold.private.jsonl" \
           "$bench/benchmark_manifest.json" \
           "$bench/SHA256SUMS.private"; do
    test -s "$f" || { echo "MISSING: $f" >&2; exit 1; }
  done
  chmod 600 "$bench"/*.private.jsonl "$bench/SHA256SUMS.private"
  (cd "$bench" && sha256sum -c SHA256SUMS.private) || exit 1
done

for f in "$QZBC/backend/app/llm/prompts/doctor_ai_chat_prompt.py" \
         "$QZBC/backend/app/llm/providers/ollama_provider.py" \
         "$QZBC/backend/app/llm/schemas.py" \
         "$QZBC/backend/app/llm/safety.py"; do
  test -s "$f" || { echo "MISSING: $f" >&2; exit 1; }
done
for d in "$FROZEN" "$CLEAN" "$RECOVERY_SFT" "$RECOVERY_DPO" "$GKD" "$CAREMODEL"; do
  test -s "$d/config.json" || { echo "BAD MODEL: $d" >&2; exit 1; }
done
```

若 v4 formal 仍在运行，先确认 PID 确实绑定旧 OUT，再发送 TERM；不要把
该旧进程作为 v5 resume 源：

```bash
V4OUT=/opt/care-workspace/care_opd/runs/private_real_thyroid_external/eval_direct_gold_structured_six_models_v4
if [[ -s "$V4OUT/driver.pid" ]]; then
  V4PID=$(cat "$V4OUT/driver.pid")
  if [[ "$V4PID" =~ ^[0-9]+$ ]] && kill -0 "$V4PID" 2>/dev/null; then
    V4CMD=$(tr '\000' ' ' < "/proc/$V4PID/cmdline")
    [[ "$V4CMD" == *thyroid_external_eval.py* && "$V4CMD" == *"$V4OUT"* ]] || {
      echo "REFUSE: stale/reused PID does not belong to v4 OUT" >&2
      exit 2
    }
    kill -TERM "$V4PID"
    for i in $(seq 1 60); do
      kill -0 "$V4PID" 2>/dev/null || break
      sleep 1
    done
    kill -0 "$V4PID" 2>/dev/null && {
      echo "v4 is still running; do not launch v5" >&2
      exit 2
    }
  fi
fi
```

定义启动函数。它一次只启动一个轨道，模型在同一逻辑 GPU 上依次加载：

```bash
launch_track () {
  local bench="$1"
  local out="$2"
  local run_mode="${3:---overwrite}"

  case "$out" in
    *v4*) echo "REFUSE: v4 OUT cannot be used by the v5 protocol: $out" >&2; return 2 ;;
  esac
  case "$run_mode" in
    --overwrite)
      [[ ! -e "$out" ]] || { echo "REFUSE: OUT already exists: $out" >&2; return 2; }
      ;;
    --resume)
      [[ -d "$out" ]] || { echo "REFUSE: missing v5 partial OUT: $out" >&2; return 2; }
      [[ ! -s "$out/public/summary.json" ]] || { echo "REFUSE: run is already complete" >&2; return 2; }
      ;;
    *) echo "REFUSE: run mode must be --overwrite or --resume" >&2; return 2 ;;
  esac

  local active
  active=$(ps -eo pid=,args= | grep '[t]hyroid_external_eval.py' || true)
  [[ -z "$active" ]] || {
    echo "REFUSE: another thyroid evaluator is already active" >&2
    echo "$active" >&2
    return 2
  }

  mkdir -p "$out"
  nohup setsid env \
    CUDA_VISIBLE_DEVICES=4 \
    PYTHONPATH=/opt/care-workspace/care_opd/src:/opt/care-workspace/care_opd \
    "$PY" -u mechanism_experiments/real_thyroid_external/thyroid_external_eval.py \
      --seeds "$bench/locked_external_qzbc.private.jsonl" \
      --field-gold "$bench/field_gold.private.jsonl" \
      --qzbc-root "$QZBC" \
      --model "frozen=$FROZEN" \
      --model "clean_sft=$CLEAN" \
      --model "recovery_sft=$RECOVERY_SFT" \
      --model "recovery_dpo=$RECOVERY_DPO" \
      --model "gkd_full=$GKD" \
      --model "care_opd=$CAREMODEL" \
      --reference-model clean_sft \
      --additional-reference-model gkd_full \
      --output-dir "$out" \
      --device cuda:0 \
      --generation-seed 42 \
      --bootstrap-replicates 10000 \
      --bootstrap-seed 20260819 \
      --acknowledge-private-clinical-data \
      "$run_mode" \
    >>"$out/driver.log" 2>&1 </dev/null &
  local pid=$!
  printf '%s\n' "$pid" | tee "$out/driver.pid"
}

# 先跑 structured 主轨。
launch_track "$BENCH_STRUCTURED" "$OUT_STRUCTURED" --overwrite
tail -n 80 -F "$OUT_STRUCTURED/driver.log"

# structured 完成且 GPU 释放后再跑 raw 次轨；不要同时启动。
test -s "$OUT_STRUCTURED/public/summary.json" || {
  echo "REFUSE: structured run is not complete" >&2
  exit 2
}
launch_track "$BENCH_RAW" "$OUT_RAW" --overwrite
tail -n 80 -F "$OUT_RAW/driver.log"
```

V5 中断恢复时，在同一个 `bench/out` 配对上把 `--overwrite` 改为
`--resume`。不要在已有输出目录中更换轨道、输入、模型路径、生成参数、
verifier 或 bootstrap 配置；manifest 会失败关闭。V4 的任意输出目录都
不是 v5 resume 源，尤其不要对
`eval_direct_gold_structured_six_models_v4` 运行 v5 `--resume`。

确认旧 PID 已退出且没有同 OUT 进程后，structured v5 的续跑命令为：

```bash
launch_track "$BENCH_STRUCTURED" "$OUT_STRUCTURED" --resume
tail -n 80 -F "$OUT_STRUCTURED/driver.log"
```

### V5 direct-gold server bundle 重生成清单

旧 v1/v4 bundle 绑定旧请求与旧指标口径，不能原样复用，也不能 resume
到 v5。新 bundle 至少重新生成/替换：

- `code/build_locked_benchmark.py`、`code/thyroid_external_eval.py`、
  `code/thyroid_verifier.py`；
- `private_benchmark_structured/{locked_external_qzbc.private.jsonl,field_gold.private.jsonl,benchmark_manifest.json}`；
- `private_benchmark_raw/{locked_external_qzbc.private.jsonl,field_gold.private.jsonl,benchmark_manifest.json}`；
- 如 bundle 保留完整审计材料，再分别加入两轨的
  `boundary_only.private.jsonl` 与 `benchmark_lock.private.json`，但它们不是
  server evaluator 的必需输入；
- `server_scripts/00_install_and_verify.sh`、`20_smoke.sh`、`30_formal.sh`、
  `31_resume.sh`、`40_validate_and_package_public.sh`，其中路径、预期计数、
  三轴 scope 和 descriptive pooled 指标都要更新；
- `SERVER_RUNBOOK_ZH.md`、`PRIVATE_BUNDLE_NOTICE.md`、
  `BUNDLE_MANIFEST.json`、`SHA256SUMS.bundle`；
- 最终私有 archive 及其独立 SHA-256 sidecar。

`qzbc_contract_source/` 和未变更的模型 schema 只有在内容 hash 仍一致时
才可复用；新的 `BUNDLE_MANIFEST.json` 必须记录实际 hash，不能沿用旧
checksum。本清单只描述需重生成的 artifact，不授权、创建或复制任何
私有 payload。

## 8. 结果与完成性检查

可公开安全汇总位于：

```text
<EVAL>/public/summary.md
<EVAL>/public/summary.json
<EVAL>/public/model_metrics.csv
<EVAL>/public/three_axis_metrics.csv
<EVAL>/public/secondary_field_metrics.csv
<EVAL>/public/paired_comparisons.csv
<EVAL>/public/run_manifest.json
```

逐报告的 prompt、gold、raw response、assessment 位于 `<EVAL>/private/`，不得公开。

PowerShell 查看：

```powershell
Get-Content "$EVAL_STRUCTURED\public\summary.md" -Encoding UTF8
Import-Csv "$EVAL_STRUCTURED\public\model_metrics.csv" | Format-Table -AutoSize
Import-Csv "$EVAL_STRUCTURED\public\three_axis_metrics.csv" | Format-Table -AutoSize
Import-Csv "$EVAL_STRUCTURED\public\paired_comparisons.csv" | Format-Table -AutoSize
```

正式完成门要求每个模型没有生成异常且使用完整 quartet：

```powershell
foreach ($evalDir in @($EVAL_STRUCTURED, $EVAL_RAW)) {
  $summary = Get-Content "$evalDir\public\summary.json" -Raw -Encoding UTF8 | ConvertFrom-Json
  foreach ($model in $summary.models) {
    if ($model.primary.family_mode -ne "four_state") { throw "Not a four-state result: $($model.model)" }
    if ($model.primary.generation_error_count -ne 0) { throw "Generation errors: $($model.model)" }
    if ($model.primary.complete_four_state_families -ne $model.primary.families) { throw "Incomplete quartet: $($model.model)" }
  }
}
```

论文主指标与公开主表统一，但 structured/raw 必须分别成表：

1. `Clean TSR`：clean 状态中 `state_task_success=1` 的 report-family 比例；
2. `Bad-history TSR`、`Source-outage TSR`、`Evidence-conflict TSR`：三个状态分别报告，不能只给 pooled 数；
3. `Conditional Retention`：只在 clean 已通过的 family 中，三个扰动状态继续通过的比例；分母为 `3 × clean-success families`；
4. `Regression Rate = 1 - Conditional Retention`；
5. `Joint four-state TSR`：同一 family 的四个状态全部通过才记 1；
6. CARE 相对 Clean-SFT、GKD-full 的 paired family-cluster bootstrap 95% CI。bootstrap 每次整组重采样四状态，绝不把 4 行当独立样本；
7. `descriptive_pooled_perturbed_task_success_rate` 仅作为描述性附加列。bad-history 要求 exact 渲染，而 outage/conflict 要求安全边界，三者合并值不能作为独立 superiority endpoint、不能替代逐 variant 结果，也不能给因果解释。

三轴与字段次级诊断只使用 `clean+bad_history`：

- exact slot/key/schema；
- exact source/key binding；
- exact content 与 grounded-content fidelity；
- field micro precision/recall/F1、report exact、schema/source-valid 和 unsupported-field rate。

structured clean/bad-history 的辅助字段必须无实质输出；即使 section
content 看似正确，额外 title/summary/warning/next-action/disclaimer 也不能
当作有效替代答案。outage/conflict 单独报告无字段幻觉、无来源引用、边
界语义与结构合规率，只进入 named boundary TSR 与 joint TSR；不要在这
两个状态上解释三轴或字段/content recall。

新真实外测尚未预注册 non-inferiority margin，因此第一版只报告绝对值、成对差值与 95% CI；不能看到结果后套用主表的 NI margin。

不应把 72 个 report groups 写成 72 名患者，也不应将 agent-adjudicated source-exact labels 写成临床专家 gold。若之后完成临床专家复核，应另建带版本号的新 benchmark，而不是覆盖本次锁定结果。

## 9. 最小开发验证

测试只使用合成数据，不读取真实报告：

```powershell
Set-Location $CARE
$env:PYTHONPATH = "$CARE\src;$CARE"
& $PY -m unittest discover `
  -s mechanism_experiments\real_thyroid_external\tests `
  -p "test_*.py" -v
```

测试只验证程序契约，不替代真实模型结果；以本次运行实际的 PASS 数为准。
