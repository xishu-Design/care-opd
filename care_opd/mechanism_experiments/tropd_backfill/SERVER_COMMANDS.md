# TROP-D 独立后补评测（仅推理）

本目录把已经训练完成、此前未进入正式评测的 `tropd_full` 补入三条既有协议：

1. QZBC grouped 主轨：原始 4,200 条请求完整推理，正式表使用既有、结果盲的
   3,900 条 family-clean 视图；4,200 条 entity-held-out 结果仅作诊断，二者不混报。
2. 真实外测 structured v5：65 个 family、260 条四状态请求。
3. 真实外测 raw v5：54 个 family、216 条四状态请求。

默认同时重新推理 `care_opd`。这不是重训，而是用同一批请求、同一 evaluator、
同一生成参数和同一硬件拓扑得到可配对的参考结果，避免把缺少任务哈希的旧汇总
直接拼到 TROP-D 新结果旁边。所有输出写入
`/opt/care-data/care_opd_tropd_backfill`，不会覆盖原实验目录。

完整工作量是 9,352 次模型请求：QZBC 4,200×2=8,400，structured
260×2=520，raw 216×2=432；其中一半是为同协议配对而 fresh 重跑的 CARE-OPD。
所有冻结请求都会进入汇总，generation error 也按失败保留；脚本没有按模型输出、
优劣或显著性做样本筛选，也不会只挑 TROP-D 获胜的子集。

## 1. 一次性环境

```bash
cd /opt/care-workspace/care_opd

export CARE_ROOT=/opt/care-workspace/care_opd
export QZBC_ROOT=/opt/care-workspace/qzbc_source/qzbc-main
export PY=/opt/care-data/venvs/mhg_eval_qwen25vl/bin/python
export TROPD_MODEL=/opt/care-workspace/care_opd/runs/qzbc_baselines_3gpu/baselines/tropd_full/final_model
export CARE_MODEL=/opt/care-workspace/care_opd/runs/qzbc_base_model_3gpu/final_model
export BACKFILL_ROOT=/opt/care-data/care_opd_tropd_backfill
export PHYSICAL_GPU=6
export LOGICAL_DEVICE=cuda:0
export RUN_MODE=fresh
export RUNNER="$CARE_ROOT/mechanism_experiments/tropd_backfill/run_tropd_backfill.sh"

test -s "$RUNNER"
test -s "$CARE_ROOT/mechanism_experiments/tropd_backfill/protocol_guard.py"
```

`PHYSICAL_GPU` 可换成任意实际卡号。脚本**不检查卡是否空闲**，也不会杀进程、
重启服务或下载模型；运行前由操作者自行确认资源。逻辑设备固定为 `cuda:0`，因为
物理卡通过 `CUDA_VISIBLE_DEVICES` 映射进单卡评测进程。
同一 `BACKFILL_ROOT` 由 `flock` 独占，第二个并发启动会立即退出，避免两个进程
同时写同一结果目录。

## 2. 无 GPU 的语法与参数展开检查

此命令只打印四条推理命令，不读取模型和数据，也不会创建输出：

```bash
DRY_RUN=1 LOGICAL_DEVICE=cpu bash "$RUNNER" all
```

正式运行仍须恢复：

```bash
export LOGICAL_DEVICE=cuda:0
```

服务器若安装了 `bash -n`，可再做静态语法检查：

```bash
bash -n "$RUNNER"
"$PY" -c 'from pathlib import Path; p=Path("/opt/care-workspace/care_opd/mechanism_experiments/tropd_backfill/protocol_guard.py"); compile(p.read_text(encoding="utf-8"), str(p), "exec"); print("PY_SYNTAX=PASS")'
"$PY" "$CARE_ROOT/mechanism_experiments/tropd_backfill/protocol_guard.py" --help
```

## 3. 预检

```bash
bash "$RUNNER" preflight
```

预检会失败关闭地完成以下事项：

- 确认 TROP-D 与 CARE-OPD 是本地完整 HF checkpoint，不触发下载；
- 用 grouped 正式协议重新审计 QZBC train/dev/test 清单、4,200 条原请求、
  3,900 条 family-clean ID 及确定性 test recompile；
- 对两个 checkpoint 的行为相关文件逐个计算 SHA-256；
- 对 structured/raw 两个 v5 benchmark 执行各自的 `SHA256SUMS.private`；
- 通过 `thyroid_external_eval.load_locked_inputs` 读取原 seeds/gold 并核对轨道、
  family 数与 case 数；不会重建、补写或替换测试集；
- 在 `$BACKFILL_ROOT/qzbc_grouped_tropd_vs_care_v1/` 固化
  `protocol_lock.json` 与 `DATA_AUDIT.json`。

任何原始资产不存在时，预检会明确打印缺失路径并停止。不要用新造数据绕过。

## 4. 正式后台运行

三条轨道串行运行，两个模型也在同一张卡上依次加载：

```bash
export RUN_LOG="$BACKFILL_ROOT/tropd_backfill_gpu${PHYSICAL_GPU}.nohup.log"
export RUN_PID="$BACKFILL_ROOT/tropd_backfill_gpu${PHYSICAL_GPU}.pid"
mkdir -p "$BACKFILL_ROOT"

nohup bash "$RUNNER" all >"$RUN_LOG" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$RUN_PID"
disown "$pid" 2>/dev/null || true

echo "PID=$pid"
echo "LOG=$RUN_LOG"
tail -n 120 -F "$RUN_LOG"
```

也可逐阶段运行：

```bash
bash "$RUNNER" qzbc
bash "$RUNNER" structured
bash "$RUNNER" raw
```

阶段名只有 `preflight`、`qzbc`、`structured`、`raw`、`all`。每次正式阶段
都会先复验输入和协议锁。

## 5. 结果位置

```text
/opt/care-data/care_opd_tropd_backfill/
├── MODEL_LOCK.json
├── qzbc_grouped_tropd_vs_care_v1/
│   ├── protocol_lock.json
│   ├── DATA_AUDIT.json
│   └── formal/
│       ├── tropd_full/{qzbc_rollouts.jsonl,run_manifest.json,COMPLETE.json,...}
│       ├── care_opd/{qzbc_rollouts.jsonl,run_manifest.json,COMPLETE.json,...}
│       ├── family_clean_views/{tropd_full,care_opd}.jsonl
│       ├── suite_summary.json
│       └── suite_summary.csv
├── real_thyroid_v5_structured/tropd_vs_care_v1/
│   ├── private/run_manifest.private.json
│   ├── public/{summary.json,model_metrics.csv,paired_comparisons.csv,run_manifest.json,...}
│   └── VERIFY.json
└── real_thyroid_v5_raw/tropd_vs_care_v1/
    ├── private/run_manifest.private.json
    ├── public/{summary.json,model_metrics.csv,paired_comparisons.csv,run_manifest.json,...}
    └── VERIFY.json
```

快速检查：

```bash
test -s "$BACKFILL_ROOT/qzbc_grouped_tropd_vs_care_v1/formal/suite_summary.json"
test -s "$BACKFILL_ROOT/real_thyroid_v5_structured/tropd_vs_care_v1/VERIFY.json"
test -s "$BACKFILL_ROOT/real_thyroid_v5_raw/tropd_vs_care_v1/VERIFY.json"

"$PY" - "$BACKFILL_ROOT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
paths = [
    root / "qzbc_grouped_tropd_vs_care_v1/formal/suite_summary.json",
    root / "real_thyroid_v5_structured/tropd_vs_care_v1/VERIFY.json",
    root / "real_thyroid_v5_raw/tropd_vs_care_v1/VERIFY.json",
]
for path in paths:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "COMPLETE" if path.name == "suite_summary.json" else payload["status"] == "PASS"
    print("RESULT_OK", path)
PY
```

## 6. 中断恢复与不覆盖规则

真实 v5 evaluator 原生支持同合同续跑。仅在对应目录是本次**同一模型、同一输入、
同一生成参数和同一 evaluator**产生的 partial run 时执行：

```bash
RUN_MODE=resume bash "$RUNNER" structured
RUN_MODE=resume bash "$RUNNER" raw
```

manifest 不同会由原 evaluator 拒绝。QZBC `qzbc_rollout` 没有同等级的逐条 resume
合同，因此 wrapper 不会静默覆盖 partial QZBC 输出。若 QZBC 中断，保留原目录并换
一个全新路径：

```bash
export QZBC_OUT_ROOT="$BACKFILL_ROOT/qzbc_grouped_tropd_vs_care_v1_rerun_$(date +%Y%m%d_%H%M%S)"
bash "$RUNNER" qzbc
```

已完成且 `COMPLETE.json`/`VERIFY.json` 复验通过的阶段会安全跳过。

## 7. 与旧基线对齐

本次新结果已经包含 fresh CARE-OPD reference，正式比较优先使用新 run 内部的配对
结果。不要从旧 CSV 手工复制某一行加入新表。

如确需验证旧 real-v5 运行能否作描述性并列，必须比较 private manifests：

```bash
export GUARD="$CARE_ROOT/mechanism_experiments/tropd_backfill/protocol_guard.py"
export OLD_STRUCTURED=/opt/care-workspace/care_opd/runs/private_real_thyroid_external/eval_direct_gold_structured_six_models_v5
export NEW_STRUCTURED="$BACKFILL_ROOT/real_thyroid_v5_structured/tropd_vs_care_v1"

"$PY" "$GUARD" compare-real-manifests \
  --left "$OLD_STRUCTURED/private/run_manifest.private.json" \
  --right "$NEW_STRUCTURED/private/run_manifest.private.json" \
  --output "$NEW_STRUCTURED/OLD_RUN_COMPATIBILITY.json"
```

只有输出 `status=PASS` 时，输入、生成、bootstrap、verifier、evaluator 与 QZBC
合同才一致，可作协议匹配的描述性并列；严格 paired delta 仍应使用同一 run 的
case-level family bootstrap。若为 `INCOMPATIBLE`，禁止拼表、求差或排序。

旧 QZBC `baseline_hf_evaluation/*/summary.json` 没有 checkpoint fingerprint 和
split manifest，不能单凭一行 pass rate 证明与本次同协议。本脚本重新跑 CARE-OPD
正是为了关闭这个缺口。正式主表使用新 summary 的 3,900 条 family-clean 视图；
4,200 条 full view 只作为诊断，不能与旧 3,900 条数字混称。

## 8. 为什么没有自动加入 TCMEval / CMB / TCMBench

已经核对以下现有入口：

- `public_eval/tcmeval_pa/run_existing_gpu4_5.sh`
- `public_eval/cmb_exam/run_existing_gpu4_5.sh`
- `public_eval/tcmbench_v2/run_existing_gpu4_5.sh`

它们的 frozen suite / preregistration lock 绑定原六模型标签，当前 evaluator 也按原
标签集合验收。把 TROP-D 冒充任一旧标签会破坏 checkpoint identity；临时改 suite
后继续沿用旧 lock 同样无效。因此本 backfill 不自动调用这些入口，也不会自动下载
公开数据。若之后要补公开选择题轨，必须新建含 `tropd_full` 的 versioned suite、
新 preregistration lock 和全新 output root，并继续使用原数据文件/协议；不得把缺失
资产自动重造为“同一测试集”。这属于可选扩展，不是本次 QZBC + 真实双轨核心运行
的完成条件。

## 9. 本地单元测试

Windows 工作区可运行：

```powershell
.venv_ood/Scripts/python.exe -m unittest `
  care_opd/mechanism_experiments/tropd_backfill/test_protocol_guard.py -v
```

测试只验证锁不可变性、协议比较、CLI 和脚本安全约束，不进行推理。
