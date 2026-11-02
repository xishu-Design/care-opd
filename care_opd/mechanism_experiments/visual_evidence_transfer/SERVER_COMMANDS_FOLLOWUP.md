# 三项有针对性的多模态补充实验

本包只新增 `followup_*.py` 和配套测试，不改变 V4 评分、模型、任务或已有结果。推荐在原 V4 完成后使用。

## 实验与可支持的结论

| 用户关注点 | 本轮执行 | 核心端点 |
|---|---|---|
| 正常证据下有效回答 | 直接审计 V4 五个方法的首次输出，不重生成 | 所有题准确率、YES/NO 覆盖率、已回答准确率，分别列 ABSTAIN/未知格式 |
| 有效换图后的更新 | 直接复用原始同题异图、真实相反标注配对 | 新旧都答对、原来答对的题更新后答对、错误翻转/弃答 |
| 局部失效时保护正确内容 | 四个文本后端在相同冻结 VLM 观察上新推理 | A 正确更新或局部弃答，B 视觉结论保持，两个 C 文本属性保持、误改与联合成功 |

第三项每块含 A、B、C 三个彼此独立的数据记录。

- A 是变化的视觉问题，三种状态为原图、图像缺失、有效相反标注换图。
- B 是未变化的第二个视觉问题，答案有独立源标注。
- C 是另一图像簇的两个非二元 QA 标注，作为明确提供的文本事实保留。优先按原始 `content_type` 区分类别，无类别时保留独立问题标识。不会编造不存在的临床属性。
- C 不属于 A 或 B，不是同一患者，不可作为 A/B 的视觉答案证据。其原始 qid、来源文件 SHA-256 进入私有溯源记录。
- 8 字符代码仅作为辅助复制控制，不代表临床内容保持能力。

这比单一代码复制更有信息，但仍是冻结视觉前端后的多来源回答控制测试，不是端到端视觉模型训练，也不能单凭此测试将变化归因于 Protect 训练项。后者需要匹配的去 Protect 消融。

原 V4 仅 10 个图像簇。为避免同一源簇重复当独立样本，本轮按身份种子顺序选择不重叠三元组，最多约 3 块。它的目的首先是确认协议、接口与失败位置，**不是用很小的开发集宣称 TMM 主实验获胜**。不生成 p 值、不把多条件/多字段扩充当作增加独立样本数。实际块数写入选择清单；没有合格块则停止，不事后降低独立性标准。

脚本没有写死 10 个簇。后续同一协议下如已完成更大、在生成前固定的 validation 运行，只需将 `SOURCE_CONFIG` 指向其配置并设全新的 `FOLLOWUP_WORK`，即可按完全相同规则处理所有合格源簇。不能根据小样本哪种条件对 CARE 更有利来选取扩大部分；正式确认仍需与开发过程隔离的数据。

## 1. 上传和独立安装

将本包 `.tar.gz` 和对应 `.sha256` 上传到 `/opt/care-workspace`。下面代码在 Bash 中执行。圆括号是子 shell，失败不会关闭交互终端。

```bash
(
set -e
cd /opt/care-workspace
PKG=CARE_OPD_TMM_VISUAL_FOLLOWUP_V1.tar.gz
sha256sum -c "$PKG.sha256"
DEST=/opt/care-workspace/care_opd_visual_followup_v1
install -d -m 700 "$DEST"
if test -n "$(find "$DEST" -mindepth 1 -maxdepth 1 -print -quit)"; then
  echo "安装目录非空；已安装过请跳过解压，不要覆盖运行中的代码：$DEST" >&2
  exit 2
fi
tar -xzf "$PKG" -C "$DEST"
)
```

本包不包含数据和权重、不启动下载，不触碰另一张卡的 QZBC。首次解压后，后续从第 2 步继续。

解压后也可一次后台完成准备、预检和推理（与后续分步指令二选一，不要同时启动）：

```bash
(
set -e
RUNNER=/opt/care-workspace/care_opd_visual_followup_v1/mechanism_experiments/visual_evidence_transfer/run_followup_v1.sh
WORK=/opt/care-data/care_opd_tmm_visual_followup_v1
test -s "$RUNNER"
install -d -m 700 "$WORK"
LOG="$WORK/followup_gpu0_$(date +%Y%m%d_%H%M%S).log"
nohup env PHYSICAL_GPU=0 FOLLOWUP_WORK="$WORK" bash "$RUNNER" all >"$LOG" 2>&1 </dev/null &
run_pid=$!
printf '%s\n' "$run_pid" >"$WORK/followup_gpu0.pid"
disown "$run_pid" 2>/dev/null || true
printf 'PID=%s\nLOG=%s\n' "$run_pid" "$LOG"
sleep 3
ps -fp "$run_pid" || true
tail -n 80 "$LOG"
)
```

## 2. CPU 审计与准备

路径全部在本段重新定义，不依赖上次终端 export。使用已通过的 Python 环境。

```bash
(
set -e
PY=/opt/care-data/venvs/mhg_eval_qwen25vl/bin/python
SRC=/opt/care-workspace/care_opd_visual_followup_v1/mechanism_experiments/visual_evidence_transfer
SOURCE_CONFIG=/opt/care-data/care_opd_tmm_visual_gpu45_v4/config_validation10.json
WORK=/opt/care-data/care_opd_tmm_visual_followup_v1
test -x "$PY"
test -s "$SOURCE_CONFIG"
export PYTHONPATH="$SRC"
export PYTHONDONTWRITEBYTECODE=1
"$PY" -B "$SRC/followup_run.py" audit --source-config "$SOURCE_CONFIG" --work "$WORK"
"$PY" -B "$SRC/followup_run.py" prepare --source-config "$SOURCE_CONFIG" --work "$WORK"
)
```

两个阶段均只读旧目录。原五个方法和共享观察必须完整、源协议必须是 V4 validation、模型/提示/任务绑定必须一致，否则拒绝启动。读取官方源标注时会校验原 selection manifest 的 SHA-256。缺少源文件要恢复相同文件，不能手填假标签。

生成后可先看：

```bash
WORK=/opt/care-data/care_opd_tmm_visual_followup_v1
cat "$WORK/source_audit/usefulness_and_errors.csv"
cat "$WORK/source_audit/valid_image_update.csv"
cat "$WORK/selection_manifest.json"
```

此时前两项已补齐，不消耗新 GPU 推理。不得把未知输出全算作弃答，不把 `no_image` 的拒答正确率混为正常回答准确率。

## 3. GPU 0 预检与后台运行

默认 GPU 0，GPU 1 留给原 QZBC。若 0 卡有其他任务，请自行协调；脚本不清理任何进程。这里不作空卡门禁，但并发占用仍可能导致 OOM。

```bash
(
set -e
PY=/opt/care-data/venvs/mhg_eval_qwen25vl/bin/python
SRC=/opt/care-workspace/care_opd_visual_followup_v1/mechanism_experiments/visual_evidence_transfer
WORK=/opt/care-data/care_opd_tmm_visual_followup_v1
export PYTHONPATH="$SRC"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"$PY" -B "$SRC/followup_run.py" preflight --work "$WORK" --full-hash
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$WORK/followup_gpu0_${STAMP}.log"
PIDFILE="$WORK/followup_gpu0.pid"
nohup "$PY" -B -u "$SRC/followup_run.py" run --work "$WORK" >"$LOG" 2>&1 </dev/null &
run_pid=$!
printf '%s\n' "$run_pid" >"$PIDFILE"
disown "$run_pid" 2>/dev/null || true
printf 'PID=%s\nLOG=%s\n' "$run_pid" "$LOG"
sleep 3
ps -fp "$run_pid" || true
tail -n 80 "$LOG"
)
```

按 clean_sft、gkd_full、tropd_full、care_opd 顺序，每次加载一个模型；所有方法同一提示、相同输入、同一 1024-token 输出预算、同一贪心解码。新任务从两节扩为五节，因此预算统一提高并在运行前冻结，不因某个模型答错追加预算。

预检首次计算完整模型哈希，可能静默一段时间；不会重跑 VLM 或构建 RAG 索引。`run` 自动从已保存行续跑；**格式错误、空回答、截断、内容错误仍是已保存的首次输出，不会重生成**。只重试尚未产生输出的技术故障。若存在锁文件，先核对其中 PID 的所属任务，再决定如何处理，不要盲目删锁。

## 4. 查看进度、汇总

```bash
PY=/opt/care-data/venvs/mhg_eval_qwen25vl/bin/python
SRC=/opt/care-workspace/care_opd_visual_followup_v1/mechanism_experiments/visual_evidence_transfer
WORK=/opt/care-data/care_opd_tmm_visual_followup_v1
env PYTHONPATH="$SRC" "$PY" -B "$SRC/followup_run.py" status --work "$WORK"
find "$WORK" -maxdepth 1 -name 'followup_gpu0_*.log' -printf '%T@ %p\n' | sort -n
```

四个后端全部完成时自动评分。也可在运行结束后单独执行：

```bash
PY=/opt/care-data/venvs/mhg_eval_qwen25vl/bin/python
SRC=/opt/care-workspace/care_opd_visual_followup_v1/mechanism_experiments/visual_evidence_transfer
WORK=/opt/care-data/care_opd_tmm_visual_followup_v1
env PYTHONPATH="$SRC" "$PY" -B "$SRC/followup_run.py" score --work "$WORK"
find "$WORK/summary" -maxdepth 2 -type f -printf '%s %p\n' | sort
```

最终汇总在 `$WORK/summary/<响应指纹前16位>/`。重点回传 `locality_metrics.json`、`locality_metrics.csv`、四个 `*_protection.csv`，以及 `$WORK/source_audit` 下的两个 CSV。`raw_rows.private.jsonl` 含原始输出与参考答案，仅供私有诊断。

## 判断是否值得进一步补正式实验

1. 正常证据下若 CARE 仍明显少答/错答，先根据逐条输出区分观察错误、后端错误、格式或输入协议问题，不把全部失败归因于格式。
2. 换图时若只是翻转多而双图均正确低，不支持有效更新能力；不能把换图改成统一拒答以提高分数。
3. 局部保护若大量依赖弃答或只复制文本成功，不能当作视觉 Protect 的优势；需要 B 正确性、条件保持分母、A 更新和联合成功共同成立。
4. 任何后续适配训练都必须只用训练集并为对照提供同等机会，协议/超参在独立留出评测前固定。本包不自动训练，不承诺正向结果，也不将此前看过的测试集包装为新留出集。

这里的规则来自实验设计 skill 的证据约束：补能区分假说的对照，而不是换口径使 CARE 显得更好。既有负向结果应保留并在论文中准确界定。
