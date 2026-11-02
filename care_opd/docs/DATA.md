# 正式训练数据

## 数据边界

`qzbc_seed_builder` 直接读取当前 qzbc 仓库中带来源标识和可见性标签的
已审校资产，再把它们放入无真实患者信息的受控上下文。它只读取：

- `herbs.json` 与 `formulas.json`；
- 人工复核的 `herb_incompatibilities.json`；
- 患者可见的 `diet_precautions.json`；
- `review_status=approved_training_only`、已脱敏且明确禁止作为临床证据的
  学生评测条目。

它不会读取 SQLite 患者库、聊天记录、真实处方、Graphiti、Mem0 或线上
检索缓存。数据的正确表述是“qzbc 已审校知识资产 + 受控合成业务/病例
上下文”，而不是“真实临床数据”或“纯模板数据”。

## 八类任务映射

| qzbc task type | 主要 qzbc 资产 | 学习目标 |
|---|---|---|
| `doctor_review_text` | 已人工复核的配伍关系 | 风险陈述、来源引用、人工复核边界；仅作 host-overridden retention |
| `doctor_ai_chat` | 方剂目录 + 合成授权业务记录 | 医生专业问答与资料不足边界 |
| `patient_education` | 患者可见饮食科普 | 通俗转述且不个体化诊疗 |
| `patient_diet_guidance` | 患者可见饮食注意资料 | 区分传统科普、临床证据与治疗方法 |
| `student_analysis` | 方剂名称、组成及来源 | 对照证据是否支持学习理由 |
| `student_quiz_explanation` | approved-training-only 题库 | 按标准答案讲解且不转成临床证据 |
| `student_ai_chat` | 药材目录字段 | 来源追溯、翻译复核与知识边界 |
| `quiz_generation` | 方剂目录字段 | 只围绕可追溯名称、组成和证据边界出题 |

每个目标仍严格遵循 qzbc 原生
`LLMRequest → GeneratedResponse`，不包含工具 action 或新 Agent 协议。

## 防泄漏切分

先对每一种知识资产做确定性 train/dev/test 切分，再从对应 split 的资产
生成任务样本。因此同一药材条目、方剂条目、配伍约束、饮食条目或题库
条目不会跨集合出现。不能只按 `source_id` 去重，因为多个目录条目可能
共享同一个来源；实现按规范化后的完整资产条目隔离，并在 manifest 中
记录：

```text
grounding_asset_split.strategy
grounding_asset_split.counts
grounding_asset_split.overlap_check = PASS
```

每条样本同时保留 `asset_id`、`source_id`、`source_record_id`、许可字段
和 qzbc contract fingerprint。

## 故障状态与训练视图

train/dev/test 的 clean seed 分别编译为：

- `clean`：原始 Evidence Packet；
- `bad_history`：插入受控的错误可信历史回答；
- `source_outage`：一个已引用证据源变为不可用；
- `evidence_conflict`：当前证据出现受控冲突。

编译器输出：

```text
data/qzbc_formal/seeds/
  qzbc_native_seeds.{train,dev,test}.jsonl
  qzbc_native_seeds.manifest.json

data/qzbc_formal/compiled/{train,dev,test}/
  qzbc_clean_mixed_sft.jsonl
  qzbc_clean_sft.jsonl
  qzbc_recovery_sft.jsonl
  qzbc_on_policy_requests.jsonl
  qzbc_stream_retention_sft.jsonl
  manifest.json
```

`doctor_review_text` 会被 qzbc 宿主结构化结果覆盖，所以只进入格式保持
视图，不计入主要模型能力增益。其余七类为 model-visible 任务。

## 正式生成

默认命令生成 12,000 个分层 clean seed，并编译为四种状态：

```bash
cd /opt/care-workspace/care_opd
source .venv/bin/activate

QZBC_ROOT=/opt/care-workspace/qzbc_source/qzbc-main \
QZBC_SEED_TOTAL=12000 \
OVERWRITE_DATA=1 \
bash scripts/01_build_formal_data.sh
```

核对 manifest 后再训练：

```bash
python3 - <<'PY'
import json
p = "data/qzbc_formal/seeds/qzbc_native_seeds.manifest.json"
d = json.load(open(p, encoding="utf-8"))
print(d["row_count"], d["split_counts"])
print(d["qzbc_grounding_assets"]["usable_counts"])
print(d["grounding_asset_split"])
PY
```

## 许可与发布边界

生成器不会删除 qzbc 原有许可和来源记录。当前目录资产中可能同时包含
Apache-2.0 的教学评测条目和带 CC-BY-NC-4.0 限制的非商业目录资产。
参赛研究、模型权重公开和商业部署不是同一许可场景：公开数据、模型或
权重前，必须依据 manifest 中的来源逐项复核原始许可、署名及衍生物
条款。没有明确许可的记录不得自行推定为可再分发。

所有合成上下文和评测结果只支持工程研究结论，不能表述为临床有效性、
诊断准确率或真实患者收益。
