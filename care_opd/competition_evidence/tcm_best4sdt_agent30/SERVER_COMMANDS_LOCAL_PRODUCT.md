# 选择中立专家盲审 `local_product` 主轨服务器运行命令（`/opt/care-workspace`，物理 GPU 3）

本页用于时间受限的30题正式专家盲审。该轨保留真实医生端角色路由、CARE-OPD、
Graphiti、Mem0、SQL 对话上下文和冻结完整 `content/data` 的本地结构化检索；不等待、
不调用 LightRAG 全量索引。原 LightRAG 建库状态不会被删除，后续仍可单独续跑增强轨。
服务器增量包中的本轮评测任务只包含已经冻结并校验哈希的30条无答案题面、选择清单和
重叠审计；TCM-BEST4SDT 完整300例源文件及其独立参考答案文件均不进入包，也不会在推理
阶段读取。冻结 QZBC 产品快照可能含与本轮无关的公开数据资源，但它们不属于本轮活跃
检索源；正式运行还会对实际启用的五类本地结构化知识源单独做重叠审计。

> **本轮必须使用全新的运行根和盲审根。** 上一轮第二例在已经形成可见答案后，因项目侧
> “具体候选”和“事实—判断链”规则被单独重采样；这会造成四系统答案选择不对称。新协议
> 对每个系统每题冻结首个用户可见答案，内容、结构、引用、长度和安全维度只作统一审计，
> 不触发拒收、重试或删题。上一轮回答和运行清单完整保留为开发审计，不得复用或编辑；
> 本轮只允许按第4节白名单复用满足“首个可见答案已冻结”条件的三个基线，“神农三言”
> 从首个预注册病例干净重跑。

## 1. 上传、校验并解压新包

将以下两个文件上传到服务器现有目录：

```text
/opt/care-workspace/care_opd/runs/public_eval/tcm_best4sdt_agent30_bundle/
```

文件名：

```text
TCM_BEST4SDT_AGENT30_LOCAL_PRODUCT_GPU3_FAIR_BLIND_REVIEW_FINAL.tar.gz
TCM_BEST4SDT_AGENT30_LOCAL_PRODUCT_GPU3_FAIR_BLIND_REVIEW_FINAL.tar.gz.sha256
```

执行：

```bash
# 不要在交互式登录终端启用 `set -e`；任一预检非零都会关闭整个 shell。
# runner 子进程内部已经启用严格模式。
set +e
set +u
set +o pipefail
umask 077

export FIX_DIR=/opt/care-workspace/care_opd/runs/public_eval/tcm_best4sdt_agent30_bundle
export FIX_NAME=TCM_BEST4SDT_AGENT30_LOCAL_PRODUCT_GPU3_FAIR_BLIND_REVIEW_FINAL.tar.gz
cd "$FIX_DIR"
sha256sum -c "$FIX_NAME.sha256"

export BUNDLE_ROOT=/opt/care-workspace/four_agent_expert_blind_bundle
export CARE_ROOT="$BUNDLE_ROOT/care_opd"
install -d -m 700 "$CARE_ROOT" /opt/care-data/four_agent_expert_blind
tar -xzf "$FIX_DIR/$FIX_NAME" -C "$CARE_ROOT"

chmod 700 \
  "$CARE_ROOT/competition_evidence/local_four_agent_server/run_four_agent_local_server.sh" \
  "$CARE_ROOT/competition_evidence/local_four_agent_server/serve_care_transformers.py" \
  "$CARE_ROOT/competition_evidence/tcm_best4sdt_agent30/run_tcm_best4sdt_agent30_gpu3.sh" \
  "$CARE_ROOT/competition_evidence/tcm_best4sdt_agent30/run_tcm_best4sdt_agent30_gpu5.sh"
```

## 2. 结束耗时的 LightRAG 建库进程，但保留全部进度

只向 `bootstrap_full_product_lightrag.py` 进程发送 TERM，不删除或移动
`runtime/lightrag`、批次清单、日志和索引目录。LightRAG 服务本身可以保持空闲；
`local_product` 不会调用它。

```bash
mapfile -t LIGHTRAG_BOOTSTRAP_PIDS < <(
  pgrep -f '[b]ootstrap_full_product_lightrag\.py' || true
)
if ((${#LIGHTRAG_BOOTSTRAP_PIDS[@]})); then
  printf '正在停止 LightRAG bootstrap：%s\n' "${LIGHTRAG_BOOTSTRAP_PIDS[*]}"
  for pid in "${LIGHTRAG_BOOTSTRAP_PIDS[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  for _ in $(seq 1 30); do
    remaining=0
    for pid in "${LIGHTRAG_BOOTSTRAP_PIDS[@]}"; do
      kill -0 "$pid" 2>/dev/null && remaining=1
    done
    ((remaining == 0)) && break
    sleep 1
  done
fi
sync
pgrep -af '[b]ootstrap_full_product_lightrag\.py' || echo 'LightRAG bootstrap 已停止；原状态已保留'
```

若最后一行仍列出进程，先查看对应 PID 和日志，不要删除索引目录，也不要同时启动正式
四系统运行。

## 3. 冻结运行环境

```bash
export BUNDLE_ROOT=/opt/care-workspace/four_agent_expert_blind_bundle
export CARE_ROOT="$BUNDLE_ROOT/care_opd"
export QZBC_ROOT="$CARE_ROOT/competition_evidence/tcm_best4sdt_agent30/qzbc_full_product"
export FULL_PRODUCT_ROOT=/opt/care-workspace/qzbc_source/qzbc-main
export MODEL_ROOT=/opt/care-data/four_agent_expert_blind/models
export CARE_MODEL=/opt/care-workspace/care_opd/runs/qzbc_base_model_3gpu/final_model

export EVAL_PY=/opt/care-workspace/opdrevi_venv/bin/python
export VLLM_BIN=/opt/care-workspace/opdrevi_venv/bin/vllm
export BASELINE_VLLM_BIN="$VLLM_BIN"
export CARE_SERVER_BACKEND=transformers
export CARE_HF_PY=/opt/care-data/venvs/mhg_eval_qwen25vl/bin/python
export BACKEND_PY=/opt/care-workspace/four_agent_expert_blind_bundle/qzbc_private_eval/.conda_eval/bin/python

export SYSTEM_API_PROFILE=local_product
export PHYSICAL_GPUS=3
# 本轮按操作者要求不检查 GPU 3 是否已有计算进程；该确认只关闭占用拦截，
# 不关闭 GPU 存在性、样本哈希、模型、服务和盲审协议检查。
export ACTIVE_GPU_PROCESS_CHECK_ACK=I_ACCEPT_GPU_CONTENTION_RISK
export CARE_MAX_MODEL_LEN=32768
export BASELINE_MAX_MODEL_LEN=8192
export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HUB_DISABLE_XET=1
export OLLAMA_BASE_URL=http://127.0.0.1:11434
export OLLAMA_HOST=127.0.0.1:11434
export QA_KNOWLEDGE_INDEX_PATH="$FULL_PRODUCT_ROOT/runtime/qa_knowledge/qa_index.db"

export TCM_BEST_TASK_ROOT="$CARE_ROOT/runs/public_eval/tcm_best4sdt_agent30_reference_free_fixed_seed20260825"
export TCM_BEST_RUN_ROOT=/opt/care-data/four_agent_expert_blind/tcm_best4sdt_agent30_fair_blind_review_four_system_runs
export TCM_BEST_REVIEW_ROOT=/opt/care-data/four_agent_expert_blind/tcm_best4sdt_agent30_fair_blind_review_expert_blind
export REVIEWERS=expert_01,expert_02
export TCM_BEST_RUNNER="$CARE_ROOT/competition_evidence/tcm_best4sdt_agent30/run_tcm_best4sdt_agent30_gpu3.sh"

test -x "$EVAL_PY"
test -x "$VLLM_BIN"
test -x "$CARE_HF_PY"
test -x "$BACKEND_PY"
test -s "$CARE_MODEL/config.json"
"$BACKEND_PY" -c 'import fastapi, sqlalchemy, uvicorn, alembic, graphiti_core, mem0, qdrant_client, redislite; print("BACKEND_STACK_OK")'
```

QA 索引是可选扩展：若上述路径已有非空索引，runner 会将其纳入重叠审计并启用；若文件
不存在，runner 会输出 `QA_KNOWLEDGE_INDEX=DISABLED_NOT_PRESENT`，并仅使用完整的冻结
结构化知识库。不要为通过预检创建空数据库或伪造 QA 记录。

确保本机 Ollama 已启动，且 Graphiti/Mem0 所需的两个冻结模型可用：

```bash
export OLLAMA_BIN="$(command -v ollama)"
test -x "$OLLAMA_BIN" || { echo '未找到 Ollama' >&2; exit 2; }

if ! curl -fsS "$OLLAMA_BASE_URL/api/tags" >/dev/null 2>&1; then
  nohup env OLLAMA_HOST="$OLLAMA_HOST" OLLAMA_CONTEXT_LENGTH=32768 \
    OLLAMA_NUM_PARALLEL=1 "$OLLAMA_BIN" serve \
    >/opt/care-data/four_agent_expert_blind/ollama_local_product.log 2>&1 </dev/null &
  echo $! >/opt/care-data/four_agent_expert_blind/ollama_local_product.pid
fi
for _ in $(seq 1 60); do
  curl -fsS "$OLLAMA_BASE_URL/api/tags" >/dev/null 2>&1 && break
  sleep 2
done
curl -fsS "$OLLAMA_BASE_URL/api/tags" >/dev/null
OLLAMA_HOST="$OLLAMA_HOST" "$OLLAMA_BIN" list | awk 'NR>1 {print $1}' | grep -Eq '^qwen2\.5:7b(:latest)?$|^qwen2\.5:7b$'
OLLAMA_HOST="$OLLAMA_HOST" "$OLLAMA_BIN" list | awk 'NR>1 {print $1}' | grep -Eq '^bge-m3(:latest)?$|^bge-m3$'
```

`local_product` 不要求启动 `start_lightrag.sh`，也不要求
`guardtcm_lightrag_docs.bootstrap_manifest.private.json` 达到 `COMPLETE`。

正式病例请求前，runner 会先通过产品自带的 `write_role_memory` 将一条不含病例事实的
角色回答格式偏好写入 Mem0，再通过同角色、同用户的 `search_role_memory` 原文回读。
该写入—回读资格检查用于证明 Mem0 实际可用，不依赖原始聊天自动晋升为长期记忆。
同一固定偏好中的中医推理组织关系会通过 Graphiti 的 `EntityNode`、`EntityEdge` 与
FalkorDB 进行类型化写入，再由同角色、同用户图分区回读；已结构化关系不再重复调用
LLM 做实体关系抽取。`local_product` 同时关闭暖身对话的异步图谱抽取，以免它占用唯一的
Graphiti 运行队列；这不关闭 Graphiti，本轮仍逐题闭锁验证真实图写入和读取。

## 4. 只复用三个已完成基线，并进行复制前后逐文件哈希一致性核对

本次修复没有改变冻结题目、三个基线模型、基线提示或生成参数，因此可复用旧运行根中
三个已完成的基线目录，避免再次占用 GPU。复用采用显式白名单：只复制下列三个目录，
严禁复制 `shennong_sanyan_full`、`runtime`、`logs`、`driver_manifest.private.txt`、
`four_agent_execution.private.json` 或任何盲审目录/盲映射/专家表格。

```bash
export OLD_RUN_ROOT=/opt/care-data/four_agent_expert_blind/tcm_best4sdt_agent30_case_evidence_v2_four_system_runs
export NEW_RUN_ROOT=/opt/care-data/four_agent_expert_blind/tcm_best4sdt_agent30_fair_blind_review_four_system_runs
export BASELINE_REUSE_AUDIT=/opt/care-data/four_agent_expert_blind/tcm_best4sdt_agent30_fair_blind_review_baseline_reuse_audit
export FROZEN_TASKS="$TCM_BEST_TASK_ROOT/tcm_best4sdt_agent30.tasks.private.jsonl"
test -s "$FROZEN_TASKS"
test "$(sha256sum "$FROZEN_TASKS" | awk '{print $1}')" = \
  01f802a2973fa615f007d2305d59e1ad0a893013614fef1f11fd7a946b8f92d5

BASELINE_SYSTEMS=(
  huatuogpt3_local_medical_agent
  glm4_local_role_model
  internlm3_local_role_model
)

# 新根和复用审计根都必须是本轮专用目录，禁止覆盖任何既有结果或审计记录。
if test -e "$NEW_RUN_ROOT" || test -e "$BASELINE_REUSE_AUDIT"; then
  echo "新运行根或复用审计根已存在，请先人工核对" >&2
  echo "RUN_ROOT=$NEW_RUN_ROOT" >&2
  echo "AUDIT_ROOT=$BASELINE_REUSE_AUDIT" >&2
  exit 2
fi
install -d -m 700 "$NEW_RUN_ROOT" "$BASELINE_REUSE_AUDIT"

# 复制前：确认三个白名单基线均为30条成功结果，并记录相对路径逐文件 SHA-256。
for system in "${BASELINE_SYSTEMS[@]}"; do
  test -s "$OLD_RUN_ROOT/$system/responses.private.jsonl" || {
    echo "旧基线不完整：$system" >&2
    exit 2
  }
  "$EVAL_PY" - "$OLD_RUN_ROOT/$system/responses.private.jsonl" "$system" "$FROZEN_TASKS" <<'PY'
import hashlib, json, sys
path, system, tasks_path = sys.argv[1:]
rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
tasks = [json.loads(line) for line in open(tasks_path, encoding="utf-8") if line.strip()]
task_hashes = {
    task["task_id"]: hashlib.sha256(
        json.dumps(task, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    for task in tasks
}
assert len(rows) == 30, (system, len(rows))
assert all(row.get("status") == "success" for row in rows), system
assert all(int(row.get("attempt_count") or 0) == 1 for row in rows), (
    system,
    "基线存在重试，无法仅凭最终文件证明冻结的是首个可见答案；该基线必须在公平协议下重跑",
)
assert len({row.get("task_id") for row in rows}) == 30, system
assert set(row.get("task_id") for row in rows) == set(task_hashes), system
assert all(row.get("task_payload_sha256") == task_hashes[row["task_id"]] for row in rows), system
print("BASELINE_COMPLETE", system, len(rows))
PY
done
(
  cd "$OLD_RUN_ROOT"
  for system in "${BASELINE_SYSTEMS[@]}"; do
    find "$system" -type f -print0 | sort -z | xargs -0 sha256sum
  done
) >"$BASELINE_REUSE_AUDIT/before_copy.sha256"

# 使用独立副本（支持时采用 reflink）；不要使用硬链接，以免后续写入污染旧证据。
for system in "${BASELINE_SYSTEMS[@]}"; do
  cp -a --reflink=auto "$OLD_RUN_ROOT/$system" "$NEW_RUN_ROOT/"
done

# 复制后：相对路径与内容哈希必须与复制前完全一致。
(
  cd "$NEW_RUN_ROOT"
  for system in "${BASELINE_SYSTEMS[@]}"; do
    find "$system" -type f -print0 | sort -z | xargs -0 sha256sum
  done
) >"$BASELINE_REUSE_AUDIT/after_copy.sha256"
diff -u \
  "$BASELINE_REUSE_AUDIT/before_copy.sha256" \
  "$BASELINE_REUSE_AUDIT/after_copy.sha256"
sha256sum \
  "$BASELINE_REUSE_AUDIT/before_copy.sha256" \
  "$BASELINE_REUSE_AUDIT/after_copy.sha256" \
  >"$BASELINE_REUSE_AUDIT/manifest_files.sha256"

# 反向确认没有把旧神农结果、运行状态、日志或清单带入新根。
for forbidden in \
  shennong_sanyan_full runtime logs \
  driver_manifest.private.txt four_agent_execution.private.json; do
  test ! -e "$NEW_RUN_ROOT/$forbidden" || {
    echo "禁止复用的旧产物进入新根：$forbidden" >&2
    exit 2
  }
done
echo "BASELINE_REUSE_VERIFIED=3"
```

上述 `before_copy.sha256` 与 `after_copy.sha256` 均须保留在回传包中。若任一基线不是
30条成功结果、任一题 `attempt_count != 1` 或 `diff` 非空，不得将其当作已完成基线跳过；
应在同一选择中立协议下重跑该基线。不能仅凭最终答案内容推断早期尝试是否已生成。

## 5. 验证已冻结的无答案样本、锁定首例并做正式预检

```bash
bash "$TCM_BEST_RUNNER" prepare
sha256sum "$TCM_BEST_TASK_ROOT/tcm_best4sdt_agent30.tasks.private.jsonl"

# 预注册首例（canary）必须保持题目、病例材料和顺序不变。
"$EVAL_PY" - "$TCM_BEST_TASK_ROOT/tcm_best4sdt_agent30.tasks.private.jsonl" <<'PY'
import hashlib, json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    first = json.loads(next(line for line in handle if line.strip()))
digest = hashlib.sha256(
    json.dumps(first, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
assert first["task_id"] == "BEST30-c061b9527239d5b57092e4e8", first["task_id"]
assert digest == "d14df2d27d04f7f5fd60e53a71a26a0b7a9482978b9266ffae121b6e31f32d2e", digest
assert first["paired_case_material_sha256"] == "e264b05f467cb2f67e26a9828812a45a1b178dae9db689410965a4a81aa6953e"
print("CANARY_LOCK_OK", first["task_id"], digest)
PY

bash "$TCM_BEST_RUNNER" preflight
preflight_rc=$?
if ((preflight_rc == 0)); then
  echo 'PRECHECK_OK：可以启动正式任务'
else
  echo "PRECHECK_FAILED：返回码=${preflight_rc}；终端保持打开，请先检查上方错误" >&2
fi
```

冻结任务文件的预期 SHA-256 为：

```text
01f802a2973fa615f007d2305d59e1ad0a893013614fef1f11fd7a946b8f92d5
```

预检必须出现 `MEMORY_ROUTE_TASKS_OK=30`，证明30个冻结病例不会被误路由到偏好记忆
响应。正式 `run` 的第一条生成固定为上述 canary；执行器会逐条校验“当前请求中的完整
病例材料—病例哈希—证据包—最终回答”闭环。若身份、题目绑定、无参考答案、产品栈调用
或持久化完整性失败，会保存执行审计并终止；若已经形成用户可见答案，则该答案立即冻结。

六节结构与非空性、病例引用、候选、事实—判断链、治法、方药、鉴别/复核对象、整体降级、
固定剂量、临床终局性、字符数和分句数全部写入私有审计，但不造成重试、替换、删题或资格
失败。它们是专家要比较的被测结果，不能先由项目侧规则筛选。三次调用只用于生成前基础设施
故障；若三次均未形成可见答案，则保留失败记录并将整轮视为未完成。修复基础设施后须从
全新运行根按同一冻结协议重跑，不能删除该题或只汇报其余病例。

这不是把标准答案或参考答案注入产品：病例材料来自公开、匿名化题面，发送给产品端的
请求明确不含参考答案；Graphiti、Mem0、SQL上下文与冻结本地结构化检索仍按完整产品
路径运行。运行结束后必须通过执行完整性报告，并生成选择中立的内容/安全审计报告，才允许
进入盲审材料生成阶段。

### 5.1 只读核验上一轮部分结果，不在原文件上续跑或编辑

上一轮部分结果应保持原位：首个病例在第二次尝试后进入结果，第二个病例的三份可见答案
均因“具体候选”和“至少两条事实—判断链”未命中项目侧正则而被丢弃。该结果证明项目臂
发生了内容触发选择，不能进入公平盲审。不能在失败行上续跑，也不能手工改状态；第4节只
白名单复制满足首个可见答案冻结条件的三个基线目录到全新运行根。下列命令只读核验该开发
记录，绝不移动、删除或修改它。

```bash
export PREVIOUS_PARTIAL_ROOT=/opt/care-data/four_agent_expert_blind/tcm_best4sdt_agent30_semantic_contract_gate_four_system_runs
export PREVIOUS_PROJECT_RESPONSES="$PREVIOUS_PARTIAL_ROOT/shennong_sanyan_full/responses.private.jsonl"
test -s "$PREVIOUS_PROJECT_RESPONSES"
"$EVAL_PY" - "$PREVIOUS_PROJECT_RESPONSES" <<'PY'
import collections, json, sys
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
counts = collections.Counter(str(row.get("status")) for row in rows)
assert len(rows) == 2, len(rows)
assert dict(counts) == {"success": 1, "failed": 1}, counts
by_task = {str(row.get("task_id")): row for row in rows}
assert set(by_task) == {
    "BEST30-c061b9527239d5b57092e4e8",
    "BEST30-b9a99321f6931bf9e4cd6162",
}, set(by_task)
assert by_task["BEST30-c061b9527239d5b57092e4e8"]["status"] == "success"
failed = [row for row in rows if row.get("status") == "failed"]
assert len(failed) == 1
assert failed[0]["task_id"] == "BEST30-b9a99321f6931bf9e4cd6162", failed[0]["task_id"]
error = str(failed[0].get("error_private") or "")
assert "positive_candidate_in_section_2" in error
assert "two_fact_to_hypothesis_links_in_section_3" in error
assert "response_detail=" in error
detail = json.loads(error.split("response_detail=", 1)[1])
assert detail["failure_dimensions"] == [
    "positive_candidate_in_section_2",
    "two_fact_to_hypothesis_links_in_section_3",
], detail
assert detail["section_character_counts"] == [101, 58, 152, 66, 66, 48], detail
assert detail["section_clause_counts"] == [3, 3, 4, 2, 3, 2], detail
assert detail["sections_below_character_guidance"] == [2], detail
assert detail["sections_below_clause_guidance"] == [], detail
assert detail["total_section_characters"] == 491, detail
assert detail["total_below_minimum"] is False, detail
print("PREVIOUS_CONTENT_SELECTION_FAILURE_PRESERVED", dict(counts), detail)
PY
sha256sum "$PREVIOUS_PROJECT_RESPONSES"
```

## 6. 后台启动：用物理 GPU 3，新根只重跑“神农三言”

以下命令按本轮要求显式跳过“GPU 3 是否已有计算进程”的拦截，不执行对应查询；若卡上
确有其他任务，显存竞争或 OOM 风险由操作者承担。三个基线仍会按冻结协议自动校验后跳过。

```bash
set +e
set +u
set +o pipefail

RUNNER=/opt/care-workspace/four_agent_expert_blind_bundle/care_opd/competition_evidence/tcm_best4sdt_agent30/run_tcm_best4sdt_agent30_gpu3.sh
RUN_LOG=/opt/care-data/four_agent_expert_blind/tcm_best4sdt_agent30_fair_blind_review_gpu3.nohup.log
RUN_PID=/opt/care-data/four_agent_expert_blind/tcm_best4sdt_agent30_fair_blind_review_gpu3.pid
RUN_ENV=(
  CARE_ROOT=/opt/care-workspace/four_agent_expert_blind_bundle/care_opd
  QZBC_ROOT=/opt/care-workspace/four_agent_expert_blind_bundle/care_opd/competition_evidence/tcm_best4sdt_agent30/qzbc_full_product
  FULL_PRODUCT_ROOT=/opt/care-workspace/qzbc_source/qzbc-main
  MODEL_ROOT=/opt/care-data/four_agent_expert_blind/models
  CARE_MODEL=/opt/care-workspace/care_opd/runs/qzbc_base_model_3gpu/final_model
  EVAL_PY=/opt/care-workspace/opdrevi_venv/bin/python
  VLLM_BIN=/opt/care-workspace/opdrevi_venv/bin/vllm
  BASELINE_VLLM_BIN=/opt/care-workspace/opdrevi_venv/bin/vllm
  CARE_SERVER_BACKEND=transformers
  CARE_HF_PY=/opt/care-data/venvs/mhg_eval_qwen25vl/bin/python
  BACKEND_PY=/opt/care-workspace/four_agent_expert_blind_bundle/qzbc_private_eval/.conda_eval/bin/python
  SYSTEM_API_PROFILE=local_product
  PHYSICAL_GPUS=3
  ACTIVE_GPU_PROCESS_CHECK_ACK=I_ACCEPT_GPU_CONTENTION_RISK
  CARE_MAX_MODEL_LEN=32768
  BASELINE_MAX_MODEL_LEN=8192
  VLLM_USE_FLASHINFER_SAMPLER=0
  HF_HUB_DISABLE_XET=1
  OLLAMA_BASE_URL=http://127.0.0.1:11434
  QA_KNOWLEDGE_INDEX_PATH=/opt/care-workspace/qzbc_source/qzbc-main/runtime/qa_knowledge/qa_index.db
  TCM_BEST_TASK_ROOT=/opt/care-workspace/four_agent_expert_blind_bundle/care_opd/runs/public_eval/tcm_best4sdt_agent30_reference_free_fixed_seed20260825
  TCM_BEST_RUN_ROOT=/opt/care-data/four_agent_expert_blind/tcm_best4sdt_agent30_fair_blind_review_four_system_runs
  TCM_BEST_REVIEW_ROOT=/opt/care-data/four_agent_expert_blind/tcm_best4sdt_agent30_fair_blind_review_expert_blind
  REVIEWERS=expert_01,expert_02
)

if ! test -x "$RUNNER"; then
  echo "缺少runner：$RUNNER；请重新解压最新修复包" >&2
else
  env "${RUN_ENV[@]}" bash "$RUNNER" preflight
  preflight_rc=$?
  if ((preflight_rc == 0)); then
    # 新根中的神农臂从固定首例开始并冻结每题首个用户可见答案；三个同时通过
    # 首次尝试与哈希核验的基线目录已复制到新根，因此不会重跑。
    # runner 的 flock 同时保证只有一个写进程；锁冲突返回码为 73。
    nohup env "${RUN_ENV[@]}" bash "$RUNNER" run >"$RUN_LOG" 2>&1 &
    pid=$!
    printf '%s\n' "$pid" >"$RUN_PID"
    disown "$pid" 2>/dev/null || true
    sleep 3
    if kill -0 "$pid" 2>/dev/null; then
      printf '已启动：PID=%s\n日志=%s\n' "$pid" "$RUN_LOG"
      tail -n 80 "$RUN_LOG"
    else
      echo '启动阶段退出，最后160行日志如下：' >&2
      tail -n 160 "$RUN_LOG" >&2
    fi
  else
    echo "预检失败，未启动正式任务；rc=$preflight_rc" >&2
  fi
fi
```

另开终端查看状态：

```bash
ps -fp "$(cat /opt/care-data/four_agent_expert_blind/tcm_best4sdt_agent30_fair_blind_review_gpu3.pid)" || true
tail -n 200 /opt/care-data/four_agent_expert_blind/tcm_best4sdt_agent30_fair_blind_review_gpu3.nohup.log
find "$TCM_BEST_RUN_ROOT" -maxdepth 3 -type f -printf '%s %p\n' 2>/dev/null | sort
```

若身份、题目绑定、无参考答案、产品栈调用或持久化完整性失败，日志会明确指出执行阶段，
且神农结果文件保留该条失败审计。若已经形成用户可见答案，则不得因六节结构、引用、候选、
事实链、治法、方药、鉴别对象、拒答、长度或安全表现重试；这些维度随冻结答案进入审计和
专家盲审。不要手工删除、替换或改写任何一行。

30例完成后，先执行全量验证并核验独立质量报告：

```bash
env "${RUN_ENV[@]}" bash "$RUNNER" verify

QUALITY_REPORT=/opt/care-data/four_agent_expert_blind/tcm_best4sdt_agent30_fair_blind_review_four_system_runs/shennong_sanyan_full/case_reasoning_quality.private.json
PROJECT_RESPONSES=/opt/care-data/four_agent_expert_blind/tcm_best4sdt_agent30_fair_blind_review_four_system_runs/shennong_sanyan_full/responses.private.jsonl
test -s "$QUALITY_REPORT"
"$EVAL_PY" - "$QUALITY_REPORT" "$PROJECT_RESPONSES" <<'PY'
import hashlib, json, sys
report_path, responses_path = sys.argv[1:]
report = json.load(open(report_path, encoding="utf-8"))
digest = hashlib.sha256(open(responses_path, "rb").read()).hexdigest()
assert report["schema_version"] == "care_opd.competition.tcm_best4sdt_case_reasoning_quality.v4"
assert report["status"] == "PASS", report.get("threshold_failures")
assert report["task_count"] == report["response_count"] == 30
assert report["responses_sha256"] == digest
assert report["reference_answer_input"] is False
assert report["output_selection_policy"] == "first_user_visible_output_frozen_content_neutral"
assert report["content_review_policy"] == "audit_only_expert_judged_no_content_triggered_regeneration"
assert report["content_review_metrics_are_selection_neutral"] is True
assert report["safety_review_metrics_are_selection_neutral"] is True
assert report["length_guidance_policy"] == "advisory_only_not_an_eligibility_or_scoring_threshold"
for name in (
    "success",
    "user_visible_response_present",
    "generated_response_present",
    "case_binding_verified",
    "raw_case_packet_verified",
    "selection_neutral_output_contract_verified",
    "row_audit_present",
):
    assert report["counts"][name] == 30, (name, report["counts"][name])
assert report["counts"]["reference_or_gold_access_violation"] == 0
for name in (
    "case_source_citation_verified",
    "exact_six_heading_structure",
    "all_six_case_source_citations",
    "all_sections_nonempty",
    "candidate_reasoning",
    "fact_to_hypothesis_links_at_least_2",
    "conditional_treatment_direction",
    "conditional_formula_direction",
    "concrete_review_target",
    "blanket_refusal",
    "fixed_dose",
    "clinical_finality",
    "length_guidance_pass",
    "all_sections_meet_character_guidance",
    "all_sections_meet_clause_guidance",
    "total_meets_character_guidance",
):
    assert 0 <= report["counts"][name] <= 30, (name, report["counts"][name])
    assert name not in report["thresholds"], name
print("CASE_REASONING_QUALITY_PASS", digest)
PY
```

只有上段输出 `CASE_REASONING_QUALITY_PASS` 后，才生成**新的**盲映射和专家 XLSX：

```bash
test ! -e /opt/care-data/four_agent_expert_blind/tcm_best4sdt_agent30_fair_blind_review_expert_blind/private/review_manifest.private.json
env "${RUN_ENV[@]}" bash "$RUNNER" review
```

旧审查包中的 `review_manifest.private.json`、`blind_mapping.private.jsonl`、
`reviewer_*.private.xlsx/csv` 和阅读 HTML 都绑定了旧回答哈希，不能复制到新盲审根，
也不能继续让专家填写。若旧表已发出，应明确作废并只发送本轮新表。

生成成功后，四系统回答位于：

```text
/opt/care-data/four_agent_expert_blind/tcm_best4sdt_agent30_fair_blind_review_four_system_runs/
```

专家盲审材料位于：

```text
/opt/care-data/four_agent_expert_blind/tcm_best4sdt_agent30_fair_blind_review_expert_blind/private/
```

向每位专家发送其自己的 `reviewer_N.private.xlsx`（没有 XLSX 时发送 CSV）和同编号
`reviewer_N.reading.private.html`。不要发送 `blind_mapping.private.jsonl` 或
`expert_blind_secret.private`。

## 7. 专家完成后的计分与结果回传

```bash
export COMPLETED_RATINGS=/opt/care-data/four_agent_expert_blind/completed/reviewer_1.completed.xlsx,/opt/care-data/four_agent_expert_blind/completed/reviewer_2.completed.xlsx
bash "$TCM_BEST_RUNNER" score
```

公开安全汇总位于：

```text
/opt/care-data/four_agent_expert_blind/tcm_best4sdt_agent30_fair_blind_review_expert_blind/summary_public/
```

打包私有回传材料：

```bash
export RETURN_ROOT=/opt/care-data/four_agent_expert_blind/tcm_best4sdt_agent30_fair_blind_review_return
test ! -e "$RETURN_ROOT" || { echo '回传目录已存在，请先人工核对' >&2; exit 2; }
install -d -m 700 "$RETURN_ROOT"
cp -a "$TCM_BEST_TASK_ROOT" "$RETURN_ROOT/frozen_task"
cp -a "$TCM_BEST_RUN_ROOT" "$RETURN_ROOT/four_system_runs"
cp -a "$TCM_BEST_REVIEW_ROOT" "$RETURN_ROOT/expert_blind_review"
cp -a "$MODEL_ROOT/suite_selection_lock.private.json" "$RETURN_ROOT/"
cp -a "$CARE_ROOT/competition_evidence/local_four_agent_server/model_suite.json" "$RETURN_ROOT/"
cp -a /opt/care-data/four_agent_expert_blind/tcm_best4sdt_agent30_fair_blind_review_baseline_reuse_audit \
  "$RETURN_ROOT/baseline_reuse_audit"
cp -a /opt/care-data/four_agent_expert_blind/tcm_best4sdt_agent30_fair_blind_review_gpu3.nohup.log "$RETURN_ROOT/"

tar -czf /opt/care-data/four_agent_expert_blind/TCM_BEST4SDT_AGENT30_FAIR_BLIND_REVIEW_RESULTS.PRIVATE.tar.gz \
  -C /opt/care-data/four_agent_expert_blind tcm_best4sdt_agent30_fair_blind_review_return
sha256sum /opt/care-data/four_agent_expert_blind/TCM_BEST4SDT_AGENT30_FAIR_BLIND_REVIEW_RESULTS.PRIVATE.tar.gz \
  >/opt/care-data/four_agent_expert_blind/TCM_BEST4SDT_AGENT30_FAIR_BLIND_REVIEW_RESULTS.PRIVATE.tar.gz.sha256
```

若专家尚未完成，可先回传推理产物和空白盲审表；不得预填、推测或挑选结果。
