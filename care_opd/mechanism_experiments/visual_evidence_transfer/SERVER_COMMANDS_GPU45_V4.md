# V4：独立目录、只做推理的提示适配验证

V4 只修改共享视觉观察与回答提示适配，不重训、不改检查点、不改评分规则、不筛选样本。V3 测试输出已被查看；V3 原始结果必须完整保留，V4 不能替换 V3 或被称为“未见测试集验证”。先运行原来的验证集 10 题族；本启动脚本不会自动运行测试集。

模型和生成参数从旧正式运行清单的 `config` 读取，保留实际解析的本地缓存路径。仅当该清单不存在时，才回退到旧验证配置；清单损坏时直接报错。旧任务、图片和模型只读，旧观察与回答不复用。默认预算仍是观察 256 token、回答 512 token；实际运行严格继承来源配置，不在这里调参。

物理 GPU 分工不变：GPU 4 先生成全体共享观察，然后 GPU 4 依次直接回答、GKD-Full，GPU 5 依次 Clean-SFT、TROPD-Full、CARE-OPD。不会检查空卡、清卡、重置 GPU 或终止其他进程；共享卡仍可能 OOM。GPU 5 在共享观察生成期间等待是正常现象。

## 1. 安装到全新的源码目录

把 `CARE_OPD_TMM_VISUAL_GPU45_V4.tar.gz` 和同名 `.sha256` 上传到 `/opt/care-workspace`。包内相对路径以 `mechanism_experiments/visual_evidence_transfer/` 开头。不要解压到旧 `/opt/care-workspace/care_opd`，也不要覆盖运行中的源码。

下面都是服务器 Bash 命令。准备块使用子 shell；失败只结束该块，不会关闭交互终端。目标目录若已存在则停止，先核实而不是删除或覆盖。

```bash
(
  set -euo pipefail
  cd /opt/care-workspace
  sha256sum -c CARE_OPD_TMM_VISUAL_GPU45_V4.tar.gz.sha256
  test ! -e /opt/care-workspace/care_opd_visual_v4
  test ! -L /opt/care-workspace/care_opd_visual_v4
  install -d -m 700 /opt/care-workspace/care_opd_visual_v4
  tar -xzf CARE_OPD_TMM_VISUAL_GPU45_V4.tar.gz -C /opt/care-workspace/care_opd_visual_v4
  test -s /opt/care-workspace/care_opd_visual_v4/mechanism_experiments/visual_evidence_transfer/run_validation_v4.sh
  bash -n /opt/care-workspace/care_opd_visual_v4/mechanism_experiments/visual_evidence_transfer/run_validation_v4.sh
)
```

固定路径：

| 用途 | 路径 |
| --- | --- |
| 新源码 | `/opt/care-workspace/care_opd_visual_v4` |
| 新工作目录 WORK | `/opt/care-data/care_opd_tmm_visual_gpu45_v4` |
| 只读旧目录 OLD | `/opt/care-data/care_opd_tmm_visual_gpu45_v3` |
| Python | `/opt/care-data/venvs/mhg_eval_qwen25vl/bin/python` |
| 首选参数来源 | `OLD/runs_test/run_manifest.json` 中的 `config` |
| 唯一回退来源 | `OLD/config_validation10.json`，仅在首选文件不存在时 |
| 复用验证任务 | `OLD/tasks_validation10`，保留其中原有图片路径 |

无需重建数据、重新下载模型或修改旧配置。不要从旧运行目录复制 `observations`、`arms`、`summary` 或 `run_manifest.json` 到 V4。

## 2. 一条命令前台验证

```bash
bash /opt/care-workspace/care_opd_visual_v4/mechanism_experiments/visual_evidence_transfer/run_validation_v4.sh
```

脚本自定位源码，依次完成新配置初始化、预检、全部验证推理与汇总。它使用 `write_json_once` 冻结配置和来源记录，并通过 V4 配置校验；配置含 `protocol_version=care_opd.visual_transfer.v4`、`evaluation_use=validation`。它拒绝与 OLD 重叠的 WORK、指向旧路径的别名、WORK 内符号链接以及未绑定身份的旧输出。

首次运行必须使用新的运行根 `WORK/runs_validation10`。同一脚本可以在基础设施故障处理后恢复相同 V4 运行；已有配置、来源、模型、代码和结果身份必须一致，已产生的回答不会重抽。不要删输出以重试低分、空文本、无效 JSON 或截断回答，也不要同时启动多份任务。

初次预检需要读取检查点并计算哈希，停留在 `HASH_MODEL` 不一定是卡住。预检通过不保证显存充足。输出状态包括 `V4_STATE=INITIALIZING`、`PREFLIGHT`、`VALIDATION_RUNNING`，最终需同时看到 `GPU45_COMPLETE`、`V4_STATE=COMPLETE` 和汇总 `status=COMPLETE`。`COMPLETE` 是运行完整性，不是分数门槛。

## 3. 或者后台启动同一验证脚本

本段与前台命令二选一，不要同时执行。时间戳日志和 PID 文件不会覆盖以前的启动记录；源码与数据检查仍在脚本中执行。

```bash
(
  set -euo pipefail
  HF_PY=/opt/care-data/venvs/mhg_eval_qwen25vl/bin/python
  WORK=/opt/care-data/care_opd_tmm_visual_gpu45_v4
  OLD=/opt/care-data/care_opd_tmm_visual_gpu45_v3
  RUNNER=/opt/care-workspace/care_opd_visual_v4/mechanism_experiments/visual_evidence_transfer/run_validation_v4.sh
  umask 077
  "$HF_PY" - "$WORK" "$OLD" <<'PY'
import sys
from pathlib import Path
work, old = (Path(p).resolve() for p in sys.argv[1:])
if work == old or old in work.parents or work in old.parents:
    raise ValueError('WORK and OLD must be disjoint')
if work.exists() and any(p.is_symlink() for p in work.rglob('*')):
    raise ValueError('V4 WORK must not contain symlinks')
logs = work / 'logs'
if logs.resolve() != logs:
    raise ValueError('log path must not be an alias')
logs.mkdir(parents=True, exist_ok=True, mode=0o700)
PY
  RUN_STAMP="$(date +%Y%m%d_%H%M%S)_$$"
  RUN_LOG="$WORK/logs/validation10_${RUN_STAMP}.nohup.log"
  RUN_PID="$WORK/logs/validation10_${RUN_STAMP}.pid"
  nohup env HF_PY="$HF_PY" WORK="$WORK" OLD="$OLD" bash "$RUNNER" >"$RUN_LOG" 2>&1 </dev/null &
  pid=$!
  printf '%s\n' "$pid" >"$RUN_PID"
  disown "$pid" 2>/dev/null || true
  printf 'PID=%s\nLOG=%s\nPID_FILE=%s\n' "$pid" "$RUN_LOG" "$RUN_PID"
  ps -fp "$pid" || true
)
```

这里只记录进程启动，不把 PID 当成实验完成。不会自动启动测试集或按得分重复验证。

## 4. 查看日志、来源和验证汇总

```bash
find /opt/care-data/care_opd_tmm_visual_gpu45_v4/logs -maxdepth 1 -type f -name '*.nohup.log' -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' | sort
```

从上面选择日志，使用 `tail -F /完整日志路径` 跟踪；`Ctrl+C` 只退出查看。分支日志按阶段生成，尚未开始的阶段没有对应文件属于正常情况：

```bash
tail -n 80 /opt/care-data/care_opd_tmm_visual_gpu45_v4/runs_validation10/logs/gpu4.observe.log
tail -n 80 /opt/care-data/care_opd_tmm_visual_gpu45_v4/runs_validation10/logs/gpu4.answers.log
tail -n 80 /opt/care-data/care_opd_tmm_visual_gpu45_v4/runs_validation10/logs/gpu5.answers.log
```

```bash
/opt/care-data/venvs/mhg_eval_qwen25vl/bin/python - <<'PY'
import json
from pathlib import Path
work = Path('/opt/care-data/care_opd_tmm_visual_gpu45_v4')
for name in ('config_validation10.json', 'config_validation10.provenance.json'):
    path = work / name
    if path.is_file():
        print(path)
        print(path.read_text(encoding='utf-8'))
summary = work / 'runs_validation10' / 'summary'
metrics = summary / 'metrics.json'
if metrics.is_file():
    print('VALIDATION_STATUS=' + str(json.loads(metrics.read_text(encoding='utf-8')).get('status')))
    for name in ('summary.md', 'metrics.csv'):
        path = summary / name
        if path.is_file():
            print(path.read_text(encoding='utf-8'))
else:
    print('验证汇总尚未生成；查看总日志和各 GPU 分支日志。')
PY
```

来源记录包含旧配置/清单路径及哈希、任务清单哈希、继承的模型路径与生成参数、启动脚本哈希。V4 自己的 `runs_validation10/run_manifest.json` 另行记录代码、环境、模型和协议身份。日志与所有原始回答均保留，包括格式失败、错误回答和低分样本。只修复基础设施错误后按原配置恢复；锁冲突需核对父进程及 `runtime/workers.jsonl`，不要直接删锁或终止其他进程。

另可手动运行只读诊断，查看观察长度与固定前缀计数、回答分布、文本保持情况和 clean/swapped 输入一致性；命令只向标准输出打印诊断，不重生成或修改结果、不读金标准，也不作为模型选择或启动测试的门槛：

```bash
/opt/care-data/venvs/mhg_eval_qwen25vl/bin/python /opt/care-workspace/care_opd_visual_v4/mechanism_experiments/visual_evidence_transfer/diagnose_outputs.py --config /opt/care-data/care_opd_tmm_visual_gpu45_v4/config_validation10.json
```

运行中的诊断是非原子快照；`snapshot_complete` 仅表示当前首输出记录覆盖情况，不能替代正式汇总的完整性校验，也不代表科学结论成立。

## 5. 可选：人工启动“旧测试集上的探索性协议修订”

默认到验证结束即停止。只有明确决定继续时，才手动执行本节。它使用相同旧测试题集，不是新的独立测试；所有报告必须写明：**V3 测试结果已被查看，V4 是事后提示协议修订后的同集探索性重复评估**。不能称为 untouched test、独立确认性结果或从 V3/V4 中择优报告。

先确认验证完整性，再创建另一个全新的配置和运行根。此处不设准确率、格式率或 CARE 胜出门槛，也不根据验证结果增删样本或方法。

```bash
(
  set -euo pipefail
  HF_PY=/opt/care-data/venvs/mhg_eval_qwen25vl/bin/python
  EXP_DIR=/opt/care-workspace/care_opd_visual_v4/mechanism_experiments/visual_evidence_transfer
  "$HF_PY" - "$EXP_DIR" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import evaluate
from common import PROTOCOL_VERSION, file_sha256, load_tasks, read_json, write_json_once
work = Path('/opt/care-data/care_opd_tmm_visual_gpu45_v4').resolve()
old = Path('/opt/care-data/care_opd_tmm_visual_gpu45_v3').resolve(strict=True)
if work == old or old in work.parents or work in old.parents:
    raise ValueError('WORK and OLD must be disjoint')
if any(p.is_symlink() for p in work.rglob('*')):
    raise ValueError('V4 WORK must not contain symlinks')
status = read_json(work / 'runs_validation10/summary/metrics.json').get('status')
if status != 'COMPLETE':
    raise ValueError(f'validation incomplete: {status}')
source_path = work / 'config_validation10.json'
source = evaluate.configuration(source_path)
if source['evaluation_use'] != 'validation':
    raise ValueError('expected frozen V4 validation config')
tasks_dir = (old / 'tasks_test').resolve(strict=True)
if old not in tasks_dir.parents:
    raise ValueError('test tasks alias a directory outside OLD')
tasks, selection = load_tasks(tasks_dir)
if selection.get('dataset', {}).get('source_split') != 'test':
    raise ValueError('expected the original frozen test split')
config_path = work / 'config_test_revision.json'
root = work / 'runs_test_revision'
provenance_path = work / 'config_test_revision.provenance.json'
if any(p.exists() for p in (config_path, root, provenance_path)):
    raise ValueError('revision paths already exist; do not overwrite or recreate them')
config = dict(source, protocol_version=PROTOCOL_VERSION,
              evaluation_use='post_test_protocol_revision',
              tasks_dir=str(tasks_dir), run_root=str(root))
evaluate.validate_evaluation_use(config, selection)
write_json_once(config_path, config)
if evaluate.configuration(config_path) != config:
    raise ValueError('unexpected configuration normalization')
write_json_once(provenance_path, {
    'evaluation_use': 'post_test_protocol_revision',
    'prior_test_outputs_inspected': True,
    'independent_untouched_test': False,
    'source_validation_config': str(source_path),
    'source_validation_config_sha256': file_sha256(source_path),
    'source_validation_provenance_sha256': file_sha256(work / 'config_validation10.provenance.json'),
    'selection_manifest_sha256': file_sha256(tasks_dir / 'selection_manifest.json'),
    'old_results_policy': 'preserved_reported_separately_no_selection_between_v3_v4',
})
print(f'EXPLORATORY_REVISION_CONFIG={config_path} tasks={len(tasks)}')
PY
  env HF_PY="$HF_PY" bash "$EXP_DIR/run_visual_transfer_gpu45.sh" preflight /opt/care-data/care_opd_tmm_visual_gpu45_v4/config_test_revision.json
)
```

确认上一步成功，再手动启动以下命令（前台；退出查看不会中止任务的需求请使用与验证相同的 `nohup` 方式，并另取 `test_revision_时间戳` 日志名）：

```bash
env HF_PY=/opt/care-data/venvs/mhg_eval_qwen25vl/bin/python bash /opt/care-workspace/care_opd_visual_v4/mechanism_experiments/visual_evidence_transfer/run_visual_transfer_gpu45.sh all /opt/care-data/care_opd_tmm_visual_gpu45_v4/config_test_revision.json
```

此运行在 `WORK/runs_test_revision` 重新生成共享观察、五种方法的回答和汇总，不使用旧推理输出。异常后续跑只重复最后的 `all` 命令，不能重建配置、删回答或换提示。检查日志与结果时，把上一节的 `runs_validation10` 换为 `runs_test_revision`；配置来源仍可追溯到旧清单。V3 与 V4 分别报告，任何差异只能作为此次协议修订下的探索性观察，不归因为重新训练或独立测试证据。
