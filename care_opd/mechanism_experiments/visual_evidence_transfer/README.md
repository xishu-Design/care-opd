# CARE-OPD：冻结视觉前端的图像证据迁移评测

## 当前版本 V4

当前入口为 [SERVER_COMMANDS_GPU45_V4.md](SERVER_COMMANDS_GPU45_V4.md) 和 `run_validation_v4.sh`。
本次是看过旧版测试输出后进行的**共享提示适配器修订**，不是新增训练，也不能事后写成预注册的未见测试。
新版安装到独立目录 `/opt/care-workspace/care_opd_visual_v4`，结果放在 `/opt/care-data/care_opd_tmm_visual_gpu45_v4`，不覆盖 V3 源码、观察或任何模型输出。

- 视觉前端以“图像概况、问题相关观察、可判定边界”保留可见事实，不用固定字数迫使其编造。
- 回答提示区分否定证据与证据不足，YES/NO 都明确要求 `image:current` 引用，ABSTAIN 保留空图像引用。
- 五个对照共享回答协议，四个文本对照共享重新生成的观察；不回填引用、不修饰返回文本、不按内容重新抽取答案。
- 沿用旧预算与评分定义，独立报告内容正确率、来源合同和回答覆盖率；观察仍是未经临床核验的模型输出。
- 先运行 validation。若复测已查看过的旧 test，必须使用 `evaluation_use=post_test_protocol_revision` 和单独运行根，明确保留该解释边界。
- `diagnose_outputs.py --config ...` 只读汇总答案分布、引用和观察状态，不读取 gold，不构成新的模型选择门槛。

下方通用设计仍适用；旧 `SERVER_COMMANDS.md` / `SERVER_COMMANDS_GPU45.md` 是历史 V1–V3 命令，不能直接用于 V4 配置。

这是**新增推理实验**，不改已有权重、不重训、不启动 LightRAG/Graphiti/Mem0，不代表完整医疗产品的横向临床质量评价。
研究对象是 `真实图像 → 冻结视觉观察 → 文本 Evidence Packet → 结构化回答`。
现有 CARE-OPD、Clean-SFT、GKD-Full、TROPD-Full 使用同一视觉前端、同一观察和同一输出要求。
另设视觉前端直接回答作为系统结构基线。直接回答与“视觉前端 + 4B”的参数和计算量不同，不能作为等预算方法消融。

## 要回答的问题

1. 不追加图文训练时，已有合同后训练检查点能否迁移到图像证据问答？
2. 视觉内容改变时，回答是否随正确答案改变，而不是仅读取人工标注的“冲突”状态？
3. 无图时是否停止视觉推断，并仍保留独立文本字段？

本实验使用 SLAKE 中文、视觉依赖的闭集 YES/NO 问题。不使用参考答案生成观察或回答。
任务标注由数据作者提供；无需新组织专家评分，但也不评价开放式处方或临床疗效。

## 条件和金标

| 条件 | 模型实际输入 | 评分目标 |
| --- | --- | --- |
| clean | 原始图像、题干、非医疗文本控制字段 | 按该图像原标注回答 YES/NO |
| no_image | 无图像、原题干、同一文本控制字段 | visual_answer=ABSTAIN，文本控制字段仍正确 |
| swapped_image | 相同题干、另一真实图像、同一文本控制字段 | 按另一图像已有标注回答；答案必须与原图相反 |

交换只选同一分区、相同规范化题干、不同图像且相反标注的已有样本。没有符合条件的配对，就不产生该条件。
**交换图不等于应拒答**：不会向模型提供 `swapped`、原答案、对照 qid 或“冲突=true”。
此条件是标签有效的视觉内容替换，不声称系统能凭空识别患者身份错配。

`context_value` 是固定生成的 8 字符非医疗代码，仅用于独立文本保持控制。
它不是真实患者信息，不提供图像答案；其保持率不是医学正确事实保持率，也不是旧 QZBC 的 Conditional Retention 指标。

## 数据隔离

- 官方来源：https://www.med-vqa.com/slake/ ，作者镜像 https://huggingface.co/datasets/BoKelvin/SLAKE 。
- 官方 train/validation/test 存在部分图像重叠，不能称其为天然图像互斥划分。
- validation 只用于试跑；正式 test 预先排除所有 validation 重复图像。这是从官方 test 派生的固定子集。
- 不使用 train 更新权重。既有基础模型是否预训练见过公开 SLAKE，未知；不能宣称预训练污染已排除。
- 样本选取、条件构造和排除规则先冻结，不依据本方法生成质量选样。
- 选择清单记录源 JSON 哈希、数据 revision、选取规则、重叠情况、任务及 gold 的哈希。
- 同图多问题以及交换图产生的相关样本按图像连通分量聚类，置信区间不能按独立问答行计算。

## 模型与协议

默认冻结视觉前端为 `Qwen/Qwen2.5-VL-7B-Instruct`。HF 适配器明确支持其原生 `qwen2_5_vl` 架构，不对任意视觉模型作兼容承诺。
参考实现：https://huggingface.co/docs/transformers/en/model_doc/qwen2_5_vl 。
推理中检查实际 `pixel_values`，有图任务若没有像素张量即停止，不自动退化成纯文本。

已有文本检查点用 `AutoModelForCausalLM` 加载，必须提供合并后的完整检查点目录，不能误传 adapter-only 目录。
默认本地 HF 推理、BF16、SDPA、greedy、无额外思考输出；视觉观察 256 new tokens，答案 512 new tokens。
这是沿用 GeneratedResponse **六个顶层字段**的新评测提示适配，不是原产品 prompt 原样执行。
所有文本方法使用同一个固定适配提示，不单独给 CARE 调优。

视觉观察是可能出错的模型文本，不是经临床验证的事实。scorer 的来源校验不验证观察在像素层面的真假。
视觉错误会传递到下游，端到端答案准确率必须保留这些错误，不能筛掉观察错误的样本后只报好看的结果。

## 第一输出冻结与恢复

- 每个任务只保存第一个成功返回的生成输出。空文本、错误 JSON、截断、拒答都不触发重采样或提示修复。
- OOM、加载失败等没有可用输出的基础设施错误进入 `infrastructure_errors.jsonl` 并停止阶段。
- 排除环境故障后执行相同命令可继续缺失任务；已完成行不得覆盖。历史基础设施错误保留。
- 任务、代码、配置、软件环境或检查点变化不得混入旧运行根；应使用新的 config/run root。
- 初次预检 SHA-256 绑定全部权重及 tokenizer/config；常规恢复核对文件大小与 mtime。`evaluate.py preflight --full-hash` 可重新逐字节校验。
- 单卡入口的所有阶段有运行根锁；不能直接用两个单卡入口同时写同一个 run root。
- 新增 GPU4/5 协调器由父进程全程持有同一运行根锁，只允许其内部 worker 分别写不同 arm，视觉观察完整生成后才开始并行。两张卡的软件环境、GPU 身份和任务分配一起冻结；不能把旧单卡运行根改用双卡恢复。
- 异常中断残留锁只可在人工核实父进程及 `runtime/workers.jsonl` 中子进程均已结束后移除，协调器不自动删除旧锁或终止其他进程。

## 输出与成本

```text
run_root/
  run_manifest.json
  observations/observations.jsonl
  arms/vlm_direct/responses.jsonl
  arms/clean_sft/responses.jsonl
  arms/gkd_full/responses.jsonl
  arms/tropd_full/responses.jsonl
  arms/care_opd/responses.jsonl
  summary/
```

报告应同时查看视觉可用条件的答案正确率、拒答/回答覆盖率、选择性准确率、输出合同成功率、文本控制保持率和配对视觉一致性。
无图条件 ABSTAIN 正确率单独报告，不与正常图像准确率混成“总医学能力”。
不同模型同一批数据做聚类配对 bootstrap，区间反映测试图像抽样，不反映训练随机性。
交换关系可能把多个图像连接成较大的簇，因此还要报告实际独立簇数；簇过少时不能把窄区间解释成充分的统计证据。
上下游分别记录 token 与延迟，并提供端到端两阶段延迟和；模型加载、数据下载、初始文件哈希、失败重试耗时不在单条推理延迟中。
缓存观察是在不同方法间共享视觉计算，不能把下游延迟写成完整系统的全部成本。

## 运行与旧实验 TROPD 补测

当前物理卡 **4、5** 的完整命令见 [SERVER_COMMANDS_GPU45_V4.md](SERVER_COMMANDS_GPU45_V4.md)，验证集入口为 `run_validation_v4.sh`，底层协调器为 `run_visual_transfer_gpu45.sh`。
卡 4 先生成共享观察，随后卡 4 顺序运行 VLM-direct / GKD，卡 5 顺序运行 Clean-SFT / TROPD / CARE。它是方法之间并行，不是双卡张量并行；观察阶段卡 5 暂时等待。
`CARE_OPD_TMM_VISUAL_GPU45_V4.tar.gz` 仅包含本视觉实验目录，不覆盖 GPU6/7 使用的 `tropd_backfill`。旧配置、旧结果、旧包都保留；V4 使用独立安装和工作目录。
原有 QZBC 和真实外测双轨的 TROPD 补测是另一组实验，见 `../tropd_backfill/SERVER_COMMANDS.md`。
两组分别汇总，不能混合分母或把图文结果替换旧 QZBC 指标。

## 证据边界

这套脚本能产生可审计的迁移比较，不能预先保证 CARE 获胜或保证 TMM 接收。
若视觉内容替换并不改变回答，或者改善只来自更高拒答率，应如实报告，不能改标签/长度阈值反复筛到通过。
模型无新增训练不等于没有实验成本；服务器仍需加载真实视觉模型、运行全部对照。
