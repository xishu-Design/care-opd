# TCMEval-PA 固定30题四系统先导实验

本目录把公开的 TCMEval-PA v3 固化为一个**生成前冻结、与结果无关**的30题先导子集，
并复用既有四系统本地执行链，依次比较：

1. 神农三言完整项目（医生端、CARE-OPD模型、角色路由与安全/证据控制）；
2. HuatuoGPT-3-8B 本地医疗模型基线；
3. GLM-4-9B-0414 本地中文模型基线；
4. InternLM3-8B-Instruct 本地中文模型基线。

三个对照是冻结的公开权重模型基线，不是复制了神农三言全部编排、工具和角色治理的
同构 Agent。该实验支持“完整项目在公开中医处方审核题上的受控补充比较”，不能单独
归因 CARE-OPD，也不能解释为临床疗效。

## 固定样本

- 数据：TCMEval-PA v3，共328题；固定 Excel SHA-256 为
  `905cb28f145ba50f3b1c912265ff6649e0bb38f56adf6cd588d77ca768db92c1`；
- 数据 DOI：`10.6084/m9.figshare.29651261.v3`，许可为 CC BY 4.0；
- 子集：30题，其中27道单选、3道多选；按类别为基础概念1、处方规范性8、
  处方适宜性21；
- 抽样：在固定“类别×题型”配额内按 SHA-256 排序取样，seed=`20260825`；
- 题单在任何四系统输出生成前固化，不以神农三言或任一对照的正确性选题；
- 任务文件不含标准答案、解释、规则摘要，评分时才从校验值锁定的 Excel 独立读取 gold。

## 运行

服务器已有四系统模型和运行环境时：

```bash
bash competition_evidence/tcmeval_pa_agent30/run_tcmeval_pa_agent30_gpu3.sh prepare
bash competition_evidence/tcmeval_pa_agent30/run_tcmeval_pa_agent30_gpu3.sh preflight
bash competition_evidence/tcmeval_pa_agent30/run_tcmeval_pa_agent30_gpu3.sh run
bash competition_evidence/tcmeval_pa_agent30/run_tcmeval_pa_agent30_gpu3.sh score
```

脚本默认使用物理 GPU 3；`CUDA_VISIBLE_DEVICES=3` 由下层四系统执行链设置，进程内
逻辑设备为 `cuda:0`。四个系统依次加载、依次释放，不会在卡3同时驻留四个模型。

若 SSH 断开，使用同一命令恢复；逐题任务哈希和断点文件不允许换题或覆盖。三个公开
模型的下载清单必须先达到 `COMPLETE`，卡3也必须没有其他计算进程。

## 评分与表述

主指标为选项集合精确匹配准确率；同时输出 Wilson 95% 区间、格式解析率、类别/题型
切片，以及神农三言对每个基线的逐题 exact McNemar 比较。30题中每题对应3.33个百分
点，故正式材料必须写成：

> TCMEval-PA v3 固定30题先导子集（非官方328题全量成绩）。

公开题库不能证明基础模型在预训练阶段从未接触；该实验也不能替代真实医疗记录外测、
三角色受控实验或专家盲评。
