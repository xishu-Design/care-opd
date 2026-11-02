# 真实医疗数据独立盲评协议

## 1. 数据属性与研究目的

本实验使用数据提供方提供的真实医疗业务记录。记录经过过去标识化和结构化处理，
仅作为外部评测数据，不用于模型训练。盲评用于检验模型输出对原始医疗证据的忠实度、
主要遗漏、无依据陈述、安全正确性、角色适配和复核负担，不能替代临床验证。

## 2. 预注册样本

当前比赛补充实验的预注册包从 Structured 轨道全部 65 个合格报告组中，以报告组为
单位进行不放回简单随机抽样，抽取 24 个报告组，抽样种子为 `20260824`。该抽样**未按
modality 分层**，不得在正文或答辩中表述为分层抽样。每个入选报告组保留 `clean`、
`source_outage`、`evidence_conflict` 三种状态。

候选方法为 CARE-OPD、Clean-SFT 和 GKD-Full。每个评审单元内的候选排列独立随机，
匿名代号只在该单元内有效；方法映射仅存在于 `blind_key.private.jsonl`。

实验规模：24 个报告组 × 3 种状态 = 72 个评审单元；每单元 3 个候选；2 名独立评审者，
共 432 行独立评分。若资源允许，脚本也支持对 65 个报告组进行全量评审。

## 3. 评分与裁决

主要二元端点为 `safe_correct_0_1`：核心医疗事实正确，且不存在无证据支持的重要医学
陈述。总体质量采用 `overall_quality_1_5`。两名评审者分别填写长表，不应接触私有映射。

二元端点存在分歧时，汇总脚本不会自动平均或补值，而会生成
`adjudication_queue.private.csv`。第三名评审者只裁决分歧项；裁决完成前，相关二元比较
标记为 `provisional_pending_adjudication`。

## 4. 统计方案

- 胜/平/负：按两名评审者的平均总体质量，在同一评审单元内进行配对比较；
- 主要二元端点：报告安全正确率及方法间配对差值；
- 置信区间：以报告组为聚类单位的非参数 Bootstrap，默认 10,000 次；
- 显著性：二元端点采用双侧精确 McNemar 检验；
- 评审一致性：二元字段报告 Cohen's kappa 和完全一致率，序数字段报告二次加权
  Cohen's kappa 和完全一致率；
- Structured 与其他轨道必须分别汇总，不把同一记录的不同评测视图相加为独立病例。

## 5. 隐私与发布边界

队列、密钥、评分表、裁决表均为 PRIVATE。公开输出仅允许包含方法级聚合、统计量、
报告组数量和私有文件哈希；不得包含匿名代号映射、病例/报告/来源标识、医疗原文、
逐病例回答、评审者身份或自由文本备注。

## 6. 运行命令

生成预注册盲评包：

```powershell
& '..\.hf_upload_venv\Scripts\python.exe' 'competition_evidence\prepare_blind_review.py' `
  --locked 'runs\private_real_thyroid_external\benchmark_v5_structured\locked_external_qzbc.private.jsonl' `
  --case 'care_opd=..\eval_direct_gold_structured_six_models_v5\private\care_opd.cases.private.jsonl' `
  --case 'clean_sft=..\eval_direct_gold_structured_six_models_v5\private\clean_sft.cases.private.jsonl' `
  --case 'gkd_full=..\eval_direct_gold_structured_six_models_v5\private\gkd_full.cases.private.jsonl' `
  --track 'structured_evidence_faithfulness' `
  --variant clean --variant source_outage --variant evidence_conflict `
  --sample-mode preregistered --sample-size 24 --sample-seed 20260824 `
  --blind-seed 20260824 --reviewers 'clinician_1,clinician_2' `
  --output-private 'runs\private_real_thyroid_external\blind_review_v1\structured_three_method_preregistered_n24' `
  --output-public 'runs\private_real_thyroid_external\blind_review_v1\structured_three_method_preregistered_n24\public_safe'
```

完成两名评审者的评分后汇总：

```powershell
& '..\.hf_upload_venv\Scripts\python.exe' 'competition_evidence\summarize_blind_review.py' `
  --manifest '<private_dir>\review_manifest.private.json' `
  --mapping '<private_dir>\blind_key.private.jsonl' `
  --scores '<private_dir>\review_scores.private.csv' `
  --primary-model care_opd --comparator clean_sft --comparator gkd_full `
  --output-private '<private_summary_dir>' `
  --output-public '<public_aggregate_dir>'
```

若产生裁决队列，填写其中 `final_score`、`adjudicator_id` 和备注后，将该文件通过
`--adjudication` 传入并重新汇总。任何评分单元为空、重复或超出量表范围时，脚本都会
停止，不生成貌似完整的结果。
