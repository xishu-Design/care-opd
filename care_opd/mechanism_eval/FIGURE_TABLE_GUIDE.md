# CARE-OPD 机制实验图件使用说明

## 主文图1：失败救回与能力回退

文件：`fig1_behavior_rescue_regression.pdf/png`

图注：以 Clean-SFT 为固定参照，在相同的 3,900 条 family-clean 请求上统计各模型对 1,144 条参照失败的救回率，以及对 2,756 条参照成功的回退率。CARE-OPD 救回 91.61% 的失败，同时回退 19.56% 的原成功样本，获得 +509 条净成功；它是所比较方法中唯一获得正向净增益的方法。该图是条件恢复诊断，不是独立公开泛化结果。

## 主文图2：局部监督与参数偏好反转

文件：`fig2_localization_and_margin.pdf/png`

图注：左图显示 verifier 选中的修复字段覆盖 63.67% 的响应 token，全响应蒸馏会额外监督 36.33% 的外围 token，相当于局部监督的 1.571 倍。右图报告在固定 200 条 Clean-SFT 失败样本上的长度归一化 oracle-minus-rejected log-probability margin；正值表示更偏好验证通过的修复回答。GKD-full 减弱但未翻转原错误偏好，CARE-OPD 的完整响应与局部字段 margin 均转为正，且局部字段达到 +0.496。

## 主文/机制附录图3：在线轮次与失败迁移

文件：`fig3_online_round_dynamics.pdf/png`

图注：三轮在线训练的总体通过率由 72.08% 升至 77.24%，但失败集合高度非平稳。Round 2 的失败中 91.35% 相对 Round 1 为新增，Round 3 中仍有 54.56% 相对 Round 2 为新增。这一现象支持每轮更新后重新进行 on-policy 采样；由于 checkpoint 与采样随机性同时变化，该转移是描述性证据而非样本级因果修复证明。

三张图已经包含正文论证所需的核心数字，因此不再配置重复表格。当前 margin 仅报告宏平均，不添加虚构置信区间；若后续从逐样本 JSONL 计算 paired bootstrap，可直接在图2上增加误差线。
