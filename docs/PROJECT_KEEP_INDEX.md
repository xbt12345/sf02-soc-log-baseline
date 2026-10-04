# 项目保留内容索引

整理日期：2026-10-02；平台交付状态补充核对日期：2026-10-04。当前状态以 [当前进度](CURRENT_PROGRESS.md)、项目 README 与只读 MCP 的实际交付、当前方向为准。

## 当前有效入口

| 内容 | 位置 | 保留意义 |
|---|---|---|
| 最新实际训练 V164 | [实际结果](V164_COMPLETE_SHORT_SUPERVISED_TRAINING_RESULTS_20261002.md)、[模型及逐行结果](../artifacts/v164_short_supervised_trajectory_20261002/) | 最新参数、各角色判错与保留能力的真实证据 |
| 最新完整质量交付 V159 | [完整结果与下一问题](V159_COMPLETE_ACTUAL_RESULTS_AND_NEXT_ISSUE.md)、[交付目录](../artifacts/v159_class_boundary_numeric_trial_20261002/) | 完整三分类结果；质量未通过，不当作已合格模型 |
| V169 下一轮 | [根审查与决定](V169_FINAL_INCREMENTAL_DECISION_20261002.md)、[本机封存](../artifacts/v169_prior_pair_training/)、[平台执行包](../platform/v169_cpu_v1/README.md) | 本机入口已封存但尚未训练；CPU 离线包已交付并完成文件核验，实际平台资格与训练结果待回传 |
| GitHub 进度与证据摘要 | [当前进度](CURRENT_PROGRESS.md)、[原始摘要副本](../evidence/2026-10-04/github_sync/) | 源码、报告和可追溯摘要入库；大型数据、模型、数组与运行时另行恢复 |
| 官方任务与数据 | [官方题目](official/)、[数据说明](../data/README.md)、[官方 parquet](../data/official/) | 原始资料和唯一工作数据保留，内容不改 |
| 历史有效结论与反例 | 本目录历史分析、[实验审查规则](EXPERIMENT_REVIEW_RULES.md)、[历史约束](../training/review_policy/) | 避免重复失败；历史成功、局部通过和最终质量分别记录 |
| 来源与实现 | [training](../training/)、[src](../src/)、[scripts](../scripts/)、[版本历史](../.git/) | 当前代码与有参考价值的历史实现，不按版本旧就删除 |
| 项目只读查询 | [MCP 使用说明](../mcp_readonly/README.md) | 当前与历史目录指定的文件保持可读取 |
| 本次清理 | [清理结果](../evidence/2026-10-02/project_cleanup/清理结果.md)、[删除前审查](../evidence/2026-10-02/project_cleanup/pre_deletion_review.json)、[逐项计划](../evidence/2026-10-02/project_cleanup/deletion_plan.json) | 删除对象、理由、哈希、保留检查和实际磁盘变化 |

## 为什么仍保留一些旧版本目录

最新训练会读取旧轮次的合法专家、输入矩阵、监督身份、已正确记录保护集合、对照与回滚模型。这些是存活依赖，不能因目录版本旧就删除。V169 当前本机封存中的全部项目内物理文件，以及平台迁移已核查的读取集合，均保护。

历史实验保留结论、关键指标、失败记录、最终逐行结果、实际选中或终点模型。采用末五状态验收的历史轮次保留必要窗口，避免只留下一个偶然正确的终点。冗余优化器续训状态、已替代的中间步检查点、可再生成的矩阵/解析缓存和旧上传副本已列入清理。

## 清理后的复现边界

当前 V169 封存与必要续跑依赖保持完整。本次没有训练或更改模型、数据、科学配置、验收门槛。

已结束的历史全步审计，可能仍在其旧 manifest 中引用本次删除的中间产物；这些 manifest 保留为当时的来源记录，不能再声称其全部旧物理件仍在。历史的全步复跑需要先按保留的原始数据和源码重建相关中间产物。本次保留检查不等于模型质量合格或平台执行已通过。
