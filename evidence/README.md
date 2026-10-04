# 证据与复现

截至 2026-10-04，最新实际训练为 V164，最新完整质量交付为 V159；V169 CPU 包已交付，实际平台执行仍待回传。GitHub 中可核对的 [原始结果与交付摘要](2026-10-04/github_sync/) 及 [当前进度](../docs/CURRENT_PROGRESS.md) 优先于下列历史阶段文字。大型训练产物保留在本地及正式执行包，摘要不替代完整逐行质量验收。

当前执行（2026-09-21）：**v7.5 已完成 8 次真实拟合，32,596 条 native_flow 恶意记录全部纳入最终重训，17 个解析格式逐类审查。** 四臂内部错误 3,450→1,044；追加稀缺对照至 887，但完整开发回放 F 错 12,953 条、M 召回 15.5565%，质量未通过，无新模型晋升。Windows/认证/CEF 稀缺切片有局部收益，unsupported 拟合缺口和 ASA 恶意退化仍存在。完整报告与下一步以 [V75_FOUR_ARM_EXECUTION.md](../docs/V75_FOUR_ARM_EXECUTION.md) 为准。以下内容为历史阶段，不覆盖本条。

最新 v3.7 全量回传证据：[审查状态](2026-09-13/v37_cloud_review/review_status.json)、[主决策指标](2026-09-13/v37_cloud_review/primary_matrix.json)、[ASA 编码损失](2026-09-13/v37_cloud_review/asa_encoding_detail.json)、[类别特征全量干预](2026-09-13/v37_cloud_review/category_ablation.json)、[完整报告](../docs/V37_CLOUD_RESULT_REVIEW.md)。10 个模型均已本地回放，质量未通过；原严格浮点核验失败及其数值原因单独保留。

v3.7 平台前执行证据：[全量输入验收](2026-09-13/v37_execution/input_audit.json)、[最终包内模型回放](2026-09-13/v37_execution/release_model_replay.json)、[执行包恢复检查](2026-09-13/v37_execution/release_bundle_check.json)、[准备阶段说明](../docs/V37_EXECUTION_REVIEW.md)。此处当时的本地配对只覆盖 AD；initial/second/third、旧 local/final/restored 文件保留准备和纠错过程，发布身份以 release 回执为准。

最新方法研究证据：[ASA 原始消息至规范化表示的分层核查](2026-09-13/v37_research/asa_observability.json)、[开源来源与固定提交](2026-09-13/v37_research/source_registry.json)、[研究报告与当前方案依据](../docs/V37_RESEARCH_REVIEW.md)。此次为读取官方数据后的信息审查与方法研究，没有新模型训练；ASA 原始同文无异标签，75/215 组来自规范化表示。

最新全量回传证据：[v3.6 回传独立审计](2026-09-13/v36_cloud_review/independent_return_audit.json)、[15 个云端模型本地回放](2026-09-13/v36_cloud_review/local_model_replay.json)、[真实采集时间反例](2026-09-13/v36_cloud_review/collector_time_stress.json)、[审查结论](../docs/V36_CLOUD_RESULT_REVIEW.md)。以下 9 月 12 日目录是平台前本地试训证据，不与本次全量成绩混用。

2026-09-13 当前证据：[v3.6 全量审计](2026-09-12/v36_execution/r13_independent_data_audit.json)、[最终试训对照](2026-09-12/v36_execution/r13_pilot_comparison.json)、[全部模型回放](2026-09-12/v36_execution/r13_model_replay.json)。[完整说明](../docs/V36_EXECUTION_REVIEW.md)。程序/内容核对通过不等于模型迁移验收。

## 历史阶段证据

以下状态保留其发生阶段含义，以文首当前入口为准。

2026-09-12 v3.2 第一轮：`artifacts/v32_ready_20260912/` 为最终已审查输入与新清单；`artifacts/v32_bundle_control_check_20260912/` 保存实际还原包后的全部外层/校准分数及 23 项复核。`2026-09-12/v32_bundle_integration.json` 记录分片还原、包身份及完整对照一致；`platform_article_access.json` 记录文章验证页阻碍。主模型资源拒绝记录在 `artifacts/v32_round_20260912_local/`；C1-W 尚未拟合。详见 [执行审查](../docs/V32_ROUND1_REVIEW.md)。中间失败准备产物保留，不能作为正式训练输入。


下列是本地实际结果，不是未来训练效果的承诺。

2026-09-12 已执行阶段见 [V0/V1 实施审查](../docs/V3_PREPARATION_REVIEW.md)：全量准备内容核对通过，但当前缓存的语义与全部角色组支持未通过，不能进入 C1 主训练。随后 [v3.2 方案](../docs/TRAINING_PLAN.md) 增加低成本分配可行性与资源试算；它没有修复解析器、发布新切分或训练模型。

| 本轮产物 | 证明范围 |
|---|---|
| [v32_efficiency_probe.json](2026-09-12/v32_efficiency_probe.json) | 固定旧分组的整组调整可行性见证、词/字符非零项抽样估计与解析调用复用空间；不是新清单或模型验收。对应 [复现脚本](2026-09-12/v32_efficiency_probe.py) 只写新的试算报告 |
| [v3_full_preparation_audit.json](2026-09-12/v3_full_preparation_audit.json) | 对官方全量原件的 ID/标签/空状态、重复分组、碰撞计数等 25 项独立内容核对；包含资源估计 |
| [v3_semantic_gate.json](2026-09-12/v3_semantic_gate.json) | 上游别名、身份残片、转义和路径反例；当前主训练不放行 |
| [v3_largest_collision_review.json](2026-09-12/v3_largest_collision_review.json) | 最大混标组的完整原文与准备文本，不改标签，不作为代表性样本 |
| [v3_control_prediction_check.json](2026-09-12/v3_control_prediction_check.json) | N 对照逐行概率/决策、混淆与文件身份回放，不是主模型质量验收 |
| [v3_parallel_semantic_check.json](2026-09-12/v3_parallel_semantic_check.json) | 17 个目的样例的串行/并行版本语义函数一致，非全量质量证据 |

| 文件 | 证明范围 |
|---|---|
| [data_summary.json](2026-09-11/data_summary.json) | 官方三份数据的全量结构与分布 |
| [shortcut_diagnostics.json](2026-09-11/shortcut_diagnostics.json) | 简单规则/浅树是否能利用来源与采集差异拿高分 |
| [existing_model_full_valid.json](2026-09-11/existing_model_full_valid.json) | 冻结旧模型在完整官方验证集上的实际输出 |
| [representation_stress.json](2026-09-11/representation_stress.json) | 固定 10 万条样本、无重训的表示变化与来源消融 |
| [移动清单](organization_moves_2026-09-11.json) | 17 份原件和审计文件移动前后 SHA-256 一致 |
| [删除清单](organization_deletions_2026-09-11.json) | 本次删去的文件、原哈希、原因与结果 |

`2026-09-11/` 还保留题目抽取全文、独立方案草稿和拆分的数据审计结果，以便检查结论来源。历史 JSON 中的旧绝对路径保留为运行记录，不代表当前入口；对应映射为 `data/data → data/official`、`analysis_2026-09-11 → evidence/2026-09-11`。三个复核脚本已只调整路径以适配新目录。

在 `SF02` 目录使用本地 `.venv`：

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONDONTWRITEBYTECODE='1'
& '.\.venv\Scripts\python.exe' '.\evidence\2026-09-11\audit_data.py'
& '.\.venv\Scripts\python.exe' '.\evidence\2026-09-11\shortcut_diagnostics.py'
& '.\.venv\Scripts\python.exe' '.\evidence\2026-09-11\evaluate_existing_model.py'
& '.\.venv\Scripts\python.exe' '.\scripts\40_counterfactual_audit.py'
```

脚本会重写其对应审计结果，不训练或替换旧模型。全量文本审计比结构化压力测试更耗内存。压力测试的 `source_ablation` 删除了可能有用的来源信息，是消融而非纯表示等价测试；不能把它的翻转数当成必须归零的要求。null 转空串操作作用于全部类别字段，尚未逐字段定位因果。

历史训练日志、旧评估和旧提交仅说明发生过相应运行；无法据此认定已完成全量训练、跨域验证或最终提交验收。

旧模型评估 JSON 的 `by_pipeline` 固定按三类计算 Macro-F1：例如只有正常样本的 `aws_cloudtrail` 会因两个零支持类得到 1/3。该数字不能理解为该来源有三类识别证据；应结合支持量和各类指标阅读，新评价按当前方案对不可估计项目标 N/A。

整理后的检查见 [organization_verification_2026-09-11.json](organization_verification_2026-09-11.json)：三份数据与模型哈希一致、当前文档链接有效、脚本语法和配置路径通过；完整 2,014,052 行验证推理重跑得到相同混淆矩阵和 Macro-F1 0.9166831553。
