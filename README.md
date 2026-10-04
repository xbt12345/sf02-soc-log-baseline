# SOC 日志威胁检测项目（SF02）

进度核对日期：2026-10-04。最新进展与 GitHub 复现边界见 [当前进度](docs/CURRENT_PROGRESS.md)；完整文件去留见 [项目保留内容索引](docs/PROJECT_KEEP_INDEX.md)。

## 当前实际状态

- **最新实际训练为 V164**：三角色短程训练完成，训练侧分类问题尚未完全解决。见 [V164 实际结果](docs/V164_COMPLETE_SHORT_SUPERVISED_TRAINING_RESULTS_20261002.md)。
- **最新完整质量交付为 V159**：完整三分类质量未通过，模型未晋升为合格成果。见 [完整结果与下一问题](docs/V159_COMPLETE_ACTUAL_RESULTS_AND_NEXT_ISSUE.md)。
- **V169 尚未训练**：本机 registration/run_seal 已保留，两次启动均在模型调用前因资源不足停止；平台迁移准备也未增加正式模型调用、拟合或更新。
- **V169 Linux aarch64 CPU 离线执行包已交付**：2026-10-02 完成 1,598 个文件的独立解压和逐文件哈希核验。平台安装、数值资格、原始零步重放及实际训练结果尚待回传。见 [平台执行包说明](platform/v169_cpu_v1/README.md) 和 [打包核验原始记录](evidence/2026-10-04/github_sync/v169_platform/final_zip_independent_review.json)。
- 三个总目标仍未完成：训练侧分类掌握、细行为同类支持、跨来源稳定修正。局部机制通过不能替代每类及独立来源分类结果。

## 使用入口

| 内容 | 入口 |
|---|---|
| 最新模型、结果、重要结论和本次清理 | [保留内容索引](docs/PROJECT_KEEP_INDEX.md) |
| 最新进度、结果摘要和 GitHub 复现边界 | [当前进度](docs/CURRENT_PROGRESS.md) |
| 官方题目与赛事资料 | [docs/official](docs/official/) |
| 原始官方数据及使用边界 | [data/README](data/README.md) |
| 当前科学方案与下一轮审查 | [V169 根决定](docs/V169_FINAL_INCREMENTAL_DECISION_20261002.md) |
| 训练约束与历史反例 | [实验审查规则](docs/EXPERIMENT_REVIEW_RULES.md)、[review_policy](training/review_policy/) |
| 只读 MCP | [mcp_readonly/README](mcp_readonly/README.md) |
| 平台使用记录 | [platform/USAGE](platform/USAGE.md) |
| V169 平台启动与结果回收源码 | [CPU 执行包说明](platform/v169_cpu_v1/README.md) |
| 历史结论与原入口全文 | [docs](docs/)、[旧 README](docs/archive/README_before_project_cleanup_20261002.md)、[旧 HANDOFF](docs/archive/HANDOFF_before_project_cleanup_20261002.md) |

## 文件保留与清理边界

原始资料、三份官方 parquet、源码、Git 历史、有效环境、最新模型/逐行输出、已验收保护集合、必要历史选中模型和结论保留。当前 V169 本机封存及已核查的平台依赖逐项保护，不按版本号删目录。

清理对象是已替代的旧中间检查点/重复逐步输出、无存活引用的派生矩阵与解析缓存、旧优化器续训状态、旧上传副本和已安装依赖的下载缓存。删除理由与逐文件哈希、保留核验和实际释放空间见 [清理记录](evidence/2026-10-02/project_cleanup/)。

历史全步复跑可能需要重建已删的中间产物；保留历史 manifest 不代表其全部旧物理件仍在。当前 V169 封存完整性与模型质量、平台执行资格分别判定，本次整理没有训练或改善模型质量。

GitHub 保存源码、实验合同、结果报告及必要核验摘要；原始 parquet、模型权重、训练数组、已安装运行时和大型离线依赖包保留在本地或正式执行包中。克隆仓库不等于恢复完整训练环境；继续训练前须恢复相应数据与模型闭包，并执行原有完整性和环境资格检查。
