# 当前训练进度

核对日期：2026-10-04。本页区分实际训练、完整质量交付和平台准备；本次 GitHub 同步没有开展新训练。

## 已完成的实际训练与检查

- **V164 是最新实际训练**：3 次拟合、5 次永久更新，三个角色分别接受 1、1、3 次更新，均因 `margin_normal_cap_stop` 停止。角色 1 相对固定基线修复 8 条原行、无新增错误；另外两个角色未产生分类修复。训练侧分类尚未掌握，模型未晋升。详见 [实际结果](V164_COMPLETE_SHORT_SUPERVISED_TRAINING_RESULTS_20261002.md) 和 [保存结果的质量审查原件副本](../evidence/2026-10-04/github_sync/v164/review.json)。
- **V159 是最新完整三分类质量交付**：完整质量未通过。V164 的短程监督结果不替代完整质量交付或独立来源泛化验收。详见 [完整结果](V159_COMPLETE_ACTUAL_RESULTS_AND_NEXT_ISSUE.md)、[交付回执](../evidence/2026-10-04/github_sync/v159/final_delivery.json) 和 [质量记录](../evidence/2026-10-04/github_sync/v159/quality.json)。
- **V165—V168 是有界诊断**，不是新的正式训练模型。下一轮科学方案见 [V169 最终决定](V169_FINAL_INCREMENTAL_DECISION_20261002.md)。

## V169 的最新交付状态

V169 本机已注册并封存；两次启动均在正式模型调用前因资源不足停止。旧本机资源等待记录保留为历史执行证据。随后在独立目录完成 CPU 迁移和平台执行包，未改变科学目标、完整参数、保护规则、预算或验收门槛。

2026-10-02 已交付 `SF02_V169_CPU_platform_v1.zip`：548,455,564 字节，1,598 个文件。包 SHA-256：

```text
7739d18e4739fb665091fd564be6d342e536c9438ace518f022f48401c75a391
```

[打包回执](../evidence/2026-10-04/github_sync/v169_platform/final_package_receipt_v1.json) 和 [独立解压核验](../evidence/2026-10-04/github_sync/v169_platform/final_zip_independent_review.json) 记录了 CRC、逐文件 SHA-256、源码清单和独立签收的一致性。该检查只证明执行包文件完整。

当前尚未取得平台实际安装、完整参数数值资格、注册时原始零步重放或正式训练结果。V169 新增正式模型调用、拟合和永久更新仍为 0；三个总目标——训练侧分类掌握、细行为同类支持、跨来源稳定修正——均未完成。下一步仍需运行完整执行包，并回传 `SF02_V169_CPU_review_v1.zip`，按原规则审查真实结果。

## GitHub 内容与复现边界

仓库同步当前源码、训练合同、历史结论、官方资料、结果报告与必要原始核验摘要。V169 CPU 启动、资格与结果回收源码的原字节副本见 [平台说明](../platform/v169_cpu_v1/README.md)。摘要副本及源文件 SHA-256 见 [同步清单](../evidence/2026-10-04/github_sync/manifest.json)。

官方 parquet、模型权重、训练数组、本机虚拟环境、已安装运行时、离线 wheel 和完整执行 ZIP 不入 Git；本地原件保留。历史文档及 MCP 目录中的 `artifacts/` 引用依赖这些本地产物，克隆仓库后不会自动存在。克隆仓库不等于完整训练环境，也不能直接对仅有源码的目录执行离线包启动器。继续训练须使用原完整执行包，或按相应原始清单恢复所有依赖并完成环境资格检查。

源码检查发现三个历史准备脚本含无效的数字开头关键字：`training/v166_coverage_core_qualification.py`、其 `_v2.py`，以及 `training/v168_decision_floor_lifecycle_qualification.py`。相应后续 `_v3.py`、`_v2.py` 的语法检查通过；这三个旧文件保留历史原字节，不作为当前执行入口。当前 V169 平台启动和训练入口的语法与交付源码哈希分别核验，源码检查没有调用模型。
