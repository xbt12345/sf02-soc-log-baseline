# 可复核的 SOC 行为试验

截至 2026-10-04，最新实际训练为 **V164**，最新完整质量交付为 **V159**，完整质量未通过。**V169 CPU 离线执行包已交付，尚无实际平台训练结果**。当前入口见 [最新进度](../docs/CURRENT_PROGRESS.md) 和 [平台执行包说明](../platform/v169_cpu_v1/README.md)。下面的“当前”“最新”均属于当时的历史阶段，不覆盖本段。

当前执行（2026-09-21）：**v7.5 已完成 8 次真实拟合，32,596 条 native_flow 恶意记录全部纳入最终重训，17 个解析格式逐类审查。** 四臂内部错误 3,450→1,044；追加稀缺对照至 887，但完整开发回放 F 错 12,953 条、M 召回 15.5565%，质量未通过，无新模型晋升。Windows/认证/CEF 稀缺切片有局部收益，unsupported 拟合缺口和 ASA 恶意退化仍存在。完整报告与下一步以 [V75_FOUR_ARM_EXECUTION.md](../docs/V75_FOUR_ARM_EXECUTION.md) 为准。以下内容为历史阶段，不覆盖本条。

最新状态：**v5.6完成38次新拟合，候选未通过继续条件。** [结果与证据](../docs/V56_SUPPORT_POOLING.md)。[主目录](../artifacts/v56_support_pooling_20260914/)保存24个组合修正、2个新基础模型及2个复用基础模型；[共享事实补充](../artifacts/v56_shared_control_20260914/)保存12个新修正。补充先做无标签覆盖检查，再固定规则训练；来源和阶段分开保存。

`v56_pooling.py`保持原行交叉熵，惩罚参数而非删除/降权记录；`v56_shared_control.py`增加跨组合单项事实，保留全部原输入。`verify_v56.py`及`verify_v56_shared.py`从原行独立重算目标/梯度和输出；`v56_information_controls.py`确保删除前后比较同一批记录。7项数学/边界测试通过；补充一个λ候选未满足更严格梯度条件，已排除选型。已有结果拒绝覆盖，无需平台原样重跑；下列为历史说明。

最新状态：**v5.5已完成18次拟合，主继续条件全部失败；新增验证能发现完整输入成绩掩盖的缺失退化。** [结果与后续边界](../docs/V55_RISK_VALIDATION.md)。[主试验冻结目录](../artifacts/v55_risk_validation_20260914/)保存16条轨迹、48个检查点和预先固定的输入/目标/选择规则；[补充分离校准对照](../artifacts/v55_calibration_control_r2_20260914/)保存另2个模型、校准选择及评价，补充假设在主结果后提出，不是新盲测。

新增 `run_v55.py`、`review_v55.py`、`diagnose_v55.py` 与独立 `verify_v55.py`；补充由 `probe_v55_calibration.py` 执行，`verify_v55_calibration.py` 重放输出和全部校准阈值排序。6项数学、实际库加权梯度和同分边界检查通过。实际加权接口在本地sklearn上执行核验，未安装新依赖，未操作平台。已有输出拒绝覆盖，不原样重跑。旧模型/提交入口未改，下面是历史说明。

最新状态：**v5.4训练与独立核验完成，全部候选质量条件失败。** [结果与下一项边界](../docs/V54_METHOD_CHANGE.md)。[冻结目录](../artifacts/v54_method_change_20260914/)保存实际SVM/MLP源码、输入、内外层划分、选择记录、模型及六视图输出。`run_v54.py`不覆盖已有结果，不要原样重跑。

新增`verify_v54.py`从支持向量/各神经层独立回放，`analyze_v54.py`比较逐类、支持和真实原文变体，`audit_v54_cold_groups.py`与`diagnose_v54_failures.py`定位冲突及错误正常判定，`v54_inference.py`仅为ASA/ACL研究接口。核2,029,249行及神经网络1,313,917行重算通过，不代表迁移通过。核分数不是概率，神经网络概率未校准。Group DRO/强化正则化对照尚未训练；以下为历史记录。

最新状态：**v5.3 ASA 专项已完成，未晋升模型。** [结果和证据](../docs/V53_ASA_EXECUTION.md)。新增 `audit_v53_asa.py`、`run_v53_asa.py`、`test_v53_inputs.py`、`verify_v53_asa.py`、`probe_v53_joint.py`、`verify_v53_joint_raw.py`、`analyze_v53_findings.py`。16个EBM与4个完整组合计数模型均已保存；独立回放覆盖2,748,960行EBM概率及2,061,720行完整组合/回退概率，另有706条真实原文干扰检查。

EBM使用[冻结配置与运行源码](../artifacts/v53_asa_factorial_r2_20260914/)，已安装interpret-core 0.7.8。工作源码在拟合前绑定SHA；完整组合的工作脚本根据`training/`定位项目，运行前校验其与冻结副本同一SHA，不能直接将冻结副本当成可迁移入口。既有输出拒绝覆盖，无需重跑或上传平台。六个候选均未通过全部质量条件；下一项分层支持/内层选择尚未实现。历史v5.2资料在[整体审查](../docs/V52_OVERALL_REDESIGN.md)，以下记录不覆盖本轮状态。

最新状态：**v5.1 参数分离已实现并完成 4 次拟合，本轮局部开发条件与回放通过。** [结果与边界](../docs/V51_PARAMETER_ISOLATION.md)。旧压力错误 894→884；原三折 6,022→6,022，逐条零退化；不适用记录概率严格不变。公共 B 未替换。

新增 `v51_residual.py`、`test_v51_residual.py`、`run_v51.py`、`analyze_v51.py`、`verify_v51_base.py`。模型与实际训练源码在[冻结目录](../artifacts/v51_fact_residual_20260914/)。`v51_residual.classify_records` 使用保存模块及原文返回三类概率和拟合支持说明，不能将旧 v4.9 包装用于这个模型版本。

四个模型已经完成训练，不需要上传平台或原样重跑。阶段调度器与输入来源在拟合前冻结；已有结果禁止覆盖。保存目标的独立重算及基线状态比较见[证据目录](../evidence/2026-09-14/v51_review/)，[核验结果](../artifacts/v51_fact_residual_20260914/verification.json)不代表整体迁移已通过。

以下为历史阶段记录，其“当前/下一步”不覆盖文首新入口。

最新状态：**v5.0 共 7 次新拟合及独立核验完成，旧压力改善、三折退化，候选不晋升。** [结论与下一步](../docs/V50_TARGETED_REVIEW.md)。旧压力错误 894→884，原三折配对错误 6,022→6,076；公共 B 未替换。

本轮新增 `v50_views.py`、`test_v50_views.py`、`run_v50.py`、`verify_v50.py`、`analyze_v50.py`、`audit_v50_icmp.py`。当前工作区调度器包含后续修正；[冻结续跑目录](../artifacts/v50_context_training_r2_20260914/)保留实际训练源码、1 个复用压力结果及 6 次配对拟合。原目录的压力模型未覆盖或重训，续跑原因和来源绑定在配置中。

[独立核验](../artifacts/v50_context_training_r2_20260914/verification.json)覆盖全部模型保存概率和指定原文变换；实际执行核验器的源码保存在[证据目录](../evidence/2026-09-14/v50_review/verify_v50.py)。冻结调度器的旧 verify 子命令曾因沿用 v4.9 版本包装而中断，应以独立 v5.0 核验结果为准。结果已存在，不要原样重跑并覆盖；现有工作不需要上传平台。

以下为历史阶段，不覆盖文首新入口。

最新状态：**v4.9 共 2 次新拟合与复核已完成，两项候选均无分类收益，不晋升、不继续三折扩训。** [结果与后续方向](../docs/V49_EVIDENCE_REVIEW.md)。观察值编码 O 与原词/字符 T 分别在独立目录固定配置比较，均为 894 条错误，逐条零修复、零退化。保留 v4.8 动作修复，停止这两个具体候选的重复试训。

本轮源码：`v49_encoding.py`、`test_v49_encoding.py`、`v49_text.py`、`run_v49.py`、`summarize_v49.py`。结果及冻结代码分别在 [O](../artifacts/v49_observed_encoding_20260914/) 和 [T](../artifacts/v49_word_character_20260914/)；[汇总诊断](../evidence/2026-09-14/v49_review/)含 314 条共同事实删减诊断。执行时必须使用各自冻结副本；当前工作区调度器包含随后增加的 T 对照支持，不覆盖 O 的冻结代码。两项都已完成，不需要原样重跑或上传平台。

下一轮先审查共同事实视图的语义，再验证完整输入与共同视图联合学习。该候选尚未实施，不将覆盖说明或拟合侧删减测试当作分类验收。以下为历史阶段，不覆盖文首新入口。

最新状态：**v4.8 动作修复、一次一致重训与核验完成；质量条件未全部通过，公共 B 未替换。** [结果与下一步](../docs/V48_INFORMATION_REPAIR.md)、[冻结目录](../artifacts/v48_information_repair_r2_20260914/)。总错误 3,988→894，但认证退化 2 条，按拟合前条件停止后续三折。

`v48_input.py` 复用冻结依赖，补回有原文依据的动作并记录字段处置；`test_v48_input.py` 验证允许/拒绝/未知/冲突与包装边界；`run_v48.py` 准备和条件调度；`verify_v48.py` 完整概率、原文推理及实际编码覆盖复核；`analyze_v48.py` 只读分析逐组贡献、正常反例与退化分数。结果目录内的配置、源码与模型禁止覆盖，拟合必须运行冻结副本并使用全新输出目录；当前已执行完毕，不需要原样重跑。

下一步优先修正正文词典覆盖和字段适用性，核查拟合侧语义支持；33 条认证正文全零和无支持事实列仍未解决。先做不训练分类器的表示验证，再逐项比较。当前本地 CPU 足够，不需要平台包；仅用官方数据，不恢复身份和绝对时间捷径。

历史 [v4.7 审查](../docs/V47_DIRECTION_RESET.md)、[v4.6 实测](../docs/V46_INTERACTION_REVIEW.md)及其冻结资料保留。以下内容属于当时阶段，不覆盖文首新入口。

历史 [v4.5](../docs/V45_MISSINGNESS_REVIEW.md)的 Z/R 模型及缺失机制证据保留。当前 v4.6 冻结目录禁止覆盖；若仅复核已有结果，应使用独立输出位置或只读比较，不再次覆盖核验凭据。准备与拟合使用 `probe_v46_interaction.py` 的 `prepare`/`fit` 两阶段，拟合运行冻结目录内副本，必须指定全新结果目录。现有结果已完成，不需原样重跑。

历史状态：**v4.4 六次现成模型拟合、原文与概率核验已完成；质量未通过，基准 B 未替换。** [报告与下一步](../docs/V44_READY_METHODS_REVIEW.md)，[冻结执行资料](../artifacts/v44_ready_methods_20260914/)。`probe_v44_ready_models.py` 实现固定配置 LR/CatBoost 对比，`verify_v44_probe.py` 复核原文及诊断决策点，`analyze_v44_results.py` 核算信息损失与保护重建。下轮仅优先验证 LR 的独立分组阈值选择；尚未执行，不得应用评价答案推导的阈值。

本轮本地 CPU 完成，未安装新依赖，不需平台操作。冻结目录内的模型、配置与源码用于复核，不应覆盖或原样重复已否决配置。

历史分析状态：**v4.3 全量信息审查和优化方案已完成，0 次新模型拟合。** [方案](../docs/V43_OPTIMIZATION_PLAN.md)推荐独立 ASA 条件分类器与分层共享；实施、内层选择和外折训练尚未执行。`audit_v43_identifiability.py` 只读核查现有观察信息、类别冲突与拟合侧支持；[结果](../artifacts/v43_information_audit_20260914/identifiability.json)附冻结审查源码，[复核](../artifacts/v43_information_audit_20260914/verification.json)独立计数得到三折最低错误数 662 / 634 / 674。以下 v4.2 为最近实际训练。

2026-09-14 当前状态：**v4.2 已完成 3 次局部拟合和 3 个不训练的固定对照，均零修复、零退化，未晋升。** 见 [执行报告](../docs/V42_EXECUTION_REVIEW.md)、[已执行方案](../docs/V42_PLAN.md)及 [结果目录](../artifacts/v42_local_r1_20260914/)。本轮本地 CPU 已完成，无需再次运行或上传平台。

本轮实现：`v42_core.py` 冻结 B，只对完整且支持充分的上下文学习有惩罚和幅度限制的恶意/可疑偏移；`run_v42.py` 冻结资料、逐折拟合和审核；`verify_v42.py` 从保存包重建拟合侧支持，复核全部评价概率、原文干扰和字段回退反例。三折训练必须复用结果目录中的冻结源码，不能改写旧模型依赖。

`diagnose_v42_scope.py` 只读计算错误覆盖、约束内理想上界及语义冲突；`audit_v42_source_tokens.py` 是正式的完整源端口槽位复核。`audit_v42_missing_context.py` 及其初版输出保留用于审查诊断错误，已被 v2 替代，不作为事实入口。完整核验见 [verification.json](../evidence/2026-09-14/v42_execution/verification.json)：6 包共 2,757,300 行概率重放；每包 31,893 行原文/变换，全部相符。

该保护方法已验证，但没有分类收益；停止原样扩训，下一步先查脱敏后的可观察证据和混标语义，不以放宽范围或增加幅度追分。原生状态的 378 行仍全部为可疑，不能作为正常误报验证。历史 [v4.1 十二次拟合](../docs/V41_EXECUTION_REVIEW.md)及其失败证据保留。

以下为历史状态与复现说明。

2026-09-13 历史状态：**v4.0 九次主拟合及真实推理复核完成，整体质量未通过，未进入压力拟合。** 见 [执行结果](../docs/V40_EXECUTION_REVIEW.md)和 [已执行方案](../docs/V40_NEXT_PLAN.md)。I 改善 ASA，却使 Duo 10 条、Windows AD 18 条原先正确的可疑记录退化；不晋升模型，不生成提交。本轮使用本地 CPU，无需用户操作平台。

本轮实现：[v40_core.py](v40_core.py)、[准备](run_v40_prepare.py)、[单次拟合](run_v40_train.py)、[输入审计](audit_v40_inputs.py)、[主比较](review_v40_primary.py)、[真实推理复核](verify_v40_execution.py)、[认证分数诊断](diagnose_v40_auth_regression.py)、[完整退化账](diagnose_v40_regressions.py)。14 项编码和边界检查通过。保存结果及 [训练前冻结源码](../artifacts/v40_local_r1_20260913/frozen_training_runtime/)供复核，九次训练全部使用该源码。不要改动源码快照或已保存配置。

只读复核命令（从项目根目录运行，输出目录必须尚不存在）：

```powershell
.\.venv\Scripts\python.exe training\verify_v40_execution.py --root . --run artifacts\v40_local_r1_20260913 --out evidence\v40_independent_verification
```

命令不训练模型。已完成的本轮复核见 [verification.json](../evidence/2026-09-13/v40_execution/verification.json)。主比较没有合格压力候选，不能手改选择记录后强行运行压力训练。源码中的运行相对路径依赖项目根目录；训练复现还必须冻结配置并使用全新结果目录，不应原样重复本轮已否决方案。

此前 [v3.9 执行报告](../docs/V39_EXECUTION_REVIEW.md)及冻结模型保留作历史基准。制定 v4.0 时使用的 `audit_v40_mechanisms.py`、`audit_v40_stress_support.py`、`audit_v40_missing_ablation.py` 仅诊断旧冻结模型；最后一个推理干预已淘汰，不能与本轮新训练结果混淆。

## 历史阶段说明

以下“当前、未训练、尚未生成”等状态仅属于相应历史阶段，以文首新入口为准。

**当前状态（2026-09-12）：v3.3.1 的 9 个任务、27 次拟合真实完成；回传审计及 6,089,952 条任务预测回放通过，来源迁移未通过。** 新增 [audit_v331_cloud_return.py](audit_v331_cloud_return.py)、[diagnose_v331_failures.py](diagnose_v331_failures.py)、[ablate_v331_carrier_features.py](ablate_v331_carrier_features.py) 已执行，只读原始训练结果。见[实测审查](../docs/V331_CLOUD_RESULT_REVIEW.md)、[当前 v3.5.1 方案](../docs/TRAINING_PLAN.md)。v3.5.1 已实现局部输入和拟合约束组件、全量审计、47,044 行候选文本补丁及新旧联合分组，完整 B1/B2 和新模型仍未完成。[最新专项审查](../docs/V351_INPUT_REVIEW.md)。以下 v3.3.1 入口保留作历史复现。

新增 [audit_v35_task_assumptions.py](audit_v35_task_assumptions.py) 已对全部官方训练行及题目原件做只读任务支持审查，结果见[整体方向复审](../docs/V35_DIRECTION_REVIEW.md)。没有新增训练。下方命令遵循当时冻结的 v3.3.1 配置，不是当前 v3.5 实施入口。

历史 v3.3.1 平台只上传 `platform/sf02_v331_update.py`。它统一调度输入修复、审计、协议检查、`run_v331_train.py` 的 9 个任务/27 次拟合，以及 `review_v331_round.py` 的逐条回放和最终结果打包。参数、原始行频率与职责隔离按当时冻结方案执行；中断按任务核验恢复。不要将首个失败的 `artifacts/v331_ready_20260912` 用作正式输入。

历史 v3.3.1 增量解析源码为 `v331_prepare.py`，全量准备为 `run_v331_prepare.py`，独立输入审计为 `audit_v331_prepare.py`，协议分配与核对分别为 `v331_protocol.py`、`verify_v331_protocol.py`。`test_v331_input.py` 覆盖 18 项输入反例，`test_v331_execution.py` 覆盖 4 项软件执行检查；合并原 v3.2 的 44 项回归共 66 项。原 `soc_v32_prepare.py`、`v32_finalize.py`、`v32_split.py` 等只在未变规则或基础工具处复用，其版本由摘要绑定。旧 v3.2 的训练、预测与审查入口保留作复现，不再是当前平台运行入口。

以下 v3.0/v1/v2 命令与状态均保留为历史复现记录。

## v3.5.1：本地专项检查，尚非训练入口

- 输入/拟合约束：[v351_safeguards.py](v351_safeguards.py)。
- 可复核全量审计：[audit_v351_inputs.py](audit_v351_inputs.py)。重新执行需指定不存在审计报告的新输出目录，避免覆盖当前证据。
- 结果：[audit.json](../evidence/2026-09-12/v351_input_review/verified/audit.json)。只使用验证输入的产品/时间列作覆盖检查，没有读取验证答案。
- 候选补丁与联合分组已生成，但旧 9 个职责均需重新分配；不可直接拿旧模型或旧划分运行新输入。
- 新入口目前只是阶段 1 的专项组件，不能称为完整 B1 或 B2。

## v3.0：历史准备和对照（未获训练资格）

本轮仅使用 `data/official/train.parquet`，未读取官方验证文件，没有外部数据。当前三个入口均为 Python 3.8 语法兼容：

- `soc_v3_prepare.py`：分块准备、原文区间、相同全文分组及角色支持；27 项针对性测试通过，**仍有全量发现的缺陷**。
- `audit_v3_prepared.py`：独立重读原件，对比 ID、标签、空状态、原文/准备全文重复约束，并重算混标统计与资源估计。
- `v3_format_control.py`：按 fit＋selection 的格式/消息族频率形成 N 对照，保存三类概率、独立校准阈值和外折决策；不训练主文本模型。

完整记录在 `artifacts/v3_prepare_parallel_20260912/` 和 `artifacts/v3_format_control_20260912/`。以下只用于复现当前**未获主训练资格**的审计版本；输出目录必须不存在，不覆盖历史结果：

```powershell
& '.\.venv\Scripts\python.exe' training/soc_v3_prepare.py --train data/official/train.parquet --output-dir artifacts/v3_prepare_replay --workers 4
& '.\.venv\Scripts\python.exe' training/audit_v3_prepared.py --train data/official/train.parquet --run-dir artifacts/v3_prepare_replay --output evidence/v3_prepare_replay_audit.json
& '.\.venv\Scripts\python.exe' training/v3_format_control.py --run-dir artifacts/v3_prepare_replay --output-dir artifacts/v3_control_replay
```

上述旧版问题已由 v3.2 修复并重建清单，详见顶部入口。`informative` 在 0.3 中只表示正文非空，不能当成已确认事实或独立事件。相同随机种子也不保证不同 sklearn 版本重建相同清单。

**当前状态：v1.1 的两端切分内容和逐行预测类别已核对通过，无需再次检查环境或重训 v1。v2 已在本地执行，但检测与迁移验收均未通过。** 完整概率数组未做跨运行环境比较；只读复核核对输入哈希记录，没有重新读取云端原始数据。最新结果与决策见 [BEHAVIOR_V2_REVIEW.md](../docs/BEHAVIOR_V2_REVIEW.md)。

## v2：已执行的行为解析与来源留出

入口：[soc_behavior_v2.py](soc_behavior_v2.py)，Python 3.8 兼容，使用已有 CPU CatBoost 环境。24 个行为字段与 21 个上下文字段分开比较；同一数据条件下重训旧表示、行为、温和权重及行为加上下文四组模型。先冻结旧/新消息组连通清单，隔离跨原用途的分量，再做内部阈值校准及整来源留出。分组是保守近似，不能把隔离行数当作确认重复数。

本地真实执行：CPU 4 线程，约 159.3 秒；4 组主试验、16 次来源留出拟合、2 次因训练缺类报告不可评价。13 项解析/分组测试通过，模型加载回放和来源排除已另行复核。结果保存在 `artifacts/behavior_v2_local_20260911/`。开发区可疑类仅 1 行；上下文方案在两个来源留出中明显失败，当前不采用任何本轮模型作为最终模型。

以下只用于复现已完成的实验，**当前不建议为了追分在云端重复此版本**。本地在项目根目录执行：

```powershell
& '.\.venv\Scripts\python.exe' 'training/soc_behavior_v2.py' --previous-run 'artifacts/signal_pilot_local_v11_20260911' --data-dir 'data/official' --output-dir 'artifacts/behavior_v2_replay' --iterations 200 --threads 4
```

云端如需复现，上传 `soc_behavior_v2.py` 单文件至 `/root/work`，保留已有 v1.1 运行目录及其中的冻结源码、清单和输入记录：

```bash
conda run --no-capture-output -n cm-model python -u /root/work/soc_behavior_v2.py --previous-run /root/work/sf02_runs/signal_pilot_v11 --data-dir /root/work/sf02_data --output-dir /root/work/sf02_runs/behavior_v2 --iterations 200 --threads 16
```

输出目录必须不存在，程序拒绝覆盖已有实验。它核验 v1.1 冻结源码和官方输入身份，不读取官方验证答案；旧校准/评价区的消息仅用于无标签分组关联，其标签不参与 v2 拟合、校准、早停或评分。未引入外部数据或 NPU 依赖。

主要产物为 `result.json`、`coverage.json`、`component_split_manifest.parquet`、`development_comparison.json`、`source_holdout_results.json`、抽取证据、特征缓存、源码快照及主试验模型。阈值结果复用 v1 指标函数，部分嵌套键仍以 `audit_` 开头；在 v2 中实际对应声明的开发区或来源留出区，不能据键名称为未见评价。

本地测试入口：`.venv\Scripts\python.exe training\test_behavior_v2.py`。下一步以分组可信度、独立类别组数和同源反例为先，详见当前训练方案；规范化文本模型、因果窗口、概率校准与独立外部评价尚未实现。

## v1：历史试验与复现说明

入口：[soc_signal_pilot.py](soc_signal_pilot.py)。Python 3.8 兼容，CPU CatBoost，无新增依赖。它是训练方案 A/B1 的有限原型，检查“当前可见动作词是否足以支持判断”，不代表完整行为解析器或已通过迁移验收的参赛模型。

### 云端核对已完成

用户已回传 v1.1 云端训练和全部 14 项只读核对通过的结果，见 [核对回传](../platform/pilot_content_verification_cloud_user.json)。以下保留只读复核命令供需要时重放，无需再次执行：

```bash
conda run --no-capture-output -n cm-model python -u /root/work/verify_pilot_result.py --run-dir /root/work/sf02_runs/signal_pilot_v11
```

这一步不训练、不写文件、不读取登录信息。它核对已有产物与记录哈希，按稳定格式对比清单内容和逐行预测类别，重新计算云端三类召回，并读取此前的报警阈值结果。输出 `all_checks_passed`，但即使全部通过，也只是本次两端结果复核，不是迁移验收或分数逐浮点位一致。

两端 Parquet 字节哈希不同可能来自编码/版本元数据，不能据此自动认为切分内容变了。此脚本对内容比较，已在本地完成完整回放及 4 项风险测试：编码/顺序改变不误判、真正的角色/组/标签变化能检出、重复行拒绝、Python 3.8 语法检查。用户回传确认清单内容和逐行预测类别一致，具体字节差异原因没有进一步定位。

以下保留首次训练的复现方式：

通过 Jupyter 把本目录的 **soc_signal_pilot.py 一个文件**上传至 `/root/work`。已有数据位于 `/root/work/sf02_data`，然后执行：

```bash
conda run --no-capture-output -n cm-model python -u /root/work/soc_signal_pilot.py --data-dir /root/work/sf02_data --output-dir /root/work/sf02_runs/signal_pilot_v11 --sample-size 100000 --iterations 200 --threads 16
```

这是有进度输出的前台试验。输出目录必须尚不存在；程序拒绝覆盖旧实验。若已有同名结果，先查看其中 `result.json` 或 `failure.json`，不要直接删除重跑。线程设 16，低于用户回传的 40 核配额；使用 CPU，未调用昇腾卡或外部服务。

完成后查看：

```bash
cat /root/work/sf02_runs/signal_pilot_v11/result.json
```

失败时查看 `failure.json`。如果状态为 `blocked_by_class_coverage`，说明当前固定分组切分缺少类别支持：查看 `coverage.json`，保留该结果，不换种子凑三类，也不退回逐行随机切分。

## 数据与判断依据

1. 完整核对两份输入的官方 SHA-256；验证输入只用于身份检查，不用于选型、拟合或调阈值，也不读取验证答案。
2. 从全部训练行的位置均匀抽取 100,000 行，随机种子固定 20260911；分块读取，只保留抽中的日志。没有按标签均衡抽样，也不是取前 10 万行。
3. 消息内已知实体、脱敏编号、IPv4/MAC/UUID、长十六进制串和指定日期格式先规范化。固定抽取 23 个动作词/明确布尔值信号。产品、时间、身份、缺失、消息长度、JSON 解析状态均不作主模型特征。
4. 对规范化消息进一步去除数字差异，按相同消息哈希分组。使用组哈希将组分配到训练/开发/校准/评价，目标比例 70/10/10/10。**按组分配不会保证行数、来源或类别比例一致**；大组和各类支持量必须一起解释。
5. 先保存切分清单并检查组不跨区、表示变化一致和四区类别覆盖，再进行一次无类别权重的 CatBoost 拟合。最多 200 棵深度 5 的树，仅开发区用于提前停止。
6. 校准区选择非正常风险分数 `1 − P(benign)` 的报警阈值。评价区只测其实际表现。阈值报警结果与三分类 argmax 结果分别报告；没有把“模型不确定”改标为 suspicious。

目前日志有部分字段名、行为词甚至 JSON 数值被脱敏标记破坏。因此本版本使用可读片段的固定词法信号，不假装能完整恢复原事件。`blocked:false` 等不能直接算作肯定动作；即使匹配到 failure、deny、powershell，也只是出现了相关文字，可能存在引用、否定或合法管理行为，不是攻击真值。

没有任何动作信号的日志会被单独计数，并在评价输出中标记 `needs_review_no_action_signal`。它们仍参与指标计算；不会被删掉来美化分数。模板分组是保守的近似，同一事件的格式变体仍可能漏归组，不等价于严格事件、实体或来源隔离。

## 如何解释结果

| 文件 | 用途 |
|---|---|
| `result.json` | 实验状态、关键指标、运行时间、版本与哈希、明确未完成项 |
| `input_identity.json` / `configuration.json` | 输入原件身份、参数、特征规则与切分契约 |
| `split_manifest.parquet` | 抽样行位置、event_id、消息组和固定用途 |
| `coverage.json` | 每来源/类别支持量、无信号比例、最大组、混合标签组、JSON 损坏诊断 |
| `representation_checks.json` | 元数据、已知实体、指定日期格式、空值表示与批次一致性检查 |
| `metrics.json` | 三类 P/R/F1/AP/ROC、全判正常对照、误报/万条、校准阈值、分来源指标、组重采样区间、Brier/log-loss |
| `audit_predictions.parquet` | 评价逐行输出和无信号复核标记 |
| `audit_error_examples.json` | 每种误判最多 10 条，含匹配片段；属于诊断材料 |
| `feature_importance.json` / `model.cbm` | 模型依赖及可重放模型；重要性不证明因果关系 |

AP 使用 sklearn 的 average precision 口径，不是梯形积分 PR-AUC。分来源缺少某个类别时，该类 recall/F1/AP 等标记为空；同时保留“缺类按零计的三类 Macro-F1”，该数不能用于把缺少类别的来源与完整来源直接排名。

固定误报预算只约束校准集上的经验误报率，不保证评价集或外部环境达到相同比例。低误报结论还受独立正常事件数量制约；程序始终明确没有这种保证。组 bootstrap 只是当前抽样来源混合内的描述性区间，不能替代外部测试。输出的模型分数尚未经概率校准。

## 下一阶段的边界

v1 本身未实现规范化文本线性模型、温和权重对照、来源留出、时间窗口、外部场景评价、概率校准或最终提交程序。后续 v2 已实现有限行为解析、温和权重和来源留出，实测范围见上文；其余仍未实现。任何根据 v1 评价错误做出的改进都会使这份评价区成为已见诊断材料；不要反复据此选模型再称其为未见测试。

官方数据足以启动此试验；要证明迁移能力仍需要独立环境评价，并按已确认缺口补充训练样本。具体补数据优先级以 [当前训练方案](../docs/TRAINING_PLAN.md) 为准。

本地风险检查入口：`.venv\Scripts\python.exe training\test_soc_signal_pilot.py`。11 项测试检查数据完整性、抽样位置、表示和评价逻辑，不证明检测质量；v1.1 真实 SOC 试验另存于 `artifacts/signal_pilot_local_v11_20260911/`，原 v1 记录保留供复核。v1.1 修正实体扰动为一对一映射，禁止把零误报 bootstrap 的 `[0,0]` 当作可信上界，并保存脚本快照；特征与模型参数未改变。
