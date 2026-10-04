# V159 版本化审查协议与真实CSR身份补充

2026-10-02。新官方分类器/意见特征/梯度/拟合/更新仍0，最新实际V158未通过质量。先前正式入口合同及输入准备保留，不回写。当前入口改为 `v159_boundary_runtime_v2.py`、`v159_boundary_train_v2.py`、`v159_boundary_evaluate_v2.py` 和新的 `v159_boundary_execution_contract_v2.json`；头仍同一个 `v159_current_input_boundary_v3.py`，函数、输入、保护、预算均不再改。

`training/experiment_review.py` 已被大量历史真实执行封存，直接编辑会破坏旧证据。因此用版本化适配模块 **`training/experiment_review_v159.py`** 实现明确的新六fit有限下降协议，导入并保持旧 `evaluate_primary`、真值/全行身份、逐类评分和历史函数原语义。它不把新试验假装为旧25轮或6+12确认流程。

新runtime真正调用 `review_plan`、`seal_run`、`require_run_seal` 和 `require_checkpoint`：检查唯一头与六fit/逐类梯度/原行角色预算、实际allowed入口、历史必做动作、全部物理依赖、实际Python/package身份及源字节；预算停止原因、实际attempt/completed和最后接受状态必须一致。封存前没有官方调用，现有seal不可覆盖；入口错误或源变化拒绝。有限守卫或零共同方向早停允许如实记录，但不声明数值收敛或稳定掌握。

`v159_protocol_synthetic_qualification.py` 实际exit0，15个真实协议合成案例：缺seal、错入口、覆盖seal、变更依赖、后验选点、额外proposal、错误停止原因、错误终点、伪造窗口、漏行、篡改gold、M收益掩盖S退化均被拒绝；两个已登记新入口及有限失败停止可正确读取toy seal，一个合法12行全三类/双折改善对照通过。没有调用头、官方数据模型或官方梯度。该协议测试使用toy seal和元数据，不冒充官方运行封存。

`v159_saved_CSR_identity_audit.py` 实际exit0：当前矩阵22546×66287，float32，nnz4,006,092；canonical/sorted都true，重复项0、显式零0、未排序行0。sum_duplicates、eliminate_zeros、sort_indices后的indptr/indices/data逐位相同。输入字节键与独立审查的CSR语义一致，无需改输入或重算特征。父独立combined guard审查得到同样结论。

父独立 `v159_independent_combined_guard_audit` 显示：OOF/deployment完整X+无序意见的精确键交集0，合并既有部署正确和纯OOF正确无多标签保护矛盾，约束OOF下界22/4/26兼容注册当前X22/6/28。它只排除精确输入保护冲突，不证明106万参数头可同时学习或来源转移；键交集0亦提示两种专家意见分布不同，不能临时改源权重/seed。

已准备225614合法OOF角色原行与全部部署保护账本；原行重叠不能当独立新样本。保持已验收225558正确部署TRAIN和所有联合范围，纯OOF保护/混类全部原频监督政策不变。官方零步仍须实际全行比较3e-12和判决0变化，并在336前向与12完整类梯度预算内重复；之后才执行固定六fit和完整验收，不用toy资格替代真实结果。

新合同和适配器资格已经源绑定。正式register/seal/preflight/fit/evaluate尚未执行；本补充是继续执行的审查材料，不是新模型或完整目标已完成。
