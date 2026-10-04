# V159 完整实际结果与下一问题

六个固定拟合、实际端点/末五重放及完整2056871原行三分类评价已完成，**质量失败、未晋升**。最新实际训练和完整验收为V159。三个问题及完整目标active；本配置不补训练、不选更好端点、不重启未用完预算。

A ASA为276M/4860S（总5136）；B为1902M/1629S（总3531）。B相对V146 A原点只修复2M、S不变；相对A0为M增加1584、S减少445，未通过M及总错误门槛。A的原频平均CE改善伴随大量S错误；B的两类OOF稳定风险虽下降，分类及来源外收益很小。

|角色|A OOF M/S|B OOF M/S|A/B 接受更新|B 纯输入错|停止|
|---|---:|---:|---:|---:|---|
|0|534/1570|1886/1414|20/10|3278|no_feasible_step|
|1|376/1268|872/1160|21/0|1960|no_feasible_step|
|2|328/842|806/892|95/19|1586|no_feasible_step|

A OOF错误4918，B7030；B纯当前输入错误6824，保护范围新错0。B只修复38个OOF错误。B角色1接受0次更新，不能造满末五窗口。完整已验收部署225558行及V138/V140/V142联合旧能力两臂均保持；这只证明登记保护范围，没有证明训练掌握、细行为支持或跨来源稳定。

实际训练13376头/意见块、342完整类梯度、171迭代、574proposals、165更新、6fit。旧预检84/4、有限诊断40/4、新预检336/12，评价110失败+360缓存真实重放+280真实继续，共750头。累计**14586头/意见块、362完整类梯度**，另计2次dummy forward/gradient、0dummy更新。封存但未启用的v6全量恢复调用0；后继v7只继续280，不重复前360。原模型、合同、失败console及费用均不改。

六个最终proposal在固定步长域内均被分类保护阻挡；只读risk-only核对仍满足可分辨有限下降。A最终各有2个部署纯输入新错；B最终各有2个OOF保护新错、部署无新错。`no_feasible_step`只限本方向和登记步长/回溯域，不是整个约束集合无解或已收敛。拒绝点逐行输出未保存，不把端点附近的14条记录直接当作实际blockers。

评价数值修复仅传播固定逐行误差及纠正source/rows配对；分类、真值、支持、错误数、学习和完整质量门槛没有放宽。前3个角色保存的是实际v5重放FIT行和来源表，完整actual q未保存；完整冻结训练q的逐行argmax已在v5真实执行断言通过（由后续失败控制流证明），后3角色v7完整q/logq已保存。不能将前3 FIT缓存称完整q缓存。外折曾查看，为开发泛化，不是盲验。

完整N/M/S、所有来源及hard578、strict51、unknown/missing ports、ICMP、V155新M16、混类组均保留逐行账本。下一唯一候选先做主动保护约束共同下降机制资格，以保存实际blocked行/法向/求解残差/有限步证据；当前不新增官方调用。零输出导致首步隐藏梯度0的初始化候选另行保留，不能同轮混改两个因素。

证据：`artifacts/v159_class_boundary_numeric_trial_20261002/final_delivery.json`、`quality.json`、`full_prediction_ledger.parquet`、`all_fixed_cohort_quality.json`；小汇总`artifacts/v159_complete_result_records_20261002/actual_result_summary.json`；分类阻挡`artifacts/v159_saved_proposal_guard_obstruction_review_20261002/review.json`。观察案例`training/review_policy/v159_observed_class_boundary_cases.json`。
