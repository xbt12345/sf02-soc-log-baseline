# V155 完整梯度一阶邻域风险对照：实际结果与决定

2026-10-01。**六次登记拟合与完整2056871行验收已实际完成；匹配效果通过=False，任务质量通过=False，模型晋升=False。** 训练侧已验收能力保持与迁移分类质量分别判断。第二、第三及完整任务仍未关闭；本轮没有创造新独立同类支持，也没有新环境泛化确认。

实际拟合梯度1800、proposal 1806、接受更新1200，拟合分类器分块前向33714；零步9完整梯度/108前向及注册1×1 dummy 1前向/梯度分别计量，不冒充拟合收益。评价实际模型分块前向{"SecondRepresentation": 1494, "Classifier": 534}，评价新增梯度0。

## 单因素、真实终点与保留能力

[设计审查](V155_NEIGHBORHOOD_TRIAL_DESIGN_AND_IMPLEMENTATION_REVIEW.md)、[零步历史](V155_ZERO_STEP_REVIEW_AND_EXECUTION_STATUS.md)与机器方案均在更新前绑定。A普通完整原频次成员CE、B完整梯度一阶扰动点CE，同折V146 A共同初始化，既有第二层22528参数；ρ=.001×初始化范数整个fit固定，无扫描。两组共享归一化方向、步长和回溯，B使用gplus而忽略ε导数；proposal内ε固定，只验收该局部代理Armijo。B多一次梯度，不称等算力。全部合法TRAIN原行频次、未知、ICMP、冲突保留，无外折答案选点或自动追加。

|折|臂|完整梯度|proposal|接受更新|登记终止|TRAIN M/S错|末五状态稳定|
|---|---|---:|---:|---:|---|---|---|
|0|A|200|401|200|accepted_update_budget|0/22|True|
|0|B|400|200|200|accepted_update_budget|0/22|True|
|1|A|200|403|200|accepted_update_budget|0/6|True|
|1|B|400|200|200|accepted_update_budget|0/6|True|
|2|A|200|402|200|accepted_update_budget|0/28|True|
|2|B|400|200|200|accepted_update_budget|0/28|True|

终点及末五状态逐模型重放，固定ε代理/普通成员风险与日志重现；旧V138/V140/V142联合保护实际执行，A/B三role及五状态联合通过=True，纯TRAIN错{'A': 0, 'B': 0}和初始正确原行新增错{'A': 0, 'B': 0}。六模型全部22546实际数值输入绕过隐藏缓存重放，argmax一致、概率最大差2.22044604925e-16，冻结reference/head张量身份不变。窗口/模型核验只证明其覆盖的TRAIN与输入执行范围，不代替来源外验收。

## 原行分类质量

|臂|ASA M错|ASA S错|ASA总错|完整人口总错|
|---|---:|---:|---:|---:|
|A|1904|1631|3535|3642|
|B|1920|1629|3549|3656|

完整人口2056871行独立真值逐行计分；ASA112807行由六注册模型真实重放，非ASA沿用V146冻结预测、独立重计107错，不能称重新运行其模型。已看开发折不能改称盲测或正式提交。A0原ASA M318/S2074，任务ASA总错上限2170，完整各类precision/recall/F1、来源/小组/未知及多折标准均未放宽。

|臂|类0=N/1=M/2=S|原行数|漏判|误报|precision|recall|F1|
|---|---|---:|---:|---:|---:|---:|---:|
|A|0|1899723|0|105|0.999944732|1.000000000|0.999972365|
|A|1|111728|1936|1633|0.985344402|0.982672204|0.984006489|
|A|2|45420|1706|1904|0.958262090|0.962439454|0.960346229|
|B|0|1899723|0|105|0.999944732|1.000000000|0.999972365|
|B|1|111728|1952|1631|0.985359986|0.982528999|0.983942456|
|B|2|45420|1704|1920|0.957927952|0.962483487|0.960200316|

匹配保护：`{"M_no_increase": false, "S_no_increase": true, "one_class_improves": true, "two_folds_improve": false, "outside_top3_S_no_increase": true}`。

任务未通过项：A `["ASA_M_absolute", "ASA_total_absolute", "paired_M_S_protected", "multiple_folds_improve", "full_class_recall_F1_protected", "full_precision_recall_F1_protected", "header682", "root2868_M"]`；B `["ASA_M_absolute", "ASA_total_absolute", "paired_M_S_protected", "multiple_folds_improve", "full_class_recall_F1_protected", "full_precision_recall_F1_protected", "header682", "root2868_M"]`。

B对匹配A逐类别修复/新增错：`{"0": {"support": 0, "before_correct": 0, "after_correct": 0, "before_errors": 0, "after_errors": 0, "repairs": 0, "new_errors": 0}, "1": {"support": 78748, "before_correct": 76844, "after_correct": 76828, "before_errors": 1904, "after_errors": 1920, "repairs": 0, "new_errors": 16}, "2": {"support": 34059, "before_correct": 32428, "after_correct": 32430, "before_errors": 1631, "after_errors": 1629, "repairs": 2, "new_errors": 0}}`。

B对上轮V146 A逐类别修复/新增错：`{"0": {"support": 0, "before_correct": 0, "after_correct": 0, "before_errors": 0, "after_errors": 0, "repairs": 0, "new_errors": 0}, "1": {"support": 78748, "before_correct": 76844, "after_correct": 76828, "before_errors": 1904, "after_errors": 1920, "repairs": 0, "new_errors": 16}, "2": {"support": 34059, "before_correct": 32430, "after_correct": 32430, "before_errors": 1629, "after_errors": 1629, "repairs": 0, "new_errors": 0}}`。

## 反例与边界

实际B接受日志有298次相邻迭代代理值上升，当前各自固定ε代理仍满足Armijo；重放脚本确认扰动身份改变。这直接阻止把局部代理验收写成移动邻域风险单调。普通loss、局部TRAIN正确、论文动机都不能覆盖实测分类门槛。反例文件 `training/review_policy/v155_observed_neighborhood_cases.json`，可执行 `training/v155_replay_observed_cases.py`；实际重放0模型前向/梯度/拟合/更新。

试验资格、分类质量及替换资格分开。本批只对这组共同初始化、固定半径和一阶适配产生证据，不宣布全部SAM无效、不挑折/种子重试、不从已看外层选择其他终点。同配置主门槛失败则不追加确认；即便局部效果通过，也没有权限关闭细行为同类支持、跨来源稳定性或完整任务。

产物 `artifacts/v155_guarded_full_gradient_sam_20261001/` 包含六fit、原行概率/预测、梯度/proposal日志、末五状态/终点、完整账本及质量拒绝原因。原V146/V154及V155封存源码、机器方案、零步报告保留。
