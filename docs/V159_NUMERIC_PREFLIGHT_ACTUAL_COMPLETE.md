# V159 新数值预检实际通过

`v159_boundary_train_v4.py preflight` 实际退出0，三个角色、A/B 同初始参数均通过原点概率与逐行argmax检查，所有完整重复梯度、概率、log概率、风险及方向报告已保存。实际消耗336头／意见块、12完整类梯度，参数未变，部署225558原行及联合旧能力保护通过。累计旧失败和有限诊断为 **460头／意见、20完整类梯度、0fits、0更新**。

自己的保存结果审查 `v159_numeric_preflight_execution_review.py` 也实际退出0：逐日志核对attempt/completed预算，逐角色重算四保存梯度及A/B方向，固定政策通过，两个方向均可分辨，源seal及全部保存文件哈希保持一致。可用剩余累计预算77736头／意见、2400训练完整类梯度、6fits；最多3600累计proposals／1200接受更新。

旁路 `v159_bind_numeric_independent_activation.py` 失败：启动者未等待其完整源校验结束就创建预检console，该脚本发现`preflight*`已存在后拒绝写入，没有事前绑定成功回执。独立数值入口审查文件本身早已存在、内容与来源已核验；事后审查明确记录实际失败和文件时间证据，不把它补写成事前成功。正式新预检是按已封存合格v4契约完成，旁路不修改模型或政策。

凭据：`artifacts/v159_class_boundary_numeric_trial_20261002/preflight.json`、`preflight_original_console.txt`、各`preflight{fold}_{arm}/calls.jsonl`及保存数组；小审查文件为`artifacts/v159_numeric_preflight_execution_review_20261002/review.json`。完整run seal不截断。

六次拟合及完整末五／2056871原行验收尚未执行。最新实际训练仍V158质量失败；数值预检通过只开放既定训练入口，不能称第一学习问题或任何完整目标已经通过。三个问题及完整三分类／泛化目标active。
