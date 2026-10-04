"""Close completed failed mechanism diagnostic from actual saved outputs."""
import json
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings
TRIAL=ROOT/'artifacts/v160_fixed_endpoint_diagnostic_20261002'
OUT=ROOT/'artifacts/v160_complete_fixed_endpoint_result_records_20261002'
REPORT=ROOT/'docs/V160_COMPLETE_FIXED_ENDPOINT_ACTUAL_RESULTS_20261002.md'

def main():
    assert not OUT.exists() and not REPORT.exists();OUT.mkdir()
    audit_path=ROOT/'artifacts/v160_saved_diagnostic_actual_result_audit_20261002/audit.json';audit=read(audit_path);check_bindings(audit['source_sha256'])
    assert not audit['all_roles_finite_pass'] and audit['actual_new_counts']['head_attempts']==732
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),audit_path,ROOT/'training/review_policy/v160_fixed_endpoint_diagnostic_contract.json',TRIAL/'run_seal.json',*(TRIAL/f'role{r}/diagnostic.json' for r in range(3))]}
    (OUT/'pre_result_bindings.json').write_text(json.dumps(dict(source_sha256=bindings,official_calls=0),indent=2)+'\n',encoding='utf-8')
    roles=[]
    for r in audit['reports']:
        passed=[p for p in r['probes'] if p['accepted']]
        roles.append(dict(role=r['role'],status=r['status'],counts=r['counts'],finite_proposals=r['finite_proposals'],QP_solves=r['QP_solves'],margin_normals=r['margin_normals'],passed_probes=passed,parameter_hash_restored=True,joint_TRAIN_retention_passed=True))
    assert len(roles[0]['passed_probes'])==1 and all(v['classification_changes_vs_endpoint']==0 for v in roles[0]['passed_probes'][0]['scopes'].values())
    result=dict(status='V160_three_fixed_endpoint_diagnostics_complete_mechanism_not_qualified_no_fit',roles=roles,actual_new_head_feature_calls=732,actual_new_class_gradients=6,actual_new_margin_gradients=10,actual_finite_proposals=32,actual_QP_solves=4,original_margin_normals=5,new_fits=0,permanent_updates=0,cumulative_V159_V160_heads=15318,cumulative_V159_V160_class_gradients=368,cumulative_V159_V160_margin_gradients=10,all_parameters_restored=True,all_joint_TRAIN_retention_passed=True,latest_completed_trained_evaluated_model='V159',latest_V159_quality='failed',quality_acceptance=False,root_goal_status='active',source_sha256=bindings)
    (OUT/'actual_result_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    report='''# V160 三固定终点实际诊断完成，机制未通过

三个固定 V159 B 终点均已按封存入口执行并退出，未产生新拟合或永久参数更新。最新实际完整训练与质量评估模型仍是 V159，质量失败；训练掌握、细行为支持和跨来源稳定三个目标继续 active。

|角色|实际停止|前向/特征|类梯度|间隔梯度|有限探针|QP|法向|
|---|---|---:|---:|---:|---:|---:|---:|
|0|有限安全探针通过，分类不变|276|2|2|10|1|1|
|1|独立方向符号证书失败|378|2|6|21|2|3|
|2|原单位 QP 对偶间隙证书失败|78|2|2|1|1|1|
|合计|三角色机制未通过|732|6|10|32|4|5|

三个终点参数均已恢复，哈希与原封存终点一致；恢复后的完整部署与联合 TRAIN 保护通过。旧 V159 消耗不重置：本谱系累计前向/特征 15318、完整类梯度 368、完整间隔法向梯度 10。未使用预算不能授权重跑或新增训练。

角色 0 在 round0/probe8、步长 0.00390625 通过全部有限风险与分类保护检查；逐行复算 OOF 和完整 22546 部署 local 的分类改变均为零，没有新增分类修复。OOF 仍有 3278 纯错误、3300 总错误，不能据两类风险下降称训练掌握。

角色 1 第一轮最小登记步长 1.9073486328125e-6 仍产生 4 个保护 OOF 原行翻转。补入全部实际新阻挡后，第二轮两类方向斜率下降，但一个保存原单位法向的 math.fsum 斜率为 -2.2917865521998593e-15，乘积及求和误差界约 6.023618080610827e-16；独立符号证书明确拒绝，未尝试第二轮有限步。不能称固定模型无容量或全局无解。

角色 2 的两类方向与保护符号通过独立方向符号检查，但原单位 QP 对偶间隙 9.77264262894649e-20 超过已固定证书界 2.2444713589465085e-20；入口停止，仅执行原方向控制探针。没有放宽证书，没有尝试新方向有限步。

`artifacts/v160_saved_diagnostic_actual_result_audit_20261002/audit.json` 已从所有保存 q/logq、原行、来源聚合、风险、方向、重复法向和完整调用日志逐项复算通过，审查新增模型或梯度调用为零。这是本地保存数据复核，不能替代独立执行者审查或完整模型验收。

下一项可执行工作是仅使用已保存完整向量，诊断角色 1/2 的方向重构与证书残差，提出一个有界数值修正并通过原反例及新实际反例。保留全部原阈值与有限保护；先完成新版本资格及调用边界，再登记必要有限探针。当前三角色机制未通过，不新增 fit、不组合初始化/架构改动。
'''
    REPORT.write_text(report,encoding='utf-8')
    print(json.dumps(dict(status=result['status'],actual_heads=732,official_record_model_calls=0)))

if __name__=='__main__':main()
