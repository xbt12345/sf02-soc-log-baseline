"""Recount completed six-fit/full-row results and publish immutable case records."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiment_review import ROOT,read,sha,check_bindings,class_counts
TRIAL=ROOT/'artifacts/v159_class_boundary_numeric_trial_20261002'
OUT=ROOT/'artifacts/v159_complete_result_records_20261002'
REPORT=ROOT/'docs/V159_COMPLETE_ACTUAL_RESULTS_AND_NEXT_ISSUE.md'
CASE=ROOT/'training/review_policy/v159_observed_class_boundary_cases.json'
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    assert not OUT.exists() and not REPORT.exists() and not CASE.exists();OUT.mkdir()
    paths=[Path(__file__).resolve()]+[TRIAL/p for p in ['final_delivery.json','quality.json','verification.json','learning_qualification.json','full_prediction_ledger.parquet','ASA_prediction_ledger.parquet','all_fixed_cohort_quality.json','all_source_class_quality.parquet','evaluation_failure.json','evaluation_failure_v5.json','evaluation_original_console.txt','evaluation_v5_original_console.txt','evaluation_v7_original_console.txt','run_seal.json']]+[ROOT/'artifacts/v159_saved_proposal_guard_obstruction_review_20261002/review.json',ROOT/'artifacts/v159_cached_evaluation_review_20261002/qualification.json',ROOT/'artifacts/v159_evaluation_cached_continuation_20261002/run_seal.json']
    for f in range(3):
        for arm in ['A','B']:paths += [TRIAL/f'fold{f}_{arm}'/name for name in ['fit.json','endpoint_OOF_rows.parquet','endpoint_deployment_rows.parquet','calls.jsonl']]
    binding={p.relative_to(ROOT).as_posix():sha(p) for p in paths};save(OUT/'pre_result_bindings.json',dict(status='before_full_result_and_case_recount_no_new_model_calls',source_sha256=binding))
    d=read(TRIAL/'final_delivery.json');q=read(TRIAL/'quality.json');assert d['classifier_fits']==6 and d['cumulative_actual_head_calls']==14586 and d['cumulative_actual_full_class_gradients']==362 and not d['quality_acceptance'] and not d['first_issue_training_qualification']
    full=pd.read_parquet(TRIAL/'full_prediction_ledger.parquet');asa=pd.read_parquet(TRIAL/'ASA_prediction_ledger.parquet');assert len(full)==2056871 and len(asa)==112807 and full.row_position.is_unique and np.array_equal(full.row_position,np.arange(2056871))
    arms={};fits=[]
    for arm in ['A','B']:
        own=[];all_oof=[]
        for f in range(3):
            folder=TRIAL/f'fold{f}_{arm}';r=read(folder/'fit.json');rows=pd.read_parquet(folder/'endpoint_OOF_rows.parquet');assert len(rows)==r['OOF_stats']['original_rows'] and int(rows.pred.ne(rows.truth).sum())==r['OOF_stats']['total_errors'];fits.append(r)
            own.append(dict(fold=f,OOF=r['OOF_stats'],initial_risks=r['initial_stable_class_risks'],final_risks=r['final_stable_class_risks'],updates=r['accepted_updates'],iterations=r['gradient_iterations'],proposals=r['proposal_evaluations'],termination=r['termination']));all_oof.append(rows)
        oof=pd.concat(all_oof,ignore_index=True);assert len(oof)==225614
        classes={str(c):dict(support=int(full.truth.eq(c).sum()),errors=int((full.truth.eq(c)&full['pred_'+arm].ne(c)).sum()),false_calls=int((full.truth.ne(c)&full['pred_'+arm].eq(c)).sum())) for c in [0,1,2]}
        err={str(c):int((asa.truth.eq(c)&asa['pred_'+arm].ne(c)).sum()) for c in [1,2]};assert err=={str(c):d['ASA_errors'][arm][name] for c,name in [(1,'M'),(2,'S')]}
        arms[arm]=dict(roles=own,OOF_original_rows=len(oof),OOF_errors=int(oof.pred.ne(oof.truth).sum()),OOF_pure_current_input_errors=int((oof.pred.ne(oof.truth)&oof.pure_current_input).sum()),OOF_new_errors=int((oof.pred.ne(oof.truth)&oof.initial_correct).sum()),OOF_protected_regressions=int((oof.pred.ne(oof.truth)&oof.protected_correct).sum()),ASA_errors=err,full_classes=classes,failed_quality_gates=[k for k,v in q[arm]['gates'].items() if not v],protected_TRAIN=d['TRAIN'][arm]['protected_TRAIN_passed'])
    assert arms['A']['OOF_errors']==4918 and arms['B']['OOF_errors']==7030 and arms['B']['OOF_protected_regressions']==0
    costs=dict(fits=6,training_heads=sum(r['counts']['head_attempts'] for r in fits),training_opinion_features=sum(r['counts']['feature_attempts'] for r in fits),training_full_class_gradients=sum(r['full_class_gradients'] for r in fits),gradient_iterations=sum(r['gradient_iterations'] for r in fits),proposals=sum(r['proposal_evaluations'] for r in fits),updates=sum(r['accepted_updates'] for r in fits),failed_first_preflight_heads=84,failed_first_preflight_gradients=4,bounded_diagnostic_heads=40,bounded_diagnostic_gradients=4,completed_new_preflight_heads=336,completed_new_preflight_gradients=12,failed_first_evaluation_heads=110,cached_actual_v5_evaluation_heads=360,remaining_actual_v7_evaluation_heads=280,total_evaluation_heads=750,cumulative_actual_heads=14586,cumulative_actual_opinion_features=14586,cumulative_actual_full_class_gradients=362,setup_dummy_forwards=2,setup_dummy_gradients=2,setup_dummy_updates=0,unused_v6_registered_model_calls=0,own_model_calls=0,own_features=0,own_gradients=0,own_fits=0,own_updates=0)
    assert costs['training_heads']==13376 and costs['training_full_class_gradients']==342 and costs['proposals']==574 and costs['updates']==165
    cohorts=read(TRIAL/'all_fixed_cohort_quality.json')['cohorts'];save(OUT/'actual_result_summary.json',dict(status='V159_six_fixed_fits_complete_full_quality_failed_not_promoted',arms=arms,costs=costs,cohorts=cohorts,first_training_issue_passed=False,second_support_issue_passed=False,third_transfer_issue_passed=False,quality_acceptance=False,goal_status='active',validation_scope=d['validation_scope'],cached_full_q_boundary=d['cached_full_q_provenance_boundary'],source_sha256=binding))
    save(CASE,dict(version='V159_actual_class_boundary_failure_cases',observed_cases=[dict(name='nonconvex_complete_input_capacity_not_equal_actual_learned_classification',observed='B OOF errors7030, pure current input errors6824; only38 OOF repairs and2 outer M repairs',action='Do not close training or transfer issues; qualify a single new mechanism before new official calls'),dict(name='risk_common_descent_not_equal_exact_classification_feasible_direction',observed='all six last proposals pass finite risk-only review but fail registered classification guard',action='Separate OOF/deployment blockers; retain exact guards; no feasibility-set or convergence claim'),dict(name='zero_output_initialization_hides_million_parameter_input_capacity',observed='all initial hidden/input/opinion gradients0; fold1B accepted0 updates',action='Candidate mechanism only, not certified causal explanation; do not mix initialization and constrained direction changes'),dict(name='source_CE_sum_needs_row_error_propagation',observed='25 saved tables exactly self-consistent; old point8eps test failed actual replay without saving actual failed sums',action='Propagate separate per-row stableCE and clippedCE envelopes; keep predictions/support/errors exact'),dict(name='source_table_must_match_its_actual_original_row_ledger',observed='endpoint rows paired with accepted-source table caused2 mismatched root5605 reductions; correct pair passes',action='Use original paired rows; save actual outputs before assert and preserve all failed costs'),dict(name='partial_cache_has_full_q_provenance_boundary',observed='first3 actual replay FIT rows saved, fullq not saved; actual fullargmax assertion inferred from later failure control flow',action='State boundary explicitly; preserve frozen q identity; do not claim full actual q cache')],automatic_extra_fit=False,source_sha256=binding))
    lines=['|角色|A OOF M/S|B OOF M/S|A/B 接受更新|B 纯输入错|停止|','|---|---:|---:|---:|---:|---|']
    for f in range(3):
        a,b=arms['A']['roles'][f],arms['B']['roles'][f];lines.append(f"|{f}|{a['OOF']['M_errors']}/{a['OOF']['S_errors']}|{b['OOF']['M_errors']}/{b['OOF']['S_errors']}|{a['updates']}/{b['updates']}|{b['OOF']['pure_errors']}|no_feasible_step|")
    text=f'''# V159 完整实际结果与下一问题

六个固定拟合、实际端点/末五重放及完整2056871原行三分类评价已完成，**质量失败、未晋升**。最新实际训练和完整验收为V159。三个问题及完整目标active；本配置不补训练、不选更好端点、不重启未用完预算。

A ASA为276M/4860S（总5136）；B为1902M/1629S（总3531）。B相对V146 A原点只修复2M、S不变；相对A0为M增加1584、S减少445，未通过M及总错误门槛。A的原频平均CE改善伴随大量S错误；B的两类OOF稳定风险虽下降，分类及来源外收益很小。

{chr(10).join(lines)}

A OOF错误4918，B7030；B纯当前输入错误6824，保护范围新错0。B只修复38个OOF错误。B角色1接受0次更新，不能造满末五窗口。完整已验收部署225558行及V138/V140/V142联合旧能力两臂均保持；这只证明登记保护范围，没有证明训练掌握、细行为支持或跨来源稳定。

实际训练13376头/意见块、342完整类梯度、171迭代、574proposals、165更新、6fit。旧预检84/4、有限诊断40/4、新预检336/12，评价110失败+360缓存真实重放+280真实继续，共750头。累计**14586头/意见块、362完整类梯度**，另计2次dummy forward/gradient、0dummy更新。封存但未启用的v6全量恢复调用0；后继v7只继续280，不重复前360。原模型、合同、失败console及费用均不改。

六个最终proposal在固定步长域内均被分类保护阻挡；只读risk-only核对仍满足可分辨有限下降。A最终各有2个部署纯输入新错；B最终各有2个OOF保护新错、部署无新错。`no_feasible_step`只限本方向和登记步长/回溯域，不是整个约束集合无解或已收敛。拒绝点逐行输出未保存，不把端点附近的14条记录直接当作实际blockers。

评价数值修复仅传播固定逐行误差及纠正source/rows配对；分类、真值、支持、错误数、学习和完整质量门槛没有放宽。前3个角色保存的是实际v5重放FIT行和来源表，完整actual q未保存；完整冻结训练q的逐行argmax已在v5真实执行断言通过（由后续失败控制流证明），后3角色v7完整q/logq已保存。不能将前3 FIT缓存称完整q缓存。外折曾查看，为开发泛化，不是盲验。

完整N/M/S、所有来源及hard578、strict51、unknown/missing ports、ICMP、V155新M16、混类组均保留逐行账本。下一唯一候选先做主动保护约束共同下降机制资格，以保存实际blocked行/法向/求解残差/有限步证据；当前不新增官方调用。零输出导致首步隐藏梯度0的初始化候选另行保留，不能同轮混改两个因素。

证据：`artifacts/v159_class_boundary_numeric_trial_20261002/final_delivery.json`、`quality.json`、`full_prediction_ledger.parquet`、`all_fixed_cohort_quality.json`；小汇总`artifacts/v159_complete_result_records_20261002/actual_result_summary.json`；分类阻挡`artifacts/v159_saved_proposal_guard_obstruction_review_20261002/review.json`。观察案例`training/review_policy/v159_observed_class_boundary_cases.json`。
'''
    REPORT.write_text(text,encoding='utf-8');check_bindings(binding);save(OUT/'record_receipt.json',dict(status='actual_results_and_failure_cases_recorded',report_sha256=sha(REPORT),case_sha256=sha(CASE),summary_sha256=sha(OUT/'actual_result_summary.json'),new_model_calls=0,new_gradients=0,quality_acceptance=False));print(json.dumps(dict(status='actual_full_V159_result_records_complete',OOF={k:v['OOF_errors'] for k,v in arms.items()},costs=costs),ensure_ascii=False))
if __name__=='__main__':main()
