"""Publish verified training evidence, explicitly keeping failed quality separate."""
from run_v75 import ROOT,read,save,sha

DEST=ROOT/'artifacts/v89_readout_support_20260927'
DOC='docs/V89_READOUT_SUPPORT_TRAINING_RESULTS.md'
STATUS='readout_support_training_completed_no_model_promoted'


def main():
    folder=ROOT/'evidence/2026-09-27/v89_readout_support';folder.mkdir(exist_ok=True)
    assert not (folder/'delivery.json').exists()
    verify=read(DEST/'verification.json');post=read(DEST/'postcheck.json');stress=read(DEST/'stress_model_replay.json')
    assert verify['status']=='passed' and not verify['prior_bound_files_changed'] and verify['source_sha256']==sha(ROOT/'training/v89_verify.py')
    assert post['status']=='passed' and post['source_sha256']==sha(ROOT/'training/v89_postcheck.py') and post['no_candidate_promoted']
    for p,h in {**verify['output_bindings'],**post['output_bindings']}.items():assert sha(DEST/p)==h,p
    for p in ['v89_stress_verify','v89_support_conflicts','v89_support_diagnose','v89_constraint_diagnose']:
        receipt={'v89_stress_verify':'stress_model_replay.json','v89_support_conflicts':'support_training_conflicts.json','v89_support_diagnose':'support_train_and_holdout_diagnosis.json','v89_constraint_diagnose':'dual_derived_certificates.json'}[p]
        assert read(DEST/receipt)['source_sha256']==sha(ROOT/'training'/(p+'.py'))
    reg=read(DEST/'registration.json')
    for p,h in {**reg['source_bindings'],**reg['input_bindings']}.items():assert sha(ROOT/p)==h,p
    ledger=read(DEST/'execution_ledger.json');assert verify['actual_classifier_fits']==10 and all(a['status']!='started' for a in ledger)
    readouts={n:read(DEST/(n+'_fit.json')) for n in ['R00','R01','R10','R11','F00']}
    assert all(a['status']=='completed' and not a['primary_eligible'] and not a['risk_converged'] for a in readouts.values())
    support=read(DEST/'support_results.json');assert support['all_conditions_converged'] and support['actual_classifier_fits']==5
    assert read(DEST/'selection.json')['primary_selected'] is None and read(DEST/'partial_expression_selection.json')['selected'] is None
    summary=('v8.9本轮新增10次分类器拟合、0次校准拟合：4个读出、5个从头支持对照、1个部分事实分支。'
        '6条困难M可见事实与逐行信息保存已修复；5个支持模型收敛，5个修正模型后续风险优化触及求解上限。'
        'F00训练错误315→120、原正确922009条零退化，但内层新增668条M错误；全部候选未通过，不扩折或全量重训。'
        '支持完整条件S0/6，细行为训练S28/80，52条训练错无同输入冲突。主模型质量未通过，无新模型晋升。'
        '最近完整开发回放仍为v7.9的5947错。')
    files=[p for p in DEST.rglob('*') if p.is_file()]+list((ROOT/'training').glob('v89*.py'))+[ROOT/'training/test_v89_solver.py',ROOT/DOC,ROOT/'artifacts/v89_preparation_failure_20260927.json']
    bound={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)}
    delivery={'status':STATUS,'quality_acceptance':False,'all_issues_solved':False,'model_promoted':False,
        'actual_new_fits':10,'actual_classifier_fits':10,'actual_calibration_fits':0,'support_classifier_fits_converged':5,
        'readout_models_with_verified_feasible_parameters':5,'readout_risk_optimization_stopped_unconverged':5,
        'solver_calls':verify['solver_calls'],'classifier_fits_aborted_without_saved_model':0,
        'new_full_data_final_fit':False,'new_full_development_replay':False,'new_full_task_pipeline_replay':False,
        'source_rotation_executed':False,'candidate_support_retraining_executed':False,'conditional_gates_reason':'No inner ASA/class-nonregression candidate; stop under registered v88 plan.',
        'platform_used':False,'external_training_data_used':False,'pseudo_labels_used':False,'development_not_blind':True,
        'latest_executed_training_delivery':'evidence/2026-09-27/v89_readout_support/delivery.json',
        'latest_full_development_delivery':'evidence/2026-09-27/v79_execution/delivery.json','latest_full_development_errors':5947,
        'validation_scope':'Original2056871 row labels and saved decisions/complete P constraints replayed; sampled CPU manual hidden check, five fresh support model gradients, fixed-basis numerical certificates and25raw pressure examples verified. No new blind/external/full raw-task replay.',
        'readout_results':{n:{k:a[k] for k in ['roles','inner_ASA','primary_eligible','risk_converged','P_original_rows','E_original_rows','full_constraint_check']} for n,a in readouts.items()},
        'support_results':support,'support_training_conflicts':read(DEST/'support_training_conflicts.json'),
        'partial_facts':{'original_row_mapping_preserved':True,'observed_sets':read(DEST/'row_facts_receipt.json')['unique_observation_sets'],'hard_M_cases':6,'route_class_cells':25},
        'verification':verify,'postcheck':post,'stress_summary':{'original_examples':25,'decision_changes':stress['decision_changes'],'remaining_R0_identity_matrix_counterexamples':1},
        'verification_helper_recovery':{'interrupted_no_fit_attempts':1,'reason':'Repeated npz decompression in witness row loop; cached arrays before successful repeat. No trained model changed.'},
        'current_plan':DOC,'artifact_sha256':bound}
    save(folder/'delivery.json',delivery)
    for relative,link in [('README.md',DOC),('docs/TRAINING_PLAN.md','V89_READOUT_SUPPORT_TRAINING_RESULTS.md')]:
        path=ROOT/relative;old=path.read_text(encoding='utf-8');assert 'v8.9本轮新增' not in old;title,body=old.split('\n',1)
        path.write_text(title+'\n\n当前训练执行结果（2026-09-27）：**'+summary+'** [实际结果、问题与停止决定]('+link+')。以下为历史阶段。\n'+body,encoding='utf-8')
    path=ROOT/'mcp_readonly/catalog.json';catalog=read(path)
    for e in catalog['documents']:
        if e['path'] in ['README.md','docs/TRAINING_PLAN.md']:e.update(sha256=sha(ROOT/e['path']),summary=summary)
        else:assert sha(ROOT/e['path'])==e['sha256'],e['path']
        if e['id']=='v88-direction-review':e['category']='historical_review'
        if e['id']=='v88-delivery':e['category']='historical_evidence'
    catalog['project'].update(as_of='2026-09-27',current_summary=summary,authoritative_direction_id='v89-direction-review',authoritative_delivery_id='v89-delivery',
        current_direction=['本轮10次真实拟合已执行，质量未通过；按预定门槛停止扩折、候选支持重训及最终全量训练。',
            '保留逐行部分事实修复；后续先独立复算受约束风险求解，不能把未收敛回退模型当目标充分优化的上限。',
            '针对52条有细行为支持、无同输入冲突的训练S检查可见上下文和直接分类监督；按原行M/S判对与退化验收，不恢复身份时间指纹或删恶意。'],
        known_limits=['5个支持模型收敛，5个修正模型风险优化触及上限；训练可行参数并非该风险收敛最优。',
            'F00训练修复195条但内层新增668条恶意错误；训练原正确保护不保证跨主体稳定。',
            '支持目标仅M28/S6且主体组成控制不完全匹配；不能推广为所有缺支持问题的因果结论。',
            '25个样例压力无判定变化仍有1个身份编码差异，现实未知格式/时间/主体泛化未解决。',
            '无新盲测、完整工程回放、平台训练或模型晋升；本地MCP测试不等于ChatGPT云端Tunnel实连。'])
    entries=[{'id':'v89-direction-review','title':'v8.9 读出、支持及部分事实真实训练结果','path':DOC,'category':'current_review','summary':summary,
         'keywords':['当前','最新','方向','下一步','训练','ASA','恶意','可疑','读出','支持','事实','退化','v8.9']},
        {'id':'v89-delivery','title':'v8.9 十次真实拟合与独立核查交付证据','path':(folder/'delivery.json').relative_to(ROOT).as_posix(),'category':'current_evidence',
         'summary':'10次真实分类器拟合，5个风险优化停止，5个支持模型收敛；质量未通过。全部原标签和决策复核，645个历史绑定文件未变。','keywords':['执行','证据','审查','训练','v8.9']}]
    for e in reversed(entries):e['sha256']=sha(ROOT/e['path']);catalog['documents'].insert(0,e)
    save(path,catalog)
    path=ROOT/'mcp_readonly/tests/test_readonly_mcp.py';s=path.read_text(encoding='utf-8').replace('v88-direction-review','v89-direction-review').replace('v88-delivery','v89-delivery')
    s=s.replace('root_cause_review_completed_no_new_fit',STATUS).replace('本轮新增0次分类器拟合','本轮新增10次分类器拟合');path.write_text(s,encoding='utf-8')
    print(__import__('json').dumps({'published':True,'classifier_fits':10,'new_bound_files':len(bound),'prior_files_unchanged':645,'quality_acceptance':False},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
