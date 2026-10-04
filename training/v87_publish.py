"""Publish actual completed and aborted fits, verified trials and stop gates."""
from run_v75 import ROOT,read,save,sha

DEST=ROOT/'artifacts/v87_solver_supervision_r2_20260927'
FIRST=ROOT/'artifacts/v87_solver_supervision_20260927'
DOC='docs/V87_SOLVER_SUPERVISION_TRAINING_RESULTS.md'
STATUS='solver_supervision_training_completed_no_model_promoted'


def main():
    folder=ROOT/'evidence/2026-09-27/v87_solver_supervision';folder.mkdir(exist_ok=True)
    assert not (DEST/'execution_receipt.json').exists() and not (folder/'delivery.json').exists()
    verification=read(DEST/'verification.json');ledger=read(DEST/'problem_ledger.json');gates=read(DEST/'completion_gates.json')
    assert verification['status']=='passed' and verification['actual_classifier_fits']==4
    assert verification['source_sha256']==sha(ROOT/'training/v87r2_verify.py')
    assert ledger['source_sha256']==sha(ROOT/'training/v87r2_problem_audit.py')
    assert ledger['verification_sha256']==sha(DEST/'verification.json')
    assert gates['source_sha256']==sha(ROOT/'training/v87r2_complete_gates.py')
    assert not gates['final_full_fit_authorized'],'Continue the authorized final stage before publishing a stop'
    failure=read(FIRST/'failure_receipt.json');assert failure['actual_classifier_fits_started']==4
    original_reg=read(FIRST/'registration.json')
    assert failure['registered_source_sha256']==original_reg['source_sha256']==sha(ROOT/'training/v87_execute.py')
    for arm in ['A0','A1','A2']:assert read(FIRST/'fold1'/(arm+'_fit.json'))['actual_classifier_fits']==1
    assert not (FIRST/'fold1/A3_fit.json').exists()
    assert read(FIRST/'fold1/A3_progress.json')[-1]['epoch']==failure['last_persisted_attempt']
    rotated=(DEST/'rotation/verification.json').exists();rotation=read(DEST/'rotation/verification.json') if rotated else None
    if rotated:
        assert rotation['status']=='passed' and rotation['source_sha256']==sha(ROOT/'training/v87r2_verify_rotation.py')
    more=rotation['actual_classifier_fits'] if rotated else 0
    started=4+4+more;completed=3+4+more
    winner=gates['primary_selected']
    if winner is None:selected_info='无主折合格候选'
    else:
        state=next(z for z in read(DEST/'fold1'/(winner['arm']+'_selection_trace.json')) if z['name']==winner['name'])
        selected_info=f"主折冻结{winner['name']}，训练修复{state['selected_fit']['positive_flips']}条、内层修复{state['inner']['positive_flips']}条且零退化"
    summary=(f'v8.7本轮实际启动{started}次分类器拟合（{completed}次完成、1次中止），0次校准拟合。'
             '首次A3完整保护复核不一致被拦截；统一数值余量后重跑四臂，保护922009条训练正确，构造19603组训练内M/S配对。'
             +selected_info+'；轮换或锁定来源门槛未通过，已停止后续扩训。'
             '投影可继续更新但收益不稳定，主模型质量未通过，无新模型晋升。最近完整开发回放仍为v7.9的5947错。')
    prior={}
    for path in ['evidence/2026-09-27/v79_execution/delivery.json','artifacts/v80_attribution_20260927/review_receipt.json',
       'evidence/2026-09-27/v81_diagnosis/delivery.json','evidence/2026-09-27/v82_capacity/delivery.json',
       'evidence/2026-09-27/v83_root_review/delivery.json','evidence/2026-09-27/v84_preservation/delivery.json',
       'evidence/2026-09-27/v85_protection/delivery.json','evidence/2026-09-27/v86_boundary_review/delivery.json']:
        prior.update(read(ROOT/path)['artifact_sha256'])
    changed=[p for p,h in prior.items() if sha(ROOT/p)!=h];assert not changed,changed
    files=[p for d in [FIRST,DEST] for p in d.rglob('*') if p.is_file()]+[ROOT/DOC]+list((ROOT/'training').glob('v87*.py'))+[ROOT/'training/test_v87_solver.py']
    bound={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)}
    save(DEST/'execution_receipt.json',{'status':'execution_frozen','actual_classifier_fits_started':started,
         'actual_classifier_fits_completed':completed,'actual_classifier_fits_aborted':1,'actual_calibration_fits':0,
         'quality_acceptance':False,'prior_bound_files_rehashed':len(prior),'prior_bound_files_changed':changed,'artifact_sha256':bound})
    bound[(DEST/'execution_receipt.json').relative_to(ROOT).as_posix()]=sha(DEST/'execution_receipt.json')
    delivery={'status':STATUS,'quality_acceptance':False,'all_issues_solved':False,'model_promoted':False,
       'actual_new_fits':started,'actual_classifier_fits':started,'actual_classifier_fits_started':started,
       'actual_classifier_fits_completed':completed,'actual_classifier_fits_aborted':1,'actual_calibration_fits':0,
       'new_teacher_fits':rotation['new_teacher_fits'] if rotated else 0,
       'new_full_data_final_fit':False,'new_full_development_replay':False,'new_full_task_pipeline_replay':False,
       'new_full_input_diagnostic_replays':verification['all_input_final_models_replayed']+(rotation['all_input_final_models_replayed'] if rotated else 0),
       'platform_used':False,'external_training_data_used':False,'pseudo_labels_used':False,'development_not_blind':True,
       'latest_full_development_delivery':'evidence/2026-09-27/v79_execution/delivery.json','latest_full_development_errors':5947,
       'validation_scope':'Pre-registered four-arm trial; first attempt retained with one aborted fit; uniform numerical repair re-ran every arm. R2 saved decisions/full-P/true rows independently replayed. Conditional fixed-epoch rotation independently replayed where executed. No final promotion or new blind test; full task pipeline remains v79.',
       'first_attempt_failure':failure,'numerical_probe':read(FIRST/'numerical_variability_probe.json'),
       'implementation_verification':verification,'rotation_verification':rotation,'final_gates':gates,
       'final_primary_diagnostics':read(DEST/'fold1/diagnosis.json'),'problem_ledger':ledger,'current_plan':DOC,
       'prior_bound_files_rehashed':len(prior),'prior_bound_files_changed':changed,'artifact_sha256':bound}
    save(folder/'delivery.json',delivery)
    for relative,link in [('README.md',DOC),('docs/TRAINING_PLAN.md','V87_SOLVER_SUPERVISION_TRAINING_RESULTS.md')]:
        path=ROOT/relative;old=path.read_text(encoding='utf-8');assert 'v8.7本轮实际启动' not in old
        title,body=old.split('\n',1)
        path.write_text(title+'\n\n当前训练结果（2026-09-27）：**'+summary+'** [四臂、轮换及问题记录]('+link+')。以下为历史阶段。\n'+body,encoding='utf-8')
    path=ROOT/'mcp_readonly/catalog.json';catalog=read(path)
    for e in catalog['documents']:
        if e['path'] in ['README.md','docs/TRAINING_PLAN.md']:e.update(sha256=sha(ROOT/e['path']),summary=summary)
        else:assert sha(ROOT/e['path'])==e['sha256'],e['path']
        if e['id']=='v86-direction-review':e['category']='historical_review'
        if e['id']=='v86-delivery':e['category']='historical_evidence'
    catalog['project'].update(as_of='2026-09-27',current_summary=summary,
       authoritative_direction_id='v87-direction-review',authoritative_delivery_id='v87-delivery',
       current_direction=['已执行四臂、统一数值修复和有条件固定轮换；按真实逐类/逐条门槛停止，无新正式模型。',
          '停止当前未复现收益的配对配置；下一轮优先直接监督困难M/S的判定间隔，保留训练旧正确保护；新增修复保护只作独立对照。',
          '完整任务效果仍须独立工程回放；训练P、内层、来源轮换、锁定回归和最终全量分别验收。'],
       known_limits=['首次A3第49次复核失败未保存提议；已保留中止记录并统一重跑，数值探针不是唯一成因证明。',
          '训练正确P保护不保证留出正确；静态P也不自动保护后来修复的E，新增收益可能再丢失。',
          '只有ASA闭合拒绝模板构造配对，未配对/混标仍在主损失；VPC-M和未见格式监督仍缺失。',
          '开发数据反复观察，无新盲测、平台训练或正式提交；最近完整任务回放仍v7.9，质量未通过。'])
    entries=[{'id':'v87-direction-review','title':'v8.7 四臂真实训练、数值恢复及固定来源轮换结果','path':DOC,'category':'current_review','summary':summary,
         'keywords':['当前','最新','方向','下一步','训练','恶意','可疑','退化','保护','投影','监督','泛化','v8.7']},
        {'id':'v87-delivery','title':'v8.7 真实拟合与独立重放、停止门槛证据','path':(folder/'delivery.json').relative_to(ROOT).as_posix(),'category':'current_evidence',
         'summary':f'{started}次启动、{completed}次完成、1次中止；独立重放、固定选模、失败不晋升。','keywords':['执行','证据','审查','恶意','可疑','v8.7']}]
    for e in reversed(entries):e['sha256']=sha(ROOT/e['path']);catalog['documents'].insert(0,e)
    save(path,catalog)
    path=ROOT/'mcp_readonly/tests/test_readonly_mcp.py';s=path.read_text(encoding='utf-8')
    s=s.replace('v86-direction-review','v87-direction-review').replace('v86-delivery','v87-delivery')
    s=s.replace('malicious_regression_review_completed_no_new_fit',STATUS).replace('本轮新增0次分类器拟合',f'本轮实际启动{started}次分类器拟合')
    path.write_text(s,encoding='utf-8')
    print(__import__('json').dumps({'published':True,'started_classifier_fits':started,'completed_classifier_fits':completed,'aborted_classifier_fits':1,'old_files_verified':len(prior),'new_bound_files':len(bound),'quality_acceptance':False},ensure_ascii=False))


if __name__=='__main__':main()
