"""Publish actual protection training, failed acceptance and its verified issue ledger."""
import json
from run_v75 import ROOT,read,save,sha
from v85_protection import DEST

DOC='docs/V85_PROTECTION_TRAINING_RESULTS.md'
STATUS='protection_training_completed_no_candidate_accepted'
SUMMARY='v8.5本轮新增4次分类器拟合（1个折内教师、3个保护修正分支），0次校准拟合。C保住65681条训练内正确记录、修复2条S，但内层修复1条S并改坏4条M；A/B也有负翻转，15个检查点无合格候选，已按登记停止扩折及最终全量拟合。定位到4条ASA-M保护记录阻挡原下降方向；局部切向投影仅为无拟合诊断，尚无分类迁移收益。主模型质量未通过，无新模型晋升。最近完整开发回放仍为v7.9的5947错。'


def main():
    folder=ROOT/'evidence/2026-09-27/v85_protection';folder.mkdir(exist_ok=True)
    if (DEST/'execution_receipt.json').exists() or (folder/'delivery.json').exists():raise FileExistsError('Already published')
    v=read(DEST/'verification.json');p=read(DEST/'problem_verification.json');d=read(DEST/'problem_ledger.json');reg=read(DEST/'registration.json')
    assert v['status']==p['status']=='passed' and v['actual_classifier_fits']==4
    for name,info in [('v85_verify.py',v),('v85_verify_problem_audit.py',p),('v85_failure_audit.py',d)]:assert info['source_sha256']==sha(ROOT/'training'/name)
    assert v['registered_source_sha256']==reg['source_sha256']==sha(ROOT/'training/v85_protection.py')
    assert p['problem_ledger_sha256']==sha(DEST/'problem_ledger.json') and p['casebook_sha256']==sha(DEST/'problem_casebook.parquet')
    cont=read(DEST/'continuation.json');assert cont['primary_selected'] is None and not cont['secondary_authorized']
    assert not cont['contrastive_training_authorized'] and not cont['final_full_fit_authorized']
    fits=[]
    for file in sorted(DEST.glob('fold*/*fit.json')):
        info=read(file);assert info['actual_classifier_fits']==1
        fits.append({'file':file.relative_to(ROOT).as_posix(),'sha256':sha(file),'actual_classifier_fits':1,
            'teacher':file.name=='teacher_fit.json','epochs':info.get('epochs_executed'),'selected':info.get('selected')})
    assert len(fits)==4
    prior={}
    for file in ['evidence/2026-09-27/v79_execution/delivery.json','artifacts/v80_attribution_20260927/review_receipt.json',
       'evidence/2026-09-27/v81_diagnosis/delivery.json','evidence/2026-09-27/v82_capacity/delivery.json',
       'evidence/2026-09-27/v83_root_review/delivery.json','evidence/2026-09-27/v84_preservation/delivery.json']:
        prior.update(read(ROOT/file)['artifact_sha256'])
    changed=[k for k,digest in prior.items() if sha(ROOT/k)!=digest];assert not changed,changed
    files=[p for p in DEST.rglob('*') if p.is_file()]+[ROOT/DOC]+list((ROOT/'training').glob('v85_*.py'))+[ROOT/'training/test_v85_protection.py']
    bound={file.relative_to(ROOT).as_posix():sha(file) for file in sorted(files)}
    save(DEST/'execution_receipt.json',{'status':'execution_frozen','quality_acceptance':False,
        'actual_classifier_fits':4,'actual_calibration_fits':0,'fits':fits,'artifact_sha256':bound,
        'prior_bound_files_rehashed':len(prior),'prior_bound_files_changed':changed})
    bound[(DEST/'execution_receipt.json').relative_to(ROOT).as_posix()]=sha(DEST/'execution_receipt.json')
    delivery={'status':STATUS,'quality_acceptance':False,'all_issues_solved':False,'model_promoted':False,
       'actual_new_fits':4,'actual_classifier_fits':4,'actual_calibration_fits':0,'fits':fits,
       'registered_arms':['A_frozen_teacher','B_selective_KL','C_final_margin_with_feasible_backtracking'],
       'actual_residual_update_attempts':180,'saved_training_checkpoints':15,'includes_zero_initial_snapshots':3,
       'new_full_data_final_fit':False,'new_full_development_replay':False,'new_full_input_diagnostic_replays':3,
       'new_full_task_pipeline_replay':False,'platform_used':False,'external_training_data_used':False,'pseudo_labels_used':False,'target_answers_read':False,
       'latest_full_development_delivery':'evidence/2026-09-27/v79_execution/delivery.json','latest_full_development_errors':5947,
       'validation_scope':'Registered fold1 three-arm teacher+residual training; all saved fit/inner decisions and three fixed final full-input diagnostics independently replayed. All 298 old bound files unchanged. No stage passed zero-negative-flip with real repair; rotation and later training stopped. Local tangent projection is no-fit diagnostic, not a new trained candidate or unknown-input guarantee.',
       'implementation_verification':{k:value for k,value in v.items() if k!='fold_checks'},
       'problem_verification':p,'stop_decision':cont,'final_diagnostics':read(DEST/'fold1/diagnosis.json'),
       'problem_ledger':d,'current_plan':DOC,'prior_bound_files_rehashed':len(prior),'prior_bound_files_changed':changed,'artifact_sha256':bound}
    save(folder/'delivery.json',delivery)
    for relative,link in [('README.md',DOC),('docs/TRAINING_PLAN.md','V85_PROTECTION_TRAINING_RESULTS.md')]:
        path=ROOT/relative;old=path.read_text(encoding='utf-8');title,body=old.split('\n',1)
        assert 'v8.5本轮新增4次' not in old
        path.write_text(title+'\n\n当前保护训练结果（2026-09-27）：**'+SUMMARY+'** [三臂结果与问题记录]('+link+')。以下为历史阶段。\n'+body,encoding='utf-8')
    path=ROOT/'mcp_readonly/catalog.json';catalog=read(path)
    for e in catalog['documents']:
        if e['path'] in ['README.md','docs/TRAINING_PLAN.md']:e.update(sha256=sha(ROOT/e['path']),summary=SUMMARY)
        else:assert sha(ROOT/e['path'])==e['sha256'],e['path']
        if e['id']=='v84-direction-review':e['category']='historical_review'
        if e['id']=='v84-delivery':e['category']='historical_evidence'
    catalog['project'].update(current_summary=SUMMARY,as_of='2026-09-27',authoritative_delivery_id='v85-delivery',authoritative_direction_id='v85-direction-review',
       current_direction=['已执行预登记三臂，训练P保护通过但内层和C仍退化；没有候选通过，保持原基线并停止扩折及后续拟合。',
          '下一项机制实验先补约束搜索方向：切向/主动约束投影必须全P核验，不以平均保护损失或局部诊断当分类收益。',
          '跨组件监督、撤类支持对照及正常扩训按保护与迁移门槛后移；全部格式逐类计数，真实修错且零退化才继续。'],
       known_limits=['C保住65681条训练内正确、仅修复2条S，内层修复1条S并改坏4条M；训练保护不等于跨来源保护。',
          'C后段28次更新被拒绝；局部投影有可行下降方向但未训练、未证明分类或迁移收益。',
          '混标下界22仅是本折编码的有限查表诊断；VPC缺M、未见格式和跨组件行为监督仍未解决。',
          '本轮4次分类器拟合、0次校准；无新全量最终拟合，无新晋升，最近完整开发回放仍v7.9。'])
    entries=[{'id':'v85-direction-review','title':'v8.5 保护三臂真实训练结果与问题记录','path':DOC,'category':'current_review','summary':SUMMARY,
       'keywords':['当前','最新','方向','下一步','训练','冻结','保护','正确','错误','恶意','可疑','监督','泛化','根因','v8.5']},
      {'id':'v85-delivery','title':'v8.5 保护训练真实执行与失败门槛证据','path':(folder/'delivery.json').relative_to(ROOT).as_posix(),
       'category':'current_evidence','summary':'4次拟合、15检查点、3最终全输入诊断独立重放；训练保护可行但内层退化，所有候选停止，无晋升。',
       'keywords':['执行','证据','审查','保护','冻结','v8.5']}]
    for e in reversed(entries):e['sha256']=sha(ROOT/e['path']);catalog['documents'].insert(0,e)
    save(path,catalog)
    path=ROOT/'mcp_readonly/tests/test_readonly_mcp.py';s=path.read_text(encoding='utf-8')
    s=s.replace('v84-direction-review','v85-direction-review').replace('v84-delivery','v85-delivery').replace('preservation_review_completed_no_new_fit',STATUS)
    s=s.replace('本轮新增0次分类器拟合','本轮新增4次分类器拟合');path.write_text(s,encoding='utf-8')
    print(json.dumps({'published':True,'actual_classifier_fits':4,'quality_acceptance':False,'new_bound_files':len(bound),'old_files_verified':len(prior)},ensure_ascii=False))


if __name__=='__main__':main()
