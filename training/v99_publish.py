"""Publish researched training plan and no-fit feasibility evidence."""
import json
from v89_common import ROOT,read,save,sha
from v99_normalization_feasibility import DEST

DOC='docs/V99_EVIDENCE_SEPARATION_RESEARCH_AND_PLAN.md'
STATUS='v99_research_and_no_fit_feasibility_completed'


def main():
    folder=ROOT/'evidence/2026-09-28/v99_research_plan';assert not (folder/'delivery.json').exists()
    verify=read(DEST/'verification.json');assert verify['status']=='passed'
    assert verify['source_sha256']==sha(ROOT/'training/v99_verify.py')
    assert verify['prior_bound_files_rehashed']==1131 and not verify['prior_bound_files_changed']
    audit=read(DEST/'normalization_feasibility.json');gradient=read(DEST/'objective_scale_audit.json')
    assert all(a['merged_mixed_label_groups']==0 for a in audit['modes'].values())
    catalog_path=ROOT/'mcp_readonly/catalog.json';catalog=read(catalog_path)
    assert catalog['project']['authoritative_direction_id']=='v98-direction-review'
    for d in catalog['documents']:assert sha(ROOT/d['path'])==d['sha256']
    sources=[
        {'name':'Cisco ASA ACL syslog explanation','url':'https://www.cisco.com/c/en/us/support/docs/security/adaptive-security-appliance-asa-software/217679-asa-access-control-list-configuration-ex.html','use':'Deny event reports ACL denial; no M/S adjudication supplied.'},
        {'name':'IANA ICMP parameters','url':'https://www.iana.org/assignments/icmp-parameters','use':'Type3/code13 protocol semantics, not security label.'},
        {'name':'Dos and Donts of ML in Computer Security','url':'https://www.usenix.org/conference/usenixsecurity22/presentation/arp','use':'Spurious correlations, baseline fairness, performance measures.'},
        {'name':'AugMix author implementation','url':'https://github.com/google-research/augmix/blob/master/cifar.py','use':'Inspected clean CE plus JSD predictions; borrow valid invariance idea only, not image transforms or coefficient12.'},
        {'name':'ERM++ paper','url':'https://arxiv.org/html/2304.01973v4','use':'Source validation training-length selection and full source retraining, not training loss as generalization.'},
        {'name':'ERM++ author code','url':'https://github.com/piotr-teterwak/erm_plusplus','use':'Implementation reference; no dependency installed or weights downloaded.'},
        {'name':'NeuralLog author code','url':'https://github.com/LogIntelligence/NeuralLog','use':'Parser-loss research; reject number/special-character removal for this SOC task.'},
        {'name':'TESSERACT','url':'https://www.usenix.org/conference/usenixsecurity19/presentation/pendlebury','use':'Spatial/temporal bias; do not assume sanitized chronology authentic.'},
        {'name':'DomainBed','url':'https://github.com/facebookresearch/DomainBed/blob/main/README.md','use':'Selection conditions must be part of fair method comparison.'}]
    save(DEST/'research_sources.json',{'retrieved_on':'2026-09-28','sources':sources,'external_training_data_used':False,
        'scope':'Primary papers, vendor registry/docs and author projects. Transfer to SOC is a proposed adaptation, not demonstrated benefit.'})
    contract={'status':'revised_plan_not_trained','source_report':DOC,'official_data_only':True,
        'supervision':'Official labels unchanged; distinct from observable facts and independent security adjudication.',
        'raw_data':'Keep original sanitized text and span provenance. Do not invent pre-redaction values.',
        'candidate_normalization':'v99_normalization_feasibility.normalize placeholder_cluster on audited ASA path only',
        'pending_before_fit':['original-span production mapping','raw-to-inference equivalence tests','fold/group/exposure receipt'],
        'views':['current_R0','N1_placeholder_cluster'],'magnitude_coefficients':[0.,.01],
        'source_population':'original A+B; V/inner/C/H excluded from training, step selection and method selection',
        'folds':3,'fold_seed':9901,'grouping':'union existing components with identical input groups from either candidate representation; no labels for grouping',
        'teacher_exposure':'one teacher per input view per fold, fit on training components only',
        'network':'64/16 residual','branch_seed':9701,'optimizer':'Adam','lr':.003,'soft_protection':.01,
        'row_weights':'original frequency; no duplicate or class reweighting',
        'training_trajectories':12,'teacher_fits_max':6,'total_source_stage_fits_max':18,
        'check_every_steps':25,'budget_endpoints':[200,'training_plateau_or_max2000'],
        'plateau':{'consecutive_checks':5,'relative_objective_change_lt':1e-4,'errors_unchanged':True},
        'classification_learned_is_not_total_gradient_small':True,'absolute_gradient_is_reported_with_denominator':True,
        'selection':'Eight predeclared family/budget states. ASA OOF M/S equal-weight F1 with per-class precision/recall/F1 and negative-flip protection. Tie: shorter budget. OOF is selection, not unbiased final test.',
        'background_scope':'Frozen nonASA outputs for change attribution only, not new OOF generalization evidence.',
        'refit_before_V':'Selected normalization family and matched raw reference, each teacher+branch on full A+B. At most4 fits; freeze budget from source folds then V once.',
        'final_update':'Compare ABV candidates against same ABV exposure; historic unequal-exposure model remains diagnostic only. Retain final old-correct safety.',
        'promotion':False,'quality_acceptance':False,'blindness_claim':False,
        'no_new_invariance_loss_until_needed':True,'no_external_security_ground_truth_available':True}
    save(DEST/'next_training_contract.json',contract)
    problems=[
        {'id':'artifact_dependency','status':'normalization_feasible_not_trained','finding':'N0/N1 add no empirical label-conflict floor; frozen deletion is not evidence of retraining gain.'},
        {'id':'training_sufficiency','status':'separate_fit_from_transfer','finding':'v92 achieved fit floor with harmful transfer. v97 terminal CE and magnitude gradients strongly oppose; no one-factor explanation certified.'},
        {'id':'loss_units','status':'accounted_in_plan','finding':'Full/ASA denominator ratio24.0886; report gradients in defined units and preserve all coefficients under equivalent scaling.'},
        {'id':'evidence_vs_labels','status':'separated','finding':'Official labels remain task targets; vendor/protocol facts do not alone establish M/S. Independent incident evidence unavailable.'},
        {'id':'fair_evaluation','status':'plan_revised_not_executed','finding':'Same folds/exposure, source-only selection, OOF selection disclosed, all inspected data remain development.'}]
    save(DEST/'problem_register.json',problems)
    execution={'status':STATUS,'actual_classifier_fits':0,'actual_calibration_fits':0,'parameter_updates':0,
        'selected_arm':None,'model_promoted':False,'quality_acceptance':False,'all_issues_solved':False,
        'old_model_retained':True,'new_full_development_replay':False,'platform_used':False,
        'external_training_data_used':False,'original_labels_changed':0,'target_answers_read':False}
    save(DEST/'execution_summary.json',execution)
    summary=('v9.9调研与方案修订完成：本轮新增0次分类器拟合、0次校准拟合。'
        '全112807条ASA的两种窄范围规范化均未新增输入标签冲突，A+B下界仍22、全ASA仍26；尚无重训收益。'
        '冻结MAG分类与幅度正则梯度余弦约-0.995；训练拟合和迁移分别验收。'
        '下一轮用原视图/N1规范化×ERM/MAG的四臂三折，来源内部选预算，再固定规则复训；事实、标签、模型行为分开记录。'
        '1131历史绑定文件未变。最新实际训练仍v9.7，质量未通过，无新模型晋升；最近完整开发回放仍为v7.9的5947错。')
    files=[p for p in DEST.rglob('*') if p.is_file()]+list((ROOT/'training').glob('v99*.py'))+[ROOT/DOC]
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)}
    delivery={**execution,'validation_scope':verify['scope'],'verification':verify,'current_plan':DOC,
        'latest_executed_training_delivery':'evidence/2026-09-28/v97_matched_training/delivery.json',
        'latest_full_development_delivery':'evidence/2026-09-27/v79_execution/delivery.json','latest_full_development_errors':5947,
        'problem_register':(DEST/'problem_register.json').relative_to(ROOT).as_posix(),'artifact_sha256':bindings}
    folder.mkdir(parents=True,exist_ok=True);save(folder/'delivery.json',delivery)
    for relative,link in [('README.md',DOC),('docs/TRAINING_PLAN.md','V99_EVIDENCE_SEPARATION_RESEARCH_AND_PLAN.md')]:
        p=ROOT/relative;s=p.read_text(encoding='utf-8');assert 'v9.9调研与方案修订完成' not in s
        head,body=s.split('\n',1);p.write_text(head+'\n\n当前调研与训练方案（2026-09-28）：**'+summary+'** [事实、监督与公平训练方案]('+link+')。以下保留历史阶段。\n'+body,encoding='utf-8')
    for d in catalog['documents']:
        if d['path'] in ['README.md','docs/TRAINING_PLAN.md']:d.update(sha256=sha(ROOT/d['path']),summary=summary)
        if d['id']=='v98-direction-review':d['category']='historical_review'
        if d['id']=='v98-delivery':d['category']='historical_evidence'
    catalog['project'].update(as_of='2026-09-28',current_summary=summary,
        authoritative_direction_id='v99-direction-review',authoritative_delivery_id='v99-delivery',
        current_direction=[
            '优先检验N1占位符簇规范化；原文和真实事实保留，现有全部ASA未增加精确输入冲突，但没有重训成绩。',
            '四臂为原视图/规范化×ERM/MAG，在A+B来源内同组件三折拟合；同轨迹记录短/长终点，源侧OOF选模。',
            '原始可观察事实、官方标签监督、模型行为及独立安全证据分开，脱敏前信息与事件真值不得臆造。',
            '相同暴露与预算比较、逐类和组件验收；当前开发资料已看过，最终保护要求保持，无新模型晋升。'],
        known_limits=[
            'v9.9零拟合；规范化无新增冲突不等于语义无损或提高分类。',
            '梯度反向是单终点诊断，不能据此删除正则或保证延长训练有效。',
            '685条ICMP S零同类训练支持和真实M/S语义仍未解决；官方标签一致不等于独立安全事实确认。',
            '来源OOF为开发选择，不能宣称盲测；完整任务和外部迁移没有新验证。'])
    entries=[{'id':'v99-direction-review','title':'v9.9 事实监督分离与训练调研方案','path':DOC,'category':'current_review',
        'summary':summary,'keywords':['当前','最新','方向','下一步','训练','证据','脱敏','公平','v9.9']},
        {'id':'v99-delivery','title':'v9.9 规范化与梯度零拟合审查证据','path':(folder/'delivery.json').relative_to(ROOT).as_posix(),
        'category':'current_evidence','summary':'无新拟合；两种规范化未新增标签冲突，1131历史文件未变；下一轮计划尚未执行。',
        'keywords':['证据','核验','零拟合','v9.9']}]
    for d in reversed(entries):d['sha256']=sha(ROOT/d['path']);catalog['documents'].insert(0,d)
    save(catalog_path,catalog)
    test=ROOT/'mcp_readonly/tests/test_readonly_mcp.py';s=test.read_text(encoding='utf-8')
    s=s.replace('v98-direction-review','v99-direction-review').replace('v98-delivery','v99-delivery').replace('v98_frozen_root_audit_completed_no_new_fit',STATUS)
    test.write_text(s,encoding='utf-8')
    for p,h in bindings.items():assert sha(ROOT/p)==h
    print(json.dumps({'published':True,'bound_files':len(bindings),'catalog_documents':len(catalog['documents']),
        'quality_acceptance':False,'new_classifier_fits':0,'delivery_sha256':sha(folder/'delivery.json')},ensure_ascii=False))


if __name__=='__main__':main()
