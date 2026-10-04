"""Verify the zero-fit preflight and publish narrowly supported plan revisions."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from v102_root_review import ROOT, OLD, sha
from v103_plan_preflight import DEST


def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    target=ROOT/'evidence/2026-09-28/v103_plan_review/delivery.json'
    assert not target.exists()
    evidence=json.loads((DEST/'preflight.json').read_text())
    oldpath=ROOT/'evidence/2026-09-28/v102_root_review/delivery.json'
    old=json.loads(oldpath.read_text())
    checks={'v102_bound_files_unchanged':all(sha(ROOT/p)==h for p,h in old['artifact_sha256'].items()),
        'preflight_inputs_unchanged':all(sha(ROOT/p)==h for p,h in evidence['input_sha256'].items()),
        'preflight_source_unchanged':sha(ROOT/'training/v103_plan_preflight.py')==evidence['source_sha256']}
    f=pd.read_parquet(DEST/'v102_proposed_full_population_folds.parquet')
    b=pd.read_parquet(DEST/'balanced_format_class_folds.parquet')
    r=pd.read_parquet(ROOT/'artifacts/v75_four_arm_20260921_r2/rows.parquet',columns=['row_position','route','label_index'])
    checks['all_original_rows_in_both_partitions']=bool(np.array_equal(f.row_position,r.row_position) and np.array_equal(b.row_position,r.row_position))
    checks['all_global_groups_isolated']=bool(f.groupby('root').proposed_fold.nunique().max()==1 and b.groupby('root').proposed_balanced_fold.nunique().max()==1)
    rootfold={int(c):int(hashlib.sha256(f'10203:{int(c)}'.encode()).hexdigest()[:8],16)%3 for c in f.root.unique()}
    checks['proposed_hash_exact']=bool(np.array_equal(f.root.map(rootfold),f.proposed_fold))
    asa=pd.read_parquet(DEST/'ASA_support_preflight.parquet')
    train=asa[asa.behavior.notna()]
    for k in range(3):
        for label,mask in [('small',train.proposed_fold.ne(k)&train.row_position.isin(set(f.loc[f.old_source,'row_position']))),('full',train.proposed_fold.ne(k))]:
            counts=train[mask].groupby(['behavior','label_index']).root.nunique().to_dict()
            query=asa[asa.proposed_fold==k]
            expected=np.array([counts.get((t,int(c)),0) if t is not None else -1 for t,c in zip(query.behavior,query.label_index)])
            checks[f'{label}_strict_support_fold{k}']=bool(np.array_equal(expected,query[label+'_exact_support']))
    t=pd.read_parquet(DEST/'113_target_support_after_split.parquet')
    checks['113_decomposition']=bool(len(t)==113 and int(((t.small_exact_support==0)&(t.full_exact_support>0)).sum())==101 and int((t.small_exact_support>0).sum())==4 and int((t.full_exact_support==0).sum())==8)
    controls=evidence['support_key_sensitivity_old_predictions']
    checks['coarse_S_support']=any(z['class']==2 and z['old_errors_strict_zero_support']==230 and z['has_coarse_same_class_support']==230 for z in controls)
    assert all(checks.values()),checks
    save(DEST/'verification.json',{'all_checks_passed':True,'checks':checks,'v102_bound_files_verified':len(old['artifact_sha256']),
        'scope':'Split, row integrity, fold-side strict support and diagnostic decomposition; no classifier fit or quality validation.'})
    plan=copy.deepcopy(json.loads((ROOT/'artifacts/v102_root_review_20260928/next_experiment_contract.json').read_text()))
    plan['version']='v103';plan['status']='evidence_based_revision_not_trained'
    plan['supersedes']='artifacts/v102_root_review_20260928/next_experiment_contract.json'
    plan['split']['manifest']='artifacts/v103_plan_preflight_20260928/v102_proposed_full_population_folds.parquet'
    plan['split']['manifest_sha256']=sha(DEST/'v102_proposed_full_population_folds.parquet')
    plan['split']['stratified_alternative_adopted']=False
    plan['split']['stratified_alternative_reason']='Improved S row balance but worsened S-unit balance; no prediction-based reason to change. Largest S unit is 62.26% of all S.'
    plan['phase_A']['purpose']='Total effect of expanding official training population, not isolated causal effect of behavior coverage.'
    plan['phase_A']['required_reports']=['Common-hold per-class confusion and positive/negative flips','Train class/format proportions and predicted class proportions','Strict/coarse support transition slices','Same-evaluation-population score ranking diagnostics; no calibration claim']
    plan['phase_A']['historic_support_targets']={'potential':113,'clean_zero_to_supported':101,'already_supported_in_new_small_pool':4,'still_zero_in_fold_train':8}
    plan['phase_B']['conditional_on']='After six phase-A fits, record a specific remaining representation/architecture question; no automatic run-to-budget requirement.'
    plan['phase_B']['candidate_input']['ensemble_first_layer']='Create distinct per-member input scaling before first mixing; shared sparse W then per-member output scaling, following BatchEnsemble semantics.'
    plan['phase_B']['candidate_input']['shared_projection_before_ensemble']=False
    plan['phase_B']['common_candidate_config'].pop('epochs')
    plan['phase_B']['common_candidate_config']['max_epochs']=100
    plan['phase_B']['common_candidate_config']['checkpoint_selection']='25/50/100 common epochs across all folds; select one global method/epoch by declared source-OOF continuation guards and pooled row MS-F1, tie prefers shorter. Selection results are developmental.'
    plan['phase_B']['deployment_gate_comparison']='OOF composition uses same-fold trained reference B boundary; historic production output is separate safety diagnosis, not unseen OOF reference.'
    plan['selection']['worst_fold_definition']='Minimum of the three per-fold MS-F1 values; also disclose each fold per-class regressions.'
    plan['selection']['pooled_reporting']='Compute row metrics over all original OOF rows and group recall over all relevant groups; do not equally average fold summaries as population estimates.'
    plan['selection']['absent_format_class']='Report N/A for missing evaluation support, never zero-error pass.'
    plan['selection']['diagnostic_only']='Strict and coarse label-conditional support is never a predictive feature or unknown-row routing input.'
    plan['revision_basis']=['Actual source support after frozen split','Paired partition dry-run without any model scores','TabM author layer-order guidance','Historical long-training and weighting failures']
    save(DEST/'revised_training_contract.json',plan)
    sources=[('TabM author implementation order','https://github.com/yandex-research/tabm#important-implementation-details','Diversity before first feature mixing; no project accuracy guarantee'),
        ('Grouped cross validation','https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data','Group integrity and class stratification are different'),
        ('ERM++','https://arxiv.org/html/2304.01973v4','Training budget is not validation-optimal duration'),
        ('Model selection bias','https://www.jmlr.org/papers/v11/cawley10a.html','Finite selection criteria can be overfit'),
        ('Label shift assumptions','https://proceedings.mlr.press/v80/lipton18a.html','Class-prior correction requires assumptions not established for this data')]
    save(DEST/'research_sources.json',{'accessed':'2026-09-28','sources':[{'title':a,'url':b,'applicability':c} for a,b,c in sources],
        'packages_installed':0,'external_training_data_downloaded':False})
    status='v103_evidence_based_plan_review_completed_no_fit'
    scope='Zero-fit support/partition preflight, primary-source review and plan corrections. Latest trained model is v101; no new model quality or external generalization claim.'
    doc='docs/V103_EVIDENCE_BASED_PLAN_REVIEW.md'
    summary=('v10.3完成有据方案评价，0次分类器拟合、0次校准。保留覆盖优先和组评价：113条潜在S目标在固定新折中实际101条从零到有、4条已有支持、8条仍无支持。'
        '230条严格键零支持S错在粗行为层都有同类支持，收紧缺监督归因。分层试划改善行平衡但恶化组平衡，未采用。'
        '修正TabM为首次线性混合前建立成员差异；先6次覆盖对照，后续9次条件启动，100周期仅预算上限。'
        '最新训练仍v10.1，质量未通过，无新模型晋升；最近完整开发回放仍为v7.9的5947错。')
    execution={'status':status,'actual_classifier_fits_this_user_request':0,'actual_calibration_fits':0,
        'quality_acceptance':False,'model_promoted':False,'validation_scope':scope,'current_review':doc,
        'latest_executed_training_delivery':'evidence/2026-09-28/v101_full_input_training/delivery.json',
        'latest_full_development_errors':5947,'external_training_data_used':False,'platform_used':False}
    save(DEST/'execution_summary.json',execution)
    paths=list(p for p in DEST.iterdir() if p.is_file())+list((ROOT/'training').glob('v103_*.py'))+[ROOT/doc]
    save(target,{**execution,'artifact_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)},'verification':checks})
    for name,link in [('README.md',doc),('docs/TRAINING_PLAN.md','V103_EVIDENCE_BASED_PLAN_REVIEW.md')]:
        p=ROOT/name;text=p.read_text(encoding='utf-8');assert 'v10.3完成有据方案' not in text
        head,rest=text.split('\n',1)
        p.write_text(head+'\n\n当前方案评价（2026-09-28）：**'+summary+'** [有据修订与预检查]( '+link+' )。下方保留历史阶段。\n'+rest,encoding='utf-8')
    cp=ROOT/'mcp_readonly/catalog.json';catalog=json.loads(cp.read_text(encoding='utf-8'))
    for e in catalog['documents']:
        if e['path'] in ['README.md','docs/TRAINING_PLAN.md']:
            e['sha256']=sha(ROOT/e['path']);e['summary']=summary
        else:assert sha(ROOT/e['path'])==e['sha256'],e['id']
        if e['category']=='current_review':e['category']='historical_review'
        if e['category']=='current_evidence':e['category']='historical_evidence'
    catalog['project'].update(as_of='2026-09-28',current_summary=summary,authoritative_delivery_id='v103-delivery',authoritative_direction_id='v103-direction-review',
        current_direction=['先做原定固定全局分组三折的6次人口扩充对照，支持按实际训练侧查询。','阶段B条件启动；TabM先建成员差异再混合输入，检查点共用全局选择规则。','不替换为未证实更好的分层，不重开降权/删字段，不放宽正式晋升保护。'],
        known_limits=['原定一折含92.89%的ASA S；最大单组本身占62.26%，分组不等于真实独立企业。','零严格行为支持不等于零可迁移信息；没有新模型实测收益。','所有已观察开发数据不能恢复盲性；正式M/旧正确保护仍未通过。'])
    for ident,title,path,category in reversed([('v103-direction-review','v10.3 有据方案评价与最小修订',doc,'current_review'),('v103-delivery','v10.3 零训练预检查与修订凭据',target.relative_to(ROOT).as_posix(),'current_evidence')]):
        catalog['documents'].insert(0,{'id':ident,'title':title,'path':path,'category':category,'summary':summary,
            'keywords':['当前','最新','方向','有据优化','TabM','训练支持','v10.3'],'sha256':sha(ROOT/path)})
    save(cp,catalog)
    test=ROOT/'mcp_readonly/tests/test_readonly_mcp.py'
    text=test.read_text(encoding='utf-8').replace('v102-direction-review','v103-direction-review').replace('v102-delivery','v103-delivery')
    text=text.replace('v102_root_review_and_official_support_audit_completed_no_fit',status)
    text=text.replace('self.assertIn("根因审查",','self.assertIn("有据评价",')
    test.write_text(text,encoding='utf-8')
    print(json.dumps({'status':status,'verification_checks':len(checks),'bound_files':len(paths),'v102_preserved':len(old['artifact_sha256']),'delivery_sha256':sha(target)},ensure_ascii=False))


if __name__=='__main__':main()
