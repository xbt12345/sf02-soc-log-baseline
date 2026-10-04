"""Publish no-fit review with bound diagnostics and a proposed next contract."""
import json
import numpy as np
import pandas as pd
from v102_root_review import ROOT, DEST, sha


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    target=ROOT/'evidence/2026-09-28/v102_root_review/delivery.json'
    assert not target.exists()
    oldpath=ROOT/'evidence/2026-09-28/v101_full_input_training/delivery.json'
    previous=json.loads(oldpath.read_text())
    oldchecks={p:sha(ROOT/p)==h for p,h in previous['artifact_sha256'].items()}
    assert all(oldchecks.values())
    verify=json.loads((DEST/'verification.json').read_text())
    assert verify['all_checks_passed']
    r=pd.read_parquet(DEST/'source_error_and_support_ledger.parquet')
    complete=r[r.behavior.notna()]
    total=complete.groupby(['behavior','label_index']).merged_root.nunique()
    held=complete.groupby(['behavior','label_index','fold']).merged_root.nunique()
    for row in r.itertuples():
        if row.behavior is None:
            assert row.train_same_class_roots==row.train_other_class_roots==-1
            continue
        for label,expected in [(row.label_index,row.train_same_class_roots),(3-row.label_index,row.train_other_class_roots)]:
            actual=total.get((row.behavior,label),0)-held.get((row.behavior,label,row.fold),0)
            assert actual==expected
    save(DEST/'final_audit_checks.json',{'all_checks_passed':True,
        'v101_bound_files_unchanged':len(oldchecks),'source_support_recomputed_rows':len(r),
        'support_method':'Total distinct merged units minus units in held fold; compared to per-fold fit-side count.',
        'initial_independent_graph_metrics_verification':verify['all_checks_passed'],
        'new_fits':0,'v101_delivery_sha256':sha(oldpath)})
    contract={'status':'proposed_plan_not_trained','version':'v102',
        'official_data_only':True,'official_training_rows':2056871,'official_ASA_rows':112807,
        'class_labels_unchanged':True,'duplicate_original_class_counts_preserved':True,
        'split':{'unit':'Full-official connected closure of historical components, R0 identity and N1 identity across all formats',
            'folds':3,'seed':10203,'rule':'sha256(f"10203:{minimum_original_component_id}") first 8 hex digits modulo 3',
            'reroll_after_metrics':False,'same_split_for_all_arms':True,
            'fresh_blind_evaluation_claim':False,'scalers_and_vocabulary_fit_on_train_only':True},
        'first_wave_max_fits':15,
        'phase_A':{'fits':6,'arms':['T-small R0 teacher: old A+B intersect new outer train','T-full R0 teacher: all official outer train'],
            'purpose':'Matched held-group comparison isolating supervised population coverage'},
        'phase_B':{'fits':9,'arms':['T-full-N1 full-format teacher','C-MLP N1 independent ASA classifier','C-TabM N1 independent ASA classifier'],
            'teacher_anchor':False,'MAG':False,'ASA_original_row_CE':True,
            'candidate_input':{'all_N1_coordinates':66287,'sparse_projection_dim':128,'facts_direct_path':True,'column_deletion':False,'hashing_or_SVD':False},
            'common_candidate_config':{'blocks':2,'width':128,'epochs':100,'batch_size':256,'lr':.002,'weight_decay':.0003,'optimizer':'AdamW','seed':10201,'checkpoints_epochs':[25,50,100]},
            'TabM_k':16,'TabM_training':'Mean of per-submodel CE losses','TabM_inference':'Mean class probabilities',
            'dependency_and_code_pin_required_before_fit':True,
            'new_custom_sparse_input_adapter_is_not_paper_reproduction':True},
        'selection':{'primary':'ASA M/S original-row equal F1; not claimed official score',
            'research_continue':['M and S row recall and F1 nondecreasing vs matched teacher','row MS-F1 improved','S root macro recall improved','worst fold nondecreasing','at least two folds net improved','specified legal wrapper invariance'],
            'record_all_positive_and_negative_flips':True,'research_continuation_is_not_model_promotion':True,
            'uncertainty_unit':'global connected unit; paired fixed-model exploratory bootstrap',
            'two_additional_seeds_for_candidate_and_matched_reference_before_scaling':'Separate conditional budget, not part of initial 15 fits'},
        'promotion':'Existing strict old-correct/per-class/full-task gates remain; no model-specific per-row oracle routing',
        'B_and_nonASA_boundary_preserved':True,'ASA_benign_generalization_not_validated':True,
        'stop':['Do not extend same residual/MAG family','No reweighting without new controls','TabM train-only gains do not warrant further scaling','No pseudo-labels or invented M/S labels'],
        'future_conditional_alternative':'Order-preserving bytes+facts only with audited supported counterexamples, no external pretrained weights in current plan'}
    save(DEST/'next_experiment_contract.json',contract)
    sources=[
        ('TabM paper','https://arxiv.org/abs/2410.24210','Candidate family, not SOC evidence'),
        ('TabM author code and training contract','https://github.com/yandex-research/tabm','Mean per-submodel loss, probability averaging, custom input interface'),
        ('Positive Congruent Training','https://openaccess.thecvf.com/content/CVPR2021/papers/Yan_Positive-Congruent_Training_Towards_Regression-Free_Model_Updates_CVPR_2021_paper.pdf','Error and negative-flip criteria are different'),
        ('Regression constraint author code','https://github.com/amazon-science/regression-constraint-model-upgrade','Compatibility approaches; no unknown-example guarantee'),
        ('ByT5','https://aclanthology.org/2022.tacl-1.17/','Ordered bytes; conditional alternative, external pretrained weights not used'),
        ('AutoLabel','https://www.usenix.org/conference/usenixsecurity25/presentation/peng-yihao','Attack ground truth uses execution auxiliary/provenance evidence unavailable here'),
        ('Security ML pitfalls','https://www.usenix.org/conference/usenixsecurity22/presentation/arp','Spurious correlations, biased selection and baseline failures'),
        ('Grouped validation','https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data','Keep groups disjoint; grouping is not class balancing')]
    save(DEST/'research_sources.json',{'accessed_date':'2026-09-28','sources':[{'title':a,'url':b,'use_and_limit':c} for a,b,c in sources],
        'external_training_data_downloads':0,'packages_installed':0,'paper_benchmarks_are_not_project_results':True})
    doc='docs/V102_ROOT_CAUSE_REVIEW_AND_RESET_PLAN.md'
    status='v102_root_review_and_official_support_audit_completed_no_fit'
    summary=('v10.2完成根因与官方覆盖审查，0次分类器拟合、0次校准。'
        'v10.1最佳候选S原行召回94.69%，实际86组的平均召回仅22.35%，60组全错；'
        '最大S组占65.21%。564错中454条零同类行为训练支持，113条S错在来源外官方组有可用同类支持。'
        '当前训练来源仅覆盖官方ASA的27.74%。修正bootstrap到实际连通组，收益区间仍跨0。'
        '下一轮先做官方覆盖匹配对照，再做独立MLP/TabM有限对照；不继续MAG步数搜索。'
        '最新实际训练仍v10.1，质量未通过，无新模型晋升；最近完整开发回放仍为v7.9的5947错。')
    execution={'status':status,'actual_classifier_fits_this_user_request':0,'actual_calibration_fits':0,
        'quality_acceptance':False,'model_promoted':False,'platform_used':False,'external_training_data_used':False,
        'latest_executed_training_delivery':str(oldpath.relative_to(ROOT)).replace('\\','/'),
        'latest_full_development_errors':5947,'current_review':doc,
        'scope':'Saved OOF read-only diagnosis, all-official descriptive support audit, primary-source research and next plan. No retraining or new model quality claim.'}
    execution['validation_scope']=execution['scope']
    save(DEST/'execution_summary.json',execution)
    paths=[p for p in DEST.iterdir() if p.is_file()]+list((ROOT/'training').glob('v102_*.py'))+[ROOT/doc]
    save(target,{**execution,'artifact_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)},
                 'v101_bound_files_unchanged':len(oldchecks),'verification':verify})
    for name,link in [('README.md',doc),('docs/TRAINING_PLAN.md','V102_ROOT_CAUSE_REVIEW_AND_RESET_PLAN.md')]:
        p=ROOT/name;content=p.read_text(encoding='utf-8');assert 'v10.2完成根因' not in content
        head,rest=content.split('\n',1)
        p.write_text(head+'\n\n当前审查与下一轮方案（2026-09-28）：**'+summary+'** [根因证据与修订方案]('+link+')。下方为历史阶段。\n'+rest,encoding='utf-8')
    catalog=ROOT/'mcp_readonly/catalog.json';data=json.loads(catalog.read_text(encoding='utf-8'))
    for e in data['documents']:
        if e['path'] in ('README.md','docs/TRAINING_PLAN.md'):
            e['sha256']=sha(ROOT/e['path']);e['summary']=summary
        else:assert sha(ROOT/e['path'])==e['sha256'],e['id']
        if e['category']=='current_review':e['category']='historical_review'
        if e['category']=='current_evidence':e['category']='historical_evidence'
    data['project'].update(as_of='2026-09-28',current_summary=summary,
        authoritative_direction_id='v102-direction-review',authoritative_delivery_id='v102-delivery',
        current_direction=['使用全官方连通组重建共同开发交叉验证；先区分有效训练覆盖收益与模型方法收益。',
            '停止原残差MAG步数搜索；独立N1-MLP与适配TabM作有限比较，未实施。',
            '原行分类与组覆盖共同报告；研究继续不等于正式晋升，M和旧正确保护不放宽。'],
        known_limits=['当前候选86个S组有60组全错；109条S错和224条M错在全官方中无第二同类行为组。',
            '全官方支持审查包含旧开发标签；新交叉验证也不恢复盲性。',
            'TabM未在本项目训练；无新模型质量通过、全任务或官方隐藏成绩。'])
    for ident,title,path,category in reversed([
        ('v102-direction-review','v10.2 根因审查与覆盖优先训练方案',doc,'current_review'),
        ('v102-delivery','v10.2 零训练诊断与核验凭据',target.relative_to(ROOT).as_posix(),'current_evidence')]):
        data['documents'].insert(0,{'id':ident,'title':title,'path':path,'category':category,'summary':summary,
            'keywords':['当前','最新','方向','ASA','覆盖','第一性原理','v10.2'],'sha256':sha(ROOT/path)})
    save(catalog,data)
    test=ROOT/'mcp_readonly/tests/test_readonly_mcp.py'
    text=test.read_text(encoding='utf-8').replace('v101-direction-review','v102-direction-review').replace('v101-delivery','v102-delivery')
    text=text.replace('v101_full_input_source_training_completed_no_candidate_qualified',status)
    text=text.replace('self.assertIn("第一性原理审查",','self.assertIn("根因审查",')
    text=text.replace('纠正后6次教师与12次分支拟合', '0次分类器拟合')
    test.write_text(text,encoding='utf-8')
    print(json.dumps({'status':status,'bound_files':len(paths),'unchanged_v101_files':len(oldchecks),'delivery_sha256':sha(target)},ensure_ascii=False))


if __name__=='__main__':main()
