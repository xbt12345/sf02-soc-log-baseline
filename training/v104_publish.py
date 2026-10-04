"""Publish the executed v10.4 round without changing historical receipts."""
import json
from pathlib import Path
from run_v75 import ROOT, read, save, sha
from v104_phase_a import DEST as ADEST
from v104_phase_b import DEST as BDEST


DOC = ROOT / 'docs/V104_OFFICIAL_COVERAGE_AND_INDEPENDENT_MODEL_TRAINING.md'
DELIVERY = ROOT / 'evidence/2026-09-28/v104_official_round/delivery.json'
STATUS = 'v104_official_coverage_and_independent_models_executed_no_promotion'


def main():
    if DELIVERY.exists():
        raise FileExistsError(DELIVERY)
    prior = read(ROOT/'evidence/2026-09-28/v103_plan_review/delivery.json')
    assert all(sha(ROOT/p)==h for p,h in prior['artifact_sha256'].items())
    verified=read(BDEST/'verification.json')
    assert verified['all_checks_passed'] and all(verified['checks'].values())
    a=read(ADEST/'phase_A_evaluation.json')
    b=read(BDEST/'phase_B_evaluation.json')
    post=read(BDEST/'postfit_error_mechanism.json')
    assert a['classifier_fits']==6 and b['classifier_fits']==9
    assert b['selected'] is None and not b['model_promoted']
    assert b['candidates']['C_TabM_epoch25']['errors']==2980
    assert post['group_concentration']['new_M_errors']['top1_rows']==948
    assert post['label_assisted_uniform_boundary_upper_bound']['unusable_for_model_selection']
    summary=('v10.4按修订方案完成15次分类器拟合、0次校准。官方全池R0在ASA的错误为5858，较匹配小池5820更差；'
        'N1教师5860错。TabM第25轮ASA降至2980错，但M判对78322→77466，S的272组仍有231组零召回，未过M召回保护。'
        '101条新增支持旧S错仅修复6条；修复和退化集中少数组。质量未通过，无模型晋升；最近旧版完整开发回放仍为v7.9的5947错，评估口径不可直接比较。')
    paths=[DOC,
        ROOT/'training/v104_phase_a.py',ROOT/'training/v104_phase_a_diagnose.py',
        ROOT/'training/v104_phase_b.py',ROOT/'training/v104_phase_b_evaluate.py',
        ROOT/'training/v104_phase_b_evaluate_r2.py',ROOT/'training/v104_postfit_audit.py',
        ROOT/'training/v104_verify.py',ROOT/'training/v104_publish.py',
        ADEST/'registration.json',ADEST/'phase_A_evaluation.json',
        ADEST/'phase_A_extended_diagnosis.json',ADEST/'phase_A_ASA_OOF_ledger.parquet',
        BDEST/'registration.json',BDEST/'phase_B_evaluation.json',
        BDEST/'phase_B_ASA_OOF_ledger.parquet',BDEST/'postfit_error_mechanism.json',
        BDEST/'verification.json']
    for k in range(3):
        for pop in ('small','full'):
            paths.append(ADEST/f'fold{k}_{pop}/fit.json')
        paths.append(BDEST/f'fold{k}_N1_teacher/fit.json')
        for arm in ('MLP','TabM'):
            paths.append(BDEST/f'fold{k}_C_{arm}/fit.json')
    assert all(p.is_file() for p in paths)
    proof={'status':STATUS,
        'actual_classifier_fits_this_user_request':15,'phase_A_fits':6,'phase_B_fits':9,
        'actual_calibration_fits':0,'official_training_rows':2056871,
        'same_ASA_OOF_rows':112807,'quality_acceptance':False,'model_promoted':False,
        'platform_used':False,'external_training_data_used':False,
        'current_report':DOC.relative_to(ROOT).as_posix(),
        'validation_scope':'Fixed global-group three-fold internal OOF. All folds previously observed; no fresh external validation.',
        'phase_A_ASA_errors':{'small':a['small']['errors'],'full':a['full']['errors']},
        'phase_B_ASA_errors':{'N1_teacher':b['N1_teacher']['errors'],
                              'TabM_25':b['candidates']['C_TabM_epoch25']['errors']},
        'phase_B_selection':None,
        'M_recall_guard_failed_for_all_six_candidate_checkpoints':True,
        'latest_full_development_replay':'v7.9, 5947 errors, different split and training scope; not directly comparable',
        'verification_checks_passed':len(verified['checks']),
        'prior_v103_bound_files_unchanged':len(prior['artifact_sha256']),
        'artifact_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in paths}}
    DELIVERY.parent.mkdir(parents=True,exist_ok=True)
    save(DELIVERY,proof)
    for rel,link in [('README.md','docs/V104_OFFICIAL_COVERAGE_AND_INDEPENDENT_MODEL_TRAINING.md'),
                     ('docs/TRAINING_PLAN.md','V104_OFFICIAL_COVERAGE_AND_INDEPENDENT_MODEL_TRAINING.md')]:
        path=ROOT/rel
        source=path.read_text(encoding='utf-8')
        assert 'v10.4按修订方案完成15次' not in source
        first,rest=source.split('\n',1)
        path.write_text(first+'\n\n当前训练结果（2026-09-28）：**'+summary+'** '
                        +f'[本轮训练、错误机制与核验]({link})。以下保留历史阶段。\n'+rest,
                        encoding='utf-8')
    catalog_path=ROOT/'mcp_readonly/catalog.json'
    catalog=read(catalog_path)
    for entry in catalog['documents']:
        if entry['path'] in ('README.md','docs/TRAINING_PLAN.md'):
            entry['sha256']=sha(ROOT/entry['path'])
            entry['summary']=summary
        else:
            assert sha(ROOT/entry['path'])==entry['sha256'],entry['id']
        if entry['category']=='current_review':entry['category']='historical_review'
        if entry['category']=='current_evidence':entry['category']='historical_evidence'
    catalog['project'].update(as_of='2026-09-28',current_summary=summary,
        current_direction=[
            '本轮15次拟合结束，无合格候选；保持正式模型不变。',
            '下一项仅在训练侧预登记并拟合统一边界，留出折只用于应用和评价，不用其标签调阈值。',
            '同时验收大组以外的S组收益及M旧正确，仍需真正外部数据检验跨来源能力。'],
        known_limits=[
            'ASA TabM25留出2980错，但M召回退化，231/272个S组零召回。',
            'S修复3640/3740集中一个组，新增M错误948/1028集中另一个组。',
            '当前三折为已反复观察的开发数据，不是三个独立企业；事后阈值仅为乐观诊断。'],
        authoritative_delivery_id='v104-delivery',
        authoritative_direction_id='v104-training-review')
    for item in reversed([
        ('v104-training-review','v10.4 官方覆盖与独立分类器训练结果',DOC,'current_review'),
        ('v104-delivery','v10.4 真实训练与独立核验凭据',DELIVERY,'current_evidence')]):
        ident,title,path,category=item
        catalog['documents'].insert(0,{'id':ident,'title':title,
            'path':path.relative_to(ROOT).as_posix(),'category':category,
            'summary':summary,
            'keywords':['当前','最新','训练','结果','TabM','恶意','可疑','v10.4'],
            'sha256':sha(path)})
    save(catalog_path,catalog)
    test=ROOT/'mcp_readonly/tests/test_readonly_mcp.py'
    source=test.read_text(encoding='utf-8')
    assert 'v103-direction-review' in source
    source=source.replace('v103-direction-review','v104-training-review')
    source=source.replace('v103-delivery','v104-delivery')
    source=source.replace('v103_evidence_based_plan_review_completed_no_fit',STATUS)
    source=source.replace('self.assertIn("有据评价",','self.assertIn("官方覆盖",')
    source=source.replace("self.assertIn('0次分类器拟合',","self.assertIn('15次分类器拟合',")
    test.write_text(source,encoding='utf-8')
    print(json.dumps({'status':STATUS,'bound_files':len(paths),
        'prior_bound_files_verified':len(prior['artifact_sha256']),
        'verification_checks':len(verified['checks']),
        'delivery_sha256':sha(DELIVERY)},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
