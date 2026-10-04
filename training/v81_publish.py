"""Publish bounded diagnostic training and next plan without rewriting past results."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'artifacts/v81_diagnosis_20260927'
DOC='docs/V81_ROOT_CAUSE_AND_TRAINING_RESET.md'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    out=ROOT/'evidence/2026-09-27/v81_diagnosis';out.mkdir(exist_ok=True)
    if (out/'delivery.json').exists():raise FileExistsError('published')
    old=read(ROOT/'evidence/2026-09-27/v79_execution/delivery.json')
    review=read(ROOT/'artifacts/v80_attribution_20260927/review_receipt.json')
    prior={**old['artifact_sha256'],**review['artifact_sha256']}
    failures=[p for p,h in prior.items() if sha(ROOT/p)!=h]
    assert not failures,failures
    verified=read(DEST/'verification.json');assert verified['status']=='passed'
    assert read(DEST/'trajectory/selection.json')['selected'] is None
    summary='v8.1本轮新增1次诊断训练（24轮、12个参数状态），0次校准拟合，无候选通过原登记条件。完成实际支持分桶、96761条VPC附加信息审计及前瞻选模契约修正；主模型质量未通过，无新模型晋升。最近完整开发回放仍为v7.9的5947错。'
    files=list(DEST.rglob('*'))+[ROOT/DOC]+list((ROOT/'training').glob('v81_*.py'))+[ROOT/'training/test_v81_contract.py']
    delivery={'status':'diagnostic_training_and_root_cause_review_completed_not_promoted','quality_acceptance':False,'all_issues_solved':False,
        'actual_new_fits':1,'actual_classifier_fits':1,'actual_calibration_fits':0,'epochs':24,'parameter_snapshots':12,
        'new_full_data_final_fit':False,'new_full_development_replay':False,'platform_used':False,'external_training_data_used':False,
        'four_arm_training_executed':False,'new_flow_features_used_by_classifier':False,
        'validation_scope':'One matched diagnostic trajectory on inspected official training roles; selected no candidate under prebound C rules. No new model H evaluation or independent environment test. Flow arithmetic/coverage checks are not model quality.',
        'latest_full_development_delivery':'evidence/2026-09-27/v79_execution/delivery.json',
        'latest_full_development_errors':5947,'previous_bound_files_rehashed':len(prior),'previous_bound_files_changed':failures,
        'findings':{'ASA_S_errors':2536,'VPC_M_errors':2664,'combined_fraction_of_full_errors':5200/5947,
                    'ASA_C_S_errors':3464,'largest_component_errors':3316,'actual_fit_seen_only_other_fact_labels_S_errors':1606,
                    'same_label_fit_ASA_S_errors':457,'VPC_M_correct_needed_even_if_other_M_perfect':341,
                    'VPC_M_with_literal_interval':262,'derived_observation_rows':96761,'clock_translation_checks':22380},
        'implementation_verification':verified,
        'prospective_selection_rule':'Overall strict improvement and supported subgroup non-regression; never retroactive promotion.',
        'artifact_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(files)) if p.is_file()}}
    (out/'delivery.json').write_text(json.dumps(delivery,ensure_ascii=False,indent=2),encoding='utf-8')
    for relative,link in [('README.md',DOC),('docs/TRAINING_PLAN.md','V81_ROOT_CAUSE_AND_TRAINING_RESET.md')]:
        p=ROOT/relative;oldtext=p.read_text(encoding='utf-8');title,body=oldtext.split('\n',1)
        assert 'v8.1本轮新增' not in oldtext
        p.write_text(title+'\n\n当前执行与审查（2026-09-27）：**'+summary+'** [根因、真实训练反证与下一轮顺序]('+link+')。以下为历史阶段。\n'+body,encoding='utf-8')
    cp=ROOT/'mcp_readonly/catalog.json';catalog=read(cp)
    for e in catalog['documents']:
        if e['path'] in ('README.md','docs/TRAINING_PLAN.md'):e.update(sha256=sha(ROOT/e['path']),summary=summary)
        else:assert sha(ROOT/e['path'])==e['sha256'],e['path']
        if e['id']=='v80-direction-review':e['category']='historical_review'
        if e['id']=='v79-delivery':e['category']='historical_evidence'
    catalog['project'].update(current_summary=summary,as_of='2026-09-27',authoritative_delivery_id='v81-delivery',authoritative_direction_id='v81-direction-review',
        current_direction=['ASA与VPC分别攻关；代表性验证先于大规模新训练，不以来源分开就等同于覆盖目标困难。',
                           '证据保全、等价呈现与真实类别对照三种监督分开验收；无标签支持不伪造VPC-M。',
                           '先做完整同输入能力对照；有效监督准备好后再补四臂，通过迁移和扩训检查后全量训练。'],
        known_limits=['本轮1次诊断训练未选出候选；新派生观察未用于分类器，未证明质量提升。',
                      'ASA校准S错误95.73%集中一隔离组件；相反标签支持和未见组合需要单列。',
                      'VPC训练M为0，开发2664条M仍全漏；只有262条有可见间隔，不能单靠它达到总目标。',
                      '前瞻修正子集门槛，不改历史失败或总任务目标；全部既有数据仍属已检查开发。'])
    for entry in reversed([
        {'id':'v81-direction-review','title':'v8.1 根因审查、诊断训练与监督重设','path':DOC,'category':'current_review','summary':summary,'keywords':['当前','最新','方向','下一步','训练','恶意','可疑','监督','泛化','根因','v8.1']},
        {'id':'v81-delivery','title':'v8.1 一次诊断训练与未晋升交付','path':'evidence/2026-09-27/v81_diagnosis/delivery.json','category':'current_evidence','summary':'24轮、12状态、1次优化运行；没有通过的候选。新派生观察和选模契约不等于分类收益。','keywords':['执行','证据','训练','v8.1']} ]):
        entry['sha256']=sha(ROOT/entry['path']);catalog['documents'].insert(0,entry)
    cp.write_text(json.dumps(catalog,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'prior_bound_files_verified':len(prior),'new_bound_files':len(delivery['artifact_sha256']),'new_fits':1,'model_promoted':False}))


if __name__=='__main__':main()
