"""Publish no-fit preservation review without changing old model artifacts."""
import json
from run_v75 import ROOT, read, save, sha

DEST=ROOT/'artifacts/v84_preservation_20260927'
DOC='docs/V84_PRESERVE_CORRECT_AND_REPAIR_PLAN.md'
STATUS='preservation_review_completed_no_new_fit'
SUMMARY='v8.4本轮新增0次分类器拟合、0次校准拟合。查明H改动18条仅2条改对、16条改错；439条困难S与已判对M存在强局部梯度冲突，但无相同输入保护冲突。下一轮先做冻结、选择性蒸馏、间隔约束三臂，以已知正确零退化且真实修错为条件，再加跨组件监督。方案尚未训练，主模型质量未通过，无新模型晋升。最近实际训练为v8.2，最近完整开发回放仍为v7.9的5947错。'


def main():
    out=ROOT/'evidence/2026-09-27/v84_preservation';out.mkdir(exist_ok=True)
    if (DEST/'review_receipt.json').exists() or (out/'delivery.json').exists():raise FileExistsError('Already published')
    v=read(DEST/'verification.json');d=read(DEST/'diagnosis.json')
    assert v['status']=='passed' and v['new_classifier_fits']==d['new_classifier_fits']==0
    assert v['source_sha256']==sha(ROOT/'training/v84_verify_preservation.py')
    assert v['diagnosis_sha256']==sha(DEST/'diagnosis.json')
    assert d['source_sha256']==sha(ROOT/'training/v84_preservation_audit.py')
    prior={}
    for p in ['evidence/2026-09-27/v79_execution/delivery.json','artifacts/v80_attribution_20260927/review_receipt.json',
              'evidence/2026-09-27/v81_diagnosis/delivery.json','evidence/2026-09-27/v82_capacity/delivery.json',
              'evidence/2026-09-27/v83_root_review/delivery.json']:
        prior.update(read(ROOT/p)['artifact_sha256'])
    changed=[p for p,h in prior.items() if sha(ROOT/p)!=h];assert not changed,changed
    files=list(DEST.glob('*'))+[ROOT/DOC]+list((ROOT/'training').glob('v84_*.py'))
    bound={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files) if p.is_file()}
    save(DEST/'review_receipt.json',{'status':'review_frozen','new_classifier_fits':0,'artifact_sha256':bound,
          'prior_bound_files_rehashed':len(prior),'prior_bound_files_changed':changed})
    bound=dict(bound);bound[(DEST/'review_receipt.json').relative_to(ROOT).as_posix()]=sha(DEST/'review_receipt.json')
    delivery={'status':STATUS,'quality_acceptance':False,'all_issues_solved':False,
       'actual_new_fits':0,'actual_classifier_fits':0,'actual_calibration_fits':0,
       'new_full_data_final_fit':False,'new_full_development_replay':False,
       'platform_used':False,'external_training_data_used':False,'pseudo_labels_used':False,'target_answers_read':False,
       'latest_training_delivery':'evidence/2026-09-27/v82_capacity/delivery.json',
       'latest_full_development_delivery':'evidence/2026-09-27/v79_execution/delivery.json','latest_full_development_errors':5947,
       'validation_scope':'No-fit negative-flip, input feasibility and local gradient diagnosis of frozen v82 models. Proposed preservation training not executed. No unknown-input guarantee.',
       'findings':d,'implementation_verification':v,'next_plan':DOC,
       'prior_bound_files_rehashed':len(prior),'prior_bound_files_changed':changed,'artifact_sha256':bound}
    save(out/'delivery.json',delivery)
    for relative,link in [('README.md',DOC),('docs/TRAINING_PLAN.md','V84_PRESERVE_CORRECT_AND_REPAIR_PLAN.md')]:
        p=ROOT/relative;previous=p.read_text(encoding='utf-8');title,body=previous.split('\n',1)
        assert 'v8.4本轮新增' not in previous
        p.write_text(title+'\n\n当前保护审查与方案（2026-09-27）：**'+SUMMARY+'** [保住正确判断与修复错误的训练方案]('+link+')。以下为历史阶段。\n'+body,encoding='utf-8')
    path=ROOT/'mcp_readonly/catalog.json';catalog=read(path)
    for e in catalog['documents']:
        if e['path'] in ['README.md','docs/TRAINING_PLAN.md']:e.update(sha256=sha(ROOT/e['path']),summary=SUMMARY)
        else:assert sha(ROOT/e['path'])==e['sha256'],e['path']
        if e['id']=='v83-direction-review':e['category']='historical_review'
        if e['id']=='v83-delivery':e['category']='historical_evidence'
    catalog['project'].update(current_summary=SUMMARY,as_of='2026-09-27',authoritative_delivery_id='v84-delivery',authoritative_direction_id='v84-direction-review',
       current_direction=['先检验保对修错：冻结基线、选择性蒸馏、最终间隔约束三臂；训练内P与验证分离，有限已知正确集零负翻转。',
         '必须真实修错才继续，保护通过后再增加跨组件条件对比；V83容量/对比四臂后移，撤类对照保留为独立支持诊断。',
         '保留原始数据和频次，逐类检查改对/改错及全部任务指标；未通过不晋升，未知输入不承诺零退化。'],
       known_limits=['梯度冲突只是旧参数处的局部平均损失证据，不等于所有模型必然发生负翻转。',
         '439条没有相同输入保护冲突；有限查表误差下界24不是原始信息极限或可部署成绩。',
         '软蒸馏、平均损失保护均不保证零翻转；新约束训练尚未执行，VPC缺M判据仍未解决。',
         '旧文件均保留，旧候选未晋升；本轮0次训练，最近实际训练v8.2，完整回放v7.9，质量未通过。'])
    entries=[{'id':'v84-direction-review','title':'v8.4 保住正确判断与修复错误的训练方案','path':DOC,'category':'current_review','summary':SUMMARY,
        'keywords':['当前','最新','方向','下一步','训练','冻结','保护','正确','错误','恶意','可疑','监督','泛化','根因','v8.4']},
       {'id':'v84-delivery','title':'v8.4 零新拟合的负翻转与保护可行性审查','path':(out/'delivery.json').relative_to(ROOT).as_posix(),
        'category':'current_evidence','summary':'0次拟合；24983种入样输入、4个角色、3类有限差分和291个旧文件核验。保护训练未执行。',
        'keywords':['执行','证据','审查','保护','冻结','v8.4']}]
    for e in reversed(entries):e['sha256']=sha(ROOT/e['path']);catalog['documents'].insert(0,e)
    save(path,catalog)
    p=ROOT/'mcp_readonly/tests/test_readonly_mcp.py';s=p.read_text(encoding='utf-8')
    s=s.replace('v83-direction-review','v84-direction-review').replace('v83-delivery','v84-delivery').replace('root_cause_review_completed_no_new_fit',STATUS)
    p.write_text(s,encoding='utf-8')
    print(json.dumps({'published':True,'new_classifier_fits':0,'new_bound_files':len(bound),
                      'old_files_verified':len(prior),'quality_acceptance':False},ensure_ascii=False))


if __name__=='__main__':main()
