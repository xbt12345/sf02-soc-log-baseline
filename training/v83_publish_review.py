"""Publish a no-fit diagnosis; retain v82 as the latest actual training."""
import json
from run_v75 import ROOT, read, save, sha

DEST=ROOT/'artifacts/v83_root_review_20260927'
DOC='docs/V83_ROOT_CAUSE_AND_SUPERVISION_PLAN.md'
STATUS='root_cause_review_completed_no_new_fit'
SUMMARY='v8.3本轮新增0次分类器拟合、0次校准拟合。复核C净修30条集中于5个组件，H多14错包含共同重训参数抵消；439条一致标签S仍未学会，ASA撤类混入类别比例变化。已制定受控撤类与参数隔离/条件对比监督方案，尚未训练。主模型质量未通过，无新模型晋升。最近实际训练为v8.2，最近完整开发回放仍为v7.9的5947错。'


def main():
    output=ROOT/'evidence/2026-09-27/v83_root_review';output.mkdir(exist_ok=True)
    if (DEST/'review_receipt.json').exists() or (output/'delivery.json').exists():raise FileExistsError('Already published')
    v=read(DEST/'verification.json');d=read(DEST/'diagnosis.json')
    assert v['status']=='passed' and v['new_classifier_fits']==d['new_classifier_fits']==0
    assert v['source_sha256']==sha(ROOT/'training/v83_verify_review.py')
    assert d['source_sha256']==sha(ROOT/'training/v83_root_diagnosis.py')
    prior={}
    for p in ['evidence/2026-09-27/v79_execution/delivery.json','artifacts/v80_attribution_20260927/review_receipt.json',
              'evidence/2026-09-27/v81_diagnosis/delivery.json','evidence/2026-09-27/v82_capacity/delivery.json']:
        prior.update(read(ROOT/p)['artifact_sha256'])
    changed=[p for p,h in prior.items() if sha(ROOT/p)!=h];assert not changed,changed
    files=list(DEST.glob('*'))+[ROOT/DOC]+list((ROOT/'training').glob('v83_*.py'))
    bound={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files) if p.is_file()}
    receipt={'status':'review_frozen','new_classifier_fits':0,'artifact_sha256':bound,
             'prior_bound_files_rehashed':len(prior),'prior_bound_files_changed':changed}
    save(DEST/'review_receipt.json',receipt)
    bound=dict(bound);bound[(DEST/'review_receipt.json').relative_to(ROOT).as_posix()]=sha(DEST/'review_receipt.json')
    delivery={'status':STATUS,'quality_acceptance':False,'all_issues_solved':False,
       'actual_new_fits':0,'actual_classifier_fits':0,'actual_calibration_fits':0,
       'new_full_data_final_fit':False,'new_full_development_replay':False,
       'platform_used':False,'external_training_data_used':False,'pseudo_labels_used':False,'target_answers_read':False,
       'latest_training_delivery':'evidence/2026-09-27/v82_capacity/delivery.json',
       'latest_full_development_delivery':'evidence/2026-09-27/v79_execution/delivery.json','latest_full_development_errors':5947,
       'validation_scope':'No-fit root cause diagnosis of frozen v82 models and previously inspected official development roles. Proposed controls and contrastive/residual arms have not been trained.',
       'findings':d,'implementation_verification':v,'next_plan':DOC,
       'prior_bound_files_rehashed':len(prior),'prior_bound_files_changed':changed,'artifact_sha256':bound}
    save(output/'delivery.json',delivery)
    for relative,link in [('README.md',DOC),('docs/TRAINING_PLAN.md','V83_ROOT_CAUSE_AND_SUPERVISION_PLAN.md')]:
        p=ROOT/relative;previous=p.read_text(encoding='utf-8');title,body=previous.split('\n',1)
        assert 'v8.3本轮新增' not in previous
        p.write_text(title+'\n\n当前根因审查与方案（2026-09-27）：**'+SUMMARY+'** [C改善而H退化的根因及训练调整]('+link+')。以下为历史阶段。\n'+body,encoding='utf-8')
    path=ROOT/'mcp_readonly/catalog.json';catalog=read(path)
    for entry in catalog['documents']:
        if entry['path'] in ['README.md','docs/TRAINING_PLAN.md']:entry.update(sha256=sha(ROOT/entry['path']),summary=SUMMARY)
        else:assert sha(ROOT/entry['path'])==entry['sha256'],entry['path']
        if entry['id']=='v82-direction-review':entry['category']='historical_review'
        if entry['id']=='v82-delivery':entry['category']='historical_evidence'
    catalog['project'].update(current_summary=SUMMARY,as_of='2026-09-27',
        authoritative_delivery_id='v83-delivery',authoritative_direction_id='v83-direction-review',
        current_direction=['先用受控撤类区分类别比例与ASA行为支持，再启动有资格监督的四臂；本轮未训练。',
          '四臂分离原分支冻结/共同更新与有无条件对比损失；保留原输入、原标签、原频次，双向保护M/S。',
          '新增原fit内组件轮换，旧C/H维持开发回归，VPC缺M真值单独保留；逐类质量与扩训稳定性通过才最终全量训练。'],
        known_limits=['C少30错主要集中于一个大组件，H多14错；参数分解是机制证据，不是修正模型已有效。',
          '439条已入样一致标签S未学会，低秩核失败不排除所有强模型；此前神经和加权也未稳定迁移。',
          '跨组件配对不是新安全真值，VPC无M训练支持仍未解决；不能把来源组件当真实组织或把历史重切称为盲测。',
          '本轮0次训练，无新模型晋升；最近实际训练v8.2，最近完整回放v7.9，质量未通过。'])
    new=[{'id':'v83-direction-review','title':'v8.3 C改善H退化根因与受控监督训练方案','path':DOC,'category':'current_review',
           'summary':SUMMARY,'keywords':['当前','最新','方向','下一步','训练','恶意','可疑','监督','泛化','根因','v8.3']},
         {'id':'v83-delivery','title':'v8.3 无新训练的根因诊断与方案证据','path':(output/'delivery.json').relative_to(ROOT).as_posix(),
          'category':'current_evidence','summary':'0次拟合；86条修复退化、220对真实入样对照、278个历史绑定文件核验。后续实验未执行。','keywords':['执行','证据','审查','监督','v8.3']}]
    for entry in reversed(new):entry['sha256']=sha(ROOT/entry['path']);catalog['documents'].insert(0,entry)
    save(path,catalog)
    p=ROOT/'mcp_readonly/tests/test_readonly_mcp.py';s=p.read_text(encoding='utf-8')
    s=s.replace('v82-direction-review','v83-direction-review').replace('v82-delivery','v83-delivery')
    s=s.replace('capacity_training_completed_not_promoted',STATUS).replace('本轮新增6次对照拟合','本轮新增0次分类器拟合')
    p.write_text(s,encoding='utf-8')
    print(json.dumps({'published':True,'new_classifier_fits':0,'new_bound_files':len(bound),'old_files_verified':len(prior),
                      'quality_acceptance':False,'raw_casebook_in_MCP':False},ensure_ascii=False))


if __name__=='__main__':main()
