"""Publish a verified review and prospective plan, without claiming a new trained model."""
from run_v75 import ROOT,read,save,sha

DEST=ROOT/'artifacts/v88_root_review_20260927'
DOC='docs/V88_ROOT_CAUSE_AND_TRAINING_PLAN.md'
STATUS='root_cause_review_completed_no_new_fit'


def main():
    folder=ROOT/'evidence/2026-09-27/v88_root_review';folder.mkdir(exist_ok=True)
    assert not (folder/'delivery.json').exists()
    v=read(DEST/'verification.json');d=read(DEST/'diagnosis.json')
    assert v['status']=='passed' and not v['prior_bound_files_changed']
    assert v['source_sha256']==sha(ROOT/'training/v88_verify.py')
    assert d['source_sha256']==sha(ROOT/'training/v88_diagnose.py')
    assert v['diagnosis_sha256']==sha(DEST/'diagnosis.json')
    assert v['support_sha256']==sha(DEST/'support_and_errors.csv')
    for path,h in v['receipt_hashes'].items():assert sha(ROOT/path)==h,path
    summary=('v8.8本轮新增0次分类器拟合、0次校准拟合。冻结权重与原行复核发现：315条错例贡献57%–59%分类损失但与原正确梯度近乎相反；'
       '第20次辅助编码器梯度约为分类梯度110倍，未直接监督输出头；冻结候选唯一内层修复来自Windows，ASA为0。'
       'ASA-S旧错174条中138条缺同类细级行为支持且均未修复；另有局部事实/辅助监督缺口。'
       '下一轮先补部分事实，做固定/自由教师读出对照及独立支持撤除控制，再按ASA跨折实效扩训。'
       '主模型质量未通过，无新模型晋升。最近实际训练仍为v8.7，最近完整开发回放仍为v7.9的5947错。')
    files=[p for p in DEST.rglob('*') if p.is_file()]+list((ROOT/'training').glob('v88*.py'))+[ROOT/DOC]
    bound={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)}
    delivery={'status':STATUS,'quality_acceptance':False,'all_issues_solved':False,'model_promoted':False,
        'actual_new_fits':0,'actual_classifier_fits':0,'actual_calibration_fits':0,
        'new_full_data_final_fit':False,'new_full_development_replay':False,'new_full_task_pipeline_replay':False,
        'platform_used':False,'external_training_data_used':False,'pseudo_labels_used':False,'development_not_blind':True,
        'latest_executed_training_delivery':'evidence/2026-09-27/v87_solver_supervision/delivery.json',
        'latest_full_development_delivery':'evidence/2026-09-27/v79_execution/delivery.json','latest_full_development_errors':5947,
        'validation_scope':'No-fit frozen gradient/input review, independent original-row attribution and semantic support recomputation; 638 historical artifacts unchanged. Prospective training remains unexecuted; no cloud or new blind validation.',
        'main_loss_rows':d['main_loss_rows'],'main_loss_error_rows':d['main_loss_error_rows'],
        'full_protected_correct_rows':d['full_protected_correct_rows'],
        'gradient_diagnostics':d['loss_and_gradient'],
        'target_support_summary':{'asa_inner_S_original_errors':174,'zero_same_class_fine_behavior_support_errors':138,'zero_support_errors_repaired_by_final_A3':0},
        'input_conflict_summary':{'blocked_errors':22,'unblocked_errors':293,'missing_port_null_vs_empty_is_not_behavior':True},
        'verification':v,'current_plan':DOC,'artifact_sha256':bound}
    save(folder/'delivery.json',delivery)
    for relative,link in [('README.md',DOC),('docs/TRAINING_PLAN.md','V88_ROOT_CAUSE_AND_TRAINING_PLAN.md')]:
        path=ROOT/relative;old=path.read_text(encoding='utf-8');assert 'v8.8本轮新增' not in old
        title,body=old.split('\n',1)
        path.write_text(title+'\n\n当前根因审查与方案（2026-09-27）：**'+summary+'** [诊断证据、训练顺序与验收]('+link+')。以下为历史阶段。\n'+body,encoding='utf-8')
    path=ROOT/'mcp_readonly/catalog.json';catalog=read(path)
    for e in catalog['documents']:
        if e['path'] in ['README.md','docs/TRAINING_PLAN.md']:e.update(sha256=sha(ROOT/e['path']),summary=summary)
        else:assert sha(ROOT/e['path'])==e['sha256'],e['path']
        if e['id']=='v87-direction-review':e['category']='historical_review'
        if e['id']=='v87-delivery':e['category']='historical_evidence'
    catalog['project'].update(as_of='2026-09-27',current_summary=summary,
       authoritative_direction_id='v88-direction-review',authoritative_delivery_id='v88-delivery',
       current_direction=['先修部分可观察事实与监督覆盖，保留原件、原标签和逐行计数；下一轮尚未训练。',
          '以匹配检查点表示做固定/自由教师系数四个小读出分支，区分表示、输出与保护可行性；不原样扩大当前配对配置。',
          '缺支持撤除与等量删除控制独立执行，不再等待候选成功；按ASA跨折实效、M/S退化和完整工程回放验收。'],
       known_limits=['原始梯度诊断未包含Adam/L2/投影，不是实际步长的唯一因果证明。',
          '部分事实解析、读出对照和支持控制尚未执行；零细级支持不等于所有语义证据不存在。',
          '原件保留不等于有限字节编码无损；NULL/空串、时间和身份不能作为修分捷径。',
          '训练P保护不保证未知来源零退化；当前无新盲测、平台训练或正式提交，质量未通过。'])
    entries=[{'id':'v88-direction-review','title':'v8.8 梯度、监督及支持根因复核与训练方案','path':DOC,'category':'current_review','summary':summary,
         'keywords':['当前','最新','方向','下一步','训练','ASA','恶意','可疑','退化','梯度','监督','读出','泛化','v8.8']},
        {'id':'v88-delivery','title':'v8.8 零拟合根因复核与历史完整性证据','path':(folder/'delivery.json').relative_to(ROOT).as_posix(),'category':'current_evidence',
         'summary':'0次新拟合；原始标签/支持/预测变更独立复算，638个历史绑定文件未变；方案未执行。','keywords':['执行','证据','审查','恶意','可疑','v8.8']}]
    for e in reversed(entries):e['sha256']=sha(ROOT/e['path']);catalog['documents'].insert(0,e)
    save(path,catalog)
    path=ROOT/'mcp_readonly/tests/test_readonly_mcp.py';s=path.read_text(encoding='utf-8')
    s=s.replace('v87-direction-review','v88-direction-review').replace('v87-delivery','v88-delivery')
    s=s.replace('solver_supervision_training_completed_no_model_promoted',STATUS).replace('本轮实际启动14次分类器拟合','本轮新增0次分类器拟合')
    path.write_text(s,encoding='utf-8')
    print(__import__('json').dumps({'published':True,'new_fits':0,'new_bound_files':len(bound),'prior_unchanged_files':v['prior_bound_files_rehashed'],'quality_acceptance':False},ensure_ascii=False))


if __name__=='__main__':main()
