"""Publish a no-fit review after independent frozen-model verification."""
from run_v75 import ROOT,read,save,sha
DEST=ROOT/'artifacts/v90_root_review_20260928'
DOC='docs/V90_REGRESSION_ROOT_CAUSE_AND_PLAN.md'
STATUS='root_cause_review_completed_no_new_fit'

def main():
    folder=ROOT/'evidence/2026-09-28/v90_root_review';folder.mkdir(exist_ok=True,parents=True)
    assert not (folder/'delivery.json').exists()
    v=read(DEST/'verification.json');d=read(DEST/'diagnosis.json');s=read(DEST/'score_decomposition.json')
    assert v['status']=='passed' and not v['prior_bound_files_changed'] and v['source_sha256']==sha(ROOT/'training/v90_verify.py')
    assert d['source_sha256']==sha(ROOT/'training/v90_diagnose.py') and s['source_sha256']==sha(ROOT/'training/v90_score_decomposition.py')
    for p,h in v['verified_output_sha256'].items():assert sha(DEST/p)==h,p
    for p,h in v['receipt_hashes'].items():assert sha(ROOT/p)==h,p
    summary=('v9.0本轮新增0次分类器拟合、0次校准拟合。独立复算2056871条原标签与F00决策，795个历史绑定文件未变。'
        'v8.9内层S修复38条、M退化668条；其中392条为UDP外部到DMZ目的514拒绝，训练同细行为仅M2/S0，去掉源端口贡献仍错360条。'
        '640条退化缺训练完整事实组合支持，576条含未见词项；删除未见项贡献反而将M退化增至856条。'
        '取消锁死最小训练修复松弛的顺序，采用可训练完整上下文及训练侧跨组件约束对照；门控须先有独立互补收益。'
        '方案未训练，最近实际训练仍为v8.9。主模型质量未通过，无新模型晋升。最近完整开发回放仍为v7.9的5947错。')
    files=[p for p in DEST.rglob('*') if p.is_file()]+list((ROOT/'training').glob('v90*.py'))+[ROOT/DOC]
    bound={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)}
    delivery={'status':STATUS,'quality_acceptance':False,'all_issues_solved':False,'model_promoted':False,
        'actual_new_fits':0,'actual_classifier_fits':0,'actual_calibration_fits':0,'new_full_data_final_fit':False,
        'new_full_development_replay':False,'new_full_task_pipeline_replay':False,'platform_used':False,'external_training_data_used':False,'pseudo_labels_used':False,'development_not_blind':True,
        'latest_executed_training_delivery':'evidence/2026-09-27/v89_readout_support/delivery.json',
        'latest_full_development_delivery':'evidence/2026-09-27/v79_execution/delivery.json','latest_full_development_errors':5947,
        'validation_scope':'No-fit review: original2056871 labels and F00 decisions, manual literal/hash score replay,36 counterfactual-role results and47 coarse failure groups verified;795 historical bound files unchanged. Partial facts reused, no renewed full raw parsing, new blind test, model or cloud connection.',
        'target_diagnostics':d['target_counts'],'largest_failure_group':s['coarse_groups'][0],
        'objective_diagnostics':{'main_risk':s['main_risk'],'zero_correction_weighted_minimum_E_slack':s['zero_correction_weighted_minimum_E_slack'],'locked_weighted_E_slack_cap':s['locked_weighted_E_slack_cap']},
        'verification':v,'current_plan':DOC,'artifact_sha256':bound}
    # Verify the current catalog before mutating its two rolling documents.
    path=ROOT/'mcp_readonly/catalog.json';catalog=read(path)
    for e in catalog['documents']:assert sha(ROOT/e['path'])==e['sha256'],e['path']
    save(folder/'delivery.json',delivery)
    for relative,link in [('README.md',DOC),('docs/TRAINING_PLAN.md','V90_REGRESSION_ROOT_CAUSE_AND_PLAN.md')]:
        p=ROOT/relative;old=p.read_text(encoding='utf-8');assert 'v9.0本轮新增' not in old;title,body=old.split('\n',1)
        p.write_text(title+'\n\n当前根因复核与下一轮方案（2026-09-28）：**'+summary+'** [退化机制、反证和执行顺序]('+link+')。以下为历史阶段。\n'+body,encoding='utf-8')
    for e in catalog['documents']:
        if e['path'] in ['README.md','docs/TRAINING_PLAN.md']:e.update(sha256=sha(ROOT/e['path']),summary=summary)
        if e['id']=='v89-direction-review':e['category']='historical_review'
        if e['id']=='v89-delivery':e['category']='historical_evidence'
    catalog['project'].update(as_of='2026-09-28',current_summary=summary,authoritative_direction_id='v90-direction-review',authoritative_delivery_id='v90-delivery',
        current_direction=['保留事实修复和旧模型，停止扩大失败F00；改正先锁死训练修复松弛再优化风险的顺序。',
            '训练人口内组件隔离参数区、约束反馈区与选择区，教师按角色重建；采用跨组件约束与可训练R0/具名事实的四格对照。',
            '52条训练S缺口与668条来源M退化分别处理；只有独立互补修复后再训练门控，按原行逐类及完整任务验收。'],
        known_limits=['本轮是零拟合诊断；新训练方案未执行，不能保证收益。',
            '640条新组合和576条未见词项是覆盖证据，不是对所有错误的单一因果证明；去掉源端口/未见项均未解决。',
            '原始信息保存不等于新分支可学习；训练保护不保证未知域零退化，已有开发角色均非新盲测。',
            'v8.9的5个风险优化未收敛、支持对照S仅6条及52条训练S错误仍待处理。',
            '无外部训练数据、平台训练、全量新模型或模型晋升；本地只读MCP验证不代表云端Tunnel实连。'])
    entries=[{'id':'v90-direction-review','title':'v9.0 恶意退化机制、训练目标复核与新方案','path':DOC,'category':'current_review','summary':summary,
        'keywords':['当前','最新','方向','下一步','训练','恶意','可疑','ASA','退化','根因','泛化','哈希','约束','v9.0']},
        {'id':'v90-delivery','title':'v9.0 冻结模型独立复算与零拟合证据','path':(folder/'delivery.json').relative_to(ROOT).as_posix(),'category':'current_evidence',
        'summary':'0次拟合；2056871条原标签/F00决策及36组诊断复算，795历史绑定文件未变；新方案未训练，质量未通过。','keywords':['证据','复核','执行','v9.0']}]
    for e in reversed(entries):e['sha256']=sha(ROOT/e['path']);catalog['documents'].insert(0,e)
    save(path,catalog)
    path=ROOT/'mcp_readonly/tests/test_readonly_mcp.py';text=path.read_text(encoding='utf-8').replace('v89-direction-review','v90-direction-review').replace('v89-delivery','v90-delivery')
    text=text.replace('readout_support_training_completed_no_model_promoted',STATUS).replace('本轮新增10次分类器拟合','本轮新增0次分类器拟合');path.write_text(text,encoding='utf-8')
    print(__import__('json').dumps({'published':True,'new_fits':0,'new_bound_files':len(bound),'prior_files_unchanged':v['prior_bound_files_rehashed'],'quality_acceptance':False},ensure_ascii=False))

if __name__=='__main__':main()
