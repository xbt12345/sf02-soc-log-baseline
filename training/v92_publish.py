"""Publish only verified executed results; quality remains separate from execution."""
from v92_common import ROOT,DEST,read,save,sha,check

DOC='docs/V92_V91_PLAN_EXECUTION_RESULTS.md'
STATUS='evidence_and_training_completed_no_model_promoted'

def main():
    check();folder=ROOT/'evidence/2026-09-28/v92_evidence_training';assert not (folder/'delivery.json').exists()
    v=read(DEST/'verification.json');s=read(DEST/'execution_summary.json');assert v['status']=='passed' and not v['prior_bound_files_changed'] and v['source_sha256']==sha(ROOT/'training/v92_verify.py')
    assert s['actual_classifier_fits']==v['actual_classifier_fits']==9 and s['selected'] is None and not s['quality_acceptance']
    summary=('v9.2本轮新增9次分类器拟合、0次校准拟合：3主对照、2数值补充、4稀缺组件轮换。'
        '720条错误、97种实际输入逐项追踪；未证实新的已知事实遗漏，U1/P1跳过。'
        'U0R达ASA训练经验下界22、旧52条S训练错误全对、训练旧正确零退化且满足全部P；内层却新增500条M错误，仅修复58条S。'
        'P0优化路径停滞，不能称P约束不可行。TCP445轮换S0/6，UDP目的未知S4/6且方向不对称。'
        '独立核验原标签和9个模型，814历史绑定文件未变。主模型质量未通过，无新模型晋升。最近完整开发回放仍为v7.9的5947错。')
    catalogpath=ROOT/'mcp_readonly/catalog.json';catalog=read(catalogpath);assert catalog['project']['authoritative_direction_id']=='v91-direction-review'
    for e in catalog['documents']:assert sha(ROOT/e['path'])==e['sha256'],e['path']
    files=[p for p in DEST.rglob('*') if p.is_file()]+list((ROOT/'training').glob('v92*.py'))+[ROOT/DOC]
    bound={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)}
    delivery={**s,'status':STATUS,'validation_scope':v['scope'],'latest_executed_training_delivery':'evidence/2026-09-28/v92_evidence_training/delivery.json',
        'previous_executed_training_delivery':'evidence/2026-09-27/v89_readout_support/delivery.json',
        'latest_full_development_delivery':'evidence/2026-09-27/v79_execution/delivery.json','latest_full_development_errors':5947,
        'verification':v,'problem_register':'artifacts/v92_evidence_training_20260928/problem_register.json','current_plan':DOC,'artifact_sha256':bound}
    folder.mkdir(parents=True,exist_ok=True);save(folder/'delivery.json',delivery)
    for relative,link in [('README.md',DOC),('docs/TRAINING_PLAN.md','V92_V91_PLAN_EXECUTION_RESULTS.md')]:
        p=ROOT/relative;old=p.read_text(encoding='utf-8');assert 'v9.2本轮新增' not in old;title,body=old.split('\n',1)
        p.write_text(title+'\n\n当前方案执行与训练结果（2026-09-28）：**'+summary+'** [实际训练、可行终点与问题记录]('+link+')。以下为历史阶段。\n'+body,encoding='utf-8')
    for e in catalog['documents']:
        if e['path'] in ['README.md','docs/TRAINING_PLAN.md']:e.update(sha256=sha(ROOT/e['path']),summary=summary)
        if e['id']=='v91-direction-review':e['category']='historical_review'
        if e['id']=='v91-delivery':e['category']='historical_evidence'
    catalog['project'].update(as_of='2026-09-28',current_summary=summary,authoritative_direction_id='v92-direction-review',authoritative_delivery_id='v92-delivery',
        current_direction=['本轮已完成9次实际拟合和720条错误追踪；没有合格候选，停止候选扩折、正式全量拟合和模型晋升。',
            '同一R0非线性分支已达到ASA训练经验界，并构造满足P的终点；优先查跨组件M/S依据与500条新M错误，不再为该训练拟合盲目增容。',
            '保留组件稀缺与unsupported不变性问题；所有实际格式监督和逐类原行结果有记录，收益以独立预测复算为准。'],
        known_limits=['U0R训练拟合成功不代表迁移成功：内层M退化500、C/H仍有M/S退化，质量未通过。',
            '52条是旧支持模型的训练难例，本轮训练池更完整；全对不作为纯容量因果证据或独立测试收益。',
            'P0路径停滞但存在可行好训练终点；约束可行与优化收敛不能混同。L0/L0R均未充分优化，不能用来断言线性能力极限。',
            '稀缺轮换每方向只有S测试，不能认证M保留率；所有数据角色仍是既有开发，不是新盲测。',
            'unsupported行45738存在身份替换不变性问题；本轮仅ASA修正，未宣称整个输入无损或身份不变。',
            '未使用平台、外部训练数据或伪标签，无新全任务成绩；本地MCP核验不代表云端Tunnel连接。'])
    entries=[{'id':'v92-direction-review','title':'v9.2 v9.1方案九次训练执行、结果与问题记录','path':DOC,'category':'current_review','summary':summary,
        'keywords':['当前','最新','方向','下一步','训练','执行','恶意','可疑','ASA','模型能力','根因','约束','v9.2']},
        {'id':'v92-delivery','title':'v9.2 原行追踪与九次实际拟合交付证据','path':(folder/'delivery.json').relative_to(ROOT).as_posix(),'category':'current_evidence',
         'summary':'9次真实拟合；720条原文、97种输入、模型独立重放，814历史绑定文件未变。ASA训练经验界达成，跨组件质量仍未通过。','keywords':['证据','执行','复核','v9.2']}]
    for e in reversed(entries):e['sha256']=sha(ROOT/e['path']);catalog['documents'].insert(0,e)
    save(catalogpath,catalog)
    p=ROOT/'mcp_readonly/tests/test_readonly_mcp.py';text=p.read_text(encoding='utf-8').replace('v91-direction-review','v92-direction-review').replace('v91-delivery','v92-delivery').replace('first_principles_review_completed_no_new_fit',STATUS).replace('本轮新增0次分类器拟合','本轮新增9次分类器拟合');p.write_text(text,encoding='utf-8')
    print(__import__('json').dumps({'published':True,'new_classifier_fits':9,'bound_files':len(bound),'quality_acceptance':False},ensure_ascii=False))

if __name__=='__main__':main()
