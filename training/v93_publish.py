"""Publish verified frozen audit and revised plan without changing training authority."""
from v89_common import ROOT,read,save,sha
from v93_regression_audit import DEST

DOC='docs/V93_500_M_REGRESSION_ROOT_CAUSE_AND_PLAN.md'
STATUS='regression_root_audit_completed_no_new_fit'

def main():
    folder=ROOT/'evidence/2026-09-28/v93_regression_audit';assert not (folder/'delivery.json').exists()
    verify=read(DEST/'verification.json');assert verify['status']=='passed' and not verify['prior_bound_files_changed']
    assert verify['source_sha256']==sha(ROOT/'training/v93_verify.py')
    problems=[
        {'id':'current_scalar_correction_path','status':'ruled_out_as_repair','finding':'Exact path: preserving58 inner S repairs retains all500 M regressions; zero M regression preserves0 S repairs.'},
        {'id':'known_input_omission_or_collision','status':'not_observed_in_target','finding':'558 raw labels and94 actual inputs verified; no known fact omissions or global same-R0 label conflict in target; not full semantic sufficiency.'},
        {'id':'cross_component_MS_decision','status':'unresolved','finding':'500 M regressions span69 inputs and60 components; fixed confidence, support and nearest rules fail.'},
        {'id':'unbounded_score_change','status':'mechanism_observed_regularized_training_pending','finding':'U0 to U0R inner absolute MS residual median5.9568 to159.3346;456/500 errors have uncalibrated softmax>=.99. Magnitude and ordering must both be tested.'},
        {'id':'protection_population_and_solver','status':'unresolved','finding':'Old full-fit P holds but does not constrain unseen decisions. New matched2x2 plan registered in report; teacher-unseen component protection and functional regularization not yet fitted.'},
        {'id':'rare_semantic_support','status':'unresolved','finding':'No official operational M/S definitions; scarce independent components remain. No guessed labels or external training data.'},
        {'id':'unsupported_identity_overlap','status':'historical_unresolved','finding':'Old unsupported row45738 pressure failure remains outside ASA correction scope.'},
    ];save(DEST/'problem_register.json',problems)
    execution={'status':STATUS,'actual_classifier_fits':0,'actual_calibration_fits':0,'actual_teacher_fits':0,'selected':None,'model_promoted':False,
        'quality_acceptance':False,'all_issues_solved':False,'new_full_development_replay':False,'platform_used':False,'external_training_data_used':False,
        'original_labels_changed':0,'target_answers_read':False,'inner_M_regressions_traced':500,'inner_S_repairs_traced':58,'exact_input_groups':94,
        'planned_next_classifier_fits':5,'planned_next_fits_executed':0,'scope':verify['scope']};save(DEST/'execution_summary.json',execution)
    summary=('v9.3本轮新增0次分类器拟合、0次校准拟合；完成v9.2新增500条M错误和58条S修复的逐行根因审查。'
        '94种输入没有已知字段遗漏或同输入异标签；456条M错误的未校准分数超过99%。'
        '精确插值证实统一降权无法保58并减少500；简单近邻、置信度和删除字段控制也失败。'
        '下一步固定完整R0，检验训练期输出变化正则与教师未见组件保护的匹配2x2对照，尚未重训。'
        '897历史绑定文件未变，主模型质量未通过，无新模型晋升；最新实际训练仍v9.2，最近完整开发回放仍为v7.9的5947错。')
    cp=ROOT/'mcp_readonly/catalog.json';catalog=read(cp);assert catalog['project']['authoritative_direction_id']=='v92-direction-review'
    for e in catalog['documents']:assert sha(ROOT/e['path'])==e['sha256'],e['path']
    files=[p for p in DEST.rglob('*') if p.is_file()]+list((ROOT/'training').glob('v93*.py'))+[ROOT/DOC]
    bound={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)}
    delivery={**execution,'validation_scope':verify['scope'],'latest_executed_training_delivery':'evidence/2026-09-28/v92_evidence_training/delivery.json',
        'latest_full_development_delivery':'evidence/2026-09-27/v79_execution/delivery.json','latest_full_development_errors':5947,'current_plan':DOC,
        'verification':verify,'problem_register':'artifacts/v93_regression_audit_20260928/problem_register.json','artifact_sha256':bound}
    folder.mkdir(parents=True,exist_ok=True);save(folder/'delivery.json',delivery)
    for relative,link in [('README.md',DOC),('docs/TRAINING_PLAN.md','V93_500_M_REGRESSION_ROOT_CAUSE_AND_PLAN.md')]:
        p=ROOT/relative;old=p.read_text(encoding='utf-8');assert 'v9.3本轮新增' not in old;title,body=old.split('\n',1)
        p.write_text(title+'\n\n当前根因审查与优化方向（2026-09-28）：**'+summary+'** [500条M退化复核与下一步方案]('+link+')。以下为历史阶段。\n'+body,encoding='utf-8')
    for e in catalog['documents']:
        if e['path'] in ['README.md','docs/TRAINING_PLAN.md']:e.update(sha256=sha(ROOT/e['path']),summary=summary)
        if e['id']=='v92-direction-review':e['category']='historical_review'
        if e['id']=='v92-delivery':e['category']='historical_evidence'
    catalog['project'].update(as_of='2026-09-28',current_summary=summary,authoritative_direction_id='v93-direction-review',authoritative_delivery_id='v93-delivery',
        current_direction=['保留完整输入与旧模型；500条M退化和58条S修复为固定回归清单，均不得加入开发答案训练。',
            '下一轮以相同教师/数据/容量对照输出变化正则与教师未见组件保护；先1教师+4分支，失败不扩训。',
            '以各类真实正确数、精确率、修复和负翻转检验；不把数值收敛、训练P零退化或高分当未知组件能力。'],
        known_limits=['本轮0次新训练；尚无通过质量门槛的新模型，根因机制证据不等于优化有效。',
            '当前固定分支沿标量路径无法保58并减500；不能推广为任意分类器的理论上限。',
            '新M错误不含已发现同输入冲突/字段遗漏，但语义充分性和M/S操作定义仍缺依据。',
            'OOF/组件角色重划仍是已观察官方开发，不是新盲测；稀缺两组件无法凑成三种独立角色。',
            '原unsupported身份压力问题、保护优化路径仍未解决；本地MCP测试不证明云端Tunnel连接。'])
    entries=[{'id':'v93-direction-review','title':'v9.3 500条恶意退化根因复核与修正计划','path':DOC,'category':'current_review','summary':summary,
        'keywords':['当前','最新','方向','下一步','500','58','退化','ASA','恶意','可疑','泛化','v9.3']},
        {'id':'v93-delivery','title':'v9.3 冻结诊断交付证据：无新增训练','path':(folder/'delivery.json').relative_to(ROOT).as_posix(),'category':'current_evidence',
         'summary':'558原行、94输入、2092标量区间角色重放；897历史文件保留。训练仍v9.2，无模型晋升。','keywords':['证据','复核','v9.3']}]
    for e in reversed(entries):e['sha256']=sha(ROOT/e['path']);catalog['documents'].insert(0,e)
    save(cp,catalog)
    p=ROOT/'mcp_readonly/tests/test_readonly_mcp.py';s=p.read_text(encoding='utf-8').replace('v92-direction-review','v93-direction-review').replace('v92-delivery','v93-delivery').replace('evidence_and_training_completed_no_model_promoted',STATUS).replace('本轮新增9次分类器拟合','本轮新增0次分类器拟合');p.write_text(s,encoding='utf-8')
    print(__import__('json').dumps({'published':True,'bound_files':len(bound),'new_fits':0,'quality_acceptance':False},ensure_ascii=False))

if __name__=='__main__':main()
