"""Publish a no-fit root audit without changing any historical bound result."""
import json
from v89_common import ROOT, read, save, sha
from v98_frozen_audit import DEST, V97

DOC='docs/V98_BOUNDARY_ROOT_AUDIT_AND_NEXT_PLAN.md'
STATUS='v98_frozen_root_audit_completed_no_new_fit'


def main():
    folder=ROOT/'evidence/2026-09-28/v98_boundary_audit'
    assert not (folder/'delivery.json').exists()
    verify=read(DEST/'verification.json')
    assert verify['status']=='passed' and verify['prior_bound_files_rehashed']==1099
    assert verify['source_sha256']==sha(ROOT/'training/v98_verify.py')
    assert verify['placeholder_affected_rows_raw_replayed']==verify['affected_rows_with_nested_original_placeholder']==4307
    pop=read(DEST/'population_surface_controls.json')
    assert sum(c['regressions'] for c in pop['controls']['literal_only']['populations'])==439
    assert sum(c['regressions'] for c in pop['controls']['whitespace_only']['populations'])==28
    catalog_path=ROOT/'mcp_readonly/catalog.json';catalog=read(catalog_path)
    assert catalog['project']['authoritative_direction_id']=='v97-direction-review'
    for d in catalog['documents']:assert sha(ROOT/d['path'])==d['sha256'],d['path']
    summary=('v9.8第一性原理审查完成：本轮新增0次分类器拟合、0次校准拟合。'
        '4307条ASA存在嵌套脱敏残留且本批全为S；冻结模型只去残留字样新增439条S错误，空格单控新增28条。'
        'A+B训练251错中229条无完整输入冲突；63条相关训练S仍错6条。'
        '历史全拟合693条负翻转中691在旧教师见过而新教师未见的V。'
        '下一轮按占位符规范化与训练侧停止规则做匹配2x2；未重训。'
        '1099历史绑定文件未变。最新实际训练仍v9.7，质量未通过，无新模型晋升；最近完整开发回放仍为v7.9的5947错。')
    problems=[
        {'id':'nested_placeholder_shortcut','status':'confirmed_sensitivity_not_fixed',
         'finding':'4307 affected original ASA rows all currently S; literal-only frozen control loses439 S, whitespace-only28. Direct inference deletion rejected.'},
        {'id':'incomplete_fit','status':'quantified_unresolved',
         'finding':'AB actual251 errors vs observed exact-input floor22;229 nonconflict errors. Six supported training S errors are nonconflicting. Step200 was not convergence evidence.'},
        {'id':'independent_behavior_support','status':'unresolved',
         'finding':'Each of four behaviors behind12 V supported errors has one S training component;685 ICMP S errors have zero AB subtype support.'},
        {'id':'new_M_boundary_crossing','status':'localized_not_fixed',
         'finding':'Four M losses share one unsupported R0, crossing residual scale0.742. Scale0.5 V posthoc repairs16 S with0 newM; not selected/calibrated/promoted.'},
        {'id':'unequal_exposure_gate','status':'next_contract_corrected_historic_preserved',
         'finding':'693 old-fit NF:2A,0B,691V;687V already lost at teacher replacement,4 newly lost at branch. V/C/H failures remain even without confounded gate.'}]
    save(DEST/'problem_register.json',problems)
    contract={'status':'next_experiment_plan_not_executed','official_data_only':True,'labels_changed':0,
        'input_factor':['current_R0','narrow_nested_placeholder_normalization_after_collision_audit'],
        'optimization_factor':['fixed200','train_only_stopping_max2000'],
        'max_teacher_fits':2,'max_branch_fits':4,'max_calibration_fits':0,
        'fixed':{'roles':'v95 A+B gradient/V development; historic inner/C/H diagnosis','seed':9701,
            'network':'64/16 residual','optimizer':'Adam','lr':.003,'teacher_correct_soft_protection':.01,'MAG_output_penalty':.01},
        'training_stopping':{'interval':25,'successive_checks':5,'relative_objective_change':1e-4,'gradient_inf_max':1e-5,
            'errors_unchanged':True,'max_steps':2000,'V_used_for_stopping':False,'budget_exhaustion_is_convergence':False},
        'teacher_exposure':'each input view has its own same A+B teacher; none uses V gradients',
        'diagnostic_only_comparison':'AB candidate versus historic ABV teacher on old fit is exposure-confounded, cannot establish method degradation',
        'promotion':'No current promotion. Matched exposure, per-class quality and original-correct safety must pass; historical failed models remain failed.',
        'support_reporting':['multiple_component_same_class','single_component_same_class','zero_same_class','incomplete_behavior'],
        'stop_expansion':'No independent quality improvement or any unmet safety gate; do not replace with adaptive V weighting.',
        'blind_test_claim':False,'raw_inputs_preserved':True,'current_report':DOC}
    save(DEST/'next_experiment_contract.json',contract)
    execution={'status':STATUS,'actual_classifier_fits':0,'actual_calibration_fits':0,'selected_arm':None,
        'model_promoted':False,'quality_acceptance':False,'old_model_retained':True,'all_issues_solved':False,
        'new_full_development_replay':False,'platform_used':False,'external_training_data_used':False,
        'original_labels_changed':0,'target_answers_read':False,'scope':verify['scope']}
    save(DEST/'execution_summary.json',execution)
    files=[p for p in DEST.rglob('*') if p.is_file()]+list((ROOT/'training').glob('v98*.py'))+[ROOT/DOC]
    bound={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)}
    delivery={**execution,'validation_scope':verify['scope'],'verification':verify,
        'latest_executed_training_delivery':'evidence/2026-09-28/v97_matched_training/delivery.json',
        'latest_full_development_delivery':'evidence/2026-09-27/v79_execution/delivery.json',
        'latest_full_development_errors':5947,'current_plan':DOC,
        'problem_register':(DEST/'problem_register.json').relative_to(ROOT).as_posix(),'artifact_sha256':bound}
    folder.mkdir(parents=True,exist_ok=True);save(folder/'delivery.json',delivery)
    for path,link in [('README.md',DOC),('docs/TRAINING_PLAN.md','V98_BOUNDARY_ROOT_AUDIT_AND_NEXT_PLAN.md')]:
        p=ROOT/path;content=p.read_text(encoding='utf-8');assert 'v9.8第一性原理审查完成' not in content
        head,body=content.split('\n',1)
        p.write_text(head+'\n\n当前根因审查与方案（2026-09-28）：**'+summary+'** [脱敏残留、边界与下一轮对照]('+link+')。以下保留历史阶段。\n'+body,encoding='utf-8')
    for d in catalog['documents']:
        if d['path'] in ['README.md','docs/TRAINING_PLAN.md']:d.update(sha256=sha(ROOT/d['path']),summary=summary)
        if d['id']=='v97-direction-review':d['category']='historical_review'
        if d['id']=='v97-delivery':d['category']='historical_evidence'
    catalog['project'].update(as_of='2026-09-28',current_summary=summary,
        authoritative_direction_id='v98-direction-review',authoritative_delivery_id='v98-delivery',
        current_direction=[
            '优先审查嵌套脱敏残留的窄范围规范化；原文保留，先检查新增输入冲突，不能直接修改当前模型的推理输入。',
            '同A+B训练曝光下做视图规范化×训练侧停止规则2x2，最多两个教师和四个分支；本轮未训练。',
            '229条无输入冲突训练错误与零支持迁移分开；每类报告召回、精确率、正负翻转和独立组件数。',
            '不同训练曝光的历史模型只作诊断。最终相同曝光比较仍保留原正确保护和来源验证，不追认v9.7通过。'],
        known_limits=[
            '4307条残留与S相关、439条冻结翻错不是规范化后重新训练的收益；尚无新训练质量结果。',
            '685条ICMP S仍0正确；单组件不能拆开伪造跨组件支持，粗行为相同不代表同标签。',
            'V/inner/C/H均长期观察，0.5缩放是事后诊断，不是盲测或已选超参数。',
            '没有新的完整开发回放、平台训练或官方提交；MCP本地验证不证明云端Tunnel已连通。'])
    entries=[{'id':'v98-direction-review','title':'v9.8 脱敏残留与拟合验证根因审查','path':DOC,
        'category':'current_review','summary':summary,'keywords':['当前','最新','方向','下一步','ASA','M/S','脱敏','根因','v9.8']},
        {'id':'v98-delivery','title':'v9.8 零拟合对照和原文核验凭据',
        'path':(folder/'delivery.json').relative_to(ROOT).as_posix(),'category':'current_evidence',
        'summary':'零拟合；4307原文和冻结控制复核，1099历史文件未变，未晋升模型。','keywords':['证据','核验','零拟合','v9.8']}]
    for d in reversed(entries):d['sha256']=sha(ROOT/d['path']);catalog['documents'].insert(0,d)
    save(catalog_path,catalog)
    test=ROOT/'mcp_readonly/tests/test_readonly_mcp.py';source=test.read_text(encoding='utf-8')
    source=source.replace('v97-direction-review','v98-direction-review').replace('v97-delivery','v98-delivery')
    source=source.replace('v97_three_fits_executed_failed_transfer_and_safety_gates',STATUS)
    source=source.replace('新增3次分类器拟合','本轮新增0次分类器拟合');test.write_text(source,encoding='utf-8')
    for path,h in bound.items():assert sha(ROOT/path)==h
    print(json.dumps({'published':True,'bound_files':len(bound),'catalog_documents':len(catalog['documents']),
        'actual_classifier_fits':0,'quality_acceptance':False,'delivery_sha256':sha(folder/'delivery.json')},ensure_ascii=False))


if __name__=='__main__':main()
