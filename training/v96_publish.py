"""Publish an explicit correction and audited training plan without promoting any model."""
import json
from run_v75 import ROOT, read, save, sha
from v96_protocol_audit import DEST, V95

DOC='docs/V96_ROOT_CAUSE_CORRECTION_AND_TRAINING_PLAN.md'
STATUS='protocol_and_experiment_root_audit_completed_no_new_fit'


def main():
    folder=ROOT/'evidence/2026-09-28/v96_root_cause_audit'
    assert not (folder/'delivery.json').exists()
    verify=read(DEST/'verification.json')
    assert verify['status']=='passed' and verify['new_classifier_fits']==0
    assert verify['source_sha256']==sha(ROOT/'training/v96_verify.py')
    assert verify['prior_bound_files_rehashed']==1031 and not verify['prior_bound_files_changed']
    representation=read(DEST/'representation_crosscheck.json')
    assert representation['R0_max_absolute_feature_difference']==0
    assert not representation['new_input_feature_training_warranted_by_missing_ICMP_fields']
    contract=read(DEST/'behavior_contract_audit.json')
    assert len(contract['eligible_AB_both_class_groups'])==1
    catalog_path=ROOT/'mcp_readonly/catalog.json';catalog=read(catalog_path)
    assert catalog['project']['authoritative_direction_id']=='v95-direction-review'
    for entry in catalog['documents']:assert sha(ROOT/entry['path'])==entry['sha256'],entry['path']
    problems=[
        {'id':'dual_parser_diagnostic_error','status':'corrected_and_raw_replayed',
         'finding':'All 2373 ASA ICMP have actual R0 type/code facts; full raw-to-R0 difference is0. Empty v89 auxiliary maps do not imply model input loss; cancel duplicate feature trial.'},
        {'id':'false_behavior_alignment','status':'contract_fixed_training_not_rerun',
         'finding':'B empty bucket M48 ICMP versus S6 UDP; revised actual-fact contract leaves one16-row cross-role M/S group, one component per cell. Stop old meta objective.'},
        {'id':'teacher_change_and_scope_regression','status':'attributed_and_frozen_control_executed',
         'finding':'Of inner/C/H final negative flips59/436/387,43/426/369 inherited A-only teacher loss. Restoring non-ASA reference yields218/3767/302 errors, still fails old194/3425/290.'},
        {'id':'ICMP_type3_code13_transfer_support','status':'measured_unresolved',
         'finding':'Only V contains type3/code13:180M and685S, each class in one component; no A/B support, but no exact full-R0 class conflict. No code13-to-S or zone-number rule allowed.'},
        {'id':'auxiliary_weight_and_meta_identification','status':'measured_control_required_if_restarted',
         'finding':'Auxiliary per-row coefficient up628x main; gradient norm ratios2.0-117.0. Nearparallel virtual-step gradients do not prove equivalence; no eta0 matched fit exists.'},
        {'id':'locked_metric_execution_boundary','status':'defect_confirmed_future_entrypoint_required',
         'finding':'20 checkpoints computed locked inner/C/H metrics before selection. Selection formula and gradients did not use them; no fresh blind-test claim.'},
        {'id':'underlying_ASA_transfer_quality','status':'unresolved_no_promotion',
         'finding':'Scope restoration still misses historical class/flip guards. Broader training/evaluation exposure must match before interpreting method gain.'}]
    save(DEST/'problem_register.json',problems)
    execution={'status':STATUS,'actual_classifier_fits':0,'actual_calibration_fits':0,
        'latest_actual_training_version':'v9.5','new_training_executed':False,
        'model_promoted':False,'quality_acceptance':False,'old_model_retained':True,'all_issues_solved':False,
        'raw_to_R0_records_replayed':2373,'frozen_nonASA_scope_control_executed':True,
        'new_full_development_replay':False,'platform_used':False,'external_training_data_used':False,
        'original_labels_changed':0,'target_answers_read':False,'scope':verify['scope']}
    save(DEST/'execution_summary.json',execution)
    summary=('v9.6本轮新增0次分类器拟合、0次校准拟合；纠正v9.5根因归属。'
        '2373条ICMP原文回放到实际R0差异0，type/code早已保留；空辅助组混合ICMP与UDP。'
        '恢复非ASA旧输出后H错误643降到302，仍差于旧参照290；C仍3767错。'
        '685条V区S对应子类型在A/B零支持，尚未解决。1031历史绑定文件未变。'
        '最新真实训练仍为v9.5，质量验收未通过，无新模型晋升；最近完整开发回放仍为v7.9的5947错。')
    files=[p for p in DEST.rglob('*') if p.is_file()]+list((ROOT/'training').glob('v96*.py'))+[ROOT/DOC]
    bound={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)}
    delivery={**execution,'validation_scope':verify['scope'],'verification':verify,
        'latest_executed_training_delivery':'evidence/2026-09-28/v95_four_arm_training/delivery.json',
        'latest_full_development_delivery':'evidence/2026-09-27/v79_execution/delivery.json',
        'latest_full_development_errors':5947,'current_plan':DOC,
        'problem_register':(DEST/'problem_register.json').relative_to(ROOT).as_posix(),
        'artifact_sha256':bound}
    folder.mkdir(parents=True,exist_ok=True);save(folder/'delivery.json',delivery)
    for relative,link in [('README.md',DOC),('docs/TRAINING_PLAN.md','V96_ROOT_CAUSE_CORRECTION_AND_TRAINING_PLAN.md')]:
        p=ROOT/relative;old=p.read_text(encoding='utf-8');assert 'v9.6本轮新增' not in old
        title,body=old.split('\n',1)
        p.write_text(title+'\n\n当前审查与方案（2026-09-28）：**'+summary+'** [根因纠正与训练方案]('+link+')。下方旧阶段文字由本次明确勘误覆盖。\n'+body,encoding='utf-8')
    for entry in catalog['documents']:
        if entry['path'] in ['README.md','docs/TRAINING_PLAN.md']:entry.update(sha256=sha(ROOT/entry['path']),summary=summary)
        if entry['id']=='v95-direction-review':entry['category']='historical_review'
        if entry['id']=='v95-delivery':entry['category']='historical_evidence'
    catalog['project'].update(as_of='2026-09-28',current_summary=summary,
        authoritative_direction_id='v96-direction-review',authoritative_delivery_id='v96-delivery',
        current_direction=[
            '取消重复ICMP特征实验：实际R0已保留type/code，2373条原文回放差异0；辅助解析与模型输入必须分清。',
            '停止空字典行为对齐和旧三组高权重元目标；完整协议事实资格仅剩16条TCP组，每个角色类别一个组件。',
            '下一轮先用共同A+B教师、同样本/初值/预算比较普通与幅度约束ASA分支，非ASA保留旧输出；教师收益、分支收益、历史安全分表报告。',
            '分开报告支持内与零支持迁移；type3/code13只有V180M+685S，不能据V标签补规则或承诺已解决。',
            '训练进程不能载入旧锁定角色标签；冻结选择后独立诊断，历史开发不能重称盲测。'],
        known_limits=[
            '本轮是零拟合审查和冻结对照，不是新训练完成；最新训练v9.5，质量仍未通过。',
            '非ASA范围恢复在历史H修复341条，但C/H仍比旧参照差，未晋升。',
            'type3/code13粗语义相同不等于完整输入冲突；全R0异标签冲突0，跨主体判据仍不足。',
            '旧内层/C/H在v9.5检查点阶段提前计算；未发现选模公式或梯度使用，但隔离执行不完整。',
            '原有unsupported身份问题、真实外部迁移、官方M/S判定语义仍未解决；本地MCP测试不证明云端Tunnel。'])
    entries=[
        {'id':'v96-direction-review','title':'v9.6 根因纠正与训练方案','path':DOC,'category':'current_review','summary':summary,
         'keywords':['当前','最新','方向','结果','下一步','ASA','ICMP','恶意','可疑','教师','泛化','v9.6']},
        {'id':'v96-delivery','title':'v9.6 原文回放与实验根因审查证据','path':(folder/'delivery.json').relative_to(ROOT).as_posix(),
         'category':'current_evidence','summary':'零拟合；2373条ICMP原文重建R0；错误流向与冻结范围对照；1031历史文件未变。',
         'keywords':['证据','纠正','复核','v9.6']}]
    for entry in reversed(entries):entry['sha256']=sha(ROOT/entry['path']);catalog['documents'].insert(0,entry)
    save(catalog_path,catalog)
    tests=ROOT/'mcp_readonly/tests/test_readonly_mcp.py'
    source=tests.read_text(encoding='utf-8').replace('v95-direction-review','v96-direction-review').replace('v95-delivery','v96-delivery')
    source=source.replace('v95_training_executed_failed_transfer_gates',STATUS).replace('本轮新增5次分类器拟合','本轮新增0次分类器拟合')
    tests.write_text(source,encoding='utf-8')
    for p,h in bound.items():assert sha(ROOT/p)==h
    print(json.dumps({'published':True,'bound_files':len(bound),'catalog_documents':len(catalog['documents']),
        'new_classifier_fits':0,'quality_acceptance':False,'delivery_sha256':sha(folder/'delivery.json')},ensure_ascii=False))


if __name__=='__main__':main()
