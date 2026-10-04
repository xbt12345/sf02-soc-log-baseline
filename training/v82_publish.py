"""Publish actual bounded fits without overwriting any frozen earlier result."""
import json
import platform
import sys
from pathlib import Path
from importlib.metadata import version
from run_v75 import ROOT, read, save, sha
from v82_capacity import DEST

DOC='docs/V82_CAPACITY_TRAINING_RESULTS.md'
SUMMARY='v8.2本轮新增6次对照拟合、0次校准拟合，16个参数状态完成重载核验。完整同输入非线性交互修复18/457条一致标签S，但C少30错、H多14错，困难条件M退化及ASA零M迁移未通过，无新模型晋升。未触发扩训或最终全量训练；最近完整开发回放仍为v7.9的5947错。'
STATUS='capacity_training_completed_not_promoted'


def main():
    out=ROOT/'evidence/2026-09-27/v82_capacity';out.mkdir(exist_ok=True)
    if (out/'delivery.json').exists():raise FileExistsError('Already published')
    prior={}
    for p in ['evidence/2026-09-27/v79_execution/delivery.json','artifacts/v80_attribution_20260927/review_receipt.json','evidence/2026-09-27/v81_diagnosis/delivery.json']:
        prior.update(read(ROOT/p)['artifact_sha256'])
    failed=[p for p,h in prior.items() if sha(ROOT/p)!=h]
    assert not failed,failed
    verification=read(DEST/'verification.json');assert verification['status']=='passed'
    assert not read(DEST/'continuation.json')['shadow_refit_authorized_by_evidence']
    files=list(DEST.rglob('*'))+[ROOT/DOC]+list((ROOT/'training').glob('v82_*.py'))+[ROOT/'training/test_v82_capacity.py']
    delivery={'status':STATUS,'quality_acceptance':False,'all_issues_solved':False,
        'actual_new_fits':6,'actual_classifier_fits':6,'actual_calibration_fits':0,'parameter_snapshots':16,
        'new_full_data_final_fit':False,'new_full_development_replay':False,'platform_used':False,
        'external_training_data_used':False,'pseudo_labels_used':False,'four_arm_training_executed':False,
        'new_flow_features_used_by_classifier':False,'target_answers_read':False,
        'validation_scope':'Matched original-R0 linear/nonlinear diagnostic training + ASA/CEF zero-M retraining on previously inspected official roles. No independent blind environment validation or full-data promotion.',
        'latest_full_development_delivery':'evidence/2026-09-27/v79_execution/delivery.json','latest_full_development_errors':5947,
        'previous_bound_files_rehashed':len(prior),'previous_bound_files_changed':failed,
        'protected_original_class_counts':[1899723,111728,45420],
        'findings':{'fit_errors_linear':491,'fit_errors_nonlinear':485,'same_label_457_correct':18,
                    'C_errors_linear':3509,'C_errors_nonlinear':3479,'H_errors_linear':271,'H_errors_nonlinear':285,
                    'opposite_support_M_reference_correct':18,'opposite_support_M_candidate_correct':14,
                    'ASA_zero_M_H_correct':0,'ASA_zero_M_H_support':16226,'VPC_train_M_support':0},
        'implementation_verification':verification,'continuation':read(DEST/'continuation.json'),
        'runtime':{'python':sys.version,'architecture':platform.machine(),
                   'packages':{name:version(name) for name in ['numpy','scipy','scikit-learn','pandas','pyarrow','joblib']}},
        'artifact_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(files)) if p.is_file()}}
    save(out/'delivery.json',delivery)
    for relative,link in [('README.md',DOC),('docs/TRAINING_PLAN.md','V82_CAPACITY_TRAINING_RESULTS.md')]:
        p=ROOT/relative;oldtext=p.read_text(encoding='utf-8');title,body=oldtext.split('\n',1)
        assert 'v8.2本轮新增' not in oldtext
        p.write_text(title+'\n\n当前执行与审查（2026-09-27）：**'+SUMMARY+'** [同输入容量训练、迁移结果与停止决定]('+link+')。以下为历史阶段。\n'+body,encoding='utf-8')
    cp=ROOT/'mcp_readonly/catalog.json';catalog=read(cp)
    for entry in catalog['documents']:
        if entry['path'] in ['README.md','docs/TRAINING_PLAN.md']:entry.update(sha256=sha(ROOT/entry['path']),summary=SUMMARY)
        else:assert sha(ROOT/entry['path'])==entry['sha256'],entry['path']
        if entry['id']=='v81-direction-review':entry['category']='historical_review'
        if entry['id']=='v81-delivery':entry['category']='historical_evidence'
    catalog['project'].update(current_summary=SUMMARY,as_of='2026-09-27',authoritative_delivery_id='v82-delivery',authoritative_direction_id='v82-direction-review',
        current_direction=['停止本轮256维完整R0核交互的原样扩训，不把内部总分改善冒充迁移成功。',
                           '继续定位439条一致标签S残余及4条条件M退化；先证明可观察跨主体判据与有效监督，再登记四臂。',
                           'VPC无M训练支持单独标为未解决；内部困难条件、缺类迁移与扩训稳定性通过后才最终全量训练。'],
        known_limits=['六次拟合完成，主非线性全部C状态未通过；线性C候选H恶意保护失败。',
                      'C少30错但H多14错，ASA零M下M0/16226；CEF只有12条H-M且无S，不能代替VPC验证。',
                      'R0原列保留，原件未改；字节袋长关系限制及有效监督缺口仍存在。',
                      '没有新的全量训练/开发回放或独立盲测，主模型质量未通过。'])
    for entry in reversed([
        {'id':'v82-direction-review','title':'v8.2 同输入容量训练、缺类迁移与停止决定','path':DOC,'category':'current_review',
         'summary':SUMMARY,'keywords':['当前','最新','方向','下一步','训练','恶意','可疑','监督','泛化','根因','v8.2']},
        {'id':'v82-delivery','title':'v8.2 六次真实对照训练与未晋升交付','path':'evidence/2026-09-27/v82_capacity/delivery.json',
         'category':'current_evidence','summary':'6次拟合、16状态、0次校准。主容量/ASA缺类迁移失败，未触发最终全量训练。','keywords':['执行','证据','训练','v8.2']} ]):
        entry['sha256']=sha(ROOT/entry['path']);catalog['documents'].insert(0,entry)
    save(cp,catalog)
    tests=ROOT/'mcp_readonly/tests/test_readonly_mcp.py';text=tests.read_text(encoding='utf-8')
    text=text.replace('v81-direction-review','v82-direction-review').replace('v81-delivery','v82-delivery')
    text=text.replace('diagnostic_training_and_root_cause_review_completed_not_promoted',STATUS)
    text=text.replace('本轮新增1次诊断训练','本轮新增6次对照拟合')
    tests.write_text(text,encoding='utf-8')
    print(json.dumps({'new_fits':6,'new_bound_files':len(delivery['artifact_sha256']),'prior_bound_files_verified':len(prior),'quality_acceptance':False},ensure_ascii=False))


if __name__=='__main__':main()
