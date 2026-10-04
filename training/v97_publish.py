"""Publish 3-fit matched trial and failed promotion gate to the read-only catalog."""
import json
from run_v75 import ROOT, read, save, sha
from v97_prepare import DEST

DOC='docs/V97_MATCHED_TRAINING_RESULTS_AND_DECISION.md'
STATUS='v97_three_fits_executed_failed_transfer_and_safety_gates'


def main():
    folder=ROOT/'evidence/2026-09-28/v97_matched_training'
    assert not (folder/'delivery.json').exists()
    verify=read(DEST/'verification.json');selection=read(DEST/'selection.json');diagnosis=read(DEST/'diagnosis.json')
    support=read(DEST/'support_audit.json')
    assert verify['status']=='passed' and verify['actual_classifier_fits']==3
    assert verify['prior_bound_files_rehashed']==1058 and not verify['prior_bound_files_changed']
    assert verify['source_sha256']==sha(ROOT/'training/v97_verify.py')
    assert selection['selected_arm'] is None and diagnosis['selected_arm'] is None
    assert not diagnosis['any_historic_safety_passed']
    assert selection['teacher_and_arms']['MAG']['correct']==[0,6154,591]
    assert support['MAG_S_errors']['same_label_AB_behavior_supported_rows']==12
    catalog_path=ROOT/'mcp_readonly/catalog.json';catalog=read(catalog_path)
    assert catalog['project']['authoritative_direction_id']=='v96-direction-review'
    for item in catalog['documents']:assert sha(ROOT/item['path'])==item['sha256'],item['path']
    problems=[
        {'id':'teacher_training_coverage','status':'matched_and_measured',
         'finding':'A+B teacher converged; A/B total errors339/107. V-ASA error763 and S recall561/1324; improved exposed fit does not establish transfer.'},
        {'id':'M_S_tradeoff_on_V','status':'failed_selection_gate',
         'finding':'ERM/MAG V-ASA errors743/737, S correct585/591, but both lose4 M originally correct. No arm selected.'},
        {'id':'magnitude_generalization','status':'small_uncertain_gain',
         'finding':'MAG adds6 correct S versus matched ERM across3 components; 2000-component bootstrap q025=0, no robust positive lower bound.'},
        {'id':'historic_safety','status':'failed',
         'finding':'With nonASA historic output frozen, ERM/MAG C errors3591/3585 versus3425 historic; C negative flips220/210. H has30/40 negative flips.'},
        {'id':'dominant_ICMP_support','status':'unresolved',
         'finding':'V type3/code13 M180 all correct, S685 all wrong; A/B lacks this behavior. R0 already has ICMP facts.'},
        {'id':'remaining_V_S_behavior_support','status':'quantified',
         'finding':'MAG has733 V-ASA S errors; only12 have same-class A+B behavior support, none same exact R0.'},
        {'id':'four_new_V_M_errors','status':'localized_unresolved',
         'finding':'Four original M errors share one TCP port5046 component and one R0, with zero same-behavior A+B labels.'},
        {'id':'unsupported_identity_and_external_transfer','status':'unresolved',
         'finding':'No new full development replay, external distribution test or official submission.'}]
    save(DEST/'problem_register.json',problems)
    execution={'status':STATUS,'actual_classifier_fits':3,'actual_teacher_fits':1,'actual_branch_fits':2,
        'actual_calibration_fits':0,'selected_arm':None,'model_promoted':False,'quality_acceptance':False,
        'old_model_retained':True,'all_issues_solved':False,'new_full_development_replay':False,
        'platform_used':False,'external_training_data_used':False,'original_labels_changed':0,
        'target_answers_read':False,'postselection_historic_diagnosis_executed':True,
        'scope':verify['scope']}
    save(DEST/'execution_summary.json',execution)
    summary=('v9.7严格匹配训练完成：新增3次分类器拟合、0次校准拟合。共同A+B教师与ERM/MAG两臂均执行200步；'
        'V ASA教师763错，ERM743错，MAG737错；MAG的S判对591/1324，但新增4条M错误，未过选模门槛。'
        'MAG比ERM多对6条S、组件重抽样下界0；旧C区MAG3585错、210条负翻转。'
        '685条ICMP S仍全错。1058历史绑定文件未变；质量验收未通过，无新模型晋升。'
        '最近完整开发回放仍为v7.9的5947错。')
    files=[p for p in DEST.rglob('*') if p.is_file()]+list((ROOT/'training').glob('v97*.py'))+[ROOT/DOC]
    bound={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)}
    delivery={**execution,'validation_scope':verify['scope'],'verification':verify,
        'latest_executed_training_delivery':(folder/'delivery.json').relative_to(ROOT).as_posix(),
        'latest_full_development_delivery':'evidence/2026-09-27/v79_execution/delivery.json',
        'latest_full_development_errors':5947,'current_plan':DOC,
        'problem_register':(DEST/'problem_register.json').relative_to(ROOT).as_posix(),
        'artifact_sha256':bound}
    folder.mkdir(parents=True,exist_ok=True);save(folder/'delivery.json',delivery)
    for relative,link in [('README.md',DOC),('docs/TRAINING_PLAN.md','V97_MATCHED_TRAINING_RESULTS_AND_DECISION.md')]:
        p=ROOT/relative;old=p.read_text(encoding='utf-8');assert 'v9.7严格匹配训练完成' not in old
        head,body=old.split('\n',1)
        p.write_text(head+'\n\n当前训练结果（2026-09-28）：**'+summary+'** [匹配训练结果与后续判定]('+link+')。下方为历史阶段。\n'+body,encoding='utf-8')
    for item in catalog['documents']:
        if item['path'] in ['README.md','docs/TRAINING_PLAN.md']:item.update(sha256=sha(ROOT/item['path']),summary=summary)
        if item['id']=='v96-direction-review':item['category']='historical_review'
        if item['id']=='v96-delivery':item['category']='historical_evidence'
    catalog['project'].update(as_of='2026-09-28',current_summary=summary,
        authoritative_direction_id='v97-direction-review',authoritative_delivery_id='v97-delivery',
        current_direction=[
            '共同A+B教师和匹配ERM/MAG训练已完成，V ASA的737错误并未达标：30条S修复伴随4条新M错误，未选模型。',
            'MAG相对ERM只多6条S，3组件且组件重抽样下界0；停止扩大这一幅度约束方法。',
            '非ASA已冻结历史输出，但C/H旧正确项仍受伤害；不能把局部总错下降当作安全晋升。',
            '记录733条V S错误中的同类训练支持：仅12条有A+B相同行为S样本；685条ICMP集中S仍0正确。',
            '下一步先研究有真实跨组件监督的少数切片及四条新M的边界证据；任何新机制先注册单一差异及历史保护。'],
        known_limits=[
            'V和旧内层/C/H均为长期观察的官方开发数据；不是盲测或真实外部泛化证明。',
            'A+B局部训练排除V；正式全官方训练时应纳入V原标签，但纳入后不得用它充作迁移测试。',
            '四条新M来自同一组件/R0，六条额外S来自三个组件；行级收益不可当独立样本。',
            '同类行为支持缺失不证明错误不可学；官方M/S判定语义和外部迁移仍需独立证据。',
            '没有新完整开发回放、平台运行或官方成绩；本地MCP检验不证明云端Tunnel。'])
    entries=[
        {'id':'v97-direction-review','title':'v9.7 共同教师与 ASA 双臂结果','path':DOC,
         'category':'current_review','summary':summary,
         'keywords':['当前','最新','方向','结果','下一步','A+B','教师','ASA','ERM','MAG','M/S','v9.7']},
        {'id':'v97-delivery','title':'v9.7 三次拟合与安全复核证据',
         'path':(folder/'delivery.json').relative_to(ROOT).as_posix(),'category':'current_evidence',
         'summary':'共同教师加两臂；V无合格模型；1058历史文件未变，质量未通过。',
         'keywords':['证据','拟合','复核','v9.7']}]
    for item in reversed(entries):item['sha256']=sha(ROOT/item['path']);catalog['documents'].insert(0,item)
    save(catalog_path,catalog)
    tests=ROOT/'mcp_readonly/tests/test_readonly_mcp.py';source=tests.read_text(encoding='utf-8')
    source=source.replace('v96-direction-review','v97-direction-review').replace('v96-delivery','v97-delivery')
    source=source.replace('protocol_and_experiment_root_audit_completed_no_new_fit',STATUS)
    source=source.replace('本轮新增0次分类器拟合','新增3次分类器拟合')
    tests.write_text(source,encoding='utf-8')
    for p,h in bound.items():assert sha(ROOT/p)==h
    print(json.dumps({'published':True,'bound_files':len(bound),'catalog_documents':len(catalog['documents']),
        'actual_classifier_fits':3,'quality_acceptance':False,'delivery_sha256':sha(folder/'delivery.json')},ensure_ascii=False))


if __name__=='__main__':main()
