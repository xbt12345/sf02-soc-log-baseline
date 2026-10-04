"""Bind the five-fit trial and publish failed quality decision to read-only MCP."""
import json
from run_v75 import ROOT, read, save, sha
from v95_prepare import DEST

DOC = 'docs/V95_FOUR_ARM_TRAINING_RESULTS_AND_FAILURE_ANALYSIS.md'
STATUS = 'v95_training_executed_failed_transfer_gates'


def main():
    folder = ROOT / 'evidence/2026-09-28/v95_four_arm_training'
    assert not (folder / 'delivery.json').exists()
    verify = read(DEST / 'verification.json')
    assert verify['status'] == 'passed' and verify['actual_classifier_fits'] == 5
    assert not verify['prior_bound_files_changed']
    assert verify['source_sha256'] == sha(ROOT / 'training/v95_verify.py')
    catalog_path = ROOT / 'mcp_readonly/catalog.json'
    catalog = read(catalog_path)
    assert catalog['project']['authoritative_direction_id'] == 'v94-direction-review'
    for entry in catalog['documents']:
        assert sha(ROOT / entry['path']) == entry['sha256'], entry['path']
    selection = read(DEST / 'selection.json')
    diagnosis = read(DEST / 'diagnosis.json')
    failure = read(DEST / 'failure_audit.json')
    assert selection['actual_classifier_fits'] == 5
    assert selection['V_selected']['arm'] == 'E10' and selection['V_selected']['step'] == 200
    assert not diagnosis['historic_all_guards_passed'] and not diagnosis['two_rotation_training_triggered']
    assert failure['V_S_top_error_components'][0]['rows'] == 685
    problems = [
        {'id': 'weak_A_only_teacher_transfer', 'status': 'measured_not_repaired',
         'finding': 'A teacher V has755 errors, all S; 685 S rows from one V component remain wrong. Old v85 teacher saw V, so its50 V errors are not an independent transfer comparator.'},
        {'id': 'cross_component_objective', 'status': 'tested_no_qualified_gain',
         'finding': 'Meta-only E01 has V737 at50 but8 training P negative flips; E11 V739 at50. Neither repairs the8 S in the parsed denial group or685 S in the no-known-facts group.'},
        {'id': 'magnitude_control', 'status': 'small_local_gain_no_promotion',
         'finding': 'E10 V737, 18 S repairs, zero new V errors; versus matched E00 only2 more correct in one component, component bootstrap includes0.'},
        {'id': 'historic_correctness', 'status': 'failed',
         'finding': 'Selected E10 has59 inner,436 C,387 H and743 old-fit negative flips versus historic v85. Only2 of old58 S repairs remain; no promotion.'},
        {'id': 'dominant_ICMP_support', 'status': 'specific_gap_found_unresolved',
         'finding': '685 V S in one component, two R0 inputs, no exact A/B training S; R0 retains ICMP type3/code13 bytes, but known fact parser yields empty map. No official semantic rule or independent same-label coverage proved.'},
        {'id': 'unsupported_identity_overlap', 'status': 'historical_unresolved',
         'finding': 'Previous unsupported row45738 problem not addressed by ASA-only branch.'}
    ]
    save(DEST / 'problem_register.json', problems)
    execution = {'status': STATUS, 'actual_classifier_fits': 5, 'actual_teacher_fits': 1,
                 'actual_branch_fits': 4, 'actual_calibration_fits': 0,
                 'V_selected': selection['V_selected'], 'selected_for_promotion': None,
                 'model_promoted': False, 'quality_acceptance': False,
                 'old_model_retained': True, 'all_issues_solved': False,
                 'new_full_development_replay': False, 'platform_used': False,
                 'external_training_data_used': False, 'original_labels_changed': 0,
                 'target_answers_read': False, 'two_rotation_training_executed': False,
                 'scope': verify['scope']}
    save(DEST / 'execution_summary.json', execution)
    summary = ('v9.5本轮新增5次分类器拟合、0次校准拟合；四臂训练已执行。A教师V区755错，'
               'E10最佳737错，仅修复18条S且较匹配ERM多对2条，差异集中1组件。'
               '旧内层/C/H分别有59/436/387条负翻转，685条来自单组件的V区S仍全错。'
               '947历史绑定文件未变；质量验收未通过，无新模型晋升，正式模型保持原状。'
               '最近完整开发回放仍为v7.9的5947错。')
    files = [p for p in DEST.rglob('*') if p.is_file()] + list((ROOT / 'training').glob('v95*.py')) + [ROOT / DOC]
    bound = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(files)}
    delivery = {**execution, 'validation_scope': verify['scope'],
                'latest_executed_training_delivery': (folder / 'delivery.json').relative_to(ROOT).as_posix(),
                'latest_full_development_delivery': 'evidence/2026-09-27/v79_execution/delivery.json',
                'latest_full_development_errors': 5947, 'current_plan': DOC, 'verification': verify,
                'problem_register': (DEST / 'problem_register.json').relative_to(ROOT).as_posix(),
                'artifact_sha256': bound}
    folder.mkdir(parents=True, exist_ok=True)
    save(folder / 'delivery.json', delivery)
    for relative, link in [('README.md', DOC), ('docs/TRAINING_PLAN.md', 'V95_FOUR_ARM_TRAINING_RESULTS_AND_FAILURE_ANALYSIS.md')]:
        p = ROOT / relative
        old = p.read_text(encoding='utf-8')
        assert 'v9.5本轮新增' not in old
        title, body = old.split('\n', 1)
        p.write_text(title + '\n\n当前训练与审查结论（2026-09-28）：**' + summary +
                     '** [四臂训练结果与失败诊断](' + link + ')。以下为历史阶段。\n' + body, encoding='utf-8')
    for entry in catalog['documents']:
        if entry['path'] in ['README.md', 'docs/TRAINING_PLAN.md']:
            entry.update(sha256=sha(ROOT / entry['path']), summary=summary)
        if entry['id'] == 'v94-direction-review': entry['category'] = 'historical_review'
        if entry['id'] == 'v94-delivery': entry['category'] = 'historical_evidence'
    catalog['project'].update(as_of='2026-09-28', current_summary=summary,
        authoritative_direction_id='v95-direction-review', authoritative_delivery_id='v95-delivery',
        current_direction=[
            '本轮训练完成但不晋升：V E10相对A教师修复18条S，相对匹配ERM仅多2条，且历史inner/C/H出现59/436/387负翻转。',
            '停止本组扩训；保留旧正式模型和旧完整回放结果，不能用A教师V收益证明整体模型提升。',
            '先审查685条单组件ICMP S的type/code事实解析与A/B独立对照；在训练支持充分时才注册下一次定向表达实验。',
            '继续按M/S逐类正确数、负翻转、组件覆盖和完整开发回放判断真实收益。'],
        known_limits=[
            '5次拟合通过执行复核，但质量门槛失败，无新模型晋升或官方成绩。',
            'V被用来选20个预定检查点，是已观察开发；历史v85教师见过V，不能拿它在V的表现做独立迁移基准。',
            '685条S重复属于一个组件两种R0；空已知事实不等于原文字节空或攻击语义不可辨。',
            '辅助组仅3个，其中一组已知事实为空、一组仅16行；元目标未修复预定对照行为。',
            '旧unsupported问题、官方M/S语义与外部真实迁移未解决；本地MCP检验不证明云端Tunnel连接。'])
    entries = [
        {'id': 'v95-direction-review', 'title': 'v9.5 四臂训练结果与失败根因', 'path': DOC,
         'category': 'current_review', 'summary': summary,
         'keywords': ['当前', '最新', '方向', '结果', '下一步', 'MLDG', 'ASA', '恶意', '可疑', '泛化', 'v9.5']},
        {'id': 'v95-delivery', 'title': 'v9.5 五次拟合与冻结复核交付',
         'path': (folder / 'delivery.json').relative_to(ROOT).as_posix(), 'category': 'current_evidence',
         'summary': '1教师+4分支、V选模及旧inner/C/H诊断；947历史文件未变；质量未通过。',
         'keywords': ['证据', '拟合', '复核', 'v9.5']}]
    for entry in reversed(entries):
        entry['sha256'] = sha(ROOT / entry['path'])
        catalog['documents'].insert(0, entry)
    save(catalog_path, catalog)
    tests = ROOT / 'mcp_readonly/tests/test_readonly_mcp.py'
    text = tests.read_text(encoding='utf-8').replace('v94-direction-review', 'v95-direction-review')
    text = text.replace('v94-delivery', 'v95-delivery').replace('deep_research_and_mechanism_review_completed_no_new_fit', STATUS)
    text = text.replace('本轮新增0次分类器拟合', '本轮新增5次分类器拟合')
    tests.write_text(text, encoding='utf-8')
    print(json.dumps({'published': True, 'bound_files': len(bound),
                      'catalog_documents': len(catalog['documents']),
                      'new_classifier_fits': 5, 'quality_acceptance': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
