"""Publish the corrected source-only training result without model promotion."""
import json

from run_v75 import ROOT, read, save, sha
from v101_prepare import DEST


DOC = 'docs/V101_FULL_INPUT_GROUP_TRAINING_RESULTS.md'
STATUS = 'v101_full_input_source_training_completed_no_candidate_qualified'


def main():
    folder = ROOT/'evidence/2026-09-28/v101_full_input_training'
    delivery_path = folder/'delivery.json'
    if delivery_path.exists():
        raise FileExistsError(delivery_path)
    verify = read(DEST/'verification.json')
    select = read(DEST/'selection.json')
    invalid = read(ROOT/'artifacts/v100_matched_n1_20260928/invalidity.json')
    assert verify['status'] == 'passed' and all(verify['checks'].values())
    assert verify['selected'] is None and select['selected'] is None and not select['qualified']
    assert verify['valid_teacher_fits'] == 6 and verify['valid_branch_fits'] == 12
    assert invalid['cross_fold_identical_R0_groups'] == 4876 and invalid['classifier_fits_in_invalid_run'] == 19
    assert sha(ROOT/'training/v101_verify.py') == verify['source_sha256']
    assert sha(ROOT/'training/v101_select.py') == select['source_sha256']
    summary = ('v10.1全输入隔离三折重训完成：纠正后6次教师与12次分支拟合。'
        'ASA来源OOF原视图教师588错，MAG第200步564错，但42条教师原正确M被翻错，增益组件区间跨0；'
        'N1消除指定包装变体翻转但分类退化。没有候选通过保护门槛，质量未通过，无新模型晋升。'
        '先前v10.0因非ASA相同输入跨折判无效，19次拟合单列；本次请求共37次分类器拟合、0次校准。'
        'V与旧开发区未用于本轮选模，最近完整开发回放仍为v7.9的5947错。')
    execution = {'status': STATUS, 'valid_teacher_fits': 6,
        'valid_branch_fits': 12, 'invalid_pilot_fits': 19,
        'actual_classifier_fits_this_user_request': 37,
        'actual_calibration_fits': 0, 'selected_arm': None,
        'quality_acceptance': False, 'model_promoted': False,
        'A_B_refit': False, 'ABV_final_fit': False,
        'new_full_development_replay': False,
        'platform_used': False, 'external_training_data_used': False,
        'original_labels_changed': 0, 'target_answers_read': False,
        'invalid_pilot_reason': '4876 identical non-ASA input groups crossed folds',
        'source_OOF_best_branch': 'R0_MAG_200',
        'source_OOF_best_branch_error_rows': 564,
        'source_OOF_teacher_error_rows': 588,
        'source_OOF_best_branch_negative_flips': 42,
        'source_OOF_best_branch_bootstrap_CI_includes_zero': True,
        'source_only_selection': True,
        'known_protocol_deviations': [
            'Historic 229 errors, six focus S rows and variant flip counts were traced at the registered endpoints, not every 25-step check.',
            'Complete raw-character span mapping was finished after the fits; replay/inference identity was checked before the fits.'
        ], 'latest_full_development_errors': 5947}
    save(DEST/'execution_summary.json', execution)
    bindings = {}
    for p in sorted(DEST.rglob('*')):
        if p.is_file():
            bindings[p.relative_to(ROOT).as_posix()] = sha(p)
    for p in sorted((ROOT/'training').glob('v101_*.py')):
        bindings[p.relative_to(ROOT).as_posix()] = sha(p)
    for p in [ROOT/DOC, ROOT/'artifacts/v100_matched_n1_20260928/invalidity.json']:
        bindings[p.relative_to(ROOT).as_posix()] = sha(p)
    delivery = {**execution,
        'validation_scope': 'Source-only A+B component OOF on ASA with exact input isolation, paired teacher/branch exposure, per-class/cluster and legal-variant checks. OOF was used for selection; no V, hidden validation or external transfer acceptance.',
        'verification': verify, 'current_review': DOC,
        'latest_executed_training_delivery': delivery_path.relative_to(ROOT).as_posix(),
        'latest_full_development_delivery': 'evidence/2026-09-27/v79_execution/delivery.json',
        'latest_full_development_errors': 5947,
        'artifact_sha256': bindings}
    folder.mkdir(parents=True, exist_ok=True)
    save(delivery_path, delivery)
    for relative, link in [('README.md', DOC),
                           ('docs/TRAINING_PLAN.md', 'V101_FULL_INPUT_GROUP_TRAINING_RESULTS.md')]:
        p = ROOT/relative
        text = p.read_text(encoding='utf-8')
        assert 'v10.1全输入隔离三折重训完成' not in text
        head, rest = text.split('\n', 1)
        p.write_text(head+'\n\n当前训练结果（2026-09-28）：**'+summary+'** '
                     '[全输入隔离训练与逐类验收]('+link+')。以下保留历史阶段。\n'+rest,
                     encoding='utf-8')
    catalog_path = ROOT/'mcp_readonly/catalog.json'
    catalog = read(catalog_path)
    assert catalog['project']['authoritative_direction_id'] == 'v99-direction-review'
    for entry in catalog['documents']:
        if entry['path'] not in ('README.md', 'docs/TRAINING_PLAN.md'):
            assert sha(ROOT/entry['path']) == entry['sha256'], entry['id']
        if entry['path'] in ('README.md', 'docs/TRAINING_PLAN.md'):
            entry['sha256'] = sha(ROOT/entry['path'])
            entry['summary'] = summary
        if entry['id'] == 'v99-direction-review':
            entry['category'] = 'historical_review'
        if entry['id'] == 'v99-delivery':
            entry['category'] = 'historical_evidence'
    catalog['project'].update(as_of='2026-09-28', current_summary=summary,
        authoritative_direction_id='v101-direction-review',
        authoritative_delivery_id='v101-delivery',
        current_direction=[
            'v10.1来源内四臂三折训练已完成；无候选通过逐类和旧正确保护门槛，保留既有正式模型。',
            'N1占位符规范化可确保指定包装等价，但本轮未改善M/S迁移；不直接替换生产推理输入。',
            '先核实错误组件的行为事实、M/S标签支持与独立同类组件，再设计新的保序表达或监督对照；不重复盲目延长训练。',
            '官方标签、可观察事实、模型行为与独立事件真值分开；来源OOF为已观察开发选择，不能声称外部迁移。'],
        known_limits=[
            '最好短终点总错虽少24条，却新增42条教师原正确M错误，组件增益区间跨0。',
            '来源ASA S在第1折仅36/157判对；OOF总体高分受少数大组件影响。',
            'v10.0无效试跑已隔离；正式v10.1共18次有效拟合。全任务/V/官方隐藏成绩未重新验证。',
            '每25步重点错例与合法变体逐行轨迹未保存，正式端点补做；独立安全事件真值不可得。'])
    entries = [
        {'id':'v101-direction-review','title':'v10.1 全输入隔离训练结果和下一步方向',
         'path': DOC, 'category':'current_review', 'summary':summary,
         'keywords':['当前','最新','方向','训练','逐类','ASA','N1','第一性原理','v10.1']},
        {'id':'v101-delivery','title':'v10.1 来源内训练与独立核验凭据',
         'path':delivery_path.relative_to(ROOT).as_posix(),
         'category':'current_evidence','summary':'18次纠正后拟合；旧正确保护门槛未通过，无模型晋升。',
         'keywords':['证据','核验','模型训练','来源OOF','v10.1']}]
    for entry in reversed(entries):
        entry['sha256'] = sha(ROOT/entry['path'])
        catalog['documents'].insert(0, entry)
    save(catalog_path, catalog)
    test = ROOT/'mcp_readonly/tests/test_readonly_mcp.py'
    text = test.read_text(encoding='utf-8')
    text = text.replace('v99-direction-review', 'v101-direction-review')
    text = text.replace('v99-delivery', 'v101-delivery')
    text = text.replace('v99_research_and_no_fit_feasibility_completed', STATUS)
    text = text.replace("self.assertIn('本轮新增0次分类器拟合', status.current_summary)",
        "self.assertIn('纠正后6次教师与12次分支拟合', status.current_summary)")
    test.write_text(text, encoding='utf-8')
    for name, digest in bindings.items():
        assert sha(ROOT/name) == digest, name
    print(json.dumps({'published': True, 'status': STATUS,
        'bound_files': len(bindings), 'catalog_documents':len(catalog['documents']),
        'quality_acceptance':False, 'model_promoted':False,
        'new_valid_fits':18, 'invalid_pilot_fits':19,
        'delivery_sha256':sha(delivery_path)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
