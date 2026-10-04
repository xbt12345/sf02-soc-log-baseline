"""Publish the revised plan and history-backed contract; execute no model fits."""
import json
from pathlib import Path
import subprocess
import sys
import pandas as pd
from v116_preflight import ROOT, DEST as TRAINED, save, sha
from v118_gradient_confirmation import DEST, CASES
from v118_training_contract import ContractViolation, require_terminal_checkpoint, require_complete_population


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    assert not (DEST / 'review_receipt.json').exists()
    registration = read(DEST / 'probe_registration.json')
    for rel, digest in registration['input_sha256'].items():
        assert sha(ROOT / rel) == digest, rel
    probe = read(DEST / 'probe_results.json')
    assert probe['registration_sha256'] == sha(DEST / 'probe_registration.json')
    assert [(r['fold'], r['epoch']) for r in probe['results']] == CASES
    assert all(r['original_frequency_preserved'] and r['parameters_unchanged'] for r in probe['results'])
    fits = []
    for path in sorted(TRAINED.glob('*_fold*/fit.json')):
        item = read(path)
        fits.append({'trajectory': path.parent.name, 'completed_epochs': item['epochs'],
                     'selected_epoch': item['selected_epoch'], 'fit_receipt_sha256': sha(path)})
    assert len(fits) == 6 and all(f['completed_epochs'] == 25 for f in fits)
    old = read(TRAINED / 'outer_fold1/fit.json')
    rejected = False
    try:
        require_terminal_checkpoint(old['epochs'], old['selected_epoch'], old['status'])
    except ContractViolation:
        rejected = True
    assert rejected
    manifest = pd.read_parquet(TRAINED / 'inner_split_manifest.parquet')
    fold = manifest[manifest.outer_fold.eq(1)]
    expected = fold[fold.role.isin(['K', 'U', 'validation_collateral'])].row_position
    scored = fold[fold.role.isin(['K', 'U'])].row_position
    omitted = len(expected) - len(scored)
    try:
        require_complete_population(expected, scored)
        raise AssertionError('Incomplete historical full-population claim should be rejected.')
    except ContractViolation:
        pass
    tests = subprocess.run([sys.executable, str(ROOT / 'training/test_v118_contract.py')], cwd=ROOT,
                           capture_output=True, text=True, encoding='utf-8')
    save(DEST / 'contract_test_result.json', {'exit_code': tests.returncode, 'output': tests.stdout + tests.stderr})
    assert tests.returncode == 0
    save(DEST / 'retrospective_evidence.json', {'all_six_actual_fits_completed_25_epochs': True,
        'fit_receipts': fits, 'earlier_selected_checkpoint_rejected_by_new_contract': rejected,
        'fold1_validation_collateral_omitted_by_old_K_U_scoring': omitted,
        'historical_quality_failure_already_recorded_in_V116': not read(TRAINED / 'evaluation.json')['all_gates_passed'],
        'new_classifier_fits': 0, 'optimizer_steps': 0,
        'probe_variance_ratio_B_over_A': [{'fold': r['fold'], 'epoch': r['epoch'], 'ratio': r['variance_ratio_B_over_A']} for r in probe['results']]})
    save(DEST / 'training_contract.json', {
        'status': 'planned_not_fitted', 'latest_executed_training': 'V116',
        'source_of_direction': 'docs/V118_RETROSPECTIVE_AND_REVISED_TRAINING_PLAN.md',
        'arms': {'A': 'uniform_unique_input_batches', 'B': 'original_class_mass_stratified_batches'},
        'unchanged': ['N1 input', 'SparseTabM k16', 'original-row-frequency three-class CE',
                      'AdamW lr=0.002 weight_decay=0.0003', 'outer folds', 'inference averaging and thresholds'],
        'terminal_epoch': 25, 'early_checkpoint_promotion_allowed': False,
        'diagnostic_checkpoints': [1, 2, 5, 10, 15, 20, 25],
        'primary': {'seed': 10201, 'folds': [0, 1, 2], 'fits': 6},
        'confirmation': {'seeds': [10202, 10203], 'fits': 12,
                         'condition': 'Every primary gate measured and true; no extra seeds after a failed primary.'},
        'maximum_new_fits': 18, 'no_best_seed_selection': True,
        'batch_rng': 'Initialize model_seed+fold once for each matched arm; advance across epochs.',
        'primary_gates': {'ASA_M_errors_at_most': 318, 'ASA_S_errors_at_most': 2094,
                          'ASA_total_errors_at_most': 2170,
                          'paired_M_S_not_worse': True, 'at_least_two_folds_improve': True,
                          'S_outside_top3_not_worse': True, 'full_each_class_recall_and_F1_not_worse': True},
        'confirmation_gates': ['No paired ASA total-error regression in any registered seed.',
                               'Summed paired M errors and summed paired S errors cannot increase.'],
        'all_heldout_rows_scored': True, 'preserve_each_training_input_class_mass': True,
        'training_executor_integration': 'Required before fitting; contract functions are tested but the next trainer is not executed.',
        'known_limits': ['Fold2 frozen gradient dispersion increased 37.95%; reduction is not universal.',
                         'Determinism not achieved; old outer folds are inspected development evidence.',
                         'Remaining support and hidden-context gaps cannot be fixed by batching alone.']})
    report = ROOT / 'docs/V118_RETROSPECTIVE_AND_REVISED_TRAINING_PLAN.md'
    files = sorted(DEST.glob('*.json')) + [report] + sorted((ROOT / 'training').glob('v118_*.py')) + [ROOT / 'training/test_v118_contract.py']
    save(DEST / 'review_receipt.json', {'status': 'v118_retrospective_and_conditional_training_plan',
        'new_classifier_fits': 0, 'optimizer_steps': 0, 'quality_acceptance': False, 'model_promoted': False,
        'latest_executed_training': 'V116', 'artifact_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in files}})
    summary = ('V118完成历史复盘、4项固定模型探针与5项防复发测试，新增训练0次、更新0步。'
        'V116六轨迹均跑完25轮，问题是第1折选用了15轮候选；退化在V116已拦截。'
        '最新实际训练仍为V116的6次新拟合（3次内层、3次外层），0次校准，质量未通过、无模型晋升。'
        '新批次法三折终点梯度离散度变化为-32.70%/-32.31%/+37.95%，不能全面推广。'
        '下一轮固定25轮，先6次匹配分类对照，全部质量门槛通过才追加12次种子确认。')
    for rel, link in [('README.md', 'docs/' + report.name), ('docs/TRAINING_PLAN.md', report.name)]:
        path = ROOT / rel; text = path.read_text(encoding='utf-8')
        assert report.name not in text
        text = text.replace('当前根因审查与下一轮方案（V117，未新增训练）', '历史根因审查与方案（V117；资源顺序与风险由V118修订）', 1)
        head, body = text.split('\n', 1)
        path.write_text(head + '\n\n当前复盘与修订训练方案（V118）：**' + summary + '** [事实纠正、历史教训与执行约束](' + link + ')。\n' + body, encoding='utf-8')
    path = ROOT / 'mcp_readonly/catalog.json'; catalog = read(path)
    assert catalog['project']['authoritative_delivery_id'] == 'v116-delivery'
    catalog['project'].update(current_summary=summary, authoritative_direction_id='v118-direction-review',
        current_direction=['停止早期检查点自动晋升；固定25轮，严格区分预算完成、拟合充分和来源外质量。',
            '新批次法只做6次主对照；通过全部实际质量门槛后，按固定种子追加12次确认。',
            '所有合法留出行计分，逐类学习与来源迁移分开记录；不以梯度下降或较低CE替代分类收益。'],
        known_limits=['第2折固定模型梯度离散度上升37.95%，新批次法没有普遍方差收益或新分类成绩。',
            '下一轮执行器尚未运行，需接入已验证的终点与完整评分检查。',
            '数值非确定性、真实跨来源标签依据及监督支持缺口尚未解决。'])
    for entry in catalog['documents']:
        if entry['id'] == 'v117-direction-review': entry['category'] = 'historical_review'
        if entry['path'] in ['README.md', 'docs/TRAINING_PLAN.md']:
            entry.update(summary=summary, sha256=sha(ROOT / entry['path']))
    new = [('v118-direction-review', report, 'V118 早期检查点复盘与修订方案'),
           ('v118-review-receipt', DEST / 'review_receipt.json', 'V118 零更新复盘凭据'),
           ('v118-training-contract', DEST / 'training_contract.json', 'V118 下一轮训练执行契约')]
    catalog['documents'] = [{'id': ident, 'title': title, 'path': p.relative_to(ROOT).as_posix(),
        'category': 'current_review' if ident == 'v118-direction-review' else 'current_evidence',
        'summary': summary, 'sha256': sha(p), 'keywords': ['V118', '当前', '复盘', '选模', '监督', '批次', '停止决定']}
        for ident, p, title in new] + catalog['documents']
    save(path, catalog)
    path = ROOT / 'mcp_readonly/tests/test_readonly_mcp.py'
    text = path.read_text(encoding='utf-8'); assert 'v117-direction-review' in text
    path.write_text(text.replace('v117-direction-review', 'v118-direction-review'), encoding='utf-8')
    print(json.dumps({'review': 'V118', 'latest_actual_training': 'V116', 'new_fits': 0,
                      'confirmation_cases': len(probe['results']), 'historical_contract_tests_exit_code': tests.returncode}))


if __name__ == '__main__':
    main()
