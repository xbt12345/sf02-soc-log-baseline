"""Verify prior evidence, publish this zero-fit review, retain training authority."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'artifacts/v80_attribution_20260927'
DOC = 'docs/V80_REGRESSION_AND_SUPERVISION_PLAN.md'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with open(path, 'rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    if (DEST / 'review_receipt.json').exists():
        raise FileExistsError('review already published')
    delivery_path = ROOT / 'evidence/2026-09-27/v79_execution/delivery.json'
    previous = read(delivery_path)
    hashes = previous['artifact_sha256']
    failures = [name for name, expected in hashes.items() if sha(ROOT / name) != expected]
    if failures:
        raise AssertionError(failures)
    chain = read(DEST / 'regression_chain.json')
    controls = read(DEST / 'branch_controls.json')
    for filename, payload in [('v80_failure_attribution.py', chain), ('v80_branch_controls.py', controls)]:
        assert payload['source_sha256'] == sha(ROOT / 'training' / filename)
        ast.parse((ROOT / 'training' / filename).read_text(encoding='utf-8'))
    assert chain['new_fits'] == controls['new_fits'] == 0
    assert chain['totals']['v78_O_sgd'] == previous['baseline']
    # Verify the new report against the frozen full regression, not rounded narrative values.
    totals = chain['totals']['v79_B_ovr']
    assert totals['rows'] == 2014052
    assert totals['cm'] == [[1959519, 0, 54], [100, 10744, 3208], [36, 2549, 37842]]
    assert sum(chain['loss_chain'][k] for k in ['old_SGD_to_same_subset_LBFGS', 'old_LBFGS_to_new_LBFGS']) == chain['loss_chain']['total']
    c = controls['cef_two_by_two']
    for prefix in ['False', 'True']:
        assert c[f'default_facts_{prefix}_clear_conflict_False'] == c[f'default_facts_{prefix}_clear_conflict_True']
    assert controls['asa_wrong_S_missing_value_contributions']['http_status']['absent_rows'] == 3464
    assert controls['asa_wrong_S_missing_value_contributions']['http_status']['default_active_value_bits'] == 10
    summary = 'v8.0本轮新增0次分类器拟合、0次校准拟合；完成退化链、分支干预及监督复核。旧同样本SGD→收敛模型已多错1290条，本轮相对它少错166条。最近真实训练仍为v7.9的15次拟合，质量未通过，无新模型晋升。'
    for relative, target in [('README.md', 'docs/V80_REGRESSION_AND_SUPERVISION_PLAN.md'), ('docs/TRAINING_PLAN.md', 'V80_REGRESSION_AND_SUPERVISION_PLAN.md')]:
        path = ROOT / relative
        old = path.read_text(encoding='utf-8')
        assert 'v8.0本轮新增' not in old
        title, rest = old.split('\n', 1)
        path.write_text(title + '\n\n当前审查（2026-09-27）：**' + summary + '** [退化归因、更有效的监督与下一轮对照](' + target + ')。以下均为历史阶段。\n' + rest, encoding='utf-8')
    catalog_path = ROOT / 'mcp_readonly/catalog.json'
    catalog = read(catalog_path)
    # Validate all existing whitelist entries, except the two explicitly updated entry points.
    changed = {'README.md', 'docs/TRAINING_PLAN.md'}
    for entry in catalog['documents']:
        actual = sha(ROOT / entry['path'])
        if entry['path'] in changed:
            entry['sha256'] = actual
            entry['summary'] = summary
        else:
            assert actual == entry['sha256'], entry['path']
        if entry['id'] == 'v79-direction-review':
            entry['category'] = 'historical_review'
    catalog['project'].update({
        'as_of': '2026-09-27', 'current_summary': summary,
        'authoritative_direction_id': 'v80-direction-review',
        'current_direction': [
            '保留旧SGD诊断参考；按同样本和内层来源验证比较，收敛不作为晋升条件。',
            '增加可核原文证据和完整语义等价呈现监督，不重跑已失败的单独观察标记编码或全局加权。',
            '监督×非线性能力四臂归因；每类正确/误报、跨来源与扩训稳定通过后才全量训练。'
        ],
        'known_limits': [
            '本轮只有零拟合诊断；新监督及容量候选尚未训练，实际分类质量仍未通过。',
            'ASA缺失状态贡献不是因果证明；CEF默认事实干预不是可交付修复。',
            'VPC官方训练无M监督；完整回放VPC-M0/2664、ASA2536条S→M仍未解决。',
            '历史开发数据已经多次检查，不是独立外部测试；两组冲突不能排除其他容量缺口。'
        ]})
    assert catalog['project']['authoritative_delivery_id'] == 'v79-delivery'
    catalog['documents'].insert(0, {
        'id': 'v80-direction-review', 'title': 'v8.0 退化归因复核与监督方案修订',
        'path': DOC, 'category': 'current_review', 'summary': summary,
        'keywords': ['当前', '最新', '方向', '下一步', '退化', '恶意', '可疑', '泛化', '监督', 'v8.0'],
        'sha256': sha(ROOT / DOC)})
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding='utf-8')
    paths = sorted(DEST.glob('*')) + [ROOT / DOC] + [ROOT / 'training' / name for name in ['v80_failure_attribution.py', 'v80_branch_controls.py', 'v80_publish_review.py']]
    receipt = {
        'status': 'zero_fit_attribution_completed_next_plan_not_trained', 'quality_acceptance': False,
        'new_classifier_fits': 0, 'new_calibration_fits': 0, 'platform_used': False,
        'previous_training_delivery_sha256': sha(delivery_path),
        'previous_bound_files_rehashed': len(hashes), 'previous_bound_files_changed': failures,
        'checks': {'source_hashes_match': True, 'full_regression_matches_frozen_delivery': True,
                   'paired_loss_chain_reconciles': True, 'cef_two_factor_controls_reproduced': True,
                   'missing_http_bits_reproduced': True},
        'scope': 'New zero-fit diagnostic execution and source verification only; no new predictive gain or external validation. Original data files not rehashed in this publication check.',
        'artifact_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in paths if p.is_file()}}
    (DEST / 'review_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': receipt['status'], 'prior_files_verified': len(hashes), 'new_fits': 0, 'direction': 'v80-direction-review', 'training_delivery': 'v79-delivery'}))


if __name__ == '__main__':
    main()
