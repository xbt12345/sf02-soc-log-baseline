"""Freeze verified no-fit v85 boundary analysis and publish its next plan."""
import json
from run_v75 import ROOT, read, save, sha

DEST = ROOT / 'artifacts/v86_boundary_review_20260927'
DOC = 'docs/V86_MALICIOUS_REGRESSION_ROOT_CAUSE_AND_PLAN.md'
STATUS = 'malicious_regression_review_completed_no_new_fit'
SUMMARY = ('v8.6本轮新增0次分类器拟合、0次校准拟合。复核v8.5 C新增6条M错误来自两个输入组；'
           '关键动作、协议、方向和端口未丢失，但细行为类别支持稀缺，训练正确组间隔被侵蚀99.45%。'
           'C全部60轮间隔惩罚为0，末28次更新完全拒绝。下一轮用实际步长约束投影与跨组件M/S配对监督做2×2对照，'
           '扩展训练正确保护并逐类验收；方案尚未训练。主模型质量未通过，无新模型晋升。'
           '最近实际训练仍为v8.5，最近完整开发回放仍为v7.9的5947错。')
PRIOR = [
    'evidence/2026-09-27/v79_execution/delivery.json',
    'artifacts/v80_attribution_20260927/review_receipt.json',
    'evidence/2026-09-27/v81_diagnosis/delivery.json',
    'evidence/2026-09-27/v82_capacity/delivery.json',
    'evidence/2026-09-27/v83_root_review/delivery.json',
    'evidence/2026-09-27/v84_preservation/delivery.json',
    'evidence/2026-09-27/v85_protection/delivery.json',
]


def main():
    folder = ROOT / 'evidence/2026-09-27/v86_boundary_review'
    if (DEST / 'review_receipt.json').exists() or (folder / 'delivery.json').exists():
        raise FileExistsError('Review already frozen')
    d = read(DEST / 'diagnosis.json')
    s = read(DEST / 'support_and_solver.json')
    v = read(DEST / 'verification.json')
    assert v['status'] == 'passed'
    assert v['no_new_classifier_fits'] and v['no_new_calibration_fits']
    assert d['new_classifier_fits'] == d['new_calibration_fits'] == s['new_classifier_fits'] == 0
    for name, info in [('v86_boundary_diagnosis.py', d), ('v86_support_diagnosis.py', s), ('v86_verify_boundary.py', v)]:
        assert info['source_sha256'] == sha(ROOT / 'training' / name), name
    assert v['diagnosis_sha256'] == sha(DEST / 'diagnosis.json')
    assert v['support_sha256'] == sha(DEST / 'support_and_solver.json')
    prior = {}
    for file in PRIOR:
        for path, digest in read(ROOT / file)['artifact_sha256'].items():
            assert path not in prior or prior[path] == digest, path
            prior[path] = digest
    changed = [path for path, digest in prior.items() if sha(ROOT / path) != digest]
    assert len(prior) == 378 and not changed, changed
    catalog_path = ROOT / 'mcp_readonly/catalog.json'
    catalog = read(catalog_path)
    for entry in catalog['documents']:
        assert sha(ROOT / entry['path']) == entry['sha256'], entry['path']
    assert catalog['project']['authoritative_direction_id'] == 'v85-direction-review'
    for relative in ['README.md', 'docs/TRAINING_PLAN.md']:
        assert 'v8.6本轮新增' not in (ROOT / relative).read_text(encoding='utf-8')
    folder.mkdir(parents=True, exist_ok=True)
    files = [p for p in DEST.rglob('*') if p.is_file()] + [ROOT / DOC] + list((ROOT / 'training').glob('v86_*.py'))
    bound = {path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(files)}
    save(DEST / 'review_receipt.json', {
        'status': 'review_frozen', 'new_classifier_fits': 0, 'new_calibration_fits': 0,
        'artifact_sha256': bound, 'prior_bound_files_rehashed': len(prior), 'prior_bound_files_changed': changed,
    })
    bound[(DEST / 'review_receipt.json').relative_to(ROOT).as_posix()] = sha(DEST / 'review_receipt.json')
    delivery = {
        'status': STATUS, 'quality_acceptance': False, 'all_issues_solved': False, 'model_promoted': False,
        'actual_new_fits': 0, 'actual_classifier_fits': 0, 'actual_calibration_fits': 0,
        'new_full_data_final_fit': False, 'new_full_development_replay': False,
        'platform_used': False, 'external_training_data_used': False, 'pseudo_labels_used': False,
        'training_target_answers_used': False, 'development_labels_used_for_diagnosis': True,
        'latest_training_delivery': 'evidence/2026-09-27/v85_protection/delivery.json',
        'latest_full_development_delivery': 'evidence/2026-09-27/v79_execution/delivery.json',
        'latest_full_development_errors': 5947,
        'validation_scope': ('No-fit saved-model replay, score decomposition, original-row support checks, CPU inference '
                             'and layer SVD. Previously inspected development labels used for diagnosis only. '
                             'The proposed solver and cross-component supervision experiment has not run; no transfer acceptance.'),
        'findings': d, 'support_and_solver': s, 'implementation_verification': v, 'next_plan': DOC,
        'local_raw_cases_published': False, 'prior_bound_files_rehashed': len(prior),
        'prior_bound_files_changed': changed, 'artifact_sha256': bound,
    }
    save(folder / 'delivery.json', delivery)
    for relative, link in [('README.md', DOC), ('docs/TRAINING_PLAN.md', 'V86_MALICIOUS_REGRESSION_ROOT_CAUSE_AND_PLAN.md')]:
        path = ROOT / relative
        title, body = path.read_text(encoding='utf-8').split('\n', 1)
        path.write_text(title + '\n\n当前恶意退化审查（2026-09-27）：**' + SUMMARY +
                        '** [底层诊断、研究与下一轮方案](' + link + ')。以下为历史阶段。\n' + body, encoding='utf-8')
    for entry in catalog['documents']:
        if entry['path'] in ['README.md', 'docs/TRAINING_PLAN.md']:
            entry.update(sha256=sha(ROOT / entry['path']), summary=SUMMARY)
        if entry['id'] == 'v85-direction-review':
            entry['category'] = 'historical_review'
        if entry['id'] == 'v85-delivery':
            entry['category'] = 'historical_evidence'
    catalog['project'].update(
        as_of='2026-09-27', current_summary=SUMMARY, authoritative_delivery_id='v86-delivery',
        authoritative_direction_id='v86-direction-review',
        current_direction=[
            '先核验实际参数步长的活跃约束求解及全训练人口正确保护，再执行求解器×跨组件M/S配对监督2×2小对照。',
            '机制安全与质量晋升分开：不能要求迁移已成功才引入迁移监督；所有候选仍须实际修错、逐类零新增错误才能推进。',
            '保留原记录、频次、真实端口与稀缺切片；按同类多/单组件、缺M支持和新组合分别检查，失败即停止预登记配置。',
        ],
        known_limits=[
            '6条C恶意退化来自2个输入组；关键字段未丢失不能证明完整语义足够，细行为支持不代表真实意图。',
            '287条困难S仅有粗行为配对候选条件，细到同目的端口后92条；均不是审核配对数或预计修复数。',
            '新投影求解器与配对监督均尚未训练，局部下降、QP成功或表示秩不构成分类收益。',
            '有限已知正确保护不保证未知输入零退化；VPC缺M、未见格式、完整开发质量仍未解决。',
        ],
    )
    entries = [
        {'id': 'v86-direction-review', 'title': 'v8.6 恶意退化根因与求解器及监督对照方案', 'path': DOC,
         'category': 'current_review', 'summary': SUMMARY,
         'keywords': ['当前', '最新', '方向', '下一步', '恶意', '可疑', '退化', '保护', '训练', '泛化', '监督', '根因', 'v8.6']},
        {'id': 'v86-delivery', 'title': 'v8.6 零新拟合的恶意退化独立诊断证据',
         'path': (folder / 'delivery.json').relative_to(ROOT).as_posix(), 'category': 'current_evidence',
         'summary': '0次拟合；15个角色重放、2050输入CPU分数复核、训练支持与配对覆盖复核，378个历史文件未改变。',
         'keywords': ['执行', '证据', '审查', '恶意', '可疑', '保护', 'v8.6']},
    ]
    for entry in reversed(entries):
        entry['sha256'] = sha(ROOT / entry['path'])
        catalog['documents'].insert(0, entry)
    save(catalog_path, catalog)
    path = ROOT / 'mcp_readonly/tests/test_readonly_mcp.py'
    text = path.read_text(encoding='utf-8')
    text = text.replace('v85-direction-review', 'v86-direction-review').replace('v85-delivery', 'v86-delivery')
    text = text.replace('protection_training_completed_no_candidate_accepted', STATUS)
    text = text.replace('本轮新增4次分类器拟合', '本轮新增0次分类器拟合')
    path.write_text(text, encoding='utf-8')
    print(json.dumps({'published': True, 'new_classifier_fits': 0, 'new_bound_files': len(bound),
                      'old_files_verified': len(prior), 'quality_acceptance': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
