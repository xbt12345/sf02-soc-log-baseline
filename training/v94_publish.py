"""Publish research and verified frozen diagnostics; preserve model authority."""
import json
import pandas as pd
from v89_common import ROOT, read, save, sha
from v94_mechanism_audit import DEST

DOC = 'docs/V94_DEEP_RESEARCH_AND_TRAINING_REVISION.md'
STATUS = 'deep_research_and_mechanism_review_completed_no_new_fit'


def main():
    folder = ROOT / 'evidence/2026-09-28/v94_deep_research'
    assert not (folder / 'delivery.json').exists()
    verify = read(DEST / 'verification.json')
    assert verify['status'] == 'passed' and not verify['prior_bound_files_changed']
    assert verify['source_sha256'] == sha(ROOT / 'training/v94_verify.py')
    catalog_path = ROOT / 'mcp_readonly/catalog.json'
    catalog = read(catalog_path)
    assert catalog['project']['authoritative_direction_id'] == 'v93-direction-review'
    for entry in catalog['documents']:
        assert sha(ROOT / entry['path']) == entry['sha256'], entry['path']

    # Training eligibility uses A/B only, never V class presence or performance.
    support = pd.read_csv(DEST / 'behavior_support.csv')
    ab = support[[f'{role}_{cl}_components' for role in ['A', 'B'] for cl in [1, 2]]].gt(0).all(axis=1)
    rows = pd.read_parquet(DEST / 'prospective_ASA_roles.parquet')
    checked = []
    for behavior, subset in rows[rows.proposed_role.isin(['A', 'B'])].groupby('behavior'):
        if all(((subset.proposed_role == role) & (subset.label_index == cl)).any() for role in ['A', 'B'] for cl in [1, 2]):
            checked.append(behavior)
    assert set(checked) == set(support.loc[ab, 'behavior']) and len(checked) == 3
    save(DEST / 'auxiliary_eligibility.json', {
        'rule': 'A and B each contain M and S; use training labels only',
        'training_supported_groups': len(checked), 'eligible_behaviors': sorted(checked),
        'groups_also_evaluable_with_both_classes_in_V': 2,
        'V_labels_used_to_choose_training_groups': False,
        'scope': 'Prospective eligibility only, not a trained method or all-format manifest'})
    sources = [
        ('MLDG', 'https://ojs.aaai.org/index.php/AAAI/article/view/11596', 'https://github.com/HAHA-DL/MLDG', 'Controlled transfer-update hypothesis; not demonstrated on SOC'),
        ('DomainBed MLDG implementation', 'https://github.com/facebookresearch/DomainBed/blob/main/domainbed/algorithms.py', None, 'First-order approximation reference; archived repository, no dependency installation'),
        ('LogitNorm', 'https://proceedings.mlr.press/v162/wei22d.html', 'https://github.com/hongxin001/logitnorm_ood', 'Magnitude control lesson; OOD detection result is not SOC classification improvement'),
        ('DFR', 'https://github.com/PolinaKirichenko/deep_feature_reweighting', 'https://github.com/PolinaKirichenko/deep_feature_reweighting/blob/main/dfr_evaluate_spurious.py', 'Balanced support and useful representation required; not default restart'),
        ('DFR limitations', 'https://arxiv.org/abs/2308.00473', None, 'Retrained readout can retain spurious dependence'),
        ('JTT', 'https://proceedings.mlr.press/v139/liu21f.html', 'https://github.com/anniesch/jtt', 'Early errors differ from late irreducible fit errors'),
        ('Learning to Reweight', 'https://proceedings.mlr.press/v80/ren18a.html', 'https://mengyeren.com/research/2018/learning-to-reweight-examples-for-robust-deep-learning/', 'Clean unbiased validation assumption not established here'),
        ('DomainBed model selection', 'https://arxiv.org/abs/2007.01434', 'https://github.com/facebookresearch/DomainBed', 'Matched ERM and independent selection roles'),
        ('Domain correspondence identifiability', 'https://proceedings.mlr.press/v162/gulrajani22a.html', None, 'Theory assumptions not claimed established for SOC'),
        ('TESSERACT', 'https://www.usenix.org/conference/usenixsecurity19/presentation/pendlebury', None, 'Spatial and temporal evaluation bias; SOC timestamps not assumed reliable'),
    ]
    save(DEST / 'research_sources.json', {
        'accessed_on': '2026-09-28', 'method': 'Primary papers and author-maintained implementations inspected through web tools',
        'external_training_data_used': False, 'external_benchmarks_executed': False,
        'sources': [dict(name=n, primary_url=u, implementation_or_author_url=c, applicability=a) for n, u, c, a in sources]})
    problems = [
        {'id': 'continuation_regression', 'status': 'mechanism_decomposed_not_repaired', 'finding': '500 M negative flips:260 already wrong at U0,192 had wrong-direction residual while still correct,48 residual direction reversals. U0-to-U0R also recovers88 prior M regressions.'},
        {'id': 'encoder_head_coupling', 'status': 'simple_head_restore_ruled_out', 'finding': 'Frozen U0R encoder/U0 head removes inner M NF but causes28 S NF and retains only2 S PF; no qualified hybrid.'},
        {'id': 'independent_behavior_support', 'status': 'quantified_unresolved', 'finding': '2502 coarse behavior groups;5 with >=3 components/class;3 A/B-supported training groups,2 also evaluable in V. Does not imply other groups unlearnable.'},
        {'id': 'cross_component_update_objective', 'status': 'hypothesis_pending_training', 'finding': 'Replace untested protection-population factor with controlled first-order MLDG-style auxiliary objective; retain magnitude control and matched ERM.'},
        {'id': 'protection_solver_path', 'status': 'protocol_revised_not_implemented', 'finding': 'Soft compatibility during candidate optimization; final strict protection and class gates unchanged. Formal model remains frozen.'},
        {'id': 'full_format_manifest', 'status': 'pending_before_training', 'finding': 'Prospective audit covers ASA only. Full-format component-safe roles and single-component support require explicit manifest.'},
        {'id': 'semantic_or_context_identifiability', 'status': 'unresolved', 'finding': 'Official M/S operational rules and reliable contextual associations absent; no guessed labels or claimed semantic upper bound.'},
        {'id': 'unsupported_identity_overlap', 'status': 'historical_unresolved', 'finding': 'Old unsupported row45738 pressure issue remains outside ASA correction scope.'},
    ]
    save(DEST / 'problem_register.json', problems)
    execution = dict(status=STATUS, actual_classifier_fits=0, actual_calibration_fits=0, actual_teacher_fits=0,
                     selected=None, model_promoted=False, quality_acceptance=False, all_issues_solved=False,
                     new_full_development_replay=False, platform_used=False, external_training_data_used=False,
                     original_labels_changed=0, target_answers_read=False, frozen_model_combinations=4,
                     planned_next_classifier_fits=5, planned_next_fits_executed=0, training_implementation_completed=False,
                     scope=verify['scope'])
    save(DEST / 'execution_summary.json', execution)
    summary = ('v9.4本轮新增0次分类器拟合、0次校准拟合；完成4种冻结编码器/末层组合重放和行为支持审计。'
               '500条M退化分为260条早已错、192条原先朝错方向但尚未越界、48条方向反转。'
               '2502行为组仅5组各类至少3组件；A/B训练支持3组，其中2组也在V具备双类评测支持。'
               '下一轮以原频次ERM对照输出正则和跨组件更新目标；训练资格只由A/B决定，最终保护门槛不变。'
               '925历史绑定文件未变，主模型质量未通过，无新模型晋升；最新实际训练仍v9.2，最近完整开发回放仍为v7.9的5947错。')
    files = [p for p in DEST.rglob('*') if p.is_file()] + list((ROOT / 'training').glob('v94*.py')) + [ROOT / DOC]
    bound = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(files)}
    delivery = {**execution, 'validation_scope': verify['scope'],
                'latest_executed_training_delivery': 'evidence/2026-09-28/v92_evidence_training/delivery.json',
                'latest_full_development_delivery': 'evidence/2026-09-27/v79_execution/delivery.json',
                'latest_full_development_errors': 5947, 'current_plan': DOC, 'verification': verify,
                'problem_register': (DEST / 'problem_register.json').relative_to(ROOT).as_posix(), 'artifact_sha256': bound}
    folder.mkdir(parents=True, exist_ok=True)
    save(folder / 'delivery.json', delivery)
    for relative, link in [('README.md', DOC), ('docs/TRAINING_PLAN.md', 'V94_DEEP_RESEARCH_AND_TRAINING_REVISION.md')]:
        p = ROOT / relative
        old = p.read_text(encoding='utf-8')
        assert 'v9.4本轮新增' not in old
        title, body = old.split('\n', 1)
        p.write_text(title + '\n\n当前深度调研与优化方向（2026-09-28）：**' + summary + '** [机制审计与下一轮方案](' + link + ')。以下为历史阶段。\n' + body, encoding='utf-8')
    for entry in catalog['documents']:
        if entry['path'] in ['README.md', 'docs/TRAINING_PLAN.md']:
            entry.update(sha256=sha(ROOT / entry['path']), summary=summary)
        if entry['id'] == 'v93-direction-review':
            entry['category'] = 'historical_review'
        if entry['id'] == 'v93-delivery':
            entry['category'] = 'historical_evidence'
    catalog['project'].update(as_of='2026-09-28', current_summary=summary,
        authoritative_direction_id='v94-direction-review', authoritative_delivery_id='v94-delivery',
        current_direction=[
            '完整输入和全部原行监督保留；500条M退化与58条S修复固定回归，正式模型冻结。',
            '先登记完整角色清单；辅助训练按A/B支持确定3组，V双类覆盖2组只用于评估，不反向选择训练组。',
            '检验输出幅度正则×跨组件更新目标的四臂；相同软兼容训练，最终逐类和负翻转门槛不变。',
            '最多1教师+4分支首轮，按实际修复、退化、覆盖及计算量决定是否扩展；外部论文成功不代表SOC成功。'],
        known_limits=[
            '本轮没有新训练或新质量收益；MLDG式更新仍是假设，训练器尚未实施。',
            '粗行为和输入派生组件不是已认证的因果域；98.86%纯标签组件不证明标签泄漏。',
            '支持审计不证明其余S不可学；缺少独立类别对照不能靠过采样补齐。',
            '既有inner/C/H为反复观察的开发数据，不是新盲测；官方M/S语义、上下文可靠性未解决。',
            '原unsupported身份问题仍未解决；本地MCP检验不代表云端Tunnel已连通。'])
    entries = [
        dict(id='v94-direction-review', title='v9.4 深层机制与跨组件训练方案', path=DOC, category='current_review', summary=summary,
             keywords=['当前', '最新', '方向', '下一步', '500', 'MLDG', '行为支持', 'ASA', '恶意', '可疑', '泛化', 'v9.4']),
        dict(id='v94-delivery', title='v9.4 调研与冻结机制审计：无新训练', path=(folder / 'delivery.json').relative_to(ROOT).as_posix(),
             category='current_evidence', summary='4组合16角色重放；2502行为组支持审计；925历史文件不变。无新模型晋升。', keywords=['证据', '复核', 'v9.4'])]
    for entry in reversed(entries):
        entry['sha256'] = sha(ROOT / entry['path'])
        catalog['documents'].insert(0, entry)
    save(catalog_path, catalog)
    tests = ROOT / 'mcp_readonly/tests/test_readonly_mcp.py'
    text = tests.read_text(encoding='utf-8').replace('v93-direction-review', 'v94-direction-review').replace('v93-delivery', 'v94-delivery').replace('regression_root_audit_completed_no_new_fit', STATUS)
    tests.write_text(text, encoding='utf-8')
    print(json.dumps(dict(published=True, bound_files=len(bound), catalog_documents=len(catalog['documents']), new_fits=0, quality_acceptance=False)))


if __name__ == '__main__':
    main()
