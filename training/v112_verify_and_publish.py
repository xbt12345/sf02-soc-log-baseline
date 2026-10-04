"""Verify V112 observed evidence and publish only a direction review."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from run_v75 import ROOT, save, sha
from v110_layer_probes import LEDGER, check_registration
from v112_fine_control_preflight import DEST, floor

REPORT = 'docs/V112_FINE_CONTROL_AND_TEXT_DEPENDENCE_PLAN.md'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def main():
    target = DEST / 'verification.json'
    assert not target.exists()
    check_registration()
    checks = {}
    old = read(ROOT / 'evidence/2026-09-29/v110_layer_probes/delivery.json')
    checks['previous_training_delivery_unchanged'] = all(sha(ROOT/p) == h for p, h in old['artifact_sha256'].items())
    sources = {'preflight.json': 'v112_fine_control_preflight.py',
        'factor_audit.json': 'v112_interface_factor_audit.py',
        'negative_evidence_audit.json': 'v112_negative_evidence_audit.py'}
    for name, source in sources.items():
        r = read(DEST/name)
        checks[name+'_zero_fits'] = r['classifier_fits'] == r['calibration_fits'] == 0
        checks[name+'_source_and_input_hashes'] = r['source_sha256'] == sha(ROOT/'training'/source) and all(sha(ROOT/p) == h for p, h in r['input_hashes'].items())
        checks[name+'_output_hashes'] = all(sha(DEST/p) == h for p, h in r['output_hashes'].items())
    pre = read(DEST/'preflight.json')
    factors = read(DEST/'factor_audit.json')
    d = pd.read_parquet(LEDGER)
    z = pd.read_parquet(DEST/'interface_row_audit.parquet')
    v = pd.read_parquet(DEST/'interface_views.parquet')
    h = pd.read_parquet(DEST/'factor_hashes.parquet')
    checks['all_event_rows_preserved_and_aligned'] = len(z) == z.row_position.nunique() == 112807 and np.array_equal(z.row_position, d.row_position) and np.array_equal(z.truth, d.truth) and np.array_equal(z.root, d.root) and np.array_equal(z.fold, d.fold)
    official = pd.read_parquet(ROOT/'data/official/train.parquet', columns=['label_binary'])
    y = official.label_binary.map({'benign': 0, 'malicious': 1, 'suspicious': 2}).to_numpy()
    checks['labels_match_official_records'] = len(y) == 2056871 and np.array_equal(y[z.row_position], z.truth)
    checks['unique_input_mapping_and_unknown_retention'] = len(v) == v.local.nunique() == 22546 and int((~z.interface_parse_verified).sum()) == 502 and bool((v.loc[~v.interface_parse_verified, 'before'] == v.loc[~v.interface_parse_verified, 'isolated']).all())
    checks['original_floor_recount'] = floor(z, 'original_full_input_hash') == pre['original_floor'] and pre['original_floor']['observed_min_errors'] == 26
    checks['isolated_floor_recount'] = floor(z, 'isolated_full_input_hash') == pre['isolated_floor'] and pre['isolated_floor']['observed_min_errors'] == 236
    checks['all_factor_floors_recount'] = all(floor(h, r['variant']) == r['observed_floor'] for r in factors['variants'])
    checks['case_only_floor_unchanged_suffix_loss_observed'] = [r['observed_floor']['observed_min_errors'] for r in factors['variants']] == [26, 206, 236]
    cross = z.groupby('isolated_full_input_hash').fold.nunique()
    checks['27_equal_input_crossfold_groups_recount'] = int((cross > 1).sum()) == 27 and int(z.isolated_full_input_hash.isin(cross[cross > 1].index).sum()) == 568
    elig = pd.read_parquet(DEST/'training_control_eligibility.parquet')
    rowfold = d.set_index('row_position').fold
    checks['no_outer_partner_anchors_in_control_tables'] = bool((elig.row_position.map(rowfold).to_numpy() != elig.audit_fold.to_numpy()).all())
    checks['all_fine_control_summary_counts_match'] = all(
        len(a := elig[(elig.audit_fold == s['fold']) & (elig.resolution == s['resolution']) & (elig.truth == (1 if s['class'] == 'M' else 2))]) == s['available_train_rows']
        and int(a.positive_cross_source_cross_name.sum()) == s['positive_rows']
        and int(a.negative_cross_source_same_name.sum()) == s['negative_rows']
        and int(a.both_controls.sum()) == s['both_rows'] for s in pre['train_fine_control_eligibility'])
    checks['S_cross_name_positive_coverage_zero_all_folds'] = not bool(elig.loc[elig.truth == 2, 'positive_cross_source_cross_name'].any())
    neg = pd.read_parquet(DEST/'negative_evidence_eligibility.parquet')
    checks['no_extra_fact_difference_in_current_candidate_negatives'] = not bool(neg.has_negative_with_non_source_port_fact_difference.any()) and bool(neg.has_cross_source_negative.any())
    # Independently re-check the decisive class of candidates by join, rather
    # than repeating the original set cache logic: for every training role,
    # each behavior/name bin with cross-source mixed labels has one non-port fact.
    fx = pd.read_parquet(ROOT/'artifacts/v75_four_arm_20260921_r2/projections.parquet', columns=['facts'])
    lookup = d.drop_duplicates('local').set_index('local').projection_id
    factmap = {}
    for loc, proj in lookup.items():
        f = json.loads(fx.facts.iat[int(proj)])
        factmap[loc] = json.dumps({k: val for k, val in f.items() if not k.startswith('src_port')}, sort_keys=True)
    z['without_port'] = z.local.map(factmap)
    independent = True
    for k in range(3):
        q = z[(z.fold != k) & z.interface_parse_verified & z.behavior.notna()]
        g = q.groupby(['behavior', 'interface_literal_pair']).agg(labels=('truth', 'nunique'), roots=('root', 'nunique'), facts=('without_port', 'nunique'))
        independent &= bool((g.loc[(g.labels == 2) & (g.roots > 1), 'facts'] == 1).all())
    checks['negative_fact_gap_independently_recounted'] = independent
    assert all(checks.values()), checks
    files = [ROOT/REPORT] + [p for p in DEST.iterdir() if p.is_file() and p.suffix in ('.json', '.parquet', '.csv')]
    files += [ROOT/'training'/s for s in sources.values()]
    ver = {'status': 'no_fit_fine_control_review_verified', 'all_checks_passed': True, 'checks': checks,
        'new_classifier_fits': 0, 'new_calibration_fits': 0, 'model_promoted': False,
        'scope': 'File identity, official row/label alignment, collision floors, training-side candidate coverage and non-source-port fact-gap recount only. No model improvement or blind transfer validation.',
        'source_sha256': sha(__file__), 'artifact_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in files}}
    save(target, ver)
    summary = ('v11.2完成零拟合精细对照审查。最新训练仍为v11.0的6次拟合、0次校准，质量未通过、无模型晋升。'
        '全名隔离把经验冲突下界26增至236；仅改大小写仍26，删编号增至206。'
        '三折S细行为跨名称正对照均为0，同名称负对照未发现源端口以外的已解析事实差异。'
        '下一轮首选仅ASA接口大小写规范化的匹配两臂；保留编号，证据不足的配对不启用附加判别损失。')
    for name, link in [('README.md', REPORT), ('docs/TRAINING_PLAN.md', Path(REPORT).name)]:
        p = ROOT/name
        s = p.read_text(encoding='utf-8'); head, body = s.split('\n', 1)
        assert 'V112_FINE_CONTROL' not in s
        body = body.replace('当前根因审查与下一轮方案（v11.1，未新增训练）', '上一轮根因审查（v11.1，其全名隔离候选已被v11.2修订）', 1)
        p.write_text(head+'\n\n当前精细对照与训练方案（v11.2，未新增训练）：**'+summary+'** [新证据、监督资格与下一轮方案]('+link+')。\n'+body, encoding='utf-8')
    catpath = ROOT/'mcp_readonly/catalog.json'; cat = read(catpath)
    assert cat['project']['authoritative_delivery_id'] == 'v110-delivery'
    assert cat['project']['authoritative_direction_id'] == 'v111-direction-review'
    cat['project'].update(as_of='2026-09-29', authoritative_direction_id='v112-direction-review', current_summary=summary,
        current_direction=['保留V110停止决定及N1开发参照；V112仅诊断与方案，无新训练。',
            '撤回全名屏蔽首选；先对照原N1与仅确认ASA接口大小写规范化，保留编号和其他正文。',
            '精细配对按类/来源核查覆盖与证据；S跨名称正例缺失、当前负例可能只强化源端口，暂不加CNC或配对判别损失。',
            '原文证据、解析/表达缺口、不可确认上下文分开；以逐类真实错误、负翻转和全任务回放验收。'],
        known_limits=['全名投影的经验冲突下界不是真实模型错误或Bayes错误；不证明接口编号有可迁移因果意义。',
            '未找到源端口以外的解析事实差异不等于官方原文无差异；当前精细配对不是安全因果真值。',
            '大小写规范化有Cisco语义依据，但分类收益尚未训练验证；现有外层均为已查看开发折。'])
    additions = [('v112-direction-review', 'v11.2 精细对照与文字依赖优化方案', REPORT, 'current_review', summary),
        ('v112-control-preflight', 'v11.2 精细对照覆盖及全名隔离冲突', 'artifacts/v112_fine_supervision_review_20260929/preflight.json', 'current_evidence', '零拟合：26到236经验冲突下界，S跨名称正例为0。'),
        ('v112-factor-audit', 'v11.2 大小写和接口编号因素拆分', 'artifacts/v112_fine_supervision_review_20260929/factor_audit.json', 'current_evidence', '只改大小写不新增冲突，删编号使下界增至206；不是模型成绩。'),
        ('v112-negative-evidence', 'v11.2 负对照事实差异资格', 'artifacts/v112_fine_supervision_review_20260929/negative_evidence_audit.json', 'current_evidence', '当前精细负对照无源端口以外的已解析事实差异，不代表原文不可分。'),
        ('v112-verification', 'v11.2 零拟合复核凭据', 'artifacts/v112_fine_supervision_review_20260929/verification.json', 'current_evidence', '身份、官方原行、冲突下界、训练侧候选和事实差异核查；不是训练完成。')]
    for item in cat['documents']:
        if item['id'] == 'v111-direction-review': item['category'] = 'historical_review'
        if item['path'] in ['README.md', 'docs/TRAINING_PLAN.md']: item.update(summary=summary, sha256=sha(ROOT/item['path']))
    cat['documents'] = [{'id': i, 'title': t, 'path': p, 'category': c, 'summary': s,
        'keywords': ['v11.2', '当前', '下一步方向', '训练方案', '精细对照', '文字路径', '大小写', '监督', 'M/S'],
        'sha256': sha(ROOT/p)} for i,t,p,c,s in additions] + cat['documents']
    save(catpath, cat)
    test = ROOT/'mcp_readonly/tests/test_readonly_mcp.py'
    test.write_text(test.read_text(encoding='utf-8').replace('v111-direction-review', 'v112-direction-review'), encoding='utf-8')
    receipt = ROOT/'evidence/2026-09-29/v112_fine_control_review/delivery.json'; receipt.parent.mkdir(parents=True, exist_ok=True)
    assert not receipt.exists()
    save(receipt, {'status': 'v112_no_fit_review_and_plan_completed', 'actual_classifier_fits_this_request': 0,
        'actual_calibration_fits': 0, 'quality_acceptance': False, 'model_promoted': False,
        'last_training_delivery': 'v110-delivery', 'summary': summary, 'validation_scope': ver['scope'],
        'artifact_sha256': {**ver['artifact_sha256'], target.relative_to(ROOT).as_posix(): sha(target)}})
    print(json.dumps({'all_checks_passed': True, 'checks': len(checks), 'new_classifier_fits': 0,
        'direction': 'v112-direction-review', 'last_training_delivery': 'v110-delivery'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
