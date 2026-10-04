"""Publish review evidence without changing the latest executed-training receipt."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from v116_preflight import ROOT, DEST as PREV, save, sha
from v117_selection_failure_audit import DEST


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def main():
    report = ROOT / 'docs/V117_REGRESSION_ROOT_CAUSE_AND_BATCH_TRAINING_PLAN.md'
    assert report.exists() and not (DEST / 'review_receipt.json').exists()
    checks = {}
    for name, mapping in [('review_inputs', read(DEST / 'input_receipt.json')),
                          ('member_inputs', read(DEST / 'frozen_member_audit.json')['input_sha256']),
                          ('gradient_inputs', read(DEST / 'gradient_batch_probe_registration.json')['input_sha256']),
                          ('prior_delivery', read(PREV / 'delivery.json')['artifact_sha256'])]:
        checks[name] = all(sha(ROOT / rel) == digest for rel, digest in mapping.items())
    a = read(DEST / 'audit.json')
    g = read(DEST / 'gradient_batch_probe.json')
    members = read(DEST / 'frozen_member_audit.json')
    oof = pd.read_parquet(PREV / 'OOF_ASA_decisions.parquet')
    checks['paired_regression_recomputed'] = int(((oof.truth == 2) & (oof.A_epoch25 == 2) & (oof.B_selected != 2)).sum()) == a['S_regressions'] == 2542
    checks['missing_parameter_regressions_recomputed'] = int(((oof.truth == 2) & oof.parameter.isna() & (oof.A_epoch25 == 2) & (oof.B_selected != 2)).sum()) == a['S_regressions_missing_parameter'] == 632
    checks['gradient_registration_bound'] = sha(DEST / 'gradient_batch_probe_registration.json') == g['registration_sha256']
    checks['gradient_mechanism_passed_not_model_quality'] = bool(g['registered_activation_passed'])
    checks['no_new_fitting_or_parameter_updates'] = a['classifier_fits'] == g['classifier_fits'] == members['new_classifier_fits'] == g['optimizer_steps'] == members['optimizer_steps'] == 0
    checks['parameters_unchanged'] = g['checkpoint_parameters_unchanged'] and members['unchanged_probe_parameters']
    # Probability-curve values are checked against the saved input arrays again.
    q = pd.read_parquet(PREV / 'inner_split_manifest.parquet')
    q = q[q.outer_fold.eq(1) & q.role.eq('U') & q.truth.eq(2)]
    ids = np.load(PREV / 'inner_fold1/inference_local_ids.npy')
    ix = pd.Series(np.arange(len(ids)), index=ids).loc[q.local].to_numpy()
    csv = pd.read_csv(DEST / 'checkpoint_probability_diagnostics.csv')
    for epoch in (15, 25):
        p = np.load(PREV / f'inner_fold1/epoch{epoch}_prob.npy')[ix]
        stat = csv[csv.scope.eq('inner') & csv.fold.eq(1) & csv.role.eq('U') & csv.truth.eq(2) & csv.epoch.eq(epoch)].iloc[0]
        checks[f'inner_U_S_epoch{epoch}_independent_recount'] = int((p.argmax(1) == 2).sum()) == stat.correct and np.isclose(-np.log(p[:, 2].astype(float)).mean(), stat.row_CE)
    checks = {k: bool(v) for k, v in checks.items()}
    assert all(checks.values()), checks
    save(DEST / 'verification.json', {'all_checks_passed': True, 'checks': checks,
        'scope': 'Identity, frozen arithmetic, input conservation and zero-update evidence; not classifier improvement or live cloud MCP acceptance.'})
    evidence = sorted(DEST.glob('*')) + [report, ROOT / 'training/test_v117.py'] + sorted((ROOT / 'training').glob('v117_*.py'))
    save(DEST / 'review_receipt.json', {'status': 'v117_zero_fit_review_and_plan', 'new_classifier_fits': 0,
        'optimizer_steps': 0, 'quality_acceptance': False, 'model_promoted': False,
        'latest_executed_training': 'V116',
        'artifact_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in evidence if p.is_file()}})
    summary = ('V117完成冻结概率、成员与批次梯度复算，新增训练0次、更新0步。最新实际训练仍V116：'
        '6次新拟合（3次内层、3次外层），0次校准；质量未通过、无模型晋升。'
        '第1折过早停止新增2542条S错误，632条缺失参数未进入选模评分；25轮拟合侧仍错531条S，只有4条涉及完整输入冲突。'
        '固定模型质量守恒分批使批次梯度离散度下降32.31%，尚无新分类收益；下一轮仅对照批次组织，停止旧epoch选模分支。')
    for rel, link in [('README.md', 'docs/' + report.name), ('docs/TRAINING_PLAN.md', report.name)]:
        p = ROOT / rel
        old = p.read_text(encoding='utf-8')
        assert report.name not in old
        old = old.replace('当前训练结果与停止决定（V116）', '最新实际训练结果与停止决定（V116）', 1)
        head, body = old.split('\n', 1)
        p.write_text(head + '\n\n当前根因审查与下一轮方案（V117，未新增训练）：**' + summary + '** [证据、研究与训练设计](' + link + ')。\n' + body, encoding='utf-8')
    catalog_path = ROOT / 'mcp_readonly/catalog.json'
    cat = read(catalog_path)
    assert cat['project']['authoritative_delivery_id'] == 'v116-delivery'
    cat['project'].update(as_of='2026-09-29', current_summary=summary,
        authoritative_direction_id='v117-direction-review',
        current_direction=['V116失败候选不晋升，保留N1开发参照，停止epoch选模分支。',
            '下一轮固定原输入、原频次CE、25轮，单独测试质量守恒批次；三折三种子两臂，最多18次拟合。',
            '逐类、缺失参数、支持资格与来源回归均计入；训练侧进步与迁移收益分开判定。'],
        known_limits=['V117只完成冻结诊断，32.31%是单检查点梯度离散度下降，不是准确率提升。',
            '确定性开关未实现数值逐位重复；当前外折已多轮观察，没有新的独立盲测。',
            '参数同类支持与隐藏上下文缺口尚未解决；批次方法不创造安全事实。'])
    for item in cat['documents']:
        if item['id'] == 'v116-stop-decision':
            item['category'] = 'executed_training_review'
        if item['path'] in ('README.md', 'docs/TRAINING_PLAN.md'):
            item.update(summary=summary, sha256=sha(ROOT / item['path']))
    additions = [('v117-direction-review', report, 'V117 退化根因与保质量批次训练方案'),
                 ('v117-review-receipt', DEST / 'review_receipt.json', 'V117 零拟合审查凭据'),
                 ('v117-gradient-probe', DEST / 'gradient_batch_probe.json', 'V117 固定模型批次梯度探针')]
    cat['documents'] = [{'id': i, 'title': title, 'path': p.relative_to(ROOT).as_posix(),
        'category': 'current_review' if i == 'v117-direction-review' else 'current_evidence',
        'summary': summary, 'sha256': sha(p), 'keywords': ['V117', '当前', '退化', '批次', '梯度', 'M/S']}
        for i, p, title in additions] + cat['documents']
    save(catalog_path, cat)
    # Direction changes; the training status and training receipt stay V116.
    tests = ROOT / 'mcp_readonly/tests/test_readonly_mcp.py'
    text = tests.read_text(encoding='utf-8')
    assert 'v116-stop-decision' in text
    tests.write_text(text.replace('v116-stop-decision', 'v117-direction-review'), encoding='utf-8')
    print(json.dumps({'checks': checks, 'direction': 'V117', 'latest_training': 'V116'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
