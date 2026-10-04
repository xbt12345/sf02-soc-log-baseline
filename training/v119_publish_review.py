"""Publish the installed review rules without reporting a new trained model."""
import json
import subprocess
import sys
import pandas as pd
from experiment_review import ROOT, read, sha, review_plan, evaluate_primary
from v119_prepare_review import POLICY, DEST, save


def main():
    assert not (DEST / 'review_receipt.json').exists()
    plan = read(POLICY / 'next_batch_plan.json'); review = review_plan(plan)
    assert review['plan_review_passed'] and not review['quality_acceptance']
    test = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'training', '-p', 'test_experiment_review.py'],
                          cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
    save(DEST / 'regression_tests.json', {'exit_code': test.returncode, 'stdout': test.stdout, 'stderr': test.stderr,
                                        'new_classifier_fits': 0, 'scope': 'Gate logic, historical data recount and inert-file binding; no classifier training.'})
    assert test.returncode == 0
    ref_path = ROOT / 'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet'
    ref = pd.read_parquet(ref_path, columns=['row_position', 'fold', 'root', 'truth']); ref['route'] = 'asa'
    historical = {}
    for version, rel, columns in [
        ('V113', 'artifacts/v113_case_training_20260929/OOF_ASA_decisions.parquet', {'A_N1_prediction': 'pred_A', 'B_case_prediction': 'pred_B'}),
        ('V116', 'artifacts/v116_nested_selection_20260929/OOF_ASA_decisions.parquet', {'A_epoch25': 'pred_A', 'B_selected': 'pred_B'})]:
        p = ROOT / rel; d = pd.read_parquet(p).rename(columns=columns)
        result = evaluate_primary(ref, d, plan['quality_profile'])
        assert not result['gates']['paired_M_S_protected']
        assert not result['primary_quality_passed']
        historical[version] = {'prediction_source': rel, 'prediction_sha256': sha(p), 'result': result}
    save(DEST / 'historical_prediction_replay.json', {'reference_sha256': sha(ref_path), 'cases': historical,
        'scope': 'Actual ASA rows only; full-population gate correctly remains false. This replay does not pretend ASA rows are the full task.'})
    report = ROOT / 'docs/V119_REVIEW_GATES_IMPLEMENTATION.md'
    evidence = list(DEST.glob('*.json')) + list(POLICY.glob('*.json')) + [
        report, ROOT / 'docs/EXPERIMENT_REVIEW_RULES.md', ROOT / 'AGENTS.md', ROOT / 'training/experiment_review.py',
        ROOT / 'training/test_experiment_review.py', ROOT / 'training/v119_prepare_review.py', ROOT / 'training/v119_publish_review.py']
    save(DEST / 'review_receipt.json', {'status': 'v119_review_rules_installed_and_replayed',
        'classifier_fits_new': 0, 'optimizer_steps': 0, 'quality_acceptance': False, 'model_promoted': False,
        'latest_actual_training': 'V116', 'actual_next_trainer_integrated': False,
        'artifact_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in evidence}})
    summary = ('V119将9项历史风险落实为选模审查、运行绑定和逐行质量检查，14项回归测试通过，新增训练0次。'
        '当前批次方案仅通过开展受限试验的前置审查，真实下一轮训练器尚未接入执行；最新实际训练仍V116：'
        '6次新拟合（3次内层、3次外层），0次校准，质量未通过、无模型晋升。'
        '后续遵循项目审查规则，固定终点属于当前实验，先6次主对照、实际门槛通过再12次确认；不得事后改门槛或隐藏反例。')
    for rel, link in [('README.md', 'docs/' + report.name), ('docs/TRAINING_PLAN.md', report.name)]:
        p = ROOT / rel; text = p.read_text(encoding='utf-8'); assert report.name not in text
        text = text.replace('当前复盘与修订训练方案（V118）', '下一轮具体训练设计（V118；执行审查由V119约束）', 1)
        head, body = text.split('\n', 1)
        p.write_text(head + '\n\n当前审查约束与执行入口（V119）：**' + summary + '** [规则落地与历史回放](' + link + ')。\n' + body, encoding='utf-8')
    p = ROOT / 'mcp_readonly/catalog.json'; catalog = read(p)
    assert catalog['project']['authoritative_delivery_id'] == 'v116-delivery'
    catalog['project'].update(current_summary=summary, authoritative_direction_id='v119-review-rules',
        current_direction=['执行前读取项目审查规则，绑定方案、真实训练器、依赖及数据划分，风险必须对应具体动作。',
            '试验资格、真实分类质量、确认与晋升分别判断；从完整逐行预测重算每类与来源结果。',
            '当前下一轮仍是V118批次两臂：先6次、通过实际门槛再12次，不启用最早平手检查点。'],
        known_limits=['14项测试证明已覆盖的审查行为，不证明分类模型改进或未来规则全部正确。',
            '新训练器还未实现接入；执行绑定不能自动发现漏报依赖或阻止绕过入口运行旧脚本。',
            '数值非确定性、低支持切片及隐藏上下文未解决；第2折梯度反例继续保留。'])
    for entry in catalog['documents']:
        if entry['id'] == 'v118-direction-review': entry['category'] = 'active_experiment_plan'
        if entry['path'] in ('README.md', 'docs/TRAINING_PLAN.md'):
            entry.update(summary=summary, sha256=sha(ROOT / entry['path']))
    entries = [('v119-review-rules', report, 'V119 历史反例与审查约束落地'),
               ('experiment-review-rules', ROOT / 'docs/EXPERIMENT_REVIEW_RULES.md', 'SF02 每轮实验审查规则'),
               ('experiment-risk-cases', POLICY / 'history_cases.json', 'SF02 历史风险与执行动作'),
               ('next-batch-review-plan', POLICY / 'next_batch_plan.json', '下一轮批次实验机器审查方案'),
               ('v119-review-receipt', DEST / 'review_receipt.json', 'V119 审查落地核验凭据')]
    catalog['documents'] = [{'id': ident, 'title': title, 'path': path.relative_to(ROOT).as_posix(),
        'category': 'current_review' if ident == 'v119-review-rules' else 'current_evidence',
        'summary': summary, 'sha256': sha(path), 'keywords': ['V119', '当前', '审查', '反例', '执行约束', '停止决定']}
        for ident, path, title in entries] + catalog['documents']
    save(p, catalog)
    p = ROOT / 'mcp_readonly/tests/test_readonly_mcp.py'; text = p.read_text(encoding='utf-8')
    assert 'v118-direction-review' in text
    p.write_text(text.replace('v118-direction-review', 'v119-review-rules').replace('self.assertIn("停止决定",', 'self.assertIn("执行约束",'), encoding='utf-8')
    print(json.dumps({'review': 'V119', 'risk_cases': 9, 'regression_tests_exit_code': test.returncode,
                      'new_fits': 0, 'actual_training': 'V116', 'next_trainer_integrated': False}))


if __name__ == '__main__':
    main()
