"""Publish V146 actual receipts without modifying sealed training inputs."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/v146_guarded_pair_training_20261001'
REPORT = ROOT / 'docs/V146_GUARDED_PAIR_TRAINING_RESULTS.md'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    d, q, v = [read(OUT / n) for n in ['final_delivery.json', 'quality.json', 'verification.json']]
    audit = read(OUT / 'objective_journal_audit.json')
    support = read(OUT / 'support_effect_audit.json')
    assert d['latest_actual'] == 'V146' and v['official_rows'] == 2056871
    assert not REPORT.exists(), 'Preserve a previously published actual report'
    summary = (f"V146完成6次既有第二层拟合、{d['full_gradient_evaluations']}次完整梯度、"
        f"{d['proposal_evaluations']}次有限试探、{d['accepted_updates']}次接受更新。"
        f"A/B注册纯TRAIN错均0、已正确原行新增错均0；ASA M/S错A={d['ASA_errors']['A']['M']}/{d['ASA_errors']['A']['S']}，"
        f"B={d['ASA_errors']['B']['M']}/{d['ASA_errors']['B']['S']}。匹配作用验收{'通过' if d['matched_effect_passed'] else '失败'}，"
        f"完整2056871行质量{'通过' if d['quality_acceptance'] else '失败'}，无模型晋升。第二、第三问题未关闭。")
    lines = ['# V146 分类保护与真实总目标回溯：实际训练结果', '', '2026-10-01。' + summary, '',
        '第一问题的结论仍仅限注册 M/S TRAIN；本轮验证其保留。B 是事前登记候选，不因 A/B、V142 或 A0 的后验结果更换终点或系数。', '',
        '## 实际执行与保护', '',
        '| 来源角色 | 组 | 完整梯度 | 有限试探 | 接受更新 | TRAIN M/S 错 | 末五状态 |',
        '|---|---|---:|---:|---:|---|---|']
    for z in v['audits']:
        s = z['endpoint_stats']
        lines.append(f"| {z['fold']} | {z['arm']} | {z['gradient_evaluations']} | {z['proposal_evaluations']} | {z['accepted_updates']} | {s['M_errors']}/{s['S_errors']} | {'通过' if z['stable_window'] else '失败'} |")
    lines += ['', f"运行前封存 {len(read(OUT/'run_seal.json')['source_sha256'])} 个物理文件；所有终点与末五个不同状态实际重放，联合 V142/V140/V138 正确原行 guard。每个模型另对全部 22546 种数值输入重新执行无缓存函数，分类完全一致。该证据不包含外部新日志部署验收。",
        '', 'A 使用原始频次成员 CE；B 只加入既定合法配对辅助，固定系数来自 V144 合法 TRAIN 初始梯度。共同使用真实目标梯度和 Armijo 回溯，每个试探要求总目标下降且实际训练分类保留。第一层、读出、原始输入/频次和模型容量不变。', '',
        f"预检另计 12 次完整 CE 梯度、6 次辅助梯度，0 更新；正式训练辅助梯度 {v['auxiliary_gradient_evaluations']} 次。全部接受与拒绝试探逐项重算 CE、辅助、封存系数和总目标，最大目标分解偏差 {max(z['objective_reconstruction_max_gap'] for z in audit['reports']):.3g}。原日志未修改；带系数的派生审计位于 all_proposals_with_bound_lambda.jsonl。",
        '', '## 完整逐行分类与三种比较', '', '| 比较：B 对参照 | M 修复 | M 新增错 | S 修复 | S 新增错 |', '|---|---:|---:|---:|---:|']
    for name, p in d['original_pairs'].items():
        lines.append(f"| {name} | {p['1']['repairs']} | {p['1']['new_errors']} | {p['2']['repairs']} | {p['2']['new_errors']} |")
    lines += ['', '| 来源折 | 匹配 A 错 | B 错 |', '|---|---:|---:|']
    for z in q['matched']['folds']:
        lines.append(f"| {z['fold']} | {z['A_errors']} | {z['B_errors']} |")
    lines += ['', '| 完整任务类别 | A 漏判 | B 漏判 | B precision | B recall | B F1 |', '|---|---:|---:|---:|---:|---:|']
    for c, label in [('0', 'N'), ('1', 'M'), ('2', 'S')]:
        a, b = q['matched']['full_task']['A'][c], q['matched']['full_task']['B'][c]
        lines.append(f"| {label} | {a['missed']} | {b['missed']} | {b['precision']:.6f} | {b['recall']:.6f} | {b['f1']:.6f} |")
    lines += ['', '匹配作用门槛：' + json.dumps(q['matched_gates'], ensure_ascii=False) + '。', '',
        'B 对正式 A0 的失败门槛：' + '、'.join(k for k, ok in q['B']['gates'].items() if not ok) + '。正常与其他格式完整保留，非 ASA 错误仍固定 107；未放宽原 A0 门槛。', '',
        '## 细支持与决策边界', '',
        '合法辅助 S 原行仅 34/18/26。578 条 V142 已知双端口困难 S 对登记配对的直接覆盖为 0；本轮没有生成新标签或补齐同类来源。该群体的实际 B 对 V142 修复/新增错误：' + json.dumps(support['original_578_known_port_S_errors']['B_vs_V142'], ensure_ascii=False) + '。', '',
        '来源开发折已反复查看，不是独立盲测；损失下降、几何作用、训练分类保留和完整任务采用分别判定。所有不利折、未知/缺失/冲突行和原频次保留。', '',
        ('该受限辅助路径已完成六次匹配试验且未通过预登记作用门槛，停止此路径；不扫描系数、换种子或追加同配置拟合。' if not d['matched_effect_passed'] else '匹配作用通过仅说明本轮间接分类收益，未补齐真实支持；完整任务门槛与未覆盖支持仍须独立解决。'),
        '完整三问题目标保持 active。下一步根据逐来源实际修复/退化和原始行为证据寻找能覆盖困难组的独立支持机制；尚无新的拟合登记，第三问题与完整任务均未通过。', '',
        '实际凭据：artifacts/v146_guarded_pair_training_20261001/{final_delivery,quality,verification,objective_journal_audit,support_effect_audit}.json；全量和 ASA 逐行账本、六终点与窗口、所有梯度和试探日志均保留。', '',
        '历史依据：[V142实际训练](V142_SECOND_LAYER_TRAINING_RESULTS.md)、[V143支持资格](V143_FINE_SUPPORT_QUALIFICATION.md)、[V144/V145原失败](V145_AUXILIARY_DIRECTION_AND_FINITE_QUALIFICATION.md)、[V146事前决定](V146_GUARDED_PAIR_PRETRAIN_DECISION.md)、[独立执行前复查](V146_INDEPENDENT_PRETRAIN_REVIEW.md)。', '']
    REPORT.write_text('\n'.join(lines), encoding='utf-8')
    p = ROOT / 'README.md'; old = p.read_text(encoding='utf-8').splitlines(); old[2] = '最新实际与当前方向（V146）：**' + summary + '** [实际六拟合与全量验收](docs/V146_GUARDED_PAIR_TRAINING_RESULTS.md)。'
    p.write_text('\n'.join(old) + '\n', encoding='utf-8')
    (ROOT / 'HANDOFF.md').write_text('\n'.join(['# SF02 续接：V146 六拟合与完整验收', '', summary, '',
        '完整三问题目标保持 active。第一问题仅注册 TRAIN 范围通过并受保护；第二、第三和完整任务未关闭。父会话只读，执行独占 GPU；不得自动委派。', '',
        '先读 AGENTS.md、docs/EXPERIMENT_REVIEW_RULES.md、本轮结果报告及只读 MCP 最新实际。使用 .venv-v61/Scripts/python.exe -X utf8；训练入口/计划/封存/原始日志不可修改或重启。', '',
        '本轮六终点全部为预登记最后接受状态；预算不续、不挑检查点或系数。v146_evaluate.py 已完成全部 2056871 原行、六函数无缓存重放和末五状态联合旧保护。', '',
        '三种逐类修复/新增错误、失败门槛及原 578 困难 S 群体现见 docs/V146_GUARDED_PAIR_TRAINING_RESULTS.md。训练 CE、辅助和参数实际改变，分类保护通过不代表细支持或来源外通过。', '',
        '后续方向按本轮失败/边界处理；无新拟合登记，不追加本受限辅助路径。检查实际原始行为与合法同类来源证据后再设计独立机制。不得用 HELD 错误调权、伪造标签、删除难例或放宽正式 A0。', '',
        '保留原末层链 12 拟合耗尽及 V142 三拟合；本轮新增六既有第二层拟合，新增参数/头拟合 0。V142 225558 正确 TRAIN 角色行（112779 独立原行）与 V140/V138 旧范围必须继续调用 v142_retention_check.py 联合保护。', '',
        '父会话新独立审查文件归父会话维护，不覆写；执行追加 objective_journal_audit/support_effect_audit，0新拟合0更新。MCP仅本地只读查询，不等于正式提交或远端平台验证。', '']), encoding='utf-8')
    catalog_path = ROOT / 'mcp_readonly/catalog.json'; cat = read(catalog_path)
    cat['project'].update(current_summary=summary, authoritative_delivery_id='v146-delivery', authoritative_direction_id='v146-direction-review',
        current_direction=['第一问题仅注册 M/S TRAIN 通过，本轮 A/B 完整保留；继续联合 V142/V140/V138 guard。',
            'V146 六拟合与全部原行评分已完成，按预登记结果处理受限辅助路径；不得继续同配置或扫系数/种子。',
            '第二问题仍需能覆盖困难行为的真实合法同类支持证据；第三问题和完整任务开放，无新拟合登记。'],
        known_limits=['已查看开发折，不是独立盲测或外部部署。', '训练分类保留和总目标下降不替代来源外分类。',
            '原 578 困难 S 对登记辅助配对直接覆盖 0；本轮不产生新的同类标签。', '正常/其他格式冻结 107 错，无模型晋升；正式 A0 质量门槛保持。',
            '原末层 12 拟合、V142 三拟合和 V146 六拟合均不自动追加。'])
    entries = [('v146-direction-review', 'docs/V146_GUARDED_PAIR_TRAINING_RESULTS.md', 'V146实际六拟合、完整失败门槛与当前方向', 'review_evidence'),
        ('v146-delivery', 'artifacts/v146_guarded_pair_training_20261001/final_delivery.json', 'V146六拟合实际交付', 'actual_delivery'),
        ('v146-quality', 'artifacts/v146_guarded_pair_training_20261001/quality.json', 'V146匹配和A0完整分类', 'review_evidence'),
        ('v146-verification', 'artifacts/v146_guarded_pair_training_20261001/verification.json', 'V146实际窗口与无缓存函数验收', 'review_evidence'),
        ('v146-objective-audit', 'artifacts/v146_guarded_pair_training_20261001/objective_journal_audit.json', 'V146每试探固定系数与目标重算', 'review_evidence'),
        ('v146-support-effect', 'artifacts/v146_guarded_pair_training_20261001/support_effect_audit.json', 'V146完整支持分层实际分类作用', 'review_evidence')]
    for ident, path, title, category in entries:
        assert not any(z['id'] == ident for z in cat['documents'])
        cat['documents'].append(dict(id=ident, path=path, title=title, category=category, summary=summary,
            sha256=sha(ROOT/path), keywords=['V146', '训练', '分类保护', '实际结果', '细支持', '当前方向']))
    for z in cat['documents']:
        if z['path'] in ['README.md', 'HANDOFF.md']:
            z['sha256'] = sha(ROOT/z['path'])
    catalog_path.write_text(json.dumps(cat, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(summary)


if __name__ == '__main__':
    main()
