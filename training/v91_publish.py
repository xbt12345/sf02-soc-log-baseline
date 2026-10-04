"""Publish verified no-fit evidence review; retain historical training artifacts."""
from run_v75 import ROOT, read, save, sha

DEST = ROOT / 'artifacts/v91_first_principles_20260928'
DOC = 'docs/V91_FIRST_PRINCIPLES_RESEARCH_AND_PLAN.md'
STATUS = 'first_principles_review_completed_no_new_fit'


def main():
    folder = ROOT / 'evidence/2026-09-28/v91_first_principles'
    assert not (folder / 'delivery.json').exists()
    v = read(DEST / 'verification.json')
    a = read(DEST / 'audit.json')
    context = read(DEST / 'current_input_context.json')
    assert v['status'] == 'passed' and not v['prior_bound_files_changed']
    assert v['source_sha256'] == sha(ROOT / 'training/v91_verify.py')
    assert a['source_sha256'] == sha(ROOT / 'training/v91_evidence_audit.py')
    for name, digest in v['outputs_sha256'].items():
        assert sha(DEST / name) == digest, name
    for name, digest in v['receipt_sha256'].items():
        assert sha(ROOT / name) == digest, name
    prior = {}
    for name in v['receipt_sha256']:
        for path, digest in read(ROOT / name)['artifact_sha256'].items():
            assert path not in prior or prior[path] == digest, path
            prior[path] = digest
    assert len(prior) == v['prior_bound_files_rehashed']
    for name, digest in prior.items():
        assert sha(ROOT / name) == digest, name
    path = ROOT / 'mcp_readonly/catalog.json'
    catalog = read(path)
    assert catalog['project']['authoritative_direction_id'] == 'v90-direction-review'
    for entry in catalog['documents']:
        assert sha(ROOT / entry['path']) == entry['sha256'], entry['path']
    summary = (
        'v9.1本轮新增0次分类器拟合、0次校准拟合。2056871条原标签复核，805个历史绑定文件未变。'
        '38771条ASA训练记录中，当前完整R0经验冲突下界22，部分事实取消哈希仍880；追加事实未降低R0冲突。'
        '52条训练S错误中14条在部分事实下与M混合，两个稀缺S切片各仅2个组件。'
        '改为先审定实际判别证据，再比较自由/保护训练与核实后的输入增量；验证按独立支持组织，最终保护标准不放宽。'
        '方案未训练，最近实际训练仍为v8.9。主模型质量未通过，无新模型晋升。最近完整开发回放仍为v7.9的5947错。'
    )
    files = [p for p in DEST.rglob('*') if p.is_file()]
    files += list((ROOT / 'training').glob('v91*.py')) + [ROOT / DOC]
    bound = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(files)}
    delivery = {
        'status': STATUS, 'quality_acceptance': False, 'all_issues_solved': False,
        'model_promoted': False, 'actual_new_fits': 0, 'actual_classifier_fits': 0,
        'actual_calibration_fits': 0, 'new_full_data_final_fit': False,
        'new_full_development_replay': False, 'new_full_task_pipeline_replay': False,
        'platform_used': False, 'external_training_data_used': False,
        'pseudo_labels_used': False, 'development_not_blind': True,
        'latest_executed_training_delivery': 'evidence/2026-09-27/v89_readout_support/delivery.json',
        'latest_full_development_delivery': 'evidence/2026-09-27/v79_execution/delivery.json',
        'latest_full_development_errors': 5947,
        'validation_scope': (
            'No-fit audit of empirical representation ambiguity and component support. '
            '9 representation-role and 3 ASA bounds independently recomputed; '
            '52 saved S errors replayed and two representative pairs traced to actual corrective text matrices. '
            '805 historical bound files unchanged. Not proof of semantic sufficiency, Bayes error, '
            'new generalization, model quality or a live cloud connection.'
        ),
        'ASA_fit_representation_bounds': context['ASA_fit_representation_bounds'],
        'support_S_error_views': a['support_S_error_views'],
        'fine_S_independent_support': a['fine_S_independent_support'],
        'verification': v, 'current_plan': DOC, 'artifact_sha256': bound,
    }
    folder.mkdir(exist_ok=True, parents=True)
    save(folder / 'delivery.json', delivery)
    for relative, link in [
        ('README.md', DOC),
        ('docs/TRAINING_PLAN.md', 'V91_FIRST_PRINCIPLES_RESEARCH_AND_PLAN.md'),
    ]:
        target = ROOT / relative
        old = target.read_text(encoding='utf-8')
        assert 'v9.1本轮新增' not in old
        title, body = old.split('\n', 1)
        target.write_text(
            title + '\n\n当前第一性原理审查与方案（2026-09-28）：**' + summary
            + '** [判别证据、调研与执行顺序](' + link + ')。以下为历史阶段。\n' + body,
            encoding='utf-8',
        )
    for entry in catalog['documents']:
        if entry['path'] in ['README.md', 'docs/TRAINING_PLAN.md']:
            entry.update(sha256=sha(ROOT / entry['path']), summary=summary)
        if entry['id'] == 'v90-direction-review':
            entry['category'] = 'historical_review'
        if entry['id'] == 'v90-delivery':
            entry['category'] = 'historical_evidence'
    catalog['project'].update(
        as_of='2026-09-28', current_summary=summary,
        authoritative_direction_id='v91-direction-review',
        authoritative_delivery_id='v91-delivery',
        current_direction=[
            '先核实错误输入差异的语义与跨组件支持；区分信息增量、表达增量与来源残片，保留原始记录。',
            '采用同输入等容量的自由/保护诊断；只有核实的遗漏才创建输入增量臂。失败求解不计成有效容量对照。',
            '验证按切片独立支持组织；两个组件不能冒充三方隔离。保留严格逐类晋升、旧模型与全任务验收。',
        ],
        known_limits=[
            '本轮0次拟合，方案未执行；没有已验证的新模型收益。',
            '880是部分事实单独在ASA训练上的经验下界，不是F00整体、原始日志或未来数据的不可约误差。',
            '完整R0输入不同不证明存在可迁移判据；仅两对错误完成实际文本与矩阵深度追踪，语义充分性未证明。',
            '组件为隔离代理；无支持行为和反复观察的开发集不能成为新盲测。',
            'v8.9内层M退化668、S修复38的问题未解决；现有质量未通过，无新模型晋升。',
            '未用外部训练数据或平台；本地只读MCP验证不代表云端Tunnel连接。',
        ],
    )
    entries = [
        {
            'id': 'v91-direction-review', 'title': 'v9.1 判别证据、独立支持与第一性原理训练方案',
            'path': DOC, 'category': 'current_review', 'summary': summary,
            'keywords': ['当前', '最新', '方向', '下一步', '训练', '恶意', '可疑', 'ASA', '根因', '调研', '约束', '独立支持', 'v9.1'],
        },
        {
            'id': 'v91-delivery', 'title': 'v9.1 表示冲突与独立支持的零拟合核验证据',
            'path': (folder / 'delivery.json').relative_to(ROOT).as_posix(),
            'category': 'current_evidence',
            'summary': '0次拟合；2056871条原标签、9组表示下界与3组ASA下界、52条S错误复算，805历史绑定文件未变；未证明模型提升。',
            'keywords': ['证据', '复核', '执行', 'v9.1'],
        },
    ]
    for entry in reversed(entries):
        entry['sha256'] = sha(ROOT / entry['path'])
        catalog['documents'].insert(0, entry)
    save(path, catalog)
    tests = ROOT / 'mcp_readonly/tests/test_readonly_mcp.py'
    text = tests.read_text(encoding='utf-8')
    text = text.replace('v90-direction-review', 'v91-direction-review').replace('v90-delivery', 'v91-delivery')
    text = text.replace('root_cause_review_completed_no_new_fit', STATUS)
    tests.write_text(text, encoding='utf-8')
    print(__import__('json').dumps({
        'published': True, 'new_fits': 0, 'new_bound_files': len(bound),
        'prior_files_unchanged': len(prior), 'quality_acceptance': False,
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
