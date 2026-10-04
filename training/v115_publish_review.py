"""Publish current research direction while retaining the actual V113 delivery."""
import json

from v115_stability_support_audit import ROOT, DEST, save, sha


def main():
    report = ROOT/'docs/V115_STABLE_MS_CRITERIA_REVIEW_AND_PLAN.md'
    j = json.loads((DEST/'diagnosis.json').read_text(encoding='utf-8'))
    s = json.loads((DEST/'supplement.json').read_text(encoding='utf-8'))
    assert s['all_checks_passed'] and j['classifier_fits'] == s['classifier_fits'] == 0
    for rel, digest in s['sources'].items():
        assert sha(ROOT/rel) == digest, rel
    summary = ('V115完成112807条ASA支持与信息保真复算，新增训练0次。'
        '最新实际训练仍V113：3次新拟合，复用旧基线共6次匹配拟合，0次校准；质量未通过、无模型晋升。'
        '已有组成的新组合28936条，旧N1只错20；候选2190错中1338涉及未见具体参数，616属于已见组合。'
        '仅保留正文解析事实的经验最少错误2510，高于旧N1实际2412；不可直接删文字。'
        '下一轮首选固定N1输入与架构，做来源/参数分层内层选模对照；隐藏上下文与标签依据单独处理，不再无差别扩训。')
    receipt = DEST/'review_receipt.json'
    assert not receipt.exists()
    paths = [report, DEST/'diagnosis.json', DEST/'supplement.json', DEST/'verification.json',
             ROOT/'training/v115_stability_support_audit.py', ROOT/'training/v115_verify_support_audit.py',
             ROOT/'training/v115_publish_review.py']
    save(receipt, {'status': 'research_and_verified_diagnosis_complete_training_not_started',
        'actual_classifier_fits_this_request': 0, 'actual_calibration_fits': 0,
        'model_promoted': False, 'new_plan_training_executed': False,
        'latest_actual_training_delivery': 'v113-delivery', 'summary': summary,
        'artifact_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}})
    for name, link in [('README.md', 'docs/'+report.name), ('docs/TRAINING_PLAN.md', report.name)]:
        p = ROOT/name
        old = p.read_text(encoding='utf-8')
        assert report.name not in old
        # Preserve the historical entry and mark its status clearly.
        old = old.replace('当前根因审查与训练方案（V114，未新增训练）', '历史根因审查（V114；下一步已由V115修订）', 1)
        head, body = old.split('\n', 1)
        p.write_text(head+'\n\n当前稳定判据审查与训练方案（V115，未新增训练）：**'+summary+'** '
            '[全量复算、方法评价与下一轮对照]('+link+')。\n'+body, encoding='utf-8')
    cp = ROOT/'mcp_readonly/catalog.json'
    cat = json.loads(cp.read_text(encoding='utf-8'))
    assert cat['project']['authoritative_delivery_id'] == 'v113-delivery'
    assert cat['project']['authoritative_direction_id'] == 'v114-direction-review'
    cat['project'].update(current_summary=summary, authoritative_direction_id='v115-direction-review',
        as_of='2026-09-29', current_direction=[
            '保留N1完整输入与开发参照，V113不晋升；区分新来源、新参数、缺失信息和标签条件改变。',
            '下一轮只检验来源/参数分层内层选模；固定N1架构、原频次CE与最大25epochs，最多六次匹配拟合。',
            '停止组合泛化主攻、删除文字和重复添加ICMP释义；真实上下文与官方M/S依据缺口另行核对。'],
        known_limits=['全部旧外折均反复查看，仍为开发证据；本轮零拟合，不代表泛化改善。',
            '诊断支持键不含全部输入；参数值未见不等于无法编码或必然不能泛化。',
            '事实投影冲突下界只适用于该投影；同事实异标签不能直接判断错标。',
            '选模改进不创造新判别信息；未确认时间/实体真实关联，未添加外部训练数据。'])
    additions = [('v115-direction-review', report, 'V115 跨来源稳定M/S判据审查与方案', 'current_review'),
                 ('v115-diagnosis', DEST/'diagnosis.json', 'V115 全量支持与信息投影诊断', 'current_evidence'),
                 ('v115-supplement', DEST/'supplement.json', 'V115 独立复算与ICMP证据核查', 'current_evidence'),
                 ('v115-review-receipt', receipt, 'V115 零拟合审查凭据', 'current_evidence')]
    for doc in cat['documents']:
        if doc['id'].startswith('v114-'):
            doc['category'] = 'historical_review' if doc['category'] == 'current_review' else 'historical_evidence'
        if doc['path'] in ('README.md', 'docs/TRAINING_PLAN.md'):
            doc.update(summary=summary, sha256=sha(ROOT/doc['path']))
    cat['documents'] = [{'id': i, 'title': title, 'path': p.relative_to(ROOT).as_posix(),
        'category': category, 'summary': summary, 'sha256': sha(p),
        'keywords': ['V115', '当前', 'M/S', '稳定判据', '来源', '参数', '选模', '停止决定']}
        for i, p, title, category in additions]+cat['documents']
    save(cp, cat)
    tests = ROOT/'mcp_readonly/tests/test_readonly_mcp.py'
    text = tests.read_text(encoding='utf-8')
    assert 'v114-direction-review' in text
    tests.write_text(text.replace('v114-direction-review', 'v115-direction-review'), encoding='utf-8')
    print(json.dumps({'published_direction': 'v115-direction-review', 'actual_delivery': 'v113-delivery',
                      'new_fits': 0, 'raw_logs_added_to_MCP': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
