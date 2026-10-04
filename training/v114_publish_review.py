"""Publish the diagnosis and direction, retaining V113 as actual training delivery."""
import json
from pathlib import Path
from run_v75 import ROOT, save, sha
from v114_mechanism_audit import DEST


def main():
    report = ROOT/'docs/V114_DEEP_MECHANISM_REVIEW_AND_TRAINING_REVISION.md'
    s = json.loads((DEST/'supplement.json').read_text(encoding='utf-8'))
    j = json.loads((DEST/'diagnosis.json').read_text(encoding='utf-8'))
    assert s['all_checks_passed'] and s['classifier_fits'] == j['new_classifier_fits'] == 0
    assert s['diagnosis_sha256'] == sha(DEST/'diagnosis.json')
    assert j['source_sha256'] == sha(ROOT/'training/v114_mechanism_audit.py')
    assert s['source_sha256'] == sha(ROOT/'training/v114_supplement_and_verify.py')
    summary = ('V114完成原文与冻结交叉推理审查，新增训练0次。最新实际训练仍V113：3次新拟合，'
        '复用旧基线共6次匹配拟合，0次校准；质量未通过、无模型晋升。'
        'B在M错误≤318的事后最佳单阈值下，S仍至少错2964；276条S修复全部有嵌套CRED形式。'
        '190条看似有多来源支持的S错中176条目的端口未知，不能当精细行为对照。'
        '先完成一次有界判别证据资格检查；有新增信息或具体表示缺陷再做匹配训练，无资格则停止同观察空间扩训。')
    receipt = DEST/'review_receipt.json'
    assert not receipt.exists()
    paths = [report, DEST/'diagnosis.json', DEST/'supplement.json', DEST/'verification.json',
             ROOT/'training/v114_mechanism_audit.py', ROOT/'training/v114_supplement_and_verify.py']
    save(receipt, {'status': 'diagnostic_review_and_conditional_plan_complete',
        'actual_classifier_fits_this_request': 0, 'actual_calibration_fits': 0,
        'model_promoted': False, 'new_plan_training_executed': False,
        'latest_actual_training_delivery': 'v113-delivery', 'summary': summary,
        'artifact_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}})
    for name, link in [('README.md', 'docs/'+report.name), ('docs/TRAINING_PLAN.md', report.name)]:
        p = ROOT/name; old = p.read_text(encoding='utf-8')
        assert report.name not in old
        head, body = old.split('\n', 1)
        p.write_text(head+'\n\n当前根因审查与训练方案（V114，未新增训练）：**'+summary+'** '
            '[新实证、研究筛选与执行条件]('+link+')。\n'+body, encoding='utf-8')
    catpath = ROOT/'mcp_readonly/catalog.json'
    cat = json.loads(catpath.read_text(encoding='utf-8'))
    assert cat['project']['authoritative_delivery_id'] == 'v113-delivery'
    assert cat['project']['authoritative_direction_id'] == 'v113-stop-decision'
    cat['project'].update(current_summary=summary, authoritative_direction_id='v114-direction-review',
        as_of='2026-09-29', current_direction=[
            '旧N1保持开发参照，V113不晋升；停止大小写调参、无新证据的末层/排序/权重重训。',
            '下一项仅执行有界的判别证据资格检查，区分已观察事实、缺失参数、训练支持与标签依据。',
            '有新增可信行为证据或具体表示缺陷才启动同折单因素训练；否则停止该支线，不猜标签。'],
        known_limits=['旧外折反复使用，只是开发证据；阈值oracle不可部署，也不是校准效果。',
            '当前精细行为缺支持不等于所有输入无信息，26条经验冲突下界不能解释全部2190错。',
            '时间/实体关联与官方M/S判据尚未确认；本轮无新模型训练或泛化保证。'])
    additions = [('v114-direction-review', report, 'V114 深层机制复核与训练方案', 'current_review'),
                 ('v114-diagnosis', DEST/'diagnosis.json', 'V114 冻结交叉与支持诊断', 'current_evidence'),
                 ('v114-supplement', DEST/'supplement.json', 'V114 独立复算及脱敏外观核对', 'current_evidence'),
                 ('v114-review-receipt', receipt, 'V114 零拟合审查凭据', 'current_evidence')]
    for d in cat['documents']:
        if d['id'] == 'v113-stop-decision': d['category'] = 'historical_review'
        if d['path'] in ('README.md', 'docs/TRAINING_PLAN.md'):
            d.update(summary=summary, sha256=sha(ROOT/d['path']))
    cat['documents'] = [{'id': i, 'title': title, 'path': p.relative_to(ROOT).as_posix(),
        'category': category, 'summary': summary, 'sha256': sha(p),
        'keywords': ['V114', '当前', '根因', '训练方案', 'M/S', '支持缺口', '停止决定']}
        for i, p, title, category in additions]+cat['documents']
    save(catpath, cat)
    tests = ROOT/'mcp_readonly/tests/test_readonly_mcp.py'
    old = tests.read_text(encoding='utf-8')
    assert 'v113-stop-decision' in old
    tests.write_text(old.replace('v113-stop-decision', 'v114-direction-review'), encoding='utf-8')
    first = ROOT/'artifacts/v114_mechanism_review_20260929/SUPERSEDED.md'
    first.write_text('本目录是已被修订的首次诊断草稿，不用于当前结论。请使用同级 v114_mechanism_review_20260929_r2。'
        '修正点：不完整行为键不再计作严格零支持；明确未知端口相同不代表行为相同。\n', encoding='utf-8')
    print(json.dumps({'published_direction': 'v114-direction-review',
        'unchanged_training_delivery': 'v113-delivery', 'new_fits': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
