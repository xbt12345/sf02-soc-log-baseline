"""Publish the verified V113 training failure to the read-only project status."""
import json
from pathlib import Path
from run_v75 import ROOT,save,sha
from v113_case_train import DEST,check


REPORT='docs/V113_CASE_TRAINING_RESULTS_AND_STOP.md'
STATUS='v113_case_only_fits_executed_failed_quality_gates'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def main():
    check()
    eval_=read(DEST/'evaluation.json')
    ver=read(DEST/'verification.json')
    diag=read(DEST/'postfit_diagnosis.json')
    assert ver['all_checks_passed'] and not eval_['all_gates_passed']
    assert eval_['classifier_fits_new']==3 and eval_['baseline_fits_reused']==3
    assert eval_['calibration_fits']==0 and not eval_['model_promoted']
    assert diag['S_repairs_total']==276 and diag['S_repairs_with_unchanged_current_input']==276
    assert all(sha(ROOT/p)==h for p,h in ver['artifact_sha256'].items())
    assert sha(DEST/'evaluation.json')==diag['evaluation_sha256']
    summary=('v11.3已在本机完成3次新拟合，复用N1三折基线共6次匹配拟合，0次校准；质量未通过、无模型晋升。'
        '仅ASA接口大小写规范化使ASA总错误2412→2190，S错误2094→1820，但M错误318→370。'
        '276条S修复全部集中一个来源且原输入未改，其他来源S略退化；改动未形成稳定跨来源收益。'
        '保留旧N1开发参照，停止大小写调参；下一步先核对原文中真实行为/上下文差异。')
    receipt=ROOT/'evidence/2026-09-29/v113_case_training/delivery.json'
    assert not receipt.exists();receipt.parent.mkdir(parents=True,exist_ok=True)
    files=[ROOT/REPORT,DEST/'evaluation.json',DEST/'verification.json',
        DEST/'postfit_diagnosis.json',ROOT/'training/v113_postfit_diagnosis.py']
    save(receipt,{'status':STATUS,'actual_classifier_fits_this_request':3,
        'baseline_fits_reused':3,'matched_arm_fits_in_comparison':6,
        'actual_calibration_fits':0,'quality_acceptance':False,'model_promoted':False,
        'best_still_for_development':'V107_N1_full_composite',
        'official_unknown_test_used':False,'platform_used':False,
        'external_training_data_used':False,'private_answer_used':False,
        'summary':summary,
        'ASA_M_errors':eval_['ASA']['class_comparison']['M']['B_errors'],
        'ASA_S_errors':eval_['ASA']['class_comparison']['S']['B_errors'],
        'gate_checks':eval_['gates'],
        'validation_scope':ver['scope'],
        'artifact_sha256':{**ver['artifact_sha256'],
            **{p.relative_to(ROOT).as_posix():sha(p) for p in files}}})
    for name,link in [('README.md',REPORT),('docs/TRAINING_PLAN.md',Path(REPORT).name)]:
        p=ROOT/name
        s=p.read_text(encoding='utf-8');head,body=s.split('\n',1)
        assert 'V113_CASE_TRAINING' not in s
        body=body.replace('当前精细对照与训练方案（v11.2，未新增训练）',
            '上一轮精细对照与训练方案（v11.2，已由v11.3执行）',1)
        p.write_text(head+'\n\n当前训练结果与停止决定（v11.3）：**'+summary+'** '
            '[三折训练、逐类结果与下一步资格]('+link+')。\n'+body,encoding='utf-8')
    catpath=ROOT/'mcp_readonly/catalog.json';cat=read(catpath)
    assert cat['project']['authoritative_delivery_id']=='v110-delivery'
    assert cat['project']['authoritative_direction_id']=='v112-direction-review'
    cat['project'].update(as_of='2026-09-29',current_summary=summary,
        authoritative_delivery_id='v113-delivery',authoritative_direction_id='v113-stop-decision',
        current_direction=['V113大小写候选未过逐类及跨来源门槛；仍以旧N1为开发参照，无模型晋升。',
            '停止大小写相关调参和全名屏蔽；对单来源S修复及52条新增M错回查官方原文中的行为/上下文差异。',
            '下一次训练先登记独立证据的来源覆盖、精细M/S反例和解析状态，再做同划分单因素比较。'],
        known_limits=['本次只有已看来源外开发折，未做新环境盲测或官方未知集提交。',
            'Cisco支持接口名大小写同义，但本轮S收益仅单来源，不能推断已学到安全判断依据。',
            '真实上下文是否可由脱敏字段关联、官方M/S规则均未确认；原文缺失与模型能力仍需区分。'])
    additions=[('v113-stop-decision','v11.3 大小写单因素训练结果与停止决定',REPORT,'current_review',summary),
        ('v113-delivery','v11.3 训练交付与未晋升凭据',receipt.relative_to(ROOT).as_posix(),'current_evidence','三次新拟合，三次旧基线复用；逐类门槛失败。'),
        ('v113-evaluation','v11.3 三折及完整三分类评估','artifacts/v113_case_training_20260929/evaluation.json','current_evidence','ASA S错误2094→1820，M错误318→370，三类完整回放。'),
        ('v113-postfit','v11.3 单来源S修复与M退化定位','artifacts/v113_case_training_20260929/postfit_diagnosis.json','current_evidence','276条S修复均来源29且本次输入未改；52条M新增错误。'),
        ('v113-verification','v11.3 独立验证','artifacts/v113_case_training_20260929/verification.json','current_evidence','官方标签、旧基线及完整回放、逐行决策、来源组和大小写同义输入核对。')]
    for item in cat['documents']:
        if item['id']=='v112-direction-review':item['category']='historical_review'
        if item['path'] in ('README.md','docs/TRAINING_PLAN.md'):
            item.update(summary=summary,sha256=sha(ROOT/item['path']))
    cat['documents']=[{'id':i,'title':t,'path':p,'category':c,'summary':s,
        'keywords':['v11.3','当前','训练结果','停止决定','M/S','大小写','来源外','下一步方向'],
        'sha256':sha(ROOT/p)} for i,t,p,c,s in additions]+cat['documents']
    save(catpath,cat)
    test=ROOT/'mcp_readonly/tests/test_readonly_mcp.py'
    s=test.read_text(encoding='utf-8')
    assert 'v112-direction-review' in s and 'v110-delivery' in s
    s=s.replace('v112-direction-review','v113-stop-decision')
    s=s.replace('v110-delivery','v113-delivery')
    s=s.replace('v110_layer_probe_fits_executed_failed_quality_gates',STATUS)
    test.write_text(s,encoding='utf-8')
    print(json.dumps({'status':'published_no_promotion','delivery':'v113-delivery',
        'direction':'v113-stop-decision','new_fits':3,'reused_fits':3,
        'all_quality_gates_passed':False},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
