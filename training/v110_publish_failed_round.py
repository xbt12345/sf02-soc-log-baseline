"""Publish the executed but failed V110 round without promoting either probe."""
import json
from datetime import datetime
from pathlib import Path

from run_v75 import ROOT, save, sha
from v110_layer_probes import DEST


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def main():
    ver=read(DEST/'verification.json');e=read(DEST/'evaluation.json')
    assert ver['all_checks_passed'] and ver['new_classifier_fits']==6 and not ver['model_promoted']
    assert e['actual_classifier_fits']==6 and not e['model_promoted']
    assert all(not e['gates'][a]['research_continuation_preliminary'] for a in ('P1','P2'))
    assert all(sha(ROOT/p)==h for p,h in ver['artifact_sha256'].items())
    report='docs/V110_LAYER_PROBE_TRAINING_RESULTS_AND_STOP_DECISION.md'
    summary=('v11.0按锁定方案完成N2两层读出×三折共6次拟合，0次校准，六次均数值收敛。'
      '但P1/P2的ASA M错误为1954/1666、S错误为4793/5068；旧N1为318/2094。'
      'P2能线性重建旧M/S平均logit，较低训练目标却对应更差来源外结果；'
      '两臂全部未过研究继续或正式替换门槛，质量未通过、无模型晋升。'
      '下一步先核清具体行为的跨来源M/S监督支持，再提名一个可验证的新证据因素。')
    for name,link in [('README.md',report),('docs/TRAINING_PLAN.md',Path(report).name)]:
        p=ROOT/name;s=p.read_text(encoding='utf-8');head,body=s.split('\n',1)
        assert 'V110_LAYER_PROBE' not in s
        body=body.replace('当前方向审查（v10.9，未新增训练）','历史方向审查（v10.9；首选实验已由v11.0执行并失败）',1)
        p.write_text(head+'\n\n当前训练结果与停止决定（v11.0）：**'+summary+'** [六次训练、根因核查与下一步边界]('+link+')。\n'+body,encoding='utf-8')
    receipt=ROOT/'evidence/2026-09-29/v110_layer_probes/delivery.json'
    assert not receipt.exists();receipt.parent.mkdir(parents=True,exist_ok=True)
    paths=[ROOT/report,DEST/'registration.json',DEST/'evaluation.json',DEST/'verification.json',
      DEST/'postfit_diagnosis_r2.json',DEST/'original_readout_feasibility.json',
      DEST/'OOF_probe_comparison.parquet',
      ROOT/'training/v110_layer_probes.py',ROOT/'training/v110_layer_probes_evaluate.py',
      ROOT/'training/v110_postfit_diagnosis.py',ROOT/'training/v110_original_readout_feasibility.py',
      ROOT/'training/v110_verify_delivery.py']
    for k in (0,1,2):
        for arm in ('P1','P2'):
            paths.extend([DEST/f'fold{k}_{arm}'/n for n in ('fit.json','probe.npz','ASA_input_S_probability.npy')])
    save(receipt,{'status':'v110_layer_probe_fits_executed_failed_quality_gates',
      'created_at':datetime.now().astimezone().isoformat(),
      'actual_classifier_fits_this_request':6,'actual_calibration_fits':0,
      'official_training_rows':2056871,'same_ASA_OOF_rows':112807,
      'validation_scope':'Source-proxy-closed three-fold internal OOF, all folds previously studied; full original-row replay but no external blind validation.',
      'quality_acceptance':False,'model_promoted':False,
      'best_still_for_development':'V107_N1_full_composite','official_unknown_test_used':False,
      'platform_used':False,'external_training_data_used':False,'private_answer_used':False,
      'source_sha256':sha(__file__),'summary':summary,
      'note':'V109 plan was executed in full. Both probes fail the predeclared M/S and full-task gates. No model is promoted.',
      'audit_corrections':['The first postfit diagnostic used a nonexistent 65536 behavior-key sentinel and gave an empty no-port slice. postfit_diagnosis_r2.json supersedes it with a correctly scoped absent-complete-port slice.',
        'The original-readout feasibility check first used len(scipy CSR), then used shape[0]; no model fit or result changed.'],
      'artifact_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in paths}})
    cp=ROOT/'mcp_readonly/catalog.json';cat=read(cp)
    assert cat['project']['authoritative_delivery_id']=='v107-delivery'
    assert cat['project']['authoritative_direction_id']=='v109-direction-review'
    cat['project'].update(as_of='2026-09-29',current_summary=summary,
      authoritative_delivery_id='v110-delivery',authoritative_direction_id='v110-stop-decision',
      current_direction=['停止N2同频次早层/末层单线性读出扩训：六次收敛拟合均未过M/S门槛，保留旧N1。',
        '先核查薄弱行为的跨来源双类监督与原文可见证据，特别是UDP 514和缺少完整目的端口的S；再登记单因素对照。',
        '下一轮必须以每类原行正确数、来源组和完整三分类对N1的真实提升为依据；现有折是开发证据。'],
      known_limits=['P1/P2用更低训练目标得到较差来源外M/S边界，表明此读出路径未解决迁移；不证明所有线性模型或官方数据无解。',
        '根2868同完整行为训练侧仅2条M和24条S跨来源对照；缺少固定端口的行为键不等于全部端口被脱敏。',
        '无官方未知测试，也无官方细化M/S规则；模型质量未验收，不能外推真实环境表现。'])
    additions=[('v110-delivery','v11.0 六次读出训练交付凭据',receipt.relative_to(ROOT).as_posix(),'current_delivery',summary),
      ('v110-stop-decision','v11.0 训练结果与停止决定',report,'current_review',summary),
      ('v110-evaluation','v11.0 全量评估与质量门槛','artifacts/v110_layer_probes_20260929/evaluation.json','current_evidence','P1/P2两层六次拟合后的ASA和全部官方行评估，均未过晋升门槛。'),
      ('v110-readout-audit','v11.0 原读出可表达性核查','artifacts/v110_layer_probes_20260929/original_readout_feasibility.json','current_evidence','旧N2平均logit可在P2线性表示中重建；新头训练目标更低，来源外错误更多。'),
      ('v110-verification','v11.0 训练凭据独立校验','artifacts/v110_layer_probes_20260929/verification.json','current_evidence','48项运行身份和指标核查通过，不表示模型质量合格。')]
    for item in cat['documents']:
        if item['id']=='v109-direction-review':item['category']='historical_review'
        if item['id']=='v109-training-contract':item['category']='historical_plan'
        if item['id']=='training-plan':item.update(summary=summary,sha256=sha(ROOT/item['path']))
    cat['documents']=[{'id':i,'title':t,'path':p,'category':c,'summary':s,
      'keywords':['v11.0','当前','训练失败','M/S','来源','停止','模型能力','下一步方向'],
      'sha256':sha(ROOT/p)} for i,t,p,c,s in additions]+cat['documents']
    save(cp,cat)
    test=ROOT/'mcp_readonly/tests/test_readonly_mcp.py'
    content=test.read_text(encoding='utf-8')
    content=content.replace('v109-direction-review','v110-stop-decision')
    content=content.replace('v107-delivery','v110-delivery')
    content=content.replace('v107_matched_training_executed_no_promotion','v110_layer_probe_fits_executed_failed_quality_gates')
    content=content.replace('"底层原因"','"停止决定"')
    test.write_text(content,encoding='utf-8')
    print(json.dumps({'status':'published_no_promotion','delivery':'v110-delivery',
      'direction':'v110-stop-decision','training_fits':6,'quality_acceptance':False},ensure_ascii=False))


if __name__=='__main__':main()
