"""Verify the no-fit review; update the read-only direction, not training delivery."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from run_v75 import ROOT,save,sha
from v110_layer_probes import LEDGER,DEST as V110,check_registration

DEST=ROOT/'artifacts/v111_root_review_20260929'
REPORT='docs/V111_ROOT_MECHANISM_REVIEW_AND_NEXT_PLAN.md'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def main():
    check_registration()
    target=DEST/'verification.json';assert not target.exists()
    checks={}
    previous=ROOT/'evidence/2026-09-29/v110_layer_probes/delivery.json'
    old=read(previous)
    checks['previous_training_delivery_unchanged']=all(sha(ROOT/p)==h for p,h in old['artifact_sha256'].items())
    for fn in ['objective_and_margin_diagnosis.json','support_eligibility.json','interface_sensitivity.json','N1_interface_control.json']:
        r=read(DEST/fn)
        checks[fn+'_zero_fits']=r['classifier_fits']==r['calibration_fits']==0
        hashes=r.get('input_hashes',{})
        checks[fn+'_input_hashes']=all(sha(ROOT/p)==h for p,h in hashes.items())
    diag=read(DEST/'objective_and_margin_diagnosis.json')
    checks['decomposition_outputs_unchanged']=all(sha(DEST/p)==h for p,h in diag['output_hashes'].items())
    source_names={'objective_and_margin_diagnosis.json':'v111_objective_diagnosis.py',
      'support_eligibility.json':'v111_support_eligibility.py',
      'interface_sensitivity.json':'v111_interface_sensitivity.py',
      'N1_interface_control.json':'v111_n1_interface_control.py'}
    checks['executed_diagnostic_source_hashes']=all(read(DEST/p)['source_sha256']==sha(ROOT/'training'/s) for p,s in source_names.items())
    d=pd.read_parquet(LEDGER)
    m=pd.read_parquet(DEST/'P2_margin_decomposition.parquet')
    o=pd.read_parquet(V110/'OOF_probe_comparison.parquet')
    checks['all_112807_rows_and_labels_aligned']=(len(m)==112807 and m.row_position.nunique()==112807
      and np.array_equal(m.row_position,d.row_position) and np.array_equal(m.truth,d.truth))
    official=pd.read_parquet(ROOT/'data/official/train.parquet',columns=['label_binary'])
    y=official.label_binary.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy()
    checks['official_labels_independently_recounted']=bool(np.array_equal(y[d.row_position.to_numpy()],d.truth.to_numpy()))
    pred=np.where(m.P2_margin.to_numpy()>=0,2,1)
    checks['P2_all_original_row_decisions_match']=bool(np.array_equal(pred,o.P2_prediction))
    checks['P2_class_error_recount']=int(((d.truth==1)&(pred!=1)).sum())==1666 and int(((d.truth==2)&(pred!=2)).sum())==5068
    delta=sum(m['delta_'+b].to_numpy() for b in ['hidden','parsed_facts','record_source_port'])+m.delta_intercept.to_numpy()
    checks['all_margin_changes_equal_sum']=bool(np.allclose(delta,m.P2_margin-m.original_margin,atol=1e-10,rtol=1e-10))
    checks['train_CE_lower_heldout_CE_higher_all_folds']=all(f['populations']['train']['P2']['CE']<f['populations']['train']['original_mean_logit']['CE'] and f['populations']['heldout']['P2']['CE']>f['populations']['heldout']['original_mean_logit']['CE'] for f in diag['folds'])
    checks['regularizer_increases_all_folds']=all(f['regularizer']['P2']>f['regularizer']['original'] for f in diag['folds'])
    supp=pd.read_parquet(DEST/'train_side_support.parquet')
    checks['support_table_hash']=sha(DEST/'train_side_support.parquet')==read(DEST/'support_eligibility.json')['support_table_sha256']
    checks['support_minima_recount']=int(((d.truth==2)&(d.N1_TabM25!=2)&(supp.behavior_M_training_roots>=3)&(supp.behavior_S_training_roots>=3)).sum())==14 and int(((d.truth==2)&(d.N1_TabM25!=2)&(supp.coarse_M_training_roots>=3)&(supp.coarse_S_training_roots>=3)).sum())==1403
    case=pd.read_parquet(DEST/'interface_intervention_predictions.parquet')
    b=case[(case.variant=='original')&(case.root==2868)].set_index('row_position')
    a=case[(case.variant=='dst_interface_to_DMZ')&(case.root==2868)].set_index('row_position')
    checks['988_flips_recount']=len(b)==1184 and len(a)==1184 and int((a.P2_prediction!=b.P2_prediction).sum())==988
    checks['24_S_counterexample']=int(((case.variant=='dst_interface_to_DMZ')&(case.root==216921)&(case.truth==2)&(case.P2_prediction==1)).sum())==24
    assert all(checks.values()),checks
    files=[ROOT/REPORT]+[p for p in DEST.iterdir() if p.is_file() and p.suffix in ('.json','.parquet','.csv')]
    files.extend(ROOT/'training'/n for n in source_names.values())
    ver={'status':'no_fit_review_verified','all_checks_passed':True,'checks':checks,
      'new_classifier_fits':0,'new_calibration_fits':0,'model_promoted':False,
      'scope':'Identity, full-row replay, algebraic score decomposition and support/intervention recount. No new model quality or blind-transfer claim.',
      'source_sha256':sha(__file__),'artifact_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in files}}
    save(target,ver)
    summary=('v11.1完成零拟合根因复核。最新训练仍为v11.0的6次拟合、0次校准，质量未通过、无模型晋升。'
      'P2三折训练CE下降35.5%至53.8%，来源外CE均上升；接口名称干预令目标组988条预测翻转，也使24条S转为M，不能当修复成绩。'
      '下一轮先做接口名称隔离的资格审查，再用原TabM目标开展两臂同划分对照，不叠加配权或继续P1/P2。')
    for name,link in [('README.md',REPORT),('docs/TRAINING_PLAN.md',Path(REPORT).name)]:
        p=ROOT/name;s=p.read_text(encoding='utf-8');head,body=s.split('\n',1)
        assert 'V111_ROOT_MECHANISM' not in s
        body=body.replace('当前训练结果与停止决定（v11.0）','最近训练结果与停止决定（v11.0）',1)
        p.write_text(head+'\n\n当前根因审查与下一轮方案（v11.1，未新增训练）：**'+summary+'** [新实证、研究筛选与单因素训练方案]('+link+')。\n'+body,encoding='utf-8')
    catpath=ROOT/'mcp_readonly/catalog.json';cat=read(catpath)
    assert cat['project']['authoritative_delivery_id']=='v110-delivery'
    assert cat['project']['authoritative_direction_id']=='v110-stop-decision'
    cat['project'].update(as_of='2026-09-29',authoritative_direction_id='v111-direction-review',current_summary=summary,
      current_direction=['不继续P1/P2；最新训练仍V110失败，保留N1开发参照。',
        '先审查接口命名路径、冲突与两臂输入等价闭合，再以TabM25原目标做原输入/接口名称隔离对照；计划上限6次拟合，尚未执行。',
        '按每类实际错误、负翻转、来源组及全任务验收；不把输入改写翻转当准确率或把已看折称为盲测。'],
      known_limits=['接口名称敏感性已实测，但名称互换是否保持安全语义未证实，不能统一改名部署。',
        '细行为双类多来源支持不足，粗行为支持不等于M/S判据相同；不直接启动配权或更大模型。',
        '没有官方M/S操作规则或新盲测；新增诊断不是新增模型性能。'])
    additions=[('v111-direction-review','v11.1 新根因证据与下一轮训练方案',REPORT,'current_review',summary),
      ('v111-objective-audit','v11.1 分类损失与正则拆分','artifacts/v111_root_review_20260929/objective_and_margin_diagnosis.json','current_evidence','三折训练CE下降但来源外CE上升；零拟合模型回放。'),
      ('v111-interface-audit','v11.1 接口命名敏感性诊断','artifacts/v111_root_review_20260929/interface_sensitivity.json','current_evidence','冻结模型文本干预，不是标签保持增强或准确率改进。'),
      ('v111-support-audit','v11.1 来源监督支持资格','artifacts/v111_root_review_20260929/support_eligibility.json','current_evidence','细行为与粗行为支持分辨率；不得当推理标签或已验证配权依据。'),
      ('v111-verification','v11.1 零拟合复核凭据','artifacts/v111_root_review_20260929/verification.json','current_evidence','历史模型未改、分数回放与计数复算；不证明模型提升。')]
    for item in cat['documents']:
        if item['id']=='v110-stop-decision':item['category']='historical_review'
        if item['id']=='training-plan':item.update(summary=summary,sha256=sha(ROOT/item['path']))
    cat['documents']=[{'id':i,'title':t,'path':p,'category':c,'summary':s,
      'keywords':['v11.1','当前','下一步方向','训练方案','根因','接口名称','M/S','损失','监督'],
      'sha256':sha(ROOT/p)} for i,t,p,c,s in additions]+cat['documents']
    save(catpath,cat)
    test=ROOT/'mcp_readonly/tests/test_readonly_mcp.py'
    s=test.read_text(encoding='utf-8').replace('v110-stop-decision','v111-direction-review')
    test.write_text(s,encoding='utf-8')
    receipt=ROOT/'evidence/2026-09-29/v111_root_review/delivery.json'
    receipt.parent.mkdir(parents=True,exist_ok=True)
    assert not receipt.exists()
    save(receipt,{'status':'v111_no_fit_review_and_plan_completed','actual_classifier_fits_this_request':0,
      'actual_calibration_fits':0,'quality_acceptance':False,'model_promoted':False,
      'last_training_delivery':'v110-delivery','summary':summary,
      'validation_scope':ver['scope'],'artifact_sha256':{**ver['artifact_sha256'],target.relative_to(ROOT).as_posix():sha(target)}})
    print(json.dumps({'all_checks_passed':True,'checks':len(checks),'new_classifier_fits':0,
      'direction':'v111-direction-review','last_training_delivery':'v110-delivery'},ensure_ascii=False))


if __name__=='__main__':main()
