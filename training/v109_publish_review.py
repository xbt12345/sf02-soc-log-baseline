"""Publish a no-training review and draft training contract, preserving V107 authority."""
import json
from datetime import datetime
from pathlib import Path
from run_v75 import ROOT, save, sha
from v109_frozen_member_audit import DEST


def main():
    verified=json.loads((DEST/'verification.json').read_text(encoding='utf-8'))
    assert verified['all_checks_passed']
    assert all(sha(DEST/p)==h for p,h in verified['artifact_sha256'].items())
    contract={'status':'plan_only_not_training_registration','version':'v109_direction',
      'new_classifier_fits':0,'new_calibration_fits':0,'model_promoted':False,
      'next_stage_max_classifier_fits':6,'external_training_data':False,
      'arms':[{'name':'P1','view':'N2','activation':'relu(first)','dimensions':2543},
              {'name':'P2','view':'N2','activation':'relu(second)','dimensions':2543}],
      'shared_input':'Ordered flatten of 16x128 frozen member activations plus the existing 495 fact/metadata coordinates.',
      'folds':[0,1,2],'sample_mass':'All original training-row counts; no deduplication mass loss or class balancing.',
      'scaling':'Training-only frequency-weighted standardization; zero std replaced with 1; no PCA or dropped coordinates.',
      'objective':'sum_i original_count_i * binary_CE_i / sum_i original_count_i + (1e-3/2)*||w||^2; intercept unpenalized',
      'solver':{'method':'L-BFGS','max_iterations':2000,'required_gradient_inf_max':1e-5},
      'decision_probability_threshold':0.5,'outer_label_selection':False,
      'trial_type':'Studied development folds, diagnostic probes; neither independent DFR set nor fresh blind test.',
      'research_continuation_necessary_gates':{'reference':'V107_N2_TabM25','ASA_M_errors_max':296,'ASA_S_errors_strictly_below':5194,
        'alternative_qualifying_entry':'Meets N1 replacement count/full-task gates; N2 M budget alone must not reject a candidate dominating N1.',
        'net_S_repairs_outside_top3_S_roots':'>0','folds_with_multiple_improving_source_groups_min':2,
        'M_and_S_source_macro_recall':'neither decreases; mean recall over roots having that class, every such root included',
        'uncertainty':'Report group-paired bootstrap interval; no statistically confirmed transfer claim if it includes zero.'},
      'replacement_necessary_gates':{'reference':'V107_N1_full_frozen_composite','ASA_M_errors_max':318,'ASA_S_errors_max':2094,
         'strict_reduction_in_at_least_one_class':True,'full_task_M_S_correct_and_class_F1':'no decrease',
         'B_false_alerts':'no increase','required_checks':['full 2056871-row replay','source and small-root checks','verified label-invariant wrapper checks']},
      'limits':['Observed roots are grouping proxies, not certified incidents or enterprises.',
        'Formal layer selection needs encoder-unseen internal sources; cached train activations do not create independence.',
        'Failure of fixed linear probes does not establish an information-theoretic impossibility.',
        'Before execution, create an immutable run registration with all inputs, source/config hashes and exact software receipts.']}
    assert not (DEST/'next_training_contract.json').exists()
    save(DEST/'next_training_contract.json',contract)
    report='docs/V109_HISTORY_SYNTHESIS_AND_CONTROLLED_TRAINING_PLAN.md'
    summary=('最新实际训练仍为v10.7：12次拟合、0次校准，质量未通过，无模型晋升。'
      'v10.9零拟合复核六个冻结模型：N2的5194条S错误中5026条为16成员全错；平均logit仅修2条S并新增2条M错。'
      '综合V89/V97/V104失败经验与中间层研究，下一步改为N2第一/第二隐藏层同宽读出三折共6头，固定原频次；'
      '研究继续与替换N1分别验收，不将读出重训或配权默认视为去偏。')
    for name,link in [('README.md',report),('docs/TRAINING_PLAN.md',Path(report).name)]:
        p=ROOT/name;text=p.read_text(encoding='utf-8');head,body=text.split('\n',1)
        assert 'V109_HISTORY_SYNTHESIS' not in text
        body=body.replace('当前方向审查（v10.8，未新增训练）','历史方向审查（v10.8；下一轮步骤已由v10.9修订）',1)
        p.write_text(head+'\n\n当前方向审查（v10.9，未新增训练）：**'+summary+'** [历史复核与修订训练方案]('+link+')。\n'+body,encoding='utf-8')
    receipt=ROOT/'evidence/2026-09-29/v109_plan_review/review.json'
    assert not receipt.exists();receipt.parent.mkdir(parents=True,exist_ok=True)
    paths=[ROOT/report,DEST/'frozen_member_audit.json',DEST/'verification.json',DEST/'next_training_contract.json',
      ROOT/'training/v109_frozen_member_audit.py',ROOT/'training/v109_verify_review.py']
    save(receipt,{'status':'v109_history_and_frozen_member_review_completed_no_fit','created_at':datetime.now().astimezone().isoformat(),
      'actual_classifier_fits_this_request':0,'actual_calibration_fits':0,'quality_acceptance':False,'model_promoted':False,
      'latest_actual_training_delivery':'v107-delivery','source_sha256':sha(__file__),
      'private_answer_used':False,'platform_used':False,'external_training_data_used':False,
      'execution_notes':['Preserved initial CSR-length failure and incomplete duplicate-forward audit; final audit captures activations from one original forward and matches saved decisions.',
       'Binary hash columns are not JSON text; a display-only serialization error did not change ledger contents.',
       '30 focused frozen audit checks pass; they do not certify future training, model quality or cloud MCP connectivity.'],
      'artifact_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in paths}})
    cp=ROOT/'mcp_readonly/catalog.json';cat=json.loads(cp.read_text(encoding='utf-8'))
    assert cat['project']['authoritative_delivery_id']=='v107-delivery'
    assert cat['project']['authoritative_direction_id']=='v108-direction-review'
    cat['project'].update(current_summary=summary,authoritative_direction_id='v109-direction-review',
      current_direction=['固定N2表示，以第一与第二隐藏层同容量M/S读出检验层间信息；初轮6头，尚未训练。',
        '保留全部官方行频次，来源和细行为支持单列；不按旧错例随意重配权或扩大损失网格。',
        '以各类真实修复/退化、来源与小组收益验收；正式替换仍须胜过N1完整组合，现有折不是盲测。'],
      known_limits=['N2的96.77%可疑错误全部16成员共错；这不证明隐藏表示不可解码。',
        '前三S组占91.19%，稀有行为的跨来源M/S支持不足；官方细化标签与脱敏关联规则仍缺。',
        '同频次末层重训不保证去偏，中间层论文的成功尚未在SF02验证；没有新独立外部成绩。'])
    additions=[('v109-direction-review','v10.9 历史复核与受控训练方案',report,'current_review',summary),
      ('v109-frozen-member-evidence','v10.9 冻结成员与梯度诊断','artifacts/v109_plan_review_20260929/frozen_member_audit.json','current_evidence','六个冻结模型输出复算，零拟合；成员共错与受限分类头导数证据。'),
      ('v109-training-contract','v10.9 下一轮训练约束（尚未执行）','artifacts/v109_plan_review_20260929/next_training_contract.json','current_plan','两层同宽探针、6次拟合预算、固定频次与分别验收；不是训练完成凭据。'),
      ('v109-review-receipt','v10.9 零拟合审查凭据',receipt.relative_to(ROOT).as_posix(),'current_evidence','最新实际训练仍由v107-delivery负责。')]
    for e in cat['documents']:
        if e['id']=='v108-direction-review':e['category']='historical_review'
        if e['id']=='training-plan':e.update(summary=summary,sha256=sha(ROOT/e['path']))
    cat['documents']=[{'id':i,'title':t,'path':p,'category':c,'summary':s,'keywords':['v10.9','当前','中间层','M/S','根因','来源','训练计划'],'sha256':sha(ROOT/p)} for i,t,p,c,s in additions]+cat['documents']
    save(cp,cat)
    test=ROOT/'mcp_readonly/tests/test_readonly_mcp.py';text=test.read_text(encoding='utf-8')
    test.write_text(text.replace('v108-direction-review','v109-direction-review'),encoding='utf-8')
    print(json.dumps({'direction':'v109-direction-review','latest_actual_training':'v107-delivery','new_fits':0,'review_sha256':sha(receipt)},ensure_ascii=False))


if __name__=='__main__':main()
