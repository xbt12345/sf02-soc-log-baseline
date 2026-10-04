"""Bind the no-fit review and publish its direction without changing training status."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_v75 import ROOT, OUT, save, sha
from v104_phase_a import FID, FOLDS
from v106_frozen_wrapper_audit import DEST


def main():
    target=DEST/'final_verification.json'; assert not target.exists()
    d=pd.read_parquet(DEST/'paired_OOF_predictions.parquet')
    previous=pd.read_parquet(ROOT/'artifacts/v104_phase_b_20260928/phase_B_ASA_OOF_ledger.parquet')
    assert previous.row_position.equals(d.row_position)
    repairs=(previous.truth==2)&(previous.N1_teacher!=2)&(previous.C_TabM_epoch25==2)
    regressions=(previous.truth==1)&(previous.N1_teacher==1)&(previous.C_TabM_epoch25!=1)
    f=pd.read_parquet(DEST/'proposed_body_closed_folds.parquet')
    r=pd.read_parquet(OUT/'rows.parquet',columns=['row_position','route','component','source_symbol','label_index'])
    fid=np.load(FID,mmap_mode='r')
    old=pd.read_parquet(FOLDS,columns=['root'])
    n1=pd.read_parquet(ROOT/'artifacts/v101_full_input_group_n1_20260928/N1_ASA_group_map.parquet')
    n2=pd.read_parquet(DEST/'collapse_input_groups.parquet')
    assert len(f)==len(r)==len(fid)==2056871
    checks={'official_rows_exact_once':bool(f.row_position.equals(r.row_position) and f.row_position.is_unique),
        'original_components_one_fold':bool(pd.DataFrame({'key':r.component,'fold':f.proposed_fold}).groupby('key').fold.nunique().max()==1),
        'original_roots_one_new_root':bool(pd.DataFrame({'key':old.root,'root':f.root}).groupby('key').root.nunique().max()==1),
        'all_R0_inputs_one_fold':bool(pd.DataFrame({'key':fid,'fold':f.proposed_fold}).groupby('key').fold.nunique().max()==1)}
    for name,group,key in [('N1',n1,'N1_group'),('N2',n2,'new_input')]:
        merged=group[['row_position',key]].merge(f,on='row_position',validate='one_to_one')
        checks[name+'_inputs_one_fold']=bool(merged.groupby(key).proposed_fold.nunique().max()==1)
        checks[name+'_inputs_one_root']=bool(merged.groupby(key).root.nunique().max()==1)
    for filename,input_field in [('registration.json','input_sha256'),
                                  ('source_context_correction.json','input_sha256'),
                                  ('body_source_split_audit.json','inputs_sha256')]:
        receipt=json.loads((DEST/filename).read_text())
        checks[filename+'_input_hashes']=all(sha(ROOT/p)==h for p,h in receipt[input_field].items())
    sources={'registration.json':'training/v106_frozen_wrapper_audit.py',
             'verification_and_N2_preflight.json':'training/v106_verify_preflight.py',
             'source_context_correction.json':'training/v106_source_context_audit.py',
             'body_source_split_audit.json':'training/v106_body_source_split_audit.py'}
    checks['audit_source_receipts_match']=all(json.loads((DEST/k).read_text())['source_sha256']==sha(ROOT/v) for k,v in sources.items())
    assert all(checks.values()),checks
    final={'status':'no_fit_review_verified','checks':checks,'all_checks_passed':True,
        'original_S_repairs':int(repairs.sum()),
        'original_S_repairs_lost_on_collapse':int((repairs&(d.collapse_pred!=2)).sum()),
        'original_M_regressions_remaining_on_collapse':int((regressions&(d.collapse_pred!=1)).sum()),
        'source_sha256':sha(__file__),'fits':0,'calibrations':0,'model_promoted':False,
        'scope':'Audit and proposed split only; no new training quality or live MCP connection claim.'}
    save(target,final)
    plan={'status':'revised_plan_not_trained','version':'v106','official_data_only':True,
        'labels_unchanged':True,'original_row_frequencies_preserved':True,
        'next_primary_question':'Does removing demonstrated wrapper dependence improve stable M/S decisions under body-source-closed folds?',
        'normalizer_source':'training/v106_frozen_wrapper_audit.py::rewrite(mode=collapse), followed by existing view/stable/N1',
        'normalizer_source_sha256':sha(ROOT/'training/v106_frozen_wrapper_audit.py'),
        'views':['N1','N2_collapse_redacted_port_chain_preserve_visible_prefix'],
        'split':{'path':'artifacts/v106_frozen_audit_20260928/proposed_body_closed_folds.parquet',
                 'sha256':sha(DEST/'proposed_body_closed_folds.parquet'),
                 'unit':'existing full-format roots plus observed ASA body-source closure; all R0/N1/N2 equal inputs isolated',
                 'folds':3,'seed':10203,'reroll':False,'fresh_blind':False,'real_actor_identity_verified':False,
                 'reuse_v104_models_as_matched_baseline':False},
        'factorial_arms':{'N1_teacher':3,'N2_teacher':3,'N1_TabM25':3,'N2_TabM25':3},
        'first_wave_fit_cap':12,'first_wave_calibration_fits':0,
        'training_config':'Reuse v104 within each model family; TabM25 only, no 50/100 search; count-weighted original-row CE.',
        'conditional_replication':{'only_after_first_wave_pass':True,'additional_TabM_seeds':[10202,10203],
                                   'paired_N1_and_N2_threefold_fit_cap':12,'not_external_validation':True},
        'mandatory_metrics':['full-task per-class numerator/denominator, precision, recall, F1, macro-F1, benign false alerts',
            'ASA MS-F1 and M/S recall on common new folds', 'non-nested wrapper stratum',
            'new-root equal recall and zero recall counts with same grouping for all arms',
            'per-fold, two historical major-root slices via row IDs, difficult ICMP/UDP514/TCP6514',
            'original-correct negative flips and original-wrong positive flips',
            'paired wrapper variants on identical underlying heldout rows'],
        'continuation_gate':['input preservation and variant invariance',
            'M/S recall and F1 not below common new-fold N1 teacher; MS-F1 improves',
            'non-wrapper S and MS-F1 improve; S group recall improves and zero-group count decreases',
            'at least two folds net improve and no hidden non-ASA or benign regression',
            'improvement outside dominant historical groups'],
        'deferred':['DFR readout only with backbone-unseen, source-separated balanced evidence inside outer train',
            'calibration only with independent train-side calibration groups',
            'context only after verified entity/clock and dual-class multi-source support'],
        'stop':['do not treat input invariance alone as quality acceptance',
                'no more depth/epochs/weight search when robust same-class support is absent',
                'do not rewrite official labels or fabricate hidden ports'],
        'source_sha256':sha(__file__)}
    save(DEST/'next_training_contract.json',plan)
    title='v10.6 冻结模型根因复核与来源隔离修订方案'
    summary=('v10.6零拟合实证：冻结TabM合并同一脱敏端口的重复包装，3486条原正确S变错；'
             '增加包装，6484条原正确M变错。3740条旧S修复有3484条失去，1028条旧M退化未修复。'
             '发现401个正文源地址跨原折（14007行），已给出更严格候选划分。'
             '下一轮在新折匹配N1/N2×线性/TabM25，初轮上限12次拟合，尚未训练。最新实际训练仍v10.4，无模型晋升。')
    doc='docs/V106_CAUSAL_WRAPPER_AUDIT_AND_REVISED_PLAN.md'
    for path,link in [(ROOT/'README.md',doc),(ROOT/'docs/TRAINING_PLAN.md',Path(doc).name)]:
        text=path.read_text(encoding='utf-8'); heading,rest=text.split('\n',1)
        path.write_text(heading+'\n\n当前根因复核与下一轮方案（2026-09-28）：**'+summary+'** ['+title+']('+link+')。下方保留历史阶段。\n'+rest,encoding='utf-8')
    cp=ROOT/'mcp_readonly/catalog.json'; catalog=json.loads(cp.read_text(encoding='utf-8'))
    assert catalog['project']['authoritative_delivery_id']=='v104-delivery'
    catalog['project'].update(current_summary=summary,authoritative_direction_id='v106-direction-review',
        current_direction=['先修复已证实的端口脱敏包装敏感性，保留可见数字、事实及原文跨度。',
            '正文源地址401个跨原折；采用保守来源连接候选折，所有匹配基线与候选重训，不能复用旧分数声称提升。',
            '下一轮N1/N2×线性/TabM25四臂三折，上限12次拟合；以无包装切片、M保护、S逐组与全任务结果共同验收。'],
        known_limits=['本轮0次新拟合；输入不变量已验证，分类改善尚未验证。',
            '原有1028条M退化未被合并包装修复；ICMP S仍缺稳定可观察判别依据。',
            '结构化src_ip与正文src存在多对多，不能混合作为真实行为实体；所有折均为已观察开发数据。'])
    new=[]
    entries=[('v106-direction-review',title,doc,'current_review',summary),
        ('v106-diagnosis','v10.6 冻结包装干预证据','artifacts/v106_frozen_audit_20260928/diagnosis.json','current_evidence','同权重原文成对干预、模型重放和逐类变化；不是重训成绩。'),
        ('v106-source-audit','v10.6 正文来源隔离缺口','artifacts/v106_frozen_audit_20260928/body_source_split_audit.json','current_evidence','401个正文源地址跨旧折；新候选隔离划分及每折分母。'),
        ('v106-next-contract','v10.6 下一轮训练契约','artifacts/v106_frozen_audit_20260928/next_training_contract.json','plan','修订方案尚未训练；12次匹配拟合上限及按实际效果验收。')]
    for i,t,p,c,s in entries:
        new.append({'id':i,'title':t,'path':p,'category':c,'summary':s,
                    'keywords':['v10.6','根因复核','脱敏','来源隔离','下一轮']+(['当前','最新','方向'] if c=='current_review' else []),'sha256':sha(ROOT/p)})
    for entry in catalog['documents']:
        if entry['id']=='v105-direction-review':entry['category']='historical_review'
        if entry['id']=='v105-diagnosis':entry['category']='historical_evidence'
        if entry['id']=='training-plan':entry.update(summary=summary,sha256=sha(ROOT/entry['path']))
    catalog['documents']=new+catalog['documents']; save(cp,catalog)
    tests=ROOT/'mcp_readonly/tests/test_readonly_mcp.py'
    tests.write_text(tests.read_text(encoding='utf-8').replace('v105-direction-review','v106-direction-review'),encoding='utf-8')
    print(json.dumps(final,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
