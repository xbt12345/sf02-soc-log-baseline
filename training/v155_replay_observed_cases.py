"""Recount real V155 evidence; no new model execution or fitting."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiment_review import ROOT,read,sha,check_bindings
from v138_runtime import save
from v138_closeout import pair

OUT=ROOT/'artifacts/v155_guarded_full_gradient_sam_20261001'
CASE=ROOT/'training/review_policy/v155_observed_neighborhood_cases.json'

def main():
    case=read(CASE);check_bindings(case['source_sha256'])
    delivery=read(OUT/'final_delivery.json');quality=read(OUT/'quality.json')
    asa=pd.read_parquet(OUT/'ASA_prediction_ledger.parquet')
    assert len(asa)==112807 and asa.row_position.is_unique
    assert pair(asa,'pred_A','pred_B')==delivery['original_pairs']['B_vs_A']
    assert pair(asa,'pred_V146_A','pred_B')==delivery['original_pairs']['B_vs_V146_A']
    for arm in ['A','B']:
        counts={str(c):int((asa.truth.eq(c)&asa['pred_'+arm].ne(c)).sum()) for c in [1,2]}
        assert counts=={str(c):quality[arm]['ASA']['B'][str(c)]['missed'] for c in [1,2]}
    rises=[];fit_count=gradient_count=proposal_count=updates=0
    for f in range(3):
        for arm in ['A','B']:
            folder=OUT/f'fold{f}_{arm}';r=read(folder/'fit.json');h=read(folder/'progress.json')
            trials=[json.loads(z) for z in (folder/'proposals.jsonl').read_text().splitlines()]
            accepted=[z for z in trials if z['accepted']]
            assert len(accepted)==r['accepted_updates']==len(h)
            assert all(z['classification_guard'] and z['frozen_epsilon_proxy_CE']<=z['proxy_Armijo_bound'] for z in accepted)
            assert all(z['stats']['new_errors_vs_start']==0 and z['stats']['pure_M_errors']==z['stats']['pure_S_errors']==0 for z in h)
            if arm=='B':
                for before,after in zip(accepted,accepted[1:]):
                    if after['frozen_epsilon_proxy_CE']>before['frozen_epsilon_proxy_CE']:
                        rises.append(dict(fold=f,previous_proposal=before['proposal'],proposal=after['proposal'],
                            previous_proxy=before['frozen_epsilon_proxy_CE'],proxy=after['frozen_epsilon_proxy_CE'],
                            current_proxy_base=after['frozen_epsilon_base_proxy_CE'],current_armijo_bound=after['proxy_Armijo_bound'],
                            epsilon_changed=before['epsilon_sha256']!=after['epsilon_sha256']))
            fit_count+=1;gradient_count+=r['full_gradient_evaluations'];proposal_count+=r['proposal_evaluations'];updates+=r['accepted_updates']
    assert rises and all(z['epsilon_changed'] and z['proxy']<=z['current_armijo_bound']<z['current_proxy_base'] for z in rises)
    assert (fit_count,gradient_count,proposal_count,updates)==tuple(delivery[k] for k in ['classifier_fits','full_gradient_evaluations','proposal_evaluations','accepted_updates'])
    assert not delivery['model_promoted'] and not delivery['issue_solved']
    failed=[k for k,v in quality['B']['gates'].items() if not v]
    assert failed==case['failed_task_gates']
    assert quality['task_acceptance']==(quality['matched_effect_passed'] and quality['B']['quality_passed'])
    result=dict(status='actual_fixed_proxy_and_original_classification_cases_replayed',
        fitting_totals=dict(fits=fit_count,full_gradients=gradient_count,proposals=proposal_count,updates=updates),
        changed_epsilon_cross_iteration_proxy_rises=len(rises),first_proxy_rise=rises[0],
        failed_task_gates=failed,matched_gates=quality['matched_gates'],
        original_pairs=delivery['original_pairs'],new_classifier_forwards=0,new_gradients=0,new_fits=0,new_updates=0,
        source_sha256={str(x.relative_to(ROOT)):sha(x) for x in [Path(__file__),CASE,OUT/'final_delivery.json',OUT/'quality.json',OUT/'ASA_prediction_ledger.parquet']})
    save(OUT/'observed_cases_replay.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','original_pairs']},ensure_ascii=False))

if __name__=='__main__':main()
