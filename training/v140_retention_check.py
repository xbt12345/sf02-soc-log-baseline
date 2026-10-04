"""Replay immutable V138 and additional V140 C-fold2 TRAIN capabilities."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from experiment_review import ROOT,read,sha
from v135_runtime import load_data,fit_context
from v137_issue_guard import protect_original_rows
from v138_retention_check import check as old_check

OUT=ROOT/'artifacts/v140_ensemble_training_round2_20261001'


def check(candidate_rows):
    old=old_check(candidate_rows);_,d=load_data();results=[]
    for item in read(OUT/'additional_verified_TRAIN_scopes.json')['scopes']:
        if sha(ROOT/item['guard'])!=item['guard_sha256'] or sha(ROOT/item['checkpoint'])!=item['checkpoint_sha256']:
            raise ValueError('Additional verified capability identity changed')
        if sha(ROOT/item['baseline_rows'])!=item['baseline_rows_sha256']:
            raise ValueError('Additional capability reference ledger changed')
        stored=pd.read_parquet(ROOT/item['guard']);fold=item['training_role'];frame,_,_,_,_=fit_context(d,fold)
        reference=frame[['row_position','truth']].copy();reference['training_role']=fold
        if not np.array_equal(stored.truth,stored.row_position.map(reference.set_index('row_position').truth)):
            raise ValueError('Stored guard differs from official truth')
        base=pd.read_parquet(ROOT/item['baseline_rows']);supplied=candidate_rows[candidate_rows.training_role.eq(fold)]
        r=protect_original_rows(reference,base[['row_position','training_role','pred']],supplied[['row_position','training_role','pred']],
                                stored[['row_position','training_role']],keys=('training_role','row_position'))
        r['scope_id']=item['id'];results.append(r)
    return {'passed':old['passed'] and all(z['repair_protection_passed'] for z in results),'V138':old,'additional_V140':results,
            'training_scope_only_not_task_quality':True}


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('candidate_role_ledger',type=Path);args=a.parse_args()
    import json
    r=check(pd.read_parquet(args.candidate_role_ledger));print(json.dumps(r,ensure_ascii=False),flush=True)
    if not r['passed']:raise SystemExit(2)
