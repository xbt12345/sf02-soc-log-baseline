"""Preserve all correct TRAIN rows after whole first-issue mastery, plus old scopes."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from experiment_review import ROOT,read,sha
from v135_runtime import load_data,fit_context
from v140_retention_check import check as previous_check
from v137_issue_guard import protect_original_rows

OUT=ROOT/'artifacts/v142_second_layer_training_20261001'


def check(candidate_rows):
    previous=previous_check(candidate_rows);_,d=load_data();results=[]
    registry=read(OUT/'verified_TRAIN_mastery_registry.json')
    if not registry['first_issue_closed']:raise ValueError('Cannot fabricate accepted first-issue scope')
    for item in registry['scopes']:
        for name,key in [('guard','guard_sha256'),('checkpoint','checkpoint_sha256'),('baseline_rows','baseline_rows_sha256')]:
            if sha(ROOT/item[name])!=item[key]:raise ValueError('Verified mastery evidence changed')
        fold=item['training_role'];frame,_,_,_,_=fit_context(d,fold);ref=frame[['row_position','truth']].copy();ref['training_role']=fold
        guard=pd.read_parquet(ROOT/item['guard']);base=pd.read_parquet(ROOT/item['baseline_rows'])
        if not np.array_equal(guard.truth,guard.row_position.map(ref.set_index('row_position').truth)):raise ValueError('Guard truth mismatch')
        supplied=candidate_rows[candidate_rows.training_role.eq(fold)]
        result=protect_original_rows(ref,base[['row_position','training_role','pred']],supplied[['row_position','training_role','pred']],
                                    guard[['row_position','training_role']],keys=('training_role','row_position'))
        result['scope_id']=item['id'];results.append(result)
    return {'passed':previous['passed'] and all(v['repair_protection_passed'] for v in results),'previous':previous,'mastered_TRAIN':results,
            'TRAIN_mastery_only_not_generalization_or_task_acceptance':True}


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('candidate_role_ledger',type=Path);args=a.parse_args()
    import json
    result=check(pd.read_parquet(args.candidate_role_ledger));print(json.dumps(result,ensure_ascii=False),flush=True)
    if not result['passed']:raise SystemExit(2)
