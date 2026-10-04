"""Future candidate guard for actually verified, scoped TRAIN capabilities."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from v138_runtime import ROOT,OUT,read,sha
from v135_runtime import load_data,fit_context
from v137_issue_guard import protect_original_rows


def check(candidate_rows):
    capabilities=read(OUT/'scoped_training_capabilities.json');_,d=load_data();results=[]
    for item in capabilities['verified_training_scopes']:
        path=ROOT/item['guard']
        if sha(path)!=item['guard_sha256'] or sha(ROOT/item['checkpoint'])!=item['checkpoint_sha256']:
            raise ValueError('Accepted scoped evidence changed')
        stored=pd.read_parquet(path);fold=int(stored.training_role.iloc[0]);frame,_,_,_,_=fit_context(d,fold)
        # Reference covers the entire legal role; truth comes from official data.
        reference=frame[['row_position','truth']].copy();reference['training_role']=fold
        if not np.array_equal(stored.truth,stored.row_position.map(reference.set_index('row_position').truth)):
            raise ValueError('Guard truth differs from independent official truth')
        baseline=pd.read_parquet(OUT/f'fold{fold}_H_L/endpoint_original_rows.parquet')
        supplied=candidate_rows[candidate_rows.training_role.eq(fold)]
        result=protect_original_rows(reference,baseline[['row_position','training_role','pred']],supplied[['row_position','training_role','pred']],
            stored[['row_position','training_role']],keys=('training_role','row_position'))
        result['scope_id']=item['id'];results.append(result)
    return {'passed':all(x['repair_protection_passed'] for x in results),'scopes':results,
        'only_scoped_TRAIN_retention_not_task_quality':True}


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('candidate_role_ledger',type=Path);p=a.parse_args()
    import json
    result=check(pd.read_parquet(p.candidate_role_ledger));print(json.dumps(result,ensure_ascii=False),flush=True)
    if not result['passed']:raise SystemExit(2)
