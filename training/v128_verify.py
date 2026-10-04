"""Independent read-only row-count and fit-receipt verification for V128."""
import numpy as np
import pandas as pd

from experiment_review import check_bindings, sha
from v128_experiment_review import read, save, require_checkpoint, require_run_seal, ROOT
from v128_train_r3 import OUT, ORDER, OFFICIAL, TRACE


def require(ok,message):
    if not ok:raise ValueError(message)


def main():
    target=OUT/'verification.json'
    if target.exists():raise FileExistsError(target)
    plan=require_run_seal(OUT/'run_seal.json',ROOT/'training/v128_train_r3.py')
    check_bindings(read(ROOT/'artifacts/v125_order_trial_20260929/run_seal.json')['source_sha256'])
    d=read(OUT/'primary_evaluation.json')
    trace=pd.read_parquet(TRACE,columns=['row_position','truth','fold','root'])
    official=pd.read_parquet(OFFICIAL,columns=['label_binary']).label_binary.map(
        {'benign':0,'malicious':1,'suspicious':2}).to_numpy(dtype=np.int8)
    asa=pd.read_parquet(OUT/'expert_ASA_predictions.parquet')
    full=pd.read_parquet(OUT/'full_prediction_ledger.parquet')
    require(len(official)==len(full)==2056871 and len(trace)==len(asa)==112807,
            'Population changed')
    require(np.array_equal(trace.row_position,asa.row_position) and
            np.array_equal(trace.truth,official[trace.row_position]) and
            np.array_equal(full.row_position,np.arange(len(full))),
            'Official row identity or truth mismatch')
    arms=('A0','AH','K_IS','K_CF','P_IS','P_CF')
    errors={};full_missed={};source_count=int(trace.root.nunique())
    for arm in arms:
        p=asa['pred_'+arm].to_numpy(dtype=np.int8)
        f=full['final_pred_'+arm].to_numpy(dtype=np.int8)
        require(np.array_equal(f[trace.row_position],p),'ASA/full decision mismatch '+arm)
        errors[arm]={str(c):int(((trace.truth.to_numpy()==c)&(p!=c)).sum()) for c in (1,2)}
        full_missed[arm]={str(c):int(((official==c)&(f!=c)).sum()) for c in (0,1,2)}
        require(all(errors[arm][str(c)]==d['metrics']['ASA'][arm][str(c)]['missed'] for c in (1,2)),
                'ASA result mismatch '+arm)
        require(all(full_missed[arm][str(c)]==d['metrics']['full_task'][arm][str(c)]['missed'] for c in (0,1,2)),
                'Full result mismatch '+arm)
    fit_receipts=[];steps=0
    for f,j,kind in ORDER:
        folder=OUT/(f'teacher{f}_{j}' if kind=='teacher' else f'fold{f}_{kind}')
        r=read(folder/'fit.json')
        require(r['status']=='fit_executed' and r['seal_sha256']==sha(OUT/'run_seal.json'),
                'Incomplete or foreign fit receipt')
        if kind=='teacher':
            require_checkpoint(plan,r,'teacher');steps+=r['optimizer_steps']
        elif kind.startswith('P_'):
            require_checkpoint(plan,r,'branch');steps+=r['optimizer_steps']
        else:
            require(r['gradient_inf']<=1e-7,'Nonstationary constant')
        fit_receipts.append({'fold':f,'inner':j,'kind':kind,'receipt_sha256':sha(folder/'fit.json')})
    require(len(fit_receipts)==21 and steps==26700,'Trial update budget not met')
    require(errors['A0']=={'1':318,'2':2074} and errors['P_CF']=={'1':1090,'2':2174},
            'Hard reference or candidate result changed')
    require(not d['primary_quality_passed'] and not d['confirmation_allowed'] and not d['model_promoted'],
            'Failed candidate wrongly promoted')
    result={'status':'independent_official_row_and_receipt_recheck_passed',
        'source_sha256':sha(__file__),'training_seal_sha256':sha(OUT/'run_seal.json'),
        'primary_evaluation_sha256':sha(OUT/'primary_evaluation.json'),
        'score_role_postmortem_sha256':sha(OUT/'score_role_postmortem.json'),
        'ASA_ledger_sha256':sha(OUT/'expert_ASA_predictions.parquet'),
        'full_ledger_sha256':sha(OUT/'full_prediction_ledger.parquet'),
        'official_original_rows':len(official),'ASA_original_rows':len(asa),
        'ASA_source_roots':source_count,'classifier_fits':len(fit_receipts),
        'neural_optimizer_steps':steps,'per_class_ASA_errors':errors,
        'per_class_full_misses':full_missed,'fit_receipts':fit_receipts,
        'quality_passed':False,'model_promoted':False,
        'scope':'Original-row decisions and saved endpoints rechecked; no blind external transfer or official submission.'}
    save(target,result)
    print({'status':result['status'],'fits':len(fit_receipts),'steps':steps,
           'A0':errors['A0'],'P_IS':errors['P_IS'],'P_CF':errors['P_CF']})


if __name__=='__main__':main()
