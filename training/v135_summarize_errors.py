"""Independent post-verification TRAIN error transitions; never updates models."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from v135_runtime import ROOT,OUT,ARMS,read,save,sha,require_run_seal

def main():
    require_run_seal(ROOT/'training/v135_train.py')
    verification=read(OUT/'verification.json');audit=read(OUT/'postflight_audit/audit.json')
    if verification['fits']!=12 or not audit['factorial_prefix_valid']:raise ValueError('Incomplete/unmatched actual run')
    dest=OUT/'error_review'
    if dest.exists():raise FileExistsError('Preserve previous review')
    dest.mkdir();summary=[];all_bad=[];bound={Path(__file__),OUT/'verification.json',OUT/'postflight_audit/audit.json'}
    for fold in range(3):
        for arm in ARMS:
            folder=OUT/f'fold{fold}_{arm}';frames=[]
            for epoch in range(96,101):
                path=folder/'epochs'/f'epoch{epoch:03d}_rows.parquet';bound.add(path)
                frames.append(pd.read_parquet(path))
            first=frames[0]
            if any(not np.array_equal(z.row_position,first.row_position) or not np.array_equal(z.truth,first.truth) for z in frames):raise ValueError('Window population mismatch')
            wrong=np.column_stack([z.pred.ne(z.truth).to_numpy() for z in frames]);pure=first.pure_TRAIN_input.to_numpy()
            old_wrong=first.old_V131_R_pred.ne(first.truth).to_numpy();union=wrong.any(1);intersection=wrong.all(1)
            endpoint=frames[-1].copy();endpoint['wrong_epochs_last5']=wrong.sum(1);endpoint['fold_role']=fold;endpoint['arm']=arm
            all_bad.append(endpoint[pure&union])
            for cl,name in [(1,'M'),(2,'S')]:
                select=pure&first.truth.eq(cl).to_numpy();old=select&old_wrong;oldgood=select&~old_wrong
                tail=endpoint[select&wrong[:,-1]]
                summary.append({'fold':fold,'arm':arm,'class':name,'pure_original_rows':int(select.sum()),
                    'old_R_pure_errors':int(old.sum()),'old_errors_repaired_all_last5':int((old&~union).sum()),
                    'old_correct_wrong_in_at_least_one_last5':int((oldgood&union).sum()),
                    'last5_error_counts':[int((select&wrong[:,j]).sum()) for j in range(5)],
                    'persistent_last5_errors':int((select&intersection).sum()),
                    'intermittent_last5_errors':int((select&union&~intersection).sum()),
                    'endpoint_errors':len(tail),'endpoint_distinct_inputs':int(tail.local.nunique()),
                    'endpoint_error_sources':int(tail.root.nunique()),
                    'endpoint_no_correct_member_rows':int(tail.correct_member_count.eq(0).sum()),
                    'endpoint_truth_probability_min':float(tail['p'+str(cl)].min()) if len(tail) else None,
                    'endpoint_truth_probability_max':float(tail['p'+str(cl)].max()) if len(tail) else None})
    pd.concat(all_bad,ignore_index=True).to_parquet(dest/'last5_nonmixed_error_rows.parquet',index=False)
    save(dest/'TRAIN_error_transitions.json',{'records':summary,'new_fits':0,'new_updates':0,
        'limits':'Original TRAIN-role counts; repeated roles are not distinct original records. Member failure or near-boundary probability alone does not establish capacity or causality.'})
    actual_path=OUT/'ASA_prediction_ledger.parquet'
    old_path=ROOT/'artifacts/v131_learning_trial_20260930/ASA_prediction_ledger.parquet'
    bound.update([actual_path,old_path])
    actual=pd.read_parquet(actual_path);old=pd.read_parquet(old_path)
    if not np.array_equal(actual[['row_position','truth','fold','root']].to_numpy(),old[['row_position','truth','fold','root']].to_numpy()):raise ValueError('HELD reference identity mismatch')
    if not np.array_equal(actual.pred_A0,old.pred_A0):raise ValueError('Original A0 changed')
    effects=[];sources=[]
    for arm in ARMS:
        pred=actual['pred_'+arm].to_numpy();y=actual.truth.to_numpy()
        for fold in [-1,0,1,2]:
            mask=np.ones(len(y),bool) if fold==-1 else actual.fold.eq(fold).to_numpy()
            for cl,name in [(1,'M'),(2,'S')]:
                select=mask&(y==cl)
                for baseline,before in [('original_A0',actual.pred_A0.to_numpy()),('old_V131_R',old.pred_R.to_numpy())]:
                    good=pred==y;wasgood=before==y
                    effects.append({'arm':arm,'fold':fold,'class':name,'baseline':baseline,'support':int(select.sum()),
                        'old_errors':int((select&~wasgood).sum()),'new_errors':int((select&~good).sum()),
                        'repaired':int((select&~wasgood&good).sum()),'regressed':int((select&wasgood&~good).sum())})
        src=actual.assign(wrong=pred!=y).groupby(['root','truth']).agg(support=('wrong','size'),errors=('wrong','sum')).reset_index()
        src['arm']=arm;sources.append(src)
    pd.DataFrame(effects).to_csv(dest/'HELD_repair_and_regression.csv',index=False)
    pd.concat(sources,ignore_index=True).to_csv(dest/'HELD_source_class_errors.csv',index=False)
    save(dest/'source_receipt.json',{'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(bound)}})
    print(json.dumps({'TRAIN_endpoint_pure_role_errors':{a:sum(z['endpoint_errors'] for z in summary if z['arm']==a) for a in ARMS},'new_updates':0}),flush=True)
if __name__=='__main__':main()
