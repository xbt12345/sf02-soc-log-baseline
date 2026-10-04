"""Zero-fit V131 training mastery audit against original legal TRAIN labels."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

from v131_common import ROOT,OUT as PARENT,REVIEW,load_data,fit_context,require_run_seal,read,save,sha

OUT=ROOT/'artifacts/v132_training_mastery_review_20260930'


def main():
    require_run_seal(ROOT/'training/v131_train.py')
    if OUT.exists():raise FileExistsError('Preserve previous review')
    receipt=read(PARENT/'diagnostics/receipt.json')
    for name,h in receipt['output_sha256'].items():
        if sha(PARENT/'diagnostics'/name)!=h:raise ValueError('Frozen diagnostics changed '+name)
    delivery=read(PARENT/'delivery.json')
    if sha(PARENT/'verification.json')!=delivery['verification_sha256']:raise ValueError('Endpoint verification changed')
    x,d=load_data();x.sort_indices()
    actual=[hashlib.sha256(x.indices[x.indptr[i]:x.indptr[i+1]].astype('<i8').tobytes()+
        x.data[x.indptr[i]:x.indptr[i+1]].astype('<f4').tobytes()).hexdigest() for i in range(x.shape[0])]
    if not np.array_equal(d.canonical_key,d.local.map(dict(enumerate(actual)))):raise ValueError('Numerical input identity changed')
    folders=[PARENT/f'fold{f}_R' for f in range(3)]+[PARENT/f'fold1_{a}' for a in ['O','C','CO']]
    sources={Path(__file__),PARENT/'delivery.json',PARENT/'verification.json',PARENT/'run_seal.json',
        PARENT/'diagnostics/receipt.json',PARENT/'diagnostics/endpoint_training_error_ledger.parquet',
        ROOT/'data/official/train.parquet',ROOT/'artifacts/v124_header_trial_20260929/B_header_ASA.npz',
        ROOT/'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet',
        REVIEW/'training_role_error_ledger.parquet'}
    for folder in folders:
        sources.update(folder/name for name in ['fit.json','started.json','progress.json','checkpoints.json',
            'epoch100_model.pt','epoch100_train_prob.npy','train_ids.npy'])
    OUT.mkdir();save(OUT/'source_receipt.json',{'new_fits':0,'new_updates':0,
        'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(sources)}})
    summaries=[];ledgers=[];conflicts=[];stability=[]
    diag=pd.read_parquet(PARENT/'diagnostics/endpoint_training_error_ledger.parquet')
    for folder in folders:
        meta=read(folder/'fit.json');f=meta['fold'];arm=meta['arm'];fit,c,pure,_,ids=fit_context(d,f)
        cp=read(folder/'checkpoints.json')[-1]
        if cp['epoch']!=100 or sha(folder/'epoch100_model.pt')!=cp['model_sha256']:raise ValueError('Wrong endpoint')
        saved=np.load(folder/'epoch100_train_prob.npy');stored_ids=np.load(folder/'train_ids.npy')
        if not np.array_equal(ids,stored_ids) or sha(folder/'epoch100_train_prob.npy')!=cp['train_prob_sha256']:raise ValueError('Training score identity changed')
        p=np.zeros((len(c),3),np.float32);p[ids]=saved;loc=fit.local.to_numpy();y=fit.truth.to_numpy();pred=p[loc].argmax(1)
        mix=fit.groupby(['canonical_key','truth']).size().unstack(fill_value=0).reindex(columns=[1,2],fill_value=0)
        mixed=mix[(mix>0).sum(1)>1]
        floor=int((mixed.sum(1)-mixed.max(1)).sum());pure_row=pure[loc].astype(bool)
        fm=int(((y==1)&(pred!=1)).sum());fs=int(((y==2)&(pred!=2)).sum())
        pm=int(((y==1)&(pred!=1)&pure_row).sum());ps=int(((y==2)&(pred!=2)&pure_row).sum())
        item={'fold':f,'arm':arm,'fit_rows':len(fit),'M_support':int((y==1).sum()),'S_support':int((y==2).sum()),
            'pure_M_support':int(((y==1)&pure_row).sum()),'pure_S_support':int(((y==2)&pure_row).sum()),
            'M_errors':fm,'S_errors':fs,'pure_M_errors':pm,'pure_S_errors':ps,
            'mixed_input_keys':len(mixed),'empirical_minimum_errors':floor,'excess_over_input_minimum':fm+fs-floor,
            'all_mixed_keys_have_M_majority':bool((mixed[1]>mixed[2]).all()),
            'endpoint_full_training_mastered':pm==ps==0 and fm+fs==floor}
        comparison=diag[(diag.outer_fit_role==f)&diag.arm.eq(arm)]
        if not np.array_equal(comparison.row_position,fit.row_position) or not np.array_equal(comparison.pred,pred):raise ValueError('Independent saved decision ledger mismatch')
        z=fit.copy();z['outer_fit_role']=f;z['arm']=arm;z['pred']=pred;z['pure_TRAIN_input']=pure_row
        z['truth_probability']=p[loc,y]
        other=p[loc].copy();other[np.arange(len(y)),y]=-np.inf
        z['probability_margin']=p[loc,y]-other.max(1)
        z['no_correct_member']=comparison.no_correct_member.to_numpy()
        z['old_B_pred']=comparison.old_B_pred.to_numpy()
        item['pure_errors_no_correct_member_rows']=int(z.loc[pure_row&(pred!=y),'no_correct_member'].sum())
        item['pure_errors_with_correct_member_rows']=pm+ps-item['pure_errors_no_correct_member_rows']
        summaries.append(item);ledgers.append(z)
        for key,row in mixed.iterrows():
            conflicts.append({'fold':f,'arm':arm,'canonical_key':key,'M':int(row[1]),'S':int(row[2]),
                'empirical_minimum_errors':int(row.min()),'majority_label':int(row.idxmax())})
        hist=read(folder/'progress.json')
        if [v['epoch'] for v in hist]!=list(range(1,101)):raise ValueError('Missing per-epoch class histories')
        for h in hist[-10:]:
            s=h['stats'];exact=s['M_errors']==0 and s['pure_S_errors']==0 and s['S_errors']==floor
            stability.append({'fold':f,'arm':arm,'epoch':h['epoch'],'M_errors':s['M_errors'],
                'S_errors':s['S_errors'],'pure_S_errors':s['pure_S_errors'],
                'input_minimum_reached':exact})
        item['last5_input_minimum_reached_every_epoch']=all(v['input_minimum_reached'] for v in stability[-5:])
        item['last10_input_minimum_reached_epochs']=sum(v['input_minimum_reached'] for v in stability[-10:])
    allrows=pd.concat(ledgers,ignore_index=True)
    allrows.to_parquet(OUT/'training_mastery_ledger.parquet',index=False)
    remain=allrows[allrows.arm.eq('R')&allrows.pure_TRAIN_input&allrows.pred.ne(allrows.truth)]
    remain.to_parquet(OUT/'R_nonmixed_remaining_errors.parquet',index=False)
    root_summary=remain.groupby(['outer_fit_role','root','truth']).size().rename('errors').reset_index()
    root_summary.to_csv(OUT/'R_remaining_error_sources.csv',index=False)
    remaining={'role_rows':len(remain),'unique_original_rows':int(remain.row_position.nunique()),
        'M_role_errors':int(remain.truth.eq(1).sum()),'S_role_errors':int(remain.truth.eq(2).sum()),
        'no_correct_member_role_errors':int(remain.no_correct_member.sum()),
        'correct_member_but_wrong_ensemble_role_errors':int((~remain.no_correct_member).sum())}
    audit={'status':'v131_training_mastery_partially_solved_not_complete','new_fits':0,'new_updates':0,
        'rows':2056871,'ASA_original_rows':len(d),'R_train_roles':len(allrows[allrows.arm.eq('R')]),
        'fold_arm_results':summaries,'mixed_inputs':conflicts,'last10_epoch_stability':stability,
        'R_nonmixed_remaining':remaining,'latest_actual_training_status':delivery['status'],
        'limits':['Current numerical representation only; unique labels do not prove semantic security truth.',
            'Repeated original rows in TRAIN roles are not independent new examples.',
            'End models were replayed in V131; this review verifies hashes and recounts stored TRAIN predictions, not a new neural replay.',
            'V131 per-epoch source CSV was overwritten; class histories and checkpoint/source ledgers remain.']}
    save(OUT/'audit.json',audit)
    save(OUT/'output_receipt.json',{'source_sha256':sha(__file__),'new_fits':0,'new_updates':0,
        'output_sha256':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()}})
    print(json.dumps({'status':audit['status'],'remaining':remaining,'results':summaries},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
