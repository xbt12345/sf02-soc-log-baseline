"""Separate read-only-model audit: epoch rows versus actual models, factorial effects."""
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from v134_runtime import ROOT,OUT,ARMS,read,save,sha,require_run_seal,load_data,fit_context
from v131_model import Classifier,infer


def main():
    require_run_seal(ROOT/'training/v134_train.py')
    destination=OUT/'postflight_audit'
    if destination.exists():raise FileExistsError('Preserve completed independent audit')
    destination.mkdir();x,d=load_data();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    bound={Path(__file__),OUT/'delivery.json',OUT/'verification.json',OUT/'run_seal.json',OUT/'learning_qualification.json'}
    results=[];prefix=[];trajectories=[];remaining=[];role_exposures=0
    for fold in range(3):
        fit,c,pure,_,ids=fit_context(d,fold)
        for arm in ARMS:
            folder=OUT/f'fold{fold}_{arm}';history=read(folder/'progress.json');bound.add(folder/'fit.json')
            for h in history:
                s=h['stats'];trajectories.append({'fold':fold,'arm':arm,'epoch':h['epoch'],'lr':h['lr'],
                    'M_errors':s['M_errors'],'S_errors':s['S_errors'],'pure_M_errors':s['pure_M_errors'],
                    'pure_S_errors':s['pure_S_errors'],'mastered':s['mastered'],'original_row_CE_online':h['row_risk_online'],
                    'pure_class_CE_online':h['pure_class_risk_online']})
            for epoch in range(96,101):
                path=folder/f'epoch{epoch}_model.pt';rows_path=folder/'epochs'/f'epoch{epoch:03d}_rows.parquet'
                bound.update([path,rows_path]);state=torch.load(path,map_location='cpu',weights_only=True)
                model=Classifier(128).to('cuda');model.load_state_dict(state['model']);q,*_=infer(model,x,ids)
                full=np.zeros((len(c),3),np.float32);full[ids]=q;expected=full[fit.local.to_numpy()]
                rows=pd.read_parquet(rows_path);actual=rows[['p0','p1','p2']].to_numpy()
                if not np.array_equal(rows.row_position,fit.row_position) or not np.array_equal(rows.truth,fit.truth):raise ValueError('Original-row identity changed')
                gap=float(np.abs(actual-expected).max())
                if gap>2e-6 or not np.array_equal(rows.pred,expected.argmax(1)):raise ValueError('Actual model versus original row journal mismatch')
                results.append({'fold':fold,'arm':arm,'epoch':epoch,'original_rows':len(rows),'model_to_original_row_max_abs':gap})
                role_exposures+=len(rows)
                if epoch==100:
                    bad=rows[rows.pure_TRAIN_input & rows.pred.ne(rows.truth)].copy();bad['fold']=fold;bad['arm']=arm
                    remaining.append(bad)
                del model;gc.collect();torch.cuda.empty_cache()
        # A genuine fixed single-factor contrast must match before the schedule diverges.
        for loss in ['R','O']:
            maximum=0.;flips=0;mass=0
            for epoch in range(1,81):
                paths=[OUT/f'fold{fold}_{loss}_{schedule}'/'epochs'/f'epoch{epoch:03d}_rows.parquet' for schedule in ['const','decay']]
                a,b=[pd.read_parquet(p,columns=['row_position','pred','p0','p1','p2']) for p in paths]
                if not np.array_equal(a.row_position,b.row_position):raise ValueError('Contrast population changed')
                maximum=max(maximum,float(np.abs(a[['p0','p1','p2']].to_numpy()-b[['p0','p1','p2']].to_numpy()).max()))
                flips+=int(a.pred.ne(b.pred).sum());mass+=len(a)
            prefix.append({'fold':fold,'loss':loss,'epochs':80,'original_row_comparisons':mass,
                'max_abs_probability':maximum,'prediction_differences':flips,'matched':maximum<=2e-6 and flips==0})
    frame=pd.DataFrame(trajectories);frame.to_csv(destination/'learning_trajectories.csv',index=False)
    pd.concat(remaining,ignore_index=True).to_parquet(destination/'endpoint_nonmixed_errors.parquet',index=False)
    contrasts=[]
    for fold in range(3):
        for population in ['TRAIN','HELD']:
            if population=='TRAIN':
                cells={arm:read(OUT/f'fold{fold}_{arm}/fit.json')['endpoint_training'] for arm in ARMS}
            else:
                ledger=pd.read_parquet(OUT/'ASA_prediction_ledger.parquet');part=ledger[ledger.fold.eq(fold)]
                cells={arm:{name+'_errors':int((part.truth.eq(cl)&part['pred_'+arm].ne(cl)).sum()) for cl,name in [(1,'M'),(2,'S')]} for arm in ARMS}
            for cl in ['M','S']:
                e={a:cells[a][cl+'_errors'] for a in ARMS}
                contrasts.append({'fold':fold,'population':population,'class':cl,**e,
                    'schedule_effect_R':e['R_decay']-e['R_const'],'schedule_effect_O':e['O_decay']-e['O_const'],
                    'supervision_effect_const':e['O_const']-e['R_const'],'supervision_effect_decay':e['O_decay']-e['R_decay'],
                    'interaction':(e['O_decay']-e['O_const'])-(e['R_decay']-e['R_const'])})
    pd.DataFrame(contrasts).to_csv(destination/'factorial_effects.csv',index=False)
    save(destination/'source_receipt.json',{'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(bound)},'new_fits':0,'new_updates':0})
    save(destination/'audit.json',{'status':'actual_original_row_window_and_factorial_prefix_audited',
        'new_fits':0,'new_updates':0,'window_replays':results,'window_original_row_comparisons':role_exposures,
        'prefix_checks':prefix,'factorial_prefix_valid':all(z['matched'] for z in prefix),
        'reproducible_or_blind_quality_proven':False,'model_promoted':False,
        'limits':['Shared observed development population; zero TRAIN errors cannot establish security semantics.',
                  'Factorial contrasts are finite-population observations, not independent-fold statistical proof.']})
    save(destination/'output_receipt.json',{'source_sha256':sha(__file__),
        'output_sha256':{p.name:sha(p) for p in destination.iterdir() if p.is_file()}})
    print({'actual_window_original_rows':role_exposures,'prefix_valid':all(z['matched'] for z in prefix),'new_fits':0},flush=True)


if __name__=='__main__':main()
