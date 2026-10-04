"""Post-terminal diagnostics only; no solver, backward, fitting, or selection."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import gc
import numpy as np
import pandas as pd
import torch
from v140_runtime import ROOT,OUT,OLD,sha,read,save,require_run_seal
from v140_train import start_model
from v138_train import configure
from v135_runtime import load_data,fit_context


def main():
    require_run_seal(ROOT/'training/v140_train.py');configure();_,d=load_data();parts=[]
    for fold in range(3):
        frame,c,pure,_,ids=fit_context(d,fold);h,f,m=start_model(fold)
        old=pd.read_parquet(OLD/f'fold{fold}_H_L/endpoint_original_rows.parquet')
        olderrors=set(old.loc[old.pure_TRAIN_input&old.pred.ne(old.truth),'local'])
        keys=frame.groupby('local').agg(sources=('root','nunique'),key=('canonical_key','first'))
        for arm in ['V138','C','E']:
            path=OLD/f'fold{fold}_H_L/endpoint_readout.pt' if arm=='V138' else OUT/f'fold{fold}_{arm}/endpoint_readout.pt'
            m.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['readout'])
            with torch.no_grad():
                for start in range(0,len(ids),2048):
                    take=ids[start:start+2048];z=m(h[take],f[take]);lp=torch.log_softmax(z,-1);pp=lp.exp();q=pp.mean(1)
                    logq=torch.logsumexp(lp,1)-np.log(16)
                    for cl in [1,2]:
                        keep=c[take,cl]>0;ii=take[keep];a=lp[keep,:,cl];resp=torch.softmax(a,1)
                        if len(ii)==0: continue
                        pred=q[keep].argmax(1);yprob=q[keep,cl]
                        others=q[keep].clone();others[:,cl]=-1
                        t=pd.DataFrame({'fold':fold,'arm':arm,'local':ii,'truth':cl,'original_mass':c[ii,cl],
                                        'pure':pure[ii].astype(bool),'V138_residual_input':[i in olderrors for i in ii],
                                        'pred':pred.cpu().numpy(),'p_true':yprob.cpu().numpy(),
                                        'probability_margin':(yprob-others.max(1).values).cpu().numpy(),
                                        'member_CE':(-a.mean(1)).cpu().numpy(),'ensemble_CE':(-logq[keep,cl]).cpu().numpy(),
                                        'effective_responsible_members':(1/resp.square().sum(1)).cpu().numpy(),
                                        'maximum_member_responsibility':resp.max(1).values.cpu().numpy(),
                                        'correct_members':pp[keep].argmax(-1).eq(cl).sum(1).cpu().numpy()})
                        t['canonical_key']=t.local.map(keys.key);t['TRAIN_source_count']=t.local.map(keys.sources);parts.append(t)
        del h,f,m;gc.collect();torch.cuda.empty_cache()
    allrows=pd.concat(parts);allrows.to_parquet(OUT/'member_responsibility_diagnostics.parquet',index=False)
    summary=[]
    for (fold,arm),a in allrows.groupby(['fold','arm']):
        for name,b in [('all_legal_TRAIN',a),('V138_residual_inputs',a[a.V138_residual_input.astype(bool)]),
                       ('pure_correct_exact2_single_source',a[a.pure&a.original_mass.eq(2)&a.TRAIN_source_count.eq(1)&a.pred.eq(a.truth)])]:
            if len(b)==0:continue
            w=b.original_mass.to_numpy();v={'fold':int(fold),'arm':arm,'slice':name,'inputs':len(b),'original_rows':int(w.sum()),
                                         'errors':int(b.loc[b.pred.ne(b.truth),'original_mass'].sum())}
            for col in ['p_true','probability_margin','member_CE','ensemble_CE','effective_responsible_members','maximum_member_responsibility','correct_members']:
                v[col+'_weighted_mean']=float(np.average(b[col],weights=w))
            summary.append(v)
    save(OUT/'readonly_diagnosis.json',{'status':'all_models_frozen_TRAIN_only_no_parameter_update','new_fits':0,'new_updates':0,
                                     'source_sha256':sha(__file__),'diagnostic_rows_sha256':sha(OUT/'member_responsibility_diagnostics.parquet'),
                                     'summary':summary,'interpretation_limit':'Responsibility concentration and drift are diagnostics, not proof of causal collusion or representation impossibility.'})
    print([v for v in summary if v['slice']=='V138_residual_inputs'],flush=True)


if __name__=='__main__':main()
