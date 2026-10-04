"""Show whether the original N2 mean-logit decision is feasible in P2's readout family."""
import json
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.special import expit
from threadpoolctl import threadpool_limits

from run_v75 import ROOT, save, sha
from v110_layer_probes import DEST, LEDGER, N2, PREV, extract, objective, check_registration


def counts_for_fold(d,k,n):
    rows=d[d.fold!=k]
    return np.bincount(rows.local.to_numpy()*3+rows.truth.to_numpy(),minlength=n*3).reshape(n,3)


def recount(y,score):
    pred=np.where(score>=0,2,1)
    return {'M_errors':int(((y==1)&(pred!=1)).sum()),
      'S_errors':int(((y==2)&(pred!=2)).sum())}


def main():
    check_registration()
    dest=DEST/'original_readout_feasibility.json'
    assert not dest.exists()
    d=pd.read_parquet(LEDGER,columns=['local','truth','fold','root'])
    x=sparse.load_npz(N2)
    out=[]
    with threadpool_limits(limits=4):
        for k in (0,1,2):
            folder=PREV/f'fold{k}_N2_TabM25_seed10201'
            state=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)['state_dict']
            hw=state['head.weight'].numpy().astype('f8')
            hb=state['head.bias'].numpy().astype('f8')
            fw=state['facts_direct.weight'].numpy().astype('f8')
            w=np.r_[(hw[:,:,2]-hw[:,:,1]).reshape(-1)/16,fw[2]-fw[1]]
            b=float((hb[:,2]-hb[:,1]).mean())
            features,_=extract(k,x)
            h=features['P2'].astype('f8')
            logits=np.load(ROOT/f'artifacts/v109_plan_review_20260929/N2_fold{k}_member_logits.npy').mean(1)
            direct=logits[:,2]-logits[:,1]
            rebuilt=h@w+b
            maxdiff=float(np.abs(direct-rebuilt).max())
            assert np.allclose(direct,rebuilt,atol=5e-5,rtol=5e-5),maxdiff
            p=np.load(DEST/f'fold{k}_P2/probe.npz')
            theta=np.r_[w*p['std'],b+float(p['mean']@w)]
            counts=counts_for_fold(d,k,x.shape[0])
            used=counts.sum(1)>0
            xt=np.ascontiguousarray((h[used]-p['mean'])/p['std'])
            m=counts[used,1].astype('f8');s=counts[used,2].astype('f8')
            old_obj,oldgrad=objective(theta,xt,m,s)
            fitted=np.r_[p['coef'],p['intercept']]
            fit_obj,fitgrad=objective(fitted,xt,m,s)
            assert fit_obj<=old_obj+1e-8,(fit_obj,old_obj)
            rows=d[d.fold==k]
            train=d[d.fold!=k]
            sc=rebuilt[rows.local.to_numpy()]
            tr_sc=rebuilt[train.local.to_numpy()]
            fitted_score=((h-p['mean'])/p['std'])@p['coef']+float(p['intercept'])
            out.append({'fold':k,'original_mean_logit_linear_reconstruction_max_abs_diff':maxdiff,
                'objective_original_mean_logit_head':old_obj,
                'objective_new_converged_P2_head':fit_obj,
                'objective_change_new_minus_original':fit_obj-old_obj,
                'original_mean_logit_heldout':recount(rows.truth.to_numpy(),sc),
                'P2_heldout':recount(rows.truth.to_numpy(),fitted_score[rows.local.to_numpy()]),
                'original_mean_logit_train':recount(train.truth.to_numpy(),tr_sc),
                'P2_train':recount(train.truth.to_numpy(),fitted_score[train.local.to_numpy()]),
                'source_model_sha256':sha(folder/'model.pt'),
                'probe_sha256':sha(DEST/f'fold{k}_P2/probe.npz')})
            print(json.dumps(out[-1],ensure_ascii=False),flush=True)
    save(dest,{'status':'no_fit_original_readout_feasibility_diagnosis','classifier_fits':0,
      'source_sha256':sha(__file__),'rows':out,
      'interpretation_limit':'Mean-logit binary difference is one feasible affine P2 readout. Lower training objective with worse heldout does not prove representation is fully sufficient or identify a unique causal field.'})


if __name__=='__main__':main()
