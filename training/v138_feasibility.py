"""Bounded TRAIN-only affine-margin sufficient-condition probe; never a candidate."""
import argparse
import time
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.optimize import linprog
from v138_runtime import ROOT,OUT,read,save,sha,require_run_seal
from v135_runtime import load_data,fit_context

HEAD_SIZE=16*3*129
PARAMETERS=HEAD_SIZE+3*495


def packed(state):
    head=np.concatenate([state['weight'].cpu().numpy().transpose(0,2,1),state['bias'].cpu().numpy()[:,:,None]],2)
    return np.concatenate([head.ravel(),state['facts'].cpu().numpy().ravel()])


def logits(theta,hidden,facts):
    head=theta[:HEAD_SIZE].reshape(16,3,129);direct=theta[HEAD_SIZE:].reshape(3,495)
    return np.einsum('nkh,kch->nkc',hidden,head[:,:,:128],optimize=True)+head[:,:,128]+(facts@direct.T)[:,None,:]


def matrix(rows,hidden,facts,truth):
    rr=[];cc=[];vv=[]
    for r,(i,m,other) in enumerate(rows):
        y=int(truth[i]);hv=np.append(hidden[i,m],1.);hi=np.flatnonzero(hv);fi=np.flatnonzero(facts[i])
        # A theta <= -1 encodes (true - competitor) >= 1.
        for cl,sign in [(y,-1.),(other,1.)]:
            indices=np.concatenate([m*3*129+cl*129+hi,HEAD_SIZE+cl*495+fi])
            values=sign*np.concatenate([hv[hi],facts[i,fi]])
            rr.extend([r]*len(indices));cc.extend(indices.tolist());vv.extend(values.tolist())
    return sparse.csr_matrix((vv,(rr,cc)),shape=(len(rows),PARAMETERS))


def probe(fold):
    require_run_seal(ROOT/'training/v138_train.py')
    torch.set_num_threads(4)
    start=time.monotonic();folder=OUT/f'probe_fold{fold}'
    if folder.exists():raise FileExistsError('One supervised probe per fold')
    folder.mkdir();fitreceipt=read(OUT/f'fold{fold}_H_L/fit.json')
    if fitreceipt['endpoint_stats']['pure_M_errors']+fitreceipt['endpoint_stats']['pure_S_errors']==0:
        raise ValueError('Conditional feasibility probe not needed')
    _,d=load_data();frame,c,pure,_,used=fit_context(d,fold)
    ids=used[pure[used].astype(bool)];truth=c[ids].argmax(1)
    hidden=np.load(OUT/f'fold{fold}_hidden.npy')[ids].astype(np.float64)
    facts=np.load(OUT/'facts.npy')[ids].astype(np.float64)
    keymap=frame[['local','canonical_key']].drop_duplicates('local').set_index('local').canonical_key
    keys=keymap.reindex(ids).to_numpy()
    state=torch.load(OUT/f'fold{fold}_H_L/endpoint_readout.pt',map_location='cpu',weights_only=True)['readout']
    theta=packed(state);active=set();history=[];certificate=False;termination='iteration_limit'
    for iteration in range(50):
        remaining=600-(time.monotonic()-start)
        if remaining<=0:termination='time_limit_inconclusive';break
        z=logits(theta,hidden,facts);margins=z[np.arange(len(ids))[:,None],np.arange(16)[None,:],truth[:,None]][:,:,None]-z
        margins[np.arange(len(ids))[:,None],np.arange(16)[None,:],truth[:,None]]=np.inf
        bad=margins<1-1e-6;minimum=float(margins.min())
        entry={'iteration':iteration,'violated_constraints':int(bad.sum()),'worst_margin':minimum,'active_constraints':len(active),'seconds':time.monotonic()-start}
        history.append(entry);save(folder/'progress.json',history)
        print({'stage':'TRAIN_margin_probe','fold':fold,**entry},flush=True)
        if not bad.any():certificate=True;termination='full_pure_population_margin_verified';break
        additions=[]
        for member in range(16):
            candidates=np.argwhere(bad[:,member,:])
            order=sorted(candidates.tolist(),key=lambda pair:(margins[pair[0],member,pair[1]],keys[pair[0]],pair[1]))
            additions.extend([(i,member,other) for i,other in order if (i,member,other) not in active][:64])
        if not additions:termination='no_new_constraints_inconclusive';break
        active.update(additions);rows=sorted(active)
        a=matrix(rows,hidden,facts,truth);remaining=600-(time.monotonic()-start)
        if remaining<=0:termination='time_limit_inconclusive';break
        result=linprog(np.zeros(PARAMETERS),A_ub=a,b_ub=-np.ones(len(rows)),bounds=(None,None),method='highs',
            options={'time_limit':remaining,'primal_feasibility_tolerance':1e-7})
        entry['solver_status']=int(result.status);entry['solver_message']=str(result.message)
        save(folder/'progress.json',history)
        if not result.success:termination='sufficient_condition_infeasible' if result.status==2 else 'solver_inconclusive';break
        theta=result.x
    if certificate:np.save(folder/'witness.npy',theta)
    save(folder/'receipt.json',{'status':'supervised_diagnostic_only','fold':fold,'solver_runs':len([h for h in history if 'solver_status' in h]),
        'pure_original_train_rows':int((c[ids].sum())),'pure_local_inputs':len(ids),'all_members':16,'competitors_each':2,
        'certificate':certificate,'termination':termination,'seconds':time.monotonic()-start,'history_sha256':sha(folder/'progress.json'),
        'legal_TRAIN_only':True,'HELD_labels_used':0,'candidate_promotable':False,
        'round2_eligible':False,'round2_note':'Pure-only certificate is insufficient; full TRAIN mixed-majority feasibility and new bound runtime required.',
        'limits':'Per-member margin >=1 is sufficient, not necessary for mean-probability classification. Infeasible/timeout cannot prove model impossible.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,required=True);a=p.parse_args();probe(a.fold)
