"""No fitting: decompose V110 P2 objective and frozen score changes."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.special import expit
from threadpoolctl import threadpool_limits

from run_v75 import ROOT, save, sha
from v110_layer_probes import DEST as V110, LEDGER, N2, PREV, extract, check_registration

DEST = ROOT/'artifacts/v111_root_review_20260929'
BLOCKS = {'hidden':slice(0,2048), 'parsed_facts':slice(2048,2525),
          'record_source_port':slice(2525,2543)}


def metrics(y,z):
    pred=np.where(z>=0,2,1)
    loss=np.where(y==2,np.logaddexp(0,-z),np.logaddexp(0,z))
    return {'rows':int(len(y)), 'CE':float(loss.mean()),
            'M_errors':int(((y==1)&(pred!=1)).sum()),
            'S_errors':int(((y==2)&(pred!=2)).sum()),
            'M_CE':float(loss[y==1].mean()) if (y==1).any() else None,
            'S_CE':float(loss[y==2].mean()) if (y==2).any() else None}


def main():
    check_registration()
    DEST.mkdir(exist_ok=True)
    target=DEST/'objective_and_margin_diagnosis.json'
    assert not target.exists()
    d=pd.read_parquet(LEDGER,columns=['row_position','local','truth','fold','root','behavior'])
    x=sparse.load_npz(N2)
    enc=joblib.load(ROOT/'artifacts/v75_four_arm_20260921_r2/facts_encoder.joblib')
    names=list(enc.names())
    assert len(names)==477
    names+=['record_source_port_'+str(i) for i in range(18)]
    rows=[]; terms=[]; fold_reports=[]; score_rows=[]
    hashes={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in [LEDGER,N2,V110/'registration.json']}
    with threadpool_limits(limits=4):
        for k in (0,1,2):
            folder=PREV/f'fold{k}_N2_TabM25_seed10201'
            for p in (folder/'model.pt',V110/f'fold{k}_P2/probe.npz'):
                hashes[p.relative_to(ROOT).as_posix()]=sha(p)
            state=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)['state_dict']
            hw=state['head.weight'].numpy().astype('f8')
            hb=state['head.bias'].numpy().astype('f8')
            fw=state['facts_direct.weight'].numpy().astype('f8')
            oldw=np.r_[(hw[:,:,2]-hw[:,:,1]).reshape(-1)/16,fw[2]-fw[1]]
            oldb=float((hb[:,2]-hb[:,1]).mean())
            feats,replay_diff=extract(k,x)
            h=feats['P2'].astype('f8');del feats
            p=np.load(V110/f'fold{k}_P2/probe.npz')
            hs=(h-p['mean'])/p['std']
            oldtheta=oldw*p['std']; oldintercept=oldb+float(p['mean']@oldw)
            newtheta=p['coef']; newintercept=float(p['intercept'])
            zo=hs@oldtheta+oldintercept; zn=hs@newtheta+newintercept
            direct=np.load(ROOT/f'artifacts/v109_plan_review_20260929/N2_fold{k}_member_logits.npy').mean(1)
            assert np.allclose(zo,direct[:,2]-direct[:,1],atol=5e-5,rtol=5e-5)
            saved=np.load(V110/f'fold{k}_P2/ASA_input_S_probability.npy')
            probe_diff=float(np.max(np.abs(expit(zn)-saved)))
            print(json.dumps({'fold':k,'probe_replay_max_abs_probability_diff':probe_diff,
                'probe_replay_decision_differences':int(np.sum((zn>=0)!=(saved>=.5)))}),flush=True)
            assert np.allclose(expit(zn),saved,atol=2e-6,rtol=2e-6)
            assert np.array_equal(zn>=0,saved>=.5)
            train=d.fold.to_numpy()!=k; held=~train
            idx=d.local.to_numpy(); y=d.truth.to_numpy()
            regularizer={'original':float(.0005*(oldtheta@oldtheta)),
                         'P2':float(.0005*(newtheta@newtheta))}
            fr={'fold':k,'old_model_replay_maxdiff':replay_diff,'probe_replay_maxdiff':probe_diff,
                'regularizer':regularizer,'populations':{},'counterfactual_blocks':{},
                'standardization':{}}
            for role,mask in [('train',train),('heldout',held)]:
                original=metrics(y[mask],zo[idx[mask]])
                new=metrics(y[mask],zn[idx[mask]])
                fr['populations'][role]={'original_mean_logit':original,'P2':new}
                if role=='train':
                    fr['objective']={a:fr['populations'][role][b]['CE']+regularizer[a]
                        for a,b in [('original','original_mean_logit'),('P2','P2')]}
            actual_fit=json.loads((V110/f'fold{k}_P2/fit.json').read_text(encoding='utf-8'))
            fr['objective_replay_abs_diff']=abs(fr['objective']['P2']-actual_fit['objective'])
            assert fr['objective_replay_abs_diff']<1e-7,fr['objective_replay_abs_diff']
            trainids=np.unique(idx[train]); heldids=idx[held]
            for block,sl in BLOCKS.items():
                trainblock=h[trainids,sl]; heldblock=h[heldids,sl]
                lo=trainblock.min(0);hi=trainblock.max(0)
                absent=(hi==lo)
                fr['standardization'][block]={
                    'train_constant_dimensions':int(absent.sum()),
                    'heldout_rows_with_new_constant_value':int(((heldblock[:,absent]!=lo[absent]).any(1)).sum()),
                    'heldout_rows_any_coordinate_outside_train_range':int(((heldblock<lo-1e-6)|(heldblock>hi+1e-6)).any(1).sum()),
                    'heldout_max_abs_standardized_value':float(abs(hs[heldids,sl]).max()),
                    'old_reg':float(.0005*np.dot(oldtheta[sl],oldtheta[sl])),
                    'new_reg':float(.0005*np.dot(newtheta[sl],newtheta[sl]))}
                # Algebraic block replacement, not an evaluated candidate selected for use.
                c=zo+hs[:,sl]@(newtheta[sl]-oldtheta[sl])
                fr['counterfactual_blocks']['replace_only_'+block]=metrics(y[held],c[idx[held]])
            fr['counterfactual_blocks']['replace_only_intercept']=metrics(y[held],zo[idx[held]]+newintercept-oldintercept)
            heldframe=d.loc[held].copy()
            heldframe['original_margin']=zo[idx[held]];heldframe['P2_margin']=zn[idx[held]]
            heldframe['delta_intercept']=newintercept-oldintercept
            for block,sl in BLOCKS.items():
                heldframe['old_'+block]=hs[idx[held],sl]@oldtheta[sl]
                heldframe['new_'+block]=hs[idx[held],sl]@newtheta[sl]
                heldframe['delta_'+block]=heldframe['new_'+block]-heldframe['old_'+block]
            reconstructed=sum(heldframe['delta_'+b].to_numpy() for b in BLOCKS)+heldframe.delta_intercept.to_numpy()
            assert np.allclose(reconstructed,heldframe.P2_margin-heldframe.original_margin,atol=1e-10)
            score_rows.append(heldframe.drop(columns=['behavior']))
            for root in (2868,29,5124,21702,20849,11083,216921):
                for role,mask in [('train',train),('heldout',held)]:
                    choose=mask&(d.root.to_numpy()==root)
                    if not choose.any():continue
                    ii=idx[choose];yy=y[choose]
                    item={'fold':k,'root':root,'role':role,
                        'original':metrics(yy,zo[ii]),'P2':metrics(yy,zn[ii]),
                        'old_margin_mean':float(zo[ii].mean()),'new_margin_mean':float(zn[ii].mean()),
                        'intercept_delta':newintercept-oldintercept,
                        'block_delta_means':{b:float((hs[ii,sl]@(newtheta[sl]-oldtheta[sl])).mean()) for b,sl in BLOCKS.items()}}
                    rows.append(item)
                    if root in (2868,11083,216921) and k==1:
                        contributions=hs[ii,2048:].mean(0)*(newtheta[2048:]-oldtheta[2048:])
                        for j in np.argsort(-abs(contributions))[:20]:
                            terms.append({'fold':k,'root':root,'role':role,'feature':names[j],
                                'mean_standardized_value':float(hs[ii,2048+j].mean()),
                                'coefficient_change':float(newtheta[2048+j]-oldtheta[2048+j]),
                                'score_change_term':float(contributions[j])})
            fold_reports.append(fr)
            print(json.dumps({'fold':k,'regularizer':regularizer,'population':fr['populations'],
                              'counterfactual':fr['counterfactual_blocks']},ensure_ascii=False),flush=True)
            del h,hs
    allscore=pd.concat(score_rows).sort_values('row_position').reset_index(drop=True)
    assert len(allscore)==112807 and allscore.row_position.nunique()==112807
    assert np.array_equal(allscore.row_position,d.row_position)
    for k in (0,1,2):
        z=allscore.loc[allscore.fold==k,'P2_margin'].to_numpy()
        old=pd.read_parquet(V110/'OOF_probe_comparison.parquet')
        assert np.array_equal(np.where(z>=0,2,1),old.loc[old.fold==k,'P2_prediction'].to_numpy())
    allscore.to_parquet(DEST/'P2_margin_decomposition.parquet',index=False)
    pd.DataFrame(terms).to_csv(DEST/'root2868_factual_score_terms.csv',index=False,encoding='utf-8-sig')
    report={'status':'no_fit_V110_objective_diagnosis','classifier_fits':0,'calibration_fits':0,
        'source_sha256':sha(__file__),'input_hashes':hashes,'folds':fold_reports,'selected_root_scores':rows,
        'scope':'Frozen model replay and algebraic attribution on already inspected development folds only. No threshold/model selected, no training, no causal feature claim.',
        'limits':['Attribution depends on feature basis and correlated features; replacing a block is an arithmetic intervention, not a valid real-world log intervention.',
                  'Original mean-logit classifier is a feasible affine reference, not exactly the original mean-probability classifier.',
                  'Training and heldout populations differ. Constant and out-of-range counts do not alone prove failure causes.'],
        'output_hashes':{f:sha(DEST/f) for f in ['P2_margin_decomposition.parquet','root2868_factual_score_terms.csv']}}
    save(target,report)


if __name__=='__main__':main()
