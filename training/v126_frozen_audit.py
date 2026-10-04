"""V126 zero-fit audit: branch collapse, fit-only constant replacement, and source errors."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from v125_model import Expert
from v125_evaluate import VIEW, TRACE, OFFICIAL, ROWS
from v124_header import old_text, HEADER

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'artifacts/v125_order_trial_20260929'
OUT=ROOT/'artifacts/v126_frozen_review_20260929'
EPOCHS=(1,2,5,10,15,20,25)


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def moments(v):
    v=np.asarray(v,dtype=np.float64)
    return {'n':len(v),'mean':float(v.mean()),'std':float(v.std()),
            'q0_10_50_90_100':np.quantile(v,[0,.1,.5,.9,1]).tolist()}


def errors(y,p):return {str(c):int(((y==c)&(p!=c)).sum()) for c in (1,2)}


def branch_outputs(branch,body,lengths):
    z=[];hidden=[]
    with torch.no_grad():
        for start in range(0,len(lengths),256):
            end=min(start+256,len(lengths));n=int(lengths[start:end].max())
            raw=torch.tensor(body[start:end,:n],device='cuda',dtype=torch.uint8)
            le=torch.tensor(lengths[start:end],device='cuda',dtype=torch.int64)
            pad=torch.arange(n,device='cuda')[None,:]>=le[:,None]
            ids=(raw.long()+1).masked_fill(pad,0)
            h=branch.embed(ids)+branch.position[:n]
            h=branch.encoder(h,src_key_padding_mask=pad)
            pooled=h.masked_fill(pad[:,:,None],0).sum(1)/le[:,None]
            z.append(branch.head(pooled).cpu().numpy());hidden.append(pooled.cpu().numpy())
    return np.concatenate(z),np.concatenate(hidden)


def main():
    if OUT.exists():raise FileExistsError(OUT)
    delivery=read(PARENT/'delivery.json')
    bound={**delivery['artifact_sha256'],**delivery['source_sha256'],delivery['report']:delivery['report_sha256'],
           delivery['risk_addendum']:delivery['risk_addendum_sha256']}
    for rel,h in bound.items():
        if sha(ROOT/rel)!=h:raise ValueError('Parent changed '+rel)
    official=pd.read_parquet(OFFICIAL,columns=['event_id','label_binary'])
    roles=pd.read_parquet(ROWS,columns=['row_position','event_id','label_index'])
    if not np.array_equal(official.event_id,roles.event_id):raise ValueError('Official event IDs changed')
    truth=official.label_binary.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy(dtype=np.int8)
    if not np.array_equal(truth,roles.label_index):raise ValueError('Official labels changed')
    t=pd.read_parquet(TRACE,columns=['row_position','local','root','fold','truth','raw_message'])
    if not np.array_equal(t.truth,truth[t.row_position]):raise ValueError('ASA labels changed')
    x=sparse.load_npz(VIEW);lengths=np.load(PARENT/'body_lengths.npy')
    local=t.local.to_numpy();y=t.truth.to_numpy();folds=t.fold.to_numpy()
    OUT.mkdir();trajectory=[];collapse=[];constant=[]
    endpred=read(PARENT/'primary_evaluation.json')
    rows_pred=pd.read_parquet(PARENT/'expert_ASA_predictions.parquet')
    for fold in range(3):
        fit=folds!=fold;held=~fit
        mass=np.bincount(local[fit],minlength=x.shape[0]).astype(np.float64)
        for arm in 'ABC':
            folder=PARENT/f'fold{fold}_{arm}'
            fitchecks={z['epoch']:z['fit_by_class'] for z in read(folder/'checkpoints.json')}
            for epoch in EPOCHS:
                p=np.load(folder/f'epoch{epoch}_prob.npy');q=p[local[held]];yy=y[held]
                row={'fold':fold,'arm':arm,'epoch':epoch,'held_rows':int(held.sum()),
                     'held_errors':errors(yy,q.argmax(1)),
                     'held_CE':{str(c):float(-np.log(np.clip(q[yy==c,c],1e-12,1)).mean()) for c in (1,2)},
                     'fit':fitchecks[epoch]}
                trajectory.append(row)
                if arm=='A':continue
                model=Expert(arm,10201).to('cuda')
                state=torch.load(folder/f'epoch{epoch}_model.pt',map_location='cpu',weights_only=True)
                model.base.load_state_dict(state['base']);model.branch.load_state_dict(state['branch']);model.eval()
                body=np.load(PARENT/('ordered_body_bytes.npy' if arm=='B' else 'shuffled_body_bytes.npy'))
                z,h=branch_outputs(model.branch,body,lengths)
                delta=z[:,2]-z[:,1]
                spread=float(np.square(h.astype(np.float64)-h.mean(0)).sum()/len(h))
                total=float(np.square(h.astype(np.float64)).sum()/len(h))
                item={'fold':fold,'arm':arm,'epoch':epoch,
                      'all_unique_MS_residual':moments(delta),
                      'fit_original_rows_MS_residual':moments(delta[local[fit]]),
                      'held_original_rows_MS_residual':moments(delta[local[held]]),
                      'pooled_variance_to_second_moment':spread/max(total,1e-30)}
                collapse.append(item)
                np.save(OUT/f'fold{fold}_{arm}_epoch{epoch}_branch_logits.npy',z)
                if epoch==25:
                    # Train-side mean has no access to held-out labels or feature frequencies.
                    const=(z.astype(np.float64)*mass[:,None]).sum(0)/mass.sum()
                    full=[];reduced=[];base=[]
                    model.branch=None
                    with torch.no_grad():
                        for start in range(0,x.shape[0],256):
                            stop=min(start+256,x.shape[0]);raw=model(x[start:stop])
                            full.append(torch.softmax(raw+torch.tensor(z[start:stop],device='cuda')[:,None,:],-1).mean(1).cpu().numpy())
                            reduced.append(torch.softmax(raw+torch.tensor(const,dtype=torch.float32,device='cuda')[None,None,:],-1).mean(1).cpu().numpy())
                            base.append(torch.softmax(raw,-1).mean(1).cpu().numpy())
                    actual=np.concatenate(full);pc=np.concatenate(reduced);pb=np.concatenate(base)
                    if not np.array_equal(actual.argmax(1),p.argmax(1)) or not np.allclose(actual,p,atol=2e-6,rtol=2e-6):
                        raise ValueError('Model/branch replay differs from saved predictions')
                    ids=local[held];yy=y[held]
                    constant.append({'fold':fold,'arm':arm,'fit_mean_logits':const.tolist(),
                        'model_replay_max_abs_diff':float(np.max(np.abs(actual-p))),
                        'held_full_errors':errors(yy,p[ids].argmax(1)),
                        'held_constant_errors':errors(yy,pc[ids].argmax(1)),
                        'held_base_only_errors':errors(yy,pb[ids].argmax(1)),
                        'held_constant_vs_actual_prediction_flips':int((pc[ids].argmax(1)!=p[ids].argmax(1)).sum()),
                        'held_constant_vs_actual_max_prob_diff':float(np.max(np.abs(pc[ids]-p[ids])))})
                    np.save(OUT/f'fold{fold}_{arm}_constant_probability.npy',pc)
                del model
            print(json.dumps({'stage':'frozen_arm_complete','fold':fold,'arm':arm}),flush=True)
    target=rows_pred.root.eq(29)&rows_pred.truth.eq(2)&rows_pred.expert_pred_A.eq(2)&rows_pred.expert_pred_B.eq(1)
    gap=pd.read_parquet(ROOT/'artifacts/v124_header_trial_20260929/header_span_ledger.parquet',columns=['row_position'])
    changed=0
    for raw in t.loc[target,'raw_message']:
        m=HEADER.match(raw)
        if m is None:raise ValueError('Unknown header')
        variant='<164>Jan 01 2000 00:00:00: '+raw[m.end():]
        changed+=old_text(raw)!=old_text(variant)
    header={'new_root29_S_errors':int(target.sum()),
            'overlap_known_header_gap':int(rows_pred.loc[target,'row_position'].isin(gap.row_position).sum()),
            'old_text_changes_after_full_header_replacement':changed,
            'interpretation':'Date-label association alone does not explain these new misses; the registered old text already masks their complete headers.'}
    save(OUT/'checkpoint_trajectory.json',trajectory);save(OUT/'branch_dynamics.json',collapse)
    save(OUT/'constant_branch_control.json',constant);save(OUT/'date_exposure_audit.json',header)
    report={'status':'frozen_audit_complete_no_fit','latest_actual_training':'V125','new_classifier_fits':0,
            'optimizer_steps':0,'calibration_fits':0,'model_promoted':False,
            'parent_bound_files_verified':len(bound),'full_official_rows':len(official),'ASA_rows':len(t),
            'source_sha256':sha(Path(__file__)),'date_exposure':header,
            'fold1_endpoint_B':next(v for v in collapse if v['fold']==1 and v['arm']=='B' and v['epoch']==25),
            'fold1_endpoint_C':next(v for v in collapse if v['fold']==1 and v['arm']=='C' and v['epoch']==25),
            'constant_controls':constant,
            'limits':['Post-fit diagnostic; no new independent blind test.',
                      'Near-constant branch output does not by itself establish learning rate, imbalance or normalization as the unique cause.',
                      'No claim that all sequence models or all official-data approaches are invalid.']}
    save(OUT/'diagnosis.json',report)
    print(json.dumps({'status':report['status'],'date_exposure':header,'constant_controls':constant},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
