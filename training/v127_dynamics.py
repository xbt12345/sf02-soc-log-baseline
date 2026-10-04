"""Post-fit V127 attention, token, pooled and residual diagnostics; no fitting."""
import json
import numpy as np
import pandas as pd
import torch

from v126_frozen_audit import ROOT, PARENT, sha, read, save
from v127_model import make_branch
from v127_train import OUT, TRACE, LENGTHS, BODY

EPOCHS=(0,1,2,5,10,15,20,25,35,50)


def require(ok,message):
    if not ok:raise ValueError(message)


def summary(x):
    a=np.asarray(x,dtype=np.float64)
    return {'mean':float(a.mean()),'std':float(a.std()),'min':float(a.min()),'max':float(a.max())}


def panel_probe(model,body,lengths,ids):
    model.eval()
    entropy=[[],[]];tokens=[];pooled=[];maxdiff=0.0
    with torch.no_grad():
        for start in range(0,len(ids),64):
            local=ids[start:start+64];length=torch.as_tensor(lengths[local],device='cuda',dtype=torch.int64)
            n=int(length.max());raw=torch.as_tensor(body[local,:n],device='cuda',dtype=torch.uint8)
            pad=torch.arange(n,device='cuda')[None,:]>=length[:,None]
            h=model.embed((raw.long()+1).masked_fill(pad,0))+model.position[:n]
            for i,layer in enumerate(model.encoder.layers):
                source=layer.norm1(h) if layer.norm_first else h
                z,w=layer.self_attn(source,source,source,key_padding_mask=pad,
                                    need_weights=True,average_attn_weights=False)
                valid=(~pad).float()[:,None,:]
                ent=-(w*w.clamp_min(1e-30).log()).sum(-1)/length.log()[:,None,None]
                entropy[i].append(((ent*valid).sum(-1)/length[:,None]).cpu().numpy())
                if layer.norm_first:
                    h=h+layer.dropout1(z)
                    f=layer.norm2(h)
                    h=h+layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(f)))))
                else:
                    h=layer.norm1(h+layer.dropout1(z))
                    h=layer.norm2(h+layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(h))))))
            h=model.final_norm(h)
            mean=h.masked_fill(pad[:,:,None],0).sum(1)/length[:,None]
            tokens.extend((h.masked_fill(pad[:,:,None],0)-mean[:,None,:]).square().mean(-1)
                          .masked_fill(pad,0).sum(-1).div(length).sqrt().cpu().numpy().tolist())
            pooled.append(mean.cpu().numpy())
            rebuilt=model.head(mean)
            actual=model(raw,length)
            maxdiff=max(maxdiff,float((rebuilt-actual).abs().max()))
    require(maxdiff<2e-5,'Manual attention path diverges')
    allpool=np.concatenate(pooled)
    return {'panel_size':len(ids),'manual_forward_max_abs_diff':maxdiff,
            'layer_head_mean_normalized_entropy':[np.concatenate(v).mean(0).tolist() for v in entropy],
            'within_input_token_spread':summary(tokens),
            'between_input_pooled_distance_to_mean':summary(np.linalg.norm(allpool-allpool.mean(0),axis=1))}


def weighted_spread(x,mass):
    a=np.asarray(x,dtype=np.float64);w=np.asarray(mass,dtype=np.float64)
    mean=np.dot(w,a)/w.sum()
    return {'original_rows':int(w.sum()),'mean':float(mean),
            'std':float(np.sqrt(np.dot(w,(a-mean)**2)/w.sum()))}


def main():
    target=OUT/'branch_dynamics.json'
    if target.exists():raise FileExistsError(target)
    trace=pd.read_parquet(TRACE,columns=['local','fold','truth'])
    local=trace.local.to_numpy();folds=trace.fold.to_numpy();truth=trace.truth.to_numpy()
    body=np.load(BODY,mmap_mode='r');lengths=np.load(LENGTHS)
    panel=pd.read_parquet(ROOT/'artifacts/v127_plan_review_20260929/attention_panel.parquet')
    torch.backends.mha.set_fastpath_enabled(False)
    records=[]
    for fold in range(3):
        fitids=np.unique(panel.loc[panel.fold==fold,'local'].to_numpy())
        heldids=[]
        for c in (1,2):
            available=np.unique(local[(folds==fold)&(truth==c)])
            count=min(256,len(available))
            heldids.extend(np.random.default_rng(12801+100*fold+c).choice(available,count,replace=False).tolist())
        heldids=np.array(sorted(set(heldids)),dtype=np.int64)
        rolemasses={role:np.bincount(local[mask],minlength=len(lengths)) for role,mask in
                    [('fit',folds!=fold),('held',folds==fold)]}
        for arm in ('W','P'):
            checks=read(OUT/f'fold{fold}_{arm}/checkpoints.json')
            require([x['epoch'] for x in checks]==list(EPOCHS),'Checkpoint list changed')
            for epoch in EPOCHS:
                path=OUT/f'fold{fold}_{arm}/epoch{epoch}_model.pt'
                model=make_branch(arm,12701,'cuda').eval()
                state=torch.load(path,map_location='cpu',weights_only=True)
                require((state['fold'],state['arm'],state['epoch'])==(fold,arm,epoch),'Wrong state')
                model.load_state_dict(state['branch'])
                extra=np.load(OUT/f'fold{fold}_{arm}/epoch{epoch}_residual.npy')
                delta=extra[:,2]-extra[:,1]
                records.append({'fold':fold,'arm':arm,'epoch':epoch,
                    'fit_attention_and_representation':panel_probe(model,body,lengths,fitids),
                    'held_attention_and_representation':panel_probe(model,body,lengths,heldids),
                    'residual_S_minus_M_original_row':{role:weighted_spread(delta,mass)
                                                    for role,mass in rolemasses.items()},
                    'model_sha256':sha(path)})
                del model
            print(json.dumps({'fold':fold,'arm':arm,'diagnostic_states':len(EPOCHS)}),flush=True)
    result={'status':'postfit_mechanism_diagnostics','latest_actual_training':'V127',
            'classifier_fits':0,'optimizer_steps':0,'states':records,
            'source_sha256':sha(__file__),
            'scope':'Fixed train/held panels for attention and representation; full original-row role masses for residual spread. Diagnostic only; no checkpoint selection or quality acceptance.'}
    save(target,result)
    print(json.dumps({'status':result['status'],'states':len(records)}),flush=True)


if __name__=='__main__':main()
