"""Read-only attention and frozen-residual diagnostics; no optimizer exists here."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from v125_model import BodyBranch
from v126_frozen_audit import ROOT, PARENT, read, sha, save
from v125_evaluate import TRACE

OUT=ROOT/'artifacts/v127_plan_review_20260929'


def describe(a):
    a=np.asarray(a,dtype=np.float64)
    return {'mean':float(a.mean()),'min':float(a.min()),'median':float(np.median(a)),
            'max':float(a.max()),'p10':float(np.quantile(a,.1)),'p90':float(np.quantile(a,.9))}


def attention_probe(model,body,lengths,ids):
    all_ent=[[],[]];all_max=[[],[]];diff=0.0
    with torch.no_grad():
        for start in range(0,len(ids),32):
            ix=ids[start:start+32];n=int(lengths[ix].max())
            le=torch.as_tensor(lengths[ix],device='cuda',dtype=torch.int64)
            raw=torch.as_tensor(body[ix,:n],device='cuda',dtype=torch.int64)
            pad=torch.arange(n,device='cuda')[None,:]>=le[:,None]
            x=model.embed((raw+1).masked_fill(pad,0))+model.position[:n]
            for k,layer in enumerate(model.encoder.layers):
                z,w=layer.self_attn(x,x,x,key_padding_mask=pad,need_weights=True,average_attn_weights=False)
                # V125 architecture is Post-LN, dropout zero. Reconstruct to expose weights.
                x=layer.norm1(x+layer.dropout1(z))
                x=layer.norm2(x+layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))))
                entropy=-(w*w.clamp_min(1e-30).log()).sum(-1)/le.log()[:,None,None]
                valid=(~pad).float()[:,None,:]
                ent=(entropy*valid).sum(-1)/le[:,None]
                peak=(w.max(-1).values*valid).sum(-1)/le[:,None]
                all_ent[k].append(ent.cpu().numpy());all_max[k].append(peak.cpu().numpy())
            pooled=x.masked_fill(pad[:,:,None],0).sum(1)/le[:,None]
            rebuilt=model.head(pooled)
            reference=model(raw.to(torch.uint8),le)
            diff=max(diff,float((rebuilt-reference).abs().max()))
    if diff>2e-5:raise ValueError('Manual attention path differs: '+str(diff))
    return {'forward_max_abs_difference':diff,'layers':[{
        'layer':i,'normalized_entropy':describe(np.concatenate(all_ent[i])),
        'per_head_mean_entropy':np.concatenate(all_ent[i]).mean(0).tolist(),
        'mean_peak_attention':describe(np.concatenate(all_max[i]))} for i in range(2)]}


def main():
    if OUT.exists():raise FileExistsError(OUT)
    prior=read(ROOT/'artifacts/v126_frozen_review_20260929/verification.json')
    for rel,h in prior['artifact_sha256'].items():
        if sha(ROOT/rel)!=h:raise ValueError('V126 source changed '+rel)
    trace=pd.read_parquet(TRACE,columns=['row_position','local','fold','truth'])
    local=trace.local.to_numpy();folds=trace.fold.to_numpy();truth=trace.truth.to_numpy()
    length=np.load(PARENT/'body_lengths.npy')
    body={a:np.load(PARENT/('ordered_body_bytes.npy' if a=='B' else 'shuffled_body_bytes.npy')) for a in 'BC'}
    OUT.mkdir();samples=[];attention=[];residual=[]
    torch.backends.mha.set_fastpath_enabled(False)
    for fold in range(3):
        fit=folds!=fold;p=np.load(PARENT/f'fold{fold}_A/epoch25_prob.npy')
        # At zero residual, gradient of mean-member CE w.r.t. shared logits is mean(prob)-one_hot.
        cl=[]
        for c in (1,2):
            ix=np.flatnonzero(fit&(truth==c));grad_l1=2*(1-p[local[ix],c].astype(np.float64))
            cl.append({'class':c,'rows':len(ix),'logit_gradient_l1':describe(grad_l1),
                       'gradient_l1_mass':float(grad_l1.sum()),
                       'fraction_gradient_l1_below_0_01':float((grad_l1<.01).mean()),
                       'wrong_rows':int((p[local[ix]].argmax(1)!=c).sum())})
        residual.append({'fold':fold,'classes':cl,
                         'S_fraction_of_logit_gradient_l1_mass':cl[1]['gradient_l1_mass']/sum(v['gradient_l1_mass'] for v in cl)})
        # Fixed stratified diagnostic panel only; does not change fit weights or select a checkpoint.
        ids=[]
        for c in (1,2):
            available=np.unique(local[fit&(truth==c)])
            rng=np.random.default_rng(12701+100*fold+c)
            chosen=np.sort(rng.choice(available,size=min(256,len(available)),replace=False))
            ids.extend(chosen.tolist())
            samples.extend({'fold':fold,'class_panel':c,'local':int(i)} for i in chosen)
        ids=np.array(sorted(set(ids)),dtype=np.int64)
        for arm in 'BC':
            for epoch in (0,1,25):
                torch.manual_seed(11182);torch.cuda.manual_seed_all(11182)
                model=BodyBranch().cuda().eval()
                if epoch:
                    state=torch.load(PARENT/f'fold{fold}_{arm}'/f'epoch{epoch}_model.pt',map_location='cpu',weights_only=True)
                    model.load_state_dict(state['branch'])
                output=attention_probe(model,body[arm],length,ids)
                attention.append({'fold':fold,'arm':arm,'epoch':epoch,'unique_fit_inputs':len(ids),**output})
                del model
        print(json.dumps({'fold_done':fold,'S_gradient_mass_fraction':residual[-1]['S_fraction_of_logit_gradient_l1_mass']}),flush=True)
    pd.DataFrame(samples).to_parquet(OUT/'attention_panel.parquet',index=False)
    report={'status':'zero_fit_mechanism_probe','latest_actual_training':'V125',
            'classifier_fits':0,'optimizer_steps':0,'calibration_fits':0,'model_promoted':False,
            'attention':attention,'zero_residual_gradient':residual,
            'source_sha256':sha(Path(__file__)),
            'parent_review_sha256':sha(ROOT/'artifacts/v126_frozen_review_20260929/verification.json'),
            'panel_sha256':sha(OUT/'attention_panel.parquet'),
            'scope':['Attention probes use a fixed diagnostic panel of up to256 unique train inputs per class/fold, not the entire population.',
                     'Normalized attention entropy is not output variance or a classification score.',
                     'Frozen-base logit gradients cover all original fit rows; parameter gradients may differ.',
                     'No optimizer steps or fitting; no claim of causal optimizer attribution.']}
    save(OUT/'mechanism_probe.json',report)
    print(json.dumps({'status':report['status'],'attention_runs':len(attention),'residual':residual}),flush=True)


if __name__=='__main__':main()
