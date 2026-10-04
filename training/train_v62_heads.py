"""Fixed-backbone mean/max versus set self-attention and query cross-attention."""
import argparse
import math
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from v61_common import read,save,sha
from v61_runtime import score,predictions
from train_v61_neural import seed_all

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/v62_capacity_20260915'


def group_metrics(df,p):
    frame=df[['label','group']].copy();frame['correct']=p.argmax(1)==df.label.to_numpy()
    source=frame.groupby(['label','group']).correct.mean().groupby('label').mean()
    return {'class_mean_source_recall':{str(k):float(v) for k,v in source.items()},
            'M_S_class_balanced_source_recall':float(source.loc[[1,2]].mean())}


class Head(nn.Module):
    def __init__(self,attention=False):
        super().__init__();self.attention=attention
        self.project=nn.Sequential(nn.Linear(130,128),nn.GELU(),nn.Linear(128,128),nn.GELU())
        self.head=nn.Sequential(nn.Linear(1168,256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,128),nn.GELU(),nn.Linear(128,3))
        if attention:
            self.blocks=nn.ModuleList([nn.TransformerEncoderLayer(128,4,256,.1,activation='gelu',batch_first=True,norm_first=True) for _ in range(2)])
            self.query=nn.Linear(896,128)
            self.cross=nn.MultiheadAttention(128,4,dropout=.1,batch_first=True)
            self.norm=nn.LayerNorm(128)

    def forward(self,text,facts,stats,nfacts,mask,relation):
        z=self.project(torch.cat([nfacts,relation],-1));valid=mask.unsqueeze(-1)
        if self.attention:
            # A zero sentinel is visible only for an empty neighbor set, avoiding
            # undefined all-masked attention. It is removed from final pooling.
            empty=~mask.any(1)
            z=torch.cat([z,torch.zeros((len(z),1,128),device=z.device,dtype=z.dtype)],1)
            allowed=torch.cat([mask,empty[:,None]],1)
            for block in self.blocks:z=block(z,src_key_padding_mask=~allowed)
            q=self.query(torch.cat([text,facts],-1)).unsqueeze(1)
            cross,_=self.cross(q,z,z,key_padding_mask=~allowed,need_weights=False)
            mean=self.norm(cross[:,0]);mean=torch.where(empty[:,None],torch.zeros_like(mean),mean)
            z=z[:,:-1]
        else:
            mean=(z*valid).sum(1)/valid.sum(1).clamp_min(1)
        maximum=z.masked_fill(~valid,-torch.inf).max(1).values
        maximum=torch.where(valid.any(1),maximum,torch.zeros_like(maximum))
        return self.head(torch.cat([text,facts,stats,mean,maximum],-1))


class Cache:
    def __init__(self):
        r=read(OUT/'cache_receipt.json')
        for name,digest in r['bindings'].items():assert sha(OUT/name)==digest
        self.df=pd.read_parquet(OUT/'rows.parquet');assert self.df.role.ne('evaluation').all()
        self.tensors={k:torch.tensor(v,device='cuda') for k,v in dict(np.load(OUT/'cache.npz')).items()}
        self.y=torch.tensor(self.df.label.to_numpy(),device='cuda',dtype=torch.long)
        self.fit=np.flatnonzero(self.df.role.eq('fit'));self.dev=np.flatnonzero(self.df.role.eq('selection'))
        n=self.tensors['neighbors'].cpu().numpy();valid=n>=0
        roles=self.df.role.to_numpy()
        assert ((roles[np.maximum(n,0)]==roles[:,None])|~valid).all()

    def batch(self,idx):
        t=self.tensors;n=t['neighbors'][idx].long()
        return {'text':t['text'][idx],'facts':t['facts'][idx],'stats':t['stats'][idx],
                'nfacts':t['facts'][n.clamp_min(0)],'mask':n>=0,'relation':t['relation'][idx]}


@torch.no_grad()
def evaluate(model,data,idx):
    model.eval();result=[]
    for start in range(0,len(idx),512):
        p=model(**data.batch(idx[start:start+512])).softmax(-1).cpu().numpy();result.append(p)
    return np.concatenate(result)


def model_checks(data):
    result={};seed_all(20260916)
    for arm in ['meanmax','attention']:
        has_neighbors=data.tensors['neighbors'].ge(0).any(1).cpu().numpy()
        idx=data.fit[has_neighbors[data.fit]][:32];assert len(idx)==32
        model=Head(arm=='attention').cuda().eval();b=data.batch(idx)
        with torch.no_grad():reference=model(**b)
        b={k:v.clone() for k,v in b.items()}
        order=torch.arange(b['mask'].shape[1]-1,-1,-1,device='cuda')
        for k in ['nfacts','mask','relation']:b[k]=b[k][:,order]
        b['nfacts'][~b['mask']]=123.45;b['relation'][~b['mask']]=-9
        with torch.no_grad():changed=model(**b)
        torch.testing.assert_close(changed,reference,rtol=1e-4,atol=1e-5)
        b['mask'][:]=False
        with torch.no_grad():empty=model(**b)
        assert torch.isfinite(empty).all()
        model.train();loss=nn.functional.cross_entropy(model(**data.batch(idx)),data.y[idx]);loss.backward()
        grads={name:float(p.grad.norm()) for name,p in model.named_parameters() if p.grad is not None}
        assert all(np.isfinite(v) for v in grads.values())
        if arm=='attention':
            assert any('cross' in k and v>0 for k,v in grads.items())
            assert any('blocks' in k and v>0 for k,v in grads.items())
        result[arm]={'permutation_and_padding_invariant':True,'empty_set_finite':True,
                     'trainable_parameters':sum(p.numel() for p in model.parameters()),'gradient_norms':grads}
        del model
    save(OUT/'head_implementation_checks.json',result)


def train(data,arm):
    seed=20260916;seed_all(seed);target=OUT/arm;assert not target.exists();target.mkdir()
    model=Head(arm=='attention').cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    epochs=12;steps=math.ceil(len(data.fit)/256)*epochs;step=0;curve=[];best=-np.inf;best_epoch=0;t=time.monotonic()
    for epoch in range(1,epochs+1):
        model.train();order=np.random.default_rng(seed+epoch).permutation(data.fit);total=0.
        for start in range(0,len(order),256):
            idx=order[start:start+256]
            ratio=min(1.,(step+1)/max(1,int(steps*.1))) if step<steps*.1 else .1+.9*.5*(1+math.cos(math.pi*(step-steps*.1)/(steps*.9)))
            for group in opt.param_groups:group['lr']=3e-4*ratio
            opt.zero_grad(set_to_none=True);z=model(**data.batch(idx));loss=nn.functional.cross_entropy(z,data.y[idx])
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();step+=1
            total+=float(loss.detach())*len(idx)
        if epoch%2:continue
        p=evaluate(model,data,data.dev);s=score(data.df.label.iloc[data.dev],p);g=group_metrics(data.df.iloc[data.dev],p)
        curve.append({'epoch':epoch,'online_fit_CE':total/len(order),'selection':s,'source_metrics':g})
        save(target/'curve.json',curve)
        value=g['M_S_class_balanced_source_recall']
        if value>best:
            best=value;best_epoch=epoch;torch.save(model.state_dict(),target/'model.pt')
            predictions(target/'selection.parquet',data.df,data.dev,p)
        print(f'{arm} epoch {epoch}/12 source-balanced recall {value:.6f}, row macro {s["ASA"]["macro_f1_M_S"]:.6f}, {time.monotonic()-t:.1f}s',flush=True)
    finalfit=evaluate(model,data,data.fit)
    final={'fit':score(data.df.label.iloc[data.fit],finalfit),'source_fit':group_metrics(data.df.iloc[data.fit],finalfit),
           'selection':curve[-1]['selection'],'source_selection':curve[-1]['source_metrics']}
    model.load_state_dict(torch.load(target/'model.pt',weights_only=True));p=evaluate(model,data,data.fit)
    predictions(target/'fit.parquet',data.df,data.fit,p)
    chosen=next(c for c in curve if c['epoch']==best_epoch)
    save(target/'result.json',{'selected_epoch':best_epoch,'selected_fit':score(data.df.label.iloc[data.fit],p),
        'selected_source_fit':group_metrics(data.df.iloc[data.fit],p),'selected_selection':chosen['selection'],
        'selected_source_selection':chosen['source_metrics'],'fixed_epoch_12':final,
        'elapsed_seconds':time.monotonic()-t,'trainable_parameters':sum(p.numel() for p in model.parameters()),
        'model_sha256':sha(target/'model.pt'),'no_external_or_old_outer_evaluation':True})
    del model,opt;torch.cuda.empty_cache()


def main():
    torch.set_num_threads(4);assert torch.cuda.is_available()
    data=Cache()
    protocol={'scope':'Capability diagnostic on previously observed development roles; no promotion claim',
        'arms':['meanmax','attention'],'cache_receipt_sha256':sha(OUT/'cache_receipt.json'),
        'shared':'Frozen same D epoch-1 text and fact backbone; same rows, states, stats, optimizer, head width and loss',
        'changed':'Explicit neighbor self-attention and target-conditioned cross-attention; parameter count NOT matched',
        'loss':'Original-row unweighted 3-class CE','epochs':12,'seed':20260916,'batch':256,
        'head_lr':.0003,'weight_decay':.01,'precision':'float32','clip_norm':1.,
        'selection':'Class-balanced mean source recall for M/S, evaluated at fixed even epochs; ties prefer earlier',
        'conditional_followup_gate':{'source_balanced_gain':.01,'S_source_recall_gain':.02,'M_source_recall_floor_delta':-.01,
            'positive_paired_source_interval_required':True},
        'same_backbone_supervised_fit_role_unchanged':True,
        'sources':{n:sha(ROOT/'training'/n) for n in ['train_v62_heads.py','prepare_v62_capacity.py','train_v61_neural.py','v61_common.py','v61_runtime.py']}}
    assert not (OUT/'head_protocol.json').exists();save(OUT/'head_protocol.json',protocol)
    model_checks(data)
    for arm in protocol['arms']:train(data,arm)


if __name__=='__main__':main()
