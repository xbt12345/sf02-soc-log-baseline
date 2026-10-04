"""Three inner source folds: frozen upstream text, matched CE/ranking heads."""
import argparse
import gc
import hashlib
import math
import time
import warnings
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from v61_common import FIELDS, REVISION, fit_vocab, encode_facts, sha, save, read
from train_v61_neural import seed_all
from train_v63_factorial import metric_summary
from v64_selection import select

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'artifacts/v61_source_factorial_20260914_r2'
ASSET=ROOT/'artifacts/v61_asset_cache/securebert2'
SEED=20260920
EPOCHS=12
BATCH=256


def load_fit():
    conf=read(DATA/'configuration.json')
    for name in ['records.parquet','context.npz']:
        assert sha(DATA/name)==conf['data_bindings'][name]
    frame=pd.read_parquet(DATA/'records.parquet',filters=[('role','==','fit')]).sort_values('row_position').reset_index(drop=True)
    assert frame.role.eq('fit').all() and len(frame)==59640
    # Only position metadata from other roles is decoded for indexing this array.
    positions=pd.read_parquet(DATA/'records.parquet',columns=['row_position']).row_position
    ix=pd.Series(np.arange(len(positions)),index=positions).loc[frame.row_position].to_numpy()
    reverse=np.full(len(positions),-1,dtype=int);reverse[ix]=np.arange(len(frame))
    with np.load(DATA/'context.npz') as z:
        stats=z['stats'][ix];old=z['neighbors'][ix];relations=z['relation'][ix]
    valid=old>=0;neighbors=np.where(valid,reverse[old.clip(min=0)],-1)
    assert ((neighbors>=0)|~valid).all()
    assert ((frame.group.to_numpy()[neighbors.clip(min=0)]==frame.group.to_numpy()[:,None])|~valid).all()
    folds=np.full(len(frame),-1,dtype=int)
    splitter=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=SEED)
    for k,(_,va) in enumerate(splitter.split(np.zeros(len(frame)),frame.label,frame.group)):folds[va]=k
    assert frame.assign(fold=folds).groupby('group').fold.nunique().max()==1
    assert frame.assign(fold=folds).groupby('body_group').fold.nunique().max()==1
    assert ((folds[neighbors.clip(min=0)]==folds[:,None])|~valid).all()
    frame['fold']=folds
    frame['behavior']=frame[FIELDS[:5]].astype(str).agg('|'.join,axis=1)
    return frame,{'stats':stats,'neighbors':neighbors,'relations':relations}


def text_cache(frame,out,device):
    receipt=read(ASSET/'download_receipt.json');assert receipt['revision']==REVISION
    for name,digest in receipt['sha256'].items():assert sha(ASSET/name)==digest
    codes,texts=pd.factorize(frame.text,sort=True)
    tokenizer=AutoTokenizer.from_pretrained(ASSET,local_files_only=True)
    raw=tokenizer(texts.tolist(),truncation=False,padding=False)
    assert max(map(len,raw['input_ids']))<=512
    assert sum(ids.count(tokenizer.unk_token_id) for ids in raw['input_ids'])==0
    assert all(tokenizer.decode(ids,skip_special_tokens=True)==txt for ids,txt in zip(raw['input_ids'],texts))
    padded=tokenizer.pad(raw,padding=True,pad_to_multiple_of=8,return_tensors='pt')
    config=AutoConfig.from_pretrained(ASSET,local_files_only=True)
    config.reference_compile=False;config._attn_implementation='sdpa'
    encoder,info=AutoModel.from_pretrained(ASSET,config=config,local_files_only=True,
                                         attn_implementation='sdpa',output_loading_info=True)
    assert not info['missing_keys'] and not info['mismatched_keys']
    encoder.eval().requires_grad_(False).to(device);parts=[]
    with torch.no_grad():
        for start in range(0,len(texts),128):
            x={k:v[start:start+128].to(device) for k,v in padded.items() if k in ['input_ids','attention_mask']}
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                hidden=encoder(**x).last_hidden_state
            mask=x['attention_mask'].unsqueeze(-1)
            parts.append(((hidden.float()*mask).sum(1)/mask.sum(1)).cpu().numpy())
    features=np.concatenate(parts).astype(np.float32)
    assert np.isfinite(features).all()
    np.save(out/'upstream_text_features.npy',features)
    pd.DataFrame({'text':texts}).to_parquet(out/'unique_text.parquet',index=False)
    save(out/'text_cache_receipt.json',{'rows':len(frame),'unique_texts':len(texts),
        'dimension':features.shape[1],'max_tokens':max(map(len,raw['input_ids'])),
        'unknown_tokens':0,'exact_token_roundtrip':True,'labels_used':False,
        'encoder_frozen':True,'revision':REVISION,'asset_receipt_sha256':sha(ASSET/'download_receipt.json'),
        'features_sha256':sha(out/'upstream_text_features.npy'),'texts_sha256':sha(out/'unique_text.parquet')})
    del encoder;gc.collect();torch.cuda.empty_cache()
    print('UPSTREAM_CACHE_COMPLETE',len(texts),flush=True)
    return features,codes


class Inputs:
    def __init__(self,frame,context,features,codes,train,device):
        facts=frame[FIELDS].astype(str).to_numpy()
        self.vocab=fit_vocab(facts[train]);encoded=encode_facts(facts,self.vocab)
        mean=context['stats'][train].mean(0);std=context['stats'][train].std(0);std[std==0]=1
        self.scaling={'mean':mean.tolist(),'std':std.tolist()}
        scaled=((context['stats']-mean)/std).astype(np.float32)
        self.device=device;self.text=torch.tensor(features,device=device)
        self.codes=torch.tensor(codes,device=device)
        self.facts=torch.tensor(encoded,device=device)
        self.stats=torch.tensor(scaled,device=device)
        self.neighbors=torch.tensor(context['neighbors'],device=device)
        self.relations=torch.tensor(context['relations'],device=device)
        self.y=torch.tensor(frame.label.to_numpy(),device=device)
        # Actual deterministic input equality, including neighbor multiplicity/order invariance.
        text_hash=[hashlib.sha256(row.tobytes()).digest() for row in features]
        fingerprints=[]
        for i in range(len(frame)):
            h=hashlib.sha256(text_hash[codes[i]]+encoded[i].tobytes()+scaled[i].tobytes())
            entries=[encoded[j].tobytes()+context['relations'][i,k].tobytes()
                     for k,j in enumerate(context['neighbors'][i]) if j>=0]
            for entry in sorted(entries):h.update(entry)
            fingerprints.append(h.hexdigest())
        self.views=np.asarray(fingerprints)

    def batch(self,indices):
        i=torch.as_tensor(indices,dtype=torch.long,device=self.device);n=self.neighbors[i]
        return {'text':self.text[self.codes[i]],'facts':self.facts[i],'stats':self.stats[i],
                'neighbor_facts':self.facts[n.clamp_min(0)],'mask':n>=0,'relations':self.relations[i]}


class Head(nn.Module):
    def __init__(self,vocab,text_dimension):
        super().__init__()
        self.embeddings=nn.ModuleList([nn.Embedding(len(v),16,padding_idx=1) for v in vocab])
        self.fact_mlp=nn.Sequential(nn.Linear(16*len(vocab),128),nn.GELU(),nn.LayerNorm(128))
        self.set_mlp=nn.Sequential(nn.Linear(130,128),nn.GELU(),nn.Linear(128,128),nn.GELU())
        self.head=nn.Sequential(nn.Linear(text_dimension+400,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,3))

    def fact_encode(self,x):
        return self.fact_mlp(torch.cat([e(x[...,j]) for j,e in enumerate(self.embeddings)],dim=-1))

    def forward(self,text,facts,stats,neighbor_facts,mask,relations):
        values=self.set_mlp(torch.cat([self.fact_encode(neighbor_facts),relations],dim=-1))
        m=mask.unsqueeze(-1);mean=(values*m).sum(1)/m.sum(1).clamp_min(1)
        maximum=values.masked_fill(~m,-torch.inf).max(1).values
        maximum=torch.where(m.any(1),maximum,torch.zeros_like(maximum))
        return self.head(torch.cat([text,self.fact_encode(facts),stats,mean,maximum],dim=-1))


def rank_loss(logits_m,logits_s):
    score_m=logits_m[:,2]-logits_m[:,1];score_s=logits_s[:,2]-logits_s[:,1]
    return nn.functional.softplus(score_m-score_s).mean()


def pair_schedule(pool,epoch,steps,fold):
    if not pool.role.eq('fit').all():raise ValueError('Pair pool may contain training rows only')
    rng=np.random.default_rng(SEED+1000*fold+epoch);pairs=[];details={}
    mixed=pool.groupby('view').label.nunique().gt(1)
    pure=pool[pool.label.isin([1,2]) & ~pool.view.map(mixed)].drop_duplicates(['behavior','group','label','view'])
    for behavior,part in pure.groupby('behavior',sort=True):
        choices={c:{int(g):rng.permutation(d['index'].to_numpy()).tolist()
                    for g,d in part[part.label.eq(c)].groupby('group')} for c in [1,2]}
        if min(len(choices[c]) for c in [1,2])<3:continue
        ss=rng.permutation(list(choices[2])).tolist();sset=set(ss)
        mm=rng.permutation(list(choices[1])).tolist();mm.sort(key=lambda g:g in sset)
        used=Counter();view_use=Counter();cursor=0;count=0
        for round_number in range(8):
            for source_s in ss:
                if used[source_s]>=8:continue
                source_m=None
                for _ in range(len(mm)):
                    candidate=mm[cursor%len(mm)];cursor+=1
                    if candidate!=source_s and used[candidate]<8:
                        source_m=candidate;break
                if source_m is None:continue
                indexes=[]
                for c,g in [(1,source_m),(2,source_s)]:
                    options=choices[c][g];indexes.append(int(options[view_use[(c,g)]%len(options)]))
                    view_use[(c,g)]+=1;used[g]+=1
                pairs.append(indexes);count+=1
        assert max(used.values(),default=0)<=8
        details[behavior]={'pairs':count,'source_participation_max':max(used.values(),default=0),
                           'M_sources_available':len(choices[1]),'S_sources_available':len(choices[2])}
    rng.shuffle(pairs)
    batches=[pairs[j:j+16] for j in range(0,len(pairs),16)]
    if len(batches)>steps:raise ValueError('Registered pair budget exceeds main steps')
    slots=np.linspace(0,steps-1,len(batches),dtype=int) if batches else []
    return {int(k):b for k,b in zip(slots,batches)},details


def measure(frame,prob):
    # Float64 re-normalization avoids a float32-roundoff warning in log_loss.
    prob=prob.astype(np.float64);prob/=prob.sum(1,keepdims=True)
    m=metric_summary(frame,prob);m['normal_support']=int(frame.label.eq(0).sum())
    m['conditional_ranking']={}
    for protocol in ['tcp','udp']:
        sub=frame[frame.transport_protocol.eq(protocol)&frame.src_role.eq('outside')&frame.dst_role.eq('dmz')]
        if sub.label.nunique()!=2:continue
        score=prob[sub.index,2]/np.maximum(prob[sub.index,1]+prob[sub.index,2],1e-30)
        y=sub.label.eq(2).to_numpy();w=np.empty(len(sub))
        for label in [1,2]:
            mask=sub.label.eq(label).to_numpy();d=sub[sub.label.eq(label)];counts=d.groupby('group').size()
            w[mask]=.5/len(counts)/d.group.map(counts).to_numpy()
        m['conditional_ranking'][protocol]={'rows':len(sub),'S_rows':int(y.sum()),
            'source_balanced_AP':float(average_precision_score(y,score,sample_weight=w)),
            'source_balanced_ROC_AUC':float(roc_auc_score(y,score,sample_weight=w))}
    return m


@torch.no_grad()
def predict(model,inputs,indices):
    model.eval();parts=[]
    for start in range(0,len(indices),256):
        parts.append(model(**inputs.batch(indices[start:start+256])).softmax(-1).cpu().numpy())
    return np.concatenate(parts)


def digest(model):
    h=hashlib.sha256()
    for name,t in model.state_dict().items():h.update(name.encode());h.update(t.cpu().numpy().tobytes())
    return h.hexdigest()


def train_arm(frame,inputs,train,valid,schedules,fold,arm,out):
    seed_all(SEED+fold);model=Head(inputs.vocab,inputs.text.shape[1]).to(inputs.device)
    initial=digest(model);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    steps=math.ceil(len(train)/BATCH)
    lr=get_linear_schedule_with_warmup(optimizer,int(steps*EPOCHS*.1),steps*EPOCHS)
    checkpoints=set([max(1,int(steps*x)) for x in [.25,.5,.75]])|{steps*x for x in range(1,EPOCHS+1)}
    curve=[];step=0;t=time.monotonic();rank_grad=None;loss_log=[]
    for epoch in range(1,EPOCHS+1):
        order=np.random.default_rng(SEED+fold*100+epoch).permutation(train)
        ce_sum=0.;aux_sum=0.;aux_batches=0
        for slot,start in enumerate(range(0,len(order),BATCH)):
            ids=order[start:start+BATCH];model.train();optimizer.zero_grad(set_to_none=True)
            ce=nn.functional.cross_entropy(model(**inputs.batch(ids)),inputs.y[ids]);ce.backward()
            ce_sum+=float(ce.detach())
            if arm=='H1' and slot in schedules[epoch-1]:
                pairs=np.array(schedules[epoch-1][slot]);n=len(pairs)
                with torch.random.fork_rng(devices=[inputs.device.index or 0]):
                    torch.manual_seed(SEED+fold*100000+epoch*1000+slot)
                    output=model(**inputs.batch(np.r_[pairs[:,0],pairs[:,1]]))
                    loss=rank_loss(output[:n],output[n:]);weighted=.25*loss
                    if rank_grad is None:
                        gradients=torch.autograd.grad(weighted,list(model.parameters()),retain_graph=True,allow_unused=True)
                        rank_grad=float(torch.stack([g.norm()**2 for g in gradients if g is not None]).sum().sqrt())
                        assert np.isfinite(rank_grad) and rank_grad>0
                    weighted.backward();aux_sum+=float(loss.detach());aux_batches+=1
            norm=float(nn.utils.clip_grad_norm_(model.parameters(),1.));assert np.isfinite(norm)
            optimizer.step();lr.step();step+=1
            if step in checkpoints:
                p=predict(model,inputs,valid)
                metrics=measure(frame.iloc[valid].reset_index(drop=True),p)
                model_file=f'step_{step:04d}.pt';prob_file=f'step_{step:04d}.parquet'
                torch.save(model.state_dict(),out/model_file)
                d=frame.iloc[valid][['row_position','label','group']].copy();d[['p_B','p_M','p_S']]=p
                d.to_parquet(out/prob_file,index=False)
                curve.append({'step':step,'epoch_fraction':step/steps,'metrics':metrics,
                              'model_file':model_file,'predictions_file':prob_file})
                save(out/'curve.json',curve)
        loss_log.append({'epoch':epoch,'main_CE':ce_sum/steps,
                         'rank_loss':aux_sum/aux_batches if aux_batches else None,'rank_batches':aux_batches})
        print(f'FOLD {fold} {arm} EPOCH {epoch} row_macro={curve[-1]["metrics"]["ASA"]["macro_f1_M_S"]:.6f} S_source={curve[-1]["metrics"]["S_source_recall"]:.6f}',flush=True)
    save(out/'training.json',{'initial_sha256':initial,'final_sha256':digest(model),
        'parameters':sum(p.numel() for p in model.parameters()),'rank_gradient_norm_first_batch':rank_grad,
        'loss_log':loss_log,'seconds':time.monotonic()-t,'encoder_trained':False})
    assert digest(model)!=initial
    del model,optimizer;gc.collect();torch.cuda.empty_cache()
    return curve


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    if a.out.exists():raise FileExistsError('Use a new directory; never overwrite a run')
    torch.set_num_threads(4);device=torch.device('cuda:0');assert torch.cuda.is_available()
    frame,context=load_fit();a.out.mkdir(parents=True)
    files=['train_v65_rank_heads.py','v64_selection.py','v61_common.py','train_v63_factorial.py','train_v61_neural.py']
    registration={'version':'v65-inner-pair-ranking-1','seed':SEED,'folds':3,'epochs':EPOCHS,
       'batch':BATCH,'pair_batch':16,'lr':.001,'weight_decay':.01,'dropout':.1,'warmup_fraction':.1,
       'rank_coefficient':.25,'clip':1.,'checkpoint_schedule':'Quarter points in epoch 1, then every full epoch; 15 per arm',
       'pair_policy':'Only training data; exclude exact full-input mixed-label views from extra ranking, retain every original CE row. At least 3 sources/class/bucket. Different sources per M/S pair. Total source participation <=8 per behavior per epoch, shared across M/S roles.',
       'context_policy':'Original source-local unlabelled fit context reused only after checking every neighbor shares source and new fold; recomputation under whole-source split is equivalent. Scaling is train-fold-only.',
       'selection':'Each H0 row-peak reference locked before its H1 is trained; also protect fit-only coarse reference via v64_selection. No feasible checkpoint => none.',
       'continuation':'All folds feasible and no hard-S decline versus stronger references; pooled subject S gain >=5pp and some hard-S improvement; only then external development confirmation, never automatic promotion.',
       'missing_normal_class':'Report unsupported; no seed search or invented operational FPR.',
       'old_selection_or_outer_labels_used':False,'upstream_revision':REVISION,
       'plan_sha256':sha(ROOT/'docs/V64_DIRECTION_AND_SELECTION_REPAIR.md'),
       'sources':{f:sha(ROOT/'training'/f) for f in files},
       'inputs':{f:sha(DATA/f) for f in ['records.parquet','context.npz']},
       'torch':torch.__version__,'quality_acceptance':False}
    save(a.out/'preregistered.json',registration)
    frame[['row_position','label','group','body_group','fold']].to_parquet(a.out/'fold_manifest.parquet',index=False)
    features,codes=text_cache(frame,a.out,device);results={}
    for fold in range(3):
        foldout=a.out/f'fold{fold}';foldout.mkdir()
        train=np.flatnonzero(frame.fold.ne(fold));valid=np.flatnonzero(frame.fold.eq(fold));train_set=set(train)
        inputs=Inputs(frame,context,features,codes,train,device)
        save(foldout/'preprocessing.json',{'vocab':inputs.vocab,'scaling':inputs.scaling,
            'train_rows':len(train),'valid_rows':len(valid),
            'train_counts':frame.iloc[train].label.value_counts().to_dict(),
            'valid_counts':frame.iloc[valid].label.value_counts().to_dict()})
        pool=frame.iloc[train][['role','behavior','group','label']].copy()
        pool['index']=train;pool['view']=inputs.views[train]
        schedules=[];exposure=[];details=[]
        for epoch in range(1,EPOCHS+1):
            schedule,detail=pair_schedule(pool,epoch,math.ceil(len(train)/BATCH),fold)
            schedules.append(schedule);details.append(detail)
            for slot,batch in schedule.items():
                for m,s in batch:
                    assert frame.label.iloc[m]==1 and frame.label.iloc[s]==2
                    assert frame.group.iloc[m]!=frame.group.iloc[s] and inputs.views[m]!=inputs.views[s]
                    assert frame.behavior.iloc[m]==frame.behavior.iloc[s]
                    assert m in train_set and s in train_set
                    exposure.append({'epoch':epoch,'slot':slot,'M_row_position':int(frame.row_position.iloc[m]),'S_row_position':int(frame.row_position.iloc[s]),
                                     'M_index':m,'S_index':s,'behavior':frame.behavior.iloc[m],
                                     'M_group':int(frame.group.iloc[m]),'S_group':int(frame.group.iloc[s])})
        pd.DataFrame(exposure).to_parquet(foldout/'pair_exposures.parquet',index=False)
        save(foldout/'pair_schedule.json',{'schedule':schedules,'support':details})
        curves={}
        for arm in ['H0','H1']:
            armout=foldout/arm;armout.mkdir()
            curves[arm]=train_arm(frame,inputs,train,valid,schedules,fold,arm,armout)
            if arm=='H0':
                peak=max(curves['H0'],key=lambda x:(x['metrics']['ASA']['macro_f1_M_S'],x['metrics']['S_source_recall'],-x['step']))
                majority=frame.iloc[train].groupby(['behavior','label']).size().unstack(fill_value=0).idxmax(axis=1)
                pred=frame.iloc[valid].behavior.map(majority).fillna(int(frame.iloc[train].label.mode().iloc[0])).to_numpy(dtype=int)
                baseline=frame.iloc[valid][['row_position','label','group']].copy();baseline['pred']=pred
                baseline.to_parquet(foldout/'coarse_reference.parquet',index=False)
                refs={'H0_row_peak':peak['metrics'],'fit_coarse':measure(frame.iloc[valid].reset_index(drop=True),np.eye(3)[pred])}
                save(foldout/'references_before_H1.json',{'references':refs,'H0_row_peak':peak,'H1_started':False})
        assert read(foldout/'H0/training.json')['initial_sha256']==read(foldout/'H1/training.json')['initial_sha256']
        chosen=select(curves['H1'],refs)
        save(foldout/'selection.json',chosen);results[str(fold)]=chosen
        print('FOLD_DONE',fold,chosen['status'],flush=True)
        del inputs;gc.collect();torch.cuda.empty_cache()
    for name,digest_value in registration['sources'].items():assert sha(ROOT/'training'/name)==digest_value
    save(a.out/'training_complete.json',{'status':'six_inner_head_fits_completed_requires_analysis',
        'quality_acceptance':False,'fold_selections':results,'outer_or_previous_selection_evaluated':False})
    print('SIX_HEAD_FITS_COMPLETE',flush=True)


if __name__=='__main__':main()
