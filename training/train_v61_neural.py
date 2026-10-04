"""B/D genuine full encoder fine-tuning; fixed CE, fit-only vocab, inner epoch choice."""
import argparse
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from v61_common import FIELDS, MODEL_ID, REVISION, fit_vocab, encode_facts, save, read, sha
from v61_runtime import load_data, score, predictions, source_receipt


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


class Inputs:
    def __init__(self,df,ctx,model_path,device,vocab=None,scaling=None):
        self.device=device;self.df=df
        fit=df.role.eq('fit').to_numpy();facts=df[FIELDS].astype(str).to_numpy()
        self.vocab=fit_vocab(facts[fit]) if vocab is None else vocab
        encoded=encode_facts(facts,self.vocab)
        self.facts=torch.tensor(encoded,device=device)
        self.y=torch.tensor(df.label.to_numpy(),device=device,dtype=torch.long)
        text_codes,texts=__import__('pandas').factorize(df.text,sort=True)
        self.tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
        tok=self.tokenizer(texts.tolist(),truncation=False,padding=False)
        lengths=np.array(list(map(len,tok['input_ids'])))
        # The bound corpus has max 46 tokens. Any future long record stops here,
        # rather than silently deleting evidence or changing the frozen experiment.
        assert lengths.max()<=512,'Long records require a separately audited chunk path.'
        padded=self.tokenizer.pad(tok,padding=True,pad_to_multiple_of=8,return_tensors='pt')
        self.tokens={k:v.to(device) for k,v in padded.items() if k in ['input_ids','attention_mask']}
        self.text_codes=torch.tensor(text_codes,device=device)
        self.length_audit={'unique_texts':len(texts),'max_tokens':int(lengths.max()),'over_512':0,
            'unknown_tokens':sum(x.count(self.tokenizer.unk_token_id) for x in tok['input_ids']),
            'all_unique_texts_lossless_token_roundtrip':all(self.tokenizer.decode(ids,skip_special_tokens=True)==txt for ids,txt in zip(tok['input_ids'],texts))}
        if scaling is None:
            mean=ctx['stats'][fit].mean(0);std=ctx['stats'][fit].std(0);std[std==0]=1
            scaling={'mean':mean.tolist(),'std':std.tolist()}
        self.scaling=scaling
        self.stats=torch.tensor((ctx['stats']-np.array(scaling['mean'],dtype=np.float32))/np.array(scaling['std'],dtype=np.float32),device=device)
        self.neighbors=torch.tensor(ctx['neighbors'].astype(np.int64),device=device)
        self.relation=torch.tensor(ctx['relation'],device=device)

    def batch(self,indices,context=True):
        i=torch.as_tensor(indices,dtype=torch.long,device=self.device);tc=self.text_codes[i]
        result={k:v[tc] for k,v in self.tokens.items()};result['facts']=self.facts[i]
        if context:
            n=self.neighbors[i];result.update(stats=self.stats[i],neighbor_facts=self.facts[n.clamp_min(0)],
                neighbor_mask=n>=0,relation=self.relation[i])
        return result


class Classifier(nn.Module):
    def __init__(self,model_path,vocab,arm,pretrained=True):
        super().__init__();self.arm=arm
        config=AutoConfig.from_pretrained(model_path,local_files_only=True)
        config.reference_compile=False
        config._attn_implementation='sdpa'
        if pretrained:
            self.encoder,info=AutoModel.from_pretrained(model_path,config=config,local_files_only=True,
                attn_implementation='sdpa',output_loading_info=True)
            assert not info['missing_keys'],info
            assert not info['mismatched_keys'],info
            self.load_info=info
        else:
            self.encoder=AutoModel.from_config(config,attn_implementation='sdpa');self.load_info={}
        self.embeddings=nn.ModuleList([nn.Embedding(len(v),16,padding_idx=1) for v in vocab])
        self.fact_mlp=nn.Sequential(nn.Linear(16*len(vocab),128),nn.GELU(),nn.LayerNorm(128))
        if arm=='D':
            self.set_mlp=nn.Sequential(nn.Linear(130,128),nn.GELU(),nn.Linear(128,128),nn.GELU())
        dim=config.hidden_size+128+(272 if arm=='D' else 0)
        self.head=nn.Sequential(nn.Linear(dim,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,3))

    def fact_encode(self,f):
        # UNKNOWN is fixed zero, distinct from the learned MISSING embedding.
        return self.fact_mlp(torch.cat([emb(f[...,j]) for j,emb in enumerate(self.embeddings)],dim=-1))

    def forward(self,input_ids,attention_mask,facts,stats=None,neighbor_facts=None,neighbor_mask=None,relation=None):
        h=self.encoder(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state
        mask=attention_mask.unsqueeze(-1);text=(h*mask).sum(1)/mask.sum(1).clamp_min(1)
        vectors=[text,self.fact_encode(facts)]
        if self.arm=='D':
            z=self.set_mlp(torch.cat([self.fact_encode(neighbor_facts),relation.to(h.dtype)],dim=-1))
            m=neighbor_mask.unsqueeze(-1)
            mean=(z*m).sum(1)/m.sum(1).clamp_min(1)
            maximum=z.masked_fill(~m,-torch.inf).max(1).values
            maximum=torch.where(m.any(1),maximum,torch.zeros_like(maximum))
            vectors.extend([stats,mean,maximum])
        return self.head(torch.cat(vectors,dim=-1))


def optimizer_for(model):
    enc=list(model.encoder.parameters());ids={id(p) for p in enc}
    return torch.optim.AdamW([{'params':enc,'lr':2e-5},{'params':[p for p in model.parameters() if id(p) not in ids],'lr':1e-4}],weight_decay=.01)


def amp(device):
    return torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda')


@torch.no_grad()
def predict(model,inputs,indices,batch=128,context_permutation=None):
    model.eval();out=[]
    for start in range(0,len(indices),batch):
        ii=indices[start:start+batch];b=inputs.batch(ii,model.arm=='D')
        if context_permutation is not None and model.arm=='D':
            other=inputs.batch(context_permutation[start:start+batch],True)
            for k in ['stats','neighbor_facts','neighbor_mask','relation']:b[k]=other[k]
        with amp(inputs.device):p=model(**b).float().softmax(-1)
        out.append(p.cpu().numpy())
    return np.concatenate(out)


def smoke(model_path,inputs,out,micro):
    seed_all(20260915);model=Classifier(model_path,inputs.vocab,'D').to(inputs.device)
    opt=optimizer_for(model)
    fit=np.flatnonzero(inputs.df.role.eq('fit'));idx=fit[:micro]
    # No dev/evaluation labels are involved in implementation checks.
    before=next(model.encoder.parameters()).detach().clone()
    initial=before.cpu()
    times=[];grad=None
    for step in range(4):
        if inputs.device.type=='cuda':torch.cuda.synchronize()
        t=time.monotonic();model.train();opt.zero_grad(set_to_none=True)
        b=inputs.batch(idx,True)
        with amp(inputs.device):logits=model(**b);loss=nn.functional.cross_entropy(logits,inputs.y[idx])
        loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.0));opt.step()
        if inputs.device.type=='cuda':torch.cuda.synchronize()
        times.append(time.monotonic()-t)
    changed=not torch.equal(initial,next(model.encoder.parameters()).detach().cpu())
    assert changed and np.isfinite(grad) and grad>0
    model.eval()
    with torch.no_grad(),amp(inputs.device):expected=model(**inputs.batch(idx,True)).float().cpu()
    checkpoint=out/'smoke_state.pt';torch.save(model.state_dict(),checkpoint)
    del model,opt;torch.cuda.empty_cache() if inputs.device.type=='cuda' else None
    reloaded=Classifier(model_path,inputs.vocab,'D',pretrained=False).to(inputs.device)
    reloaded.load_state_dict(torch.load(checkpoint,map_location=inputs.device,weights_only=True));reloaded.eval()
    with torch.no_grad(),amp(inputs.device):actual=reloaded(**inputs.batch(idx,True)).float().cpu()
    torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-5)
    # Permuting set order and changing padding facts cannot alter the output.
    b=inputs.batch(idx,True);permutation=torch.arange(b['neighbor_mask'].shape[1]-1,-1,-1,device=inputs.device)
    for k in ['neighbor_facts','neighbor_mask','relation']:b[k]=b[k][:,permutation]
    b['neighbor_facts'][~b['neighbor_mask']]=0
    with torch.no_grad(),amp(inputs.device):permuted=reloaded(**b).float().cpu()
    torch.testing.assert_close(permuted,actual,atol=.003,rtol=.003)
    receipt={'actual_forward_backward_save_reload_passed':True,'encoder_parameters_changed':changed,
        'positive_finite_gradient_norm':grad,'set_order_padding_test_passed':True,
        'microbatch':micro,'step_seconds':times,'estimated_two_arms_three_epochs_training_seconds':float(np.median(times[1:])*len(fit)/micro*6),
        'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated() if inputs.device.type=='cuda' else None,
        'tokenization':inputs.length_audit,'torch':torch.__version__,'device':str(inputs.device),
        'smoke_is_not_model_quality':True}
    save(out/'smoke.json',receipt);print(receipt,flush=True)
    checkpoint.unlink();del reloaded;torch.cuda.empty_cache() if inputs.device.type=='cuda' else None

    # A separate fit-side capacity check uses only non-conflicting observable inputs.
    # It is explicitly counted separately and discarded before every core fit.
    eligible=inputs.df.iloc[fit].copy()
    keys=eligible[['text']+FIELDS].astype(str).agg('\x1f'.join,axis=1)
    unique_counts=eligible.assign(k=keys).groupby('k').label.nunique()
    eligible=eligible.assign(k=keys);eligible=eligible[eligible.k.isin(unique_counts[unique_counts==1].index)].drop_duplicates('k')
    tiny=np.concatenate([eligible[eligible.label.eq(c)].head(8).index.to_numpy() for c in range(3)])
    assert len(set(inputs.df.label.iloc[tiny]))==3
    seed_all(20260915);small=Classifier(model_path,inputs.vocab,'B').to(inputs.device);opt=optimizer_for(small)
    final_acc=0.;tiny_curve=[]
    for step in range(120):
        small.train();opt.zero_grad(set_to_none=True)
        with amp(inputs.device):
            z=small(**inputs.batch(tiny,False));loss=nn.functional.cross_entropy(z,inputs.y[tiny])
        loss.backward();torch.nn.utils.clip_grad_norm_(small.parameters(),1.0);opt.step()
        if (step+1)%10==0:
            prob=predict(small,inputs,tiny);final_acc=float((prob.argmax(1)==inputs.df.label.iloc[tiny]).mean())
            tiny_curve.append({'step':step+1,'loss':float(loss.detach()),'accuracy':final_acc})
            if final_acc>=.95 and float(loss.detach())<.2:break
    receipt['tiny_fit']={'rows':len(tiny),'classes':inputs.df.label.iloc[tiny].value_counts().to_dict(),
        'curve':tiny_curve,'passed':final_acc>=.95,'discarded_before_core_fits':True}
    save(out/'smoke.json',receipt)
    assert receipt['tiny_fit']['passed'],'Fit-side capacity check failed; inspect before core training'
    print('Fit-only non-conflict tiny overfit check passed.',flush=True)
    del small,opt;torch.cuda.empty_cache() if inputs.device.type=='cuda' else None


def train_arm(a,inputs,arm):
    seed_all(a.seed);target=a.out/(arm+'_'+str(a.seed))
    if target.exists():assert a.resume,'Existing run; use --resume only with unchanged sources'
    else:target.mkdir()
    fit=np.flatnonzero(inputs.df.role.eq('fit'));dev=np.flatnonzero(inputs.df.role.eq('selection'))
    params={'arm':arm,'seed':a.seed,'epochs':3,'effective_batch':64,'microbatch':a.microbatch,
        'encoder_lr':2e-5,'head_lr':1e-4,'weight_decay':.01,'warmup_fraction':.1,
        'loss':'unweighted original-row CE','epoch_selection':'maximum inner ASA M/S Macro-F1; ties prefer earlier epoch',
        'model':MODEL_ID,'revision':REVISION,'attention':'sdpa','reference_compile':False,
        'precision':'bf16 autocast with fp32 weights and optimizer' if inputs.device.type=='cuda' else 'fp32',
        'D_context':'Full C statistics plus <=32 distinct fact/equality states; no identities or labels',
        'data_configuration_sha256':sha(a.data/'configuration.json'),
        'source_sha256':source_receipt(['train_v61_neural.py','v61_runtime.py','v61_common.py']),
        'pretrained_asset_receipt':read(a.model/'download_receipt.json')}
    if (target/'preregistered.json').exists():assert read(target/'preregistered.json')==params
    else:save(target/'preregistered.json',params)
    if (target/'selection.json').exists():
        assert sha(target/'model.pt')==read(target/'selection.json')['model_sha256'];return
    save(target/'preprocessing.json',{'vocab':inputs.vocab,'scaling':inputs.scaling,'tokenization':inputs.length_audit})
    model=Classifier(a.model,inputs.vocab,arm).to(inputs.device);opt=optimizer_for(model)
    nsteps=math.ceil(len(fit)/64)*3
    scheduler=get_linear_schedule_with_warmup(opt,int(nsteps*.1),nsteps)
    curve=[];best=-float('inf');best_epoch=0;t=time.monotonic();start_epoch=1
    if a.resume and (target/'latest_training_state.pt').exists():
        state=torch.load(target/'latest_training_state.pt',map_location=inputs.device,weights_only=True)
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        curve=state['curve'];start_epoch=state['epoch']+1
        best_epoch=max(range(len(curve)),key=lambda i:curve[i]['selection']['ASA']['macro_f1_M_S'])+1
        best=curve[best_epoch-1]['selection']['ASA']['macro_f1_M_S']
        torch.set_rng_state(state['torch_rng'].cpu())
        if inputs.device.type=='cuda':torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
        del state
    for epoch in range(start_epoch,4):
        order=np.random.default_rng(a.seed+epoch).permutation(fit)
        model.train();total=0.;seen=0;epoch_t=time.monotonic()
        for start in range(0,len(order),64):
            group=order[start:start+64];opt.zero_grad(set_to_none=True)
            for m in range(0,len(group),a.microbatch):
                idx=group[m:m+a.microbatch]
                with amp(inputs.device):
                    logits=model(**inputs.batch(idx,arm=='D'))
                    loss=nn.functional.cross_entropy(logits,inputs.y[idx],reduction='sum')/len(group)
                loss.backward();total+=float(loss.detach())*len(group);seen+=len(idx)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();scheduler.step()
            if start//64%200==0:
                print(f'{arm} epoch {epoch}/3 rows {min(start+64,len(order))}/{len(order)} CE {total/seen:.5f} elapsed {time.monotonic()-epoch_t:.1f}s',flush=True)
        p=predict(model,inputs,dev);s=score(inputs.df.label.iloc[dev],p)
        item={'epoch':epoch,'online_train_CE':total/seen,'selection':s,'epoch_seconds':time.monotonic()-epoch_t}
        curve.append(item);save(target/'learning_curve.json',curve)
        value=s['ASA']['macro_f1_M_S']
        if value>best:
            best=value;best_epoch=epoch
            torch.save(model.state_dict(),target/'model.pt')
            predictions(target/'selection.parquet',inputs.df,dev,p)
        # Resumable latest state, including optimizer; only local fit order is repeated on recovery.
        torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),
                    'epoch':epoch,'seed':a.seed,'curve':curve,'torch_rng':torch.get_rng_state(),
                    'cuda_rng':torch.cuda.get_rng_state_all() if inputs.device.type=='cuda' else []},target/'latest_training_state.pt')
        print(f'{arm} epoch {epoch} saved; inner Macro-F1 {value:.6f}',flush=True)
    model.load_state_dict(torch.load(target/'model.pt',map_location=inputs.device,weights_only=True))
    selected=curve[best_epoch-1]['selection'];fp=predict(model,inputs,fit)
    save(target/'selection.json',{'arm':arm,'seed':a.seed,'status':'inner_selection_only','selected_epoch':best_epoch,
        'selection':selected,'fit':score(inputs.df.label.iloc[fit],fp),'elapsed_seconds':time.monotonic()-t,
        'model_sha256':sha(target/'model.pt'),'predictions_sha256':sha(target/'selection.parquet'),
        'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated() if inputs.device.type=='cuda' else None})
    del model,opt,scheduler;torch.cuda.empty_cache() if inputs.device.type=='cuda' else None


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--model',type=Path,required=True);ap.add_argument('--seed',type=int,default=20260915)
    ap.add_argument('--microbatch',type=int,default=32);ap.add_argument('--smoke-only',action='store_true')
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--arms',nargs='+',choices=['B','D'],default=['B','D']);a=ap.parse_args()
    assert 64%a.microbatch==0;a.out.mkdir(parents=True,exist_ok=True)
    receipt=read(a.model/'download_receipt.json');assert receipt['revision']==REVISION
    for name,digest in receipt['sha256'].items():assert sha(a.model/name)==digest,name
    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':assert torch.cuda.is_bf16_supported()
    df,ctx,_=load_data(a.data);inputs=Inputs(df,ctx,a.model,device)
    if a.smoke_only:smoke(a.model,inputs,a.out,a.microbatch);return
    assert read(a.out/'smoke.json')['actual_forward_backward_save_reload_passed']
    assert read(a.out/'smoke.json')['tiny_fit']['passed']
    for arm in a.arms:train_arm(a,inputs,arm)


if __name__=='__main__':main()
