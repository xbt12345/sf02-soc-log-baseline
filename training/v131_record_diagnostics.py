"""Read-only frozen-checkpoint diagnostics; never fits or selects a model."""
import gc
import json
import os
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import pandas as pd
import torch

from v131_common import ROOT, OUT, REVIEW, read, save, sha, load_data, fit_context, require_run_seal
from v131_model import make_model, infer, batch, objective


def extra_norms(model):
    values = {}
    for key, p in model.named_parameters():
        g = p.grad
        if g is None:
            values[key] = None
            continue
        if key == 'first.weight': extra = g[128:]
        elif key in ['first.s', 'first.bias']: extra = g[:, 128:]
        elif key == 'second.weight':
            values['second.extra_output'] = float(g[128:].norm())
            values['second.extra_to_common'] = float(g[:128, 128:].norm())
            extra = g[:, 128:]
        elif key in ['second.r', 'second.s', 'second.bias']: extra = g[:, 128:]
        elif key == 'head.weight': extra = g[:, 128:]
        else: continue
        values[key] = float(extra.norm())
    return values


def main():
    require_run_seal(ROOT/'training/v131_train.py')
    delivery = read(OUT/'delivery.json')
    target = OUT/'diagnostics'
    if target.exists(): raise FileExistsError('Preserve earlier diagnostics')
    target.mkdir()
    save(target/'binding.json', {'status':'frozen_diagnostics_bound_before_execution',
        'source_sha256':sha(__file__), 'delivery_sha256':sha(OUT/'delivery.json'),
        'run_seal_sha256':sha(OUT/'run_seal.json'), 'new_optimizer_updates':0,
        'authority':'Explain recorded outcomes only; no selection or threshold changes.'})
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False
    x, d = load_data(); panel = pd.read_parquet(REVIEW/'fold1_train_only_panel.parquet')
    hard = set(panel.loc[panel.truth==2, 'row_position'])
    endpoint_ledger=[]; trajectory=[]
    folders=[OUT/f'fold1_{a}' for a in ['R','O','C','CO']]+[OUT/f'fold{f}_R' for f in [0,2]]
    for folder in folders:
        if not (folder/'fit.json').exists(): continue
        receipt=read(folder/'fit.json'); fold=receipt['fold']; arm=receipt['arm'];stem=folder.name
        fit,c,pure,totals,ids=fit_context(d,fold)
        old=np.load(ROOT/f'artifacts/v124_header_trial_20260929/fold{fold}_B/epoch25_prob.npy')[fit.local].argmax(1)
        for checkpoint in read(folder/'checkpoints.json'):
            ep=checkpoint['epoch']; model_path=folder/f'epoch{ep}_model.pt'
            if sha(model_path)!=checkpoint['model_sha256']: raise ValueError('Checkpoint changed')
            state=torch.load(model_path,map_location='cpu',weights_only=True)
            model=make_model(receipt['hidden'],'cuda'); model.load_state_dict(state['model'])
            q,no_member,margin,_,zeros=infer(model,x,ids)
            member_pred=[]
            with torch.no_grad():
                for start in range(0,len(ids),256):
                    member_pred.append(batch(model,x,ids[start:start+256]).argmax(-1).cpu().numpy().astype(np.int8))
            np.save(target/f'{stem}_epoch{ep}_member_pred.npy',np.concatenate(member_pred))
            np.save(target/f'{stem}_train_input_ids.npy',ids)
            lookup=np.full(x.shape[0],-1,np.int64); lookup[ids]=np.arange(len(ids))
            take=lookup[fit.local.to_numpy()]; y=fit.truth.to_numpy(); pred=q[take].argmax(1)
            ledger=fit.copy();ledger['arm']=arm;ledger['epoch']=ep;ledger['outer_fit_role']=fold
            ledger['pred']=pred;ledger['old_B_pred']=old;ledger['pure_TRAIN_input']=pure[fit.local].astype(bool)
            ledger['truth_probability']=q[take,y];ledger['mean_logit_margin']=margin[take,y]
            ledger['no_correct_member']=no_member[take,y]
            ledger['error_status']=np.select([(old!=y)&(pred==y),(old!=y)&(pred!=y),(old==y)&(pred!=y)],
                ['repaired','persistent','newly_wrong'],default='retained_correct')
            ledger.to_parquet(target/f'{stem}_epoch{ep}_training_ledger.parquet',index=False)
            for cl in [1,2]:
                rows=ledger[ledger.truth==cl]
                trajectory.append({'arm':arm,'fold':fold,'epoch':ep,'class':cl,'support':len(rows),
                    'errors':int(rows.pred.ne(cl).sum()),'no_correct_member_rows':int(rows.no_correct_member.sum()),
                    'truth_probability_quantiles':rows.truth_probability.quantile([0,.1,.5,.9,1]).tolist(),
                    'margin_quantiles':rows.mean_logit_margin.quantile([0,.1,.5,.9,1]).tolist(),
                    'negative_mean_logit_margin_rows':int(rows.mean_logit_margin.lt(0).sum()),
                    'error_status':{str(k):int(v) for k,v in rows.error_status.value_counts().items()},
                    'activation_zero_fraction':zeros})
            if ep==100:
                endpoint_ledger.append(ledger)
                roots=ledger.assign(wrong=ledger.pred!=ledger.truth).groupby(['root','truth']).agg(
                    support=('row_position','size'),wrong=('wrong','sum'),no_correct_member=('no_correct_member','sum'))
                roots.to_csv(target/f'{stem}_endpoint_training_sources.csv')
            del model;gc.collect();torch.cuda.empty_cache()
    save(target/'checkpoint_learning_diagnostics.json',trajectory)
    if endpoint_ledger:
        combined=pd.concat(endpoint_ledger,ignore_index=True)
        combined.to_parquet(target/'endpoint_training_error_ledger.parquet',index=False)
        remaining=combined[combined.outer_fit_role.eq(1)&combined.row_position.isin(hard)&combined.pred.ne(2)]
        remaining.to_parquet(target/'remaining_hard_S_errors.parquet',index=False)
    probe=pd.read_parquet(REVIEW/'tiny64_train_only_inputs.parquet'); ids=probe.local.to_numpy(np.int64)
    _,counts,_,_,_=fit_context(d,1); counts[:]=0;counts[probe.local,probe.truth]=probe.original_row_mass
    extra=[]
    for step in [0,1,10,100]:
        folder=OUT/'probe_C256';cp=next(v for v in read(folder/'checkpoints.json') if v['epoch']==step)
        path=folder/f'epoch{step}_model.pt'
        if sha(path)!=cp['model_sha256']: raise ValueError('Probe checkpoint changed')
        state=torch.load(path,map_location='cpu',weights_only=True)
        model=make_model(256,'cuda');model.load_state_dict(state['model']);model.zero_grad(set_to_none=True)
        z=batch(model,x,ids)
        loss,_,_=objective(z,torch.as_tensor(counts[ids],device='cuda',dtype=torch.float32),
            torch.ones(len(ids),device='cuda'),float(counts.sum()),counts.sum(0),1,False)
        loss.backward()
        extra.append({'checkpoint_update':step,'gradient_scope':'fixed probe objective at frozen checkpoint; no optimizer step',
            'extra_channel_gradient_norms':extra_norms(model)})
        del model;gc.collect();torch.cuda.empty_cache()
    save(target/'wide_extra_channel_gradients.json',extra)
    artifacts={p.name:sha(p) for p in target.iterdir() if p.is_file()}
    save(target/'receipt.json',{'status':'frozen_checkpoint_diagnostics_complete','source_sha256':sha(__file__),
        'new_fits':0,'new_optimizer_updates':0,'output_sha256':artifacts,
        'scope':'TRAIN-only checkpoint member judgments, original-row margins/repairs/regressions and expanded-channel gradients; not new blind validation.'})
    print(json.dumps({'diagnostic_records':len(trajectory),'new_fits':0,'new_updates':0}),flush=True)


if __name__=='__main__': main()
