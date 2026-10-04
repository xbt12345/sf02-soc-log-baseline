"""Frozen V135 endpoint/member and original-mass diagnostics. No optimizer exists."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from v135_runtime import ROOT, OUT, ARMS, read, save, sha, require_run_seal, load_data, fit_context
from v135_model import Classifier, batch, objective

DEST = ROOT / 'artifacts/v136_root_review_20260930'


def weighted_quantiles(values, weights):
    order = np.argsort(values); v = values[order]; w = weights[order]
    return {str(p): float(v[min(np.searchsorted(np.cumsum(w), p*w.sum()),len(v)-1)])
            for p in [0., .25, .5, .75, 1.]} if len(values) else {}


def panel(frame, pure, q, mp, lp, avgz, prefix):
    loc = frame.local.to_numpy(); y = frame.truth.to_numpy()
    pred = q[loc].argmax(1); votes = np.stack([(mp==j).sum(1) for j in range(3)],1)
    vote_pred = votes.argmax(1); logit_pred = avgz.argmax(1)
    ties = (votes == votes.max(1,keepdims=True)).sum(1)>1
    records=[]
    for cl in [1,2]:
        for status in ['all','wrong','correct']:
            m = (y==cl) & pure[loc].astype(bool)
            if status=='wrong': m &= pred!=y
            if status=='correct': m &= pred==y
            ii=loc[m]; yy=y[m]
            counts=votes[ii,yy]; conf=q[ii,yy]
            records.append({'population':prefix,'class':cl,'status':status,'original_rows':int(m.sum()),
                'unique_inputs':len(np.unique(ii)), 'source_roots':int(frame.loc[m,'root'].nunique()),
                'correct_members_histogram':{str(k):int((counts==k).sum()) for k in range(17)},
                'minority_correct_rows':int(((counts>0)&(counts<8)).sum()),
                'strict_majority_correct_rows':int((counts>8).sum()),
                'tie_rows':int(ties[ii].sum()),
                'probability_mean_errors':int((pred[m]!=yy).sum()),
                'vote_errors':int((vote_pred[ii]!=yy).sum()),
                'vote_repairs_vs_mean':int(((pred[m]!=yy)&(vote_pred[ii]==yy)).sum()),
                'vote_regressions_vs_mean':int(((pred[m]==yy)&(vote_pred[ii]!=yy)).sum()),
                'logit_mean_errors':int((logit_pred[ii]!=yy).sum()),
                'truth_probability_quantiles':weighted_quantiles(conf,np.ones(len(ii))),
                'member_CE_mean':float(-lp[ii,yy].mean()) if len(ii) else None,
                'ensemble_CE_mean':float(-np.log(conf.clip(1e-30)).mean()) if len(ii) else None})
    return records


def main():
    require_run_seal(ROOT/'training/v135_train.py')
    if DEST.exists() and any(DEST.iterdir()): raise FileExistsError('Preserve existing review')
    DEST.mkdir(exist_ok=True); x,d=load_data()
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    outputs=[]; residuals=[]; mass_reports=[]; gradients=[]
    bound={Path(__file__),OUT/'final_delivery.json',OUT/'run_seal.json'}
    bound.update(ROOT/k for k in read(OUT/'run_seal.json')['source_sha256'])
    for fold in range(3):
        fit,c,pure,totals,used=fit_context(d,fold); held=d[d.fold==fold]
        mass=c.sum(1)[used]; ss=fit.groupby(['local','truth']).size().rename('same_input_class_rows')
        keyroots=fit.groupby(['local','truth']).root.nunique().rename('same_input_class_roots')
        mass_reports.append({'fold':fold,'unique_inputs':len(used),'original_rows':len(fit),
            'original_mass_quantiles_by_unique_input':weighted_quantiles(mass,np.ones(len(mass))),
            'original_mass_quantiles_by_original_row':weighted_quantiles(mass,mass),
            'single_input_max_original_mass':int(mass.max()),
            'single_input_max_fraction':float(mass.max()/mass.sum()),
            'effective_input_count_by_mass':float(mass.sum()**2/np.square(mass.astype(float)).sum())})
        for arm in ARMS:
            folder=OUT/f'fold{fold}_{arm}'; path=folder/'epoch100_model.pt'; rows_path=folder/'epochs/epoch100_rows.parquet'
            bound.update([path,rows_path]); state=torch.load(path,map_location='cpu',weights_only=True)
            model=Classifier(128).to('cuda'); model.load_state_dict(state['model']); model.eval()
            q=np.zeros((x.shape[0],3),np.float32); lp=np.zeros_like(q); avgz=np.zeros_like(q)
            mp=np.zeros((x.shape[0],16),np.int8); conf=np.zeros((x.shape[0],16,3),np.float32)
            with torch.no_grad():
                for start in range(0,x.shape[0],256):
                    ids=np.arange(start,min(start+256,x.shape[0])); z=batch(model,x,ids)
                    p=torch.softmax(z,-1); q[ids]=p.mean(1).cpu().numpy()
                    conf[ids]=p.cpu().numpy(); lp[ids]=torch.log_softmax(z,-1).mean(1).cpu().numpy()
                    avgz[ids]=z.mean(1).cpu().numpy(); mp[ids]=z.argmax(-1).cpu().numpy()
            rows=pd.read_parquet(rows_path)
            gap=float(np.abs(rows[['p0','p1','p2']].to_numpy()-q[fit.local]).max())
            if gap>2e-6 or not np.array_equal(rows.pred,q[fit.local].argmax(1)): raise ValueError('Actual TRAIN replay mismatch')
            reports=panel(fit,pure,q,mp,lp,avgz,'TRAIN_pure')
            # HELD class labels are diagnostic only: no aggregation or candidate selected.
            reports+=panel(held,np.ones(x.shape[0]),q,mp,lp,avgz,'HELD_observed')
            for r in reports: r.update(fold=fold,arm=arm,actual_TRAIN_probability_gap=gap)
            outputs+=reports
            bad=rows[rows.pure_TRAIN_input & rows.pred.ne(rows.truth)].copy()
            bad['fold']=fold; bad['arm']=arm
            bad=bad.merge(ss.reset_index(),on=['local','truth'],how='left').merge(keyroots.reset_index(),on=['local','truth'],how='left')
            votes=np.stack([(mp==j).sum(1) for j in range(3)],1)
            ii=bad.local.to_numpy(); yy=bad.truth.to_numpy()
            bad['vote_pred']=votes[ii].argmax(1);bad['logit_mean_pred']=avgz[ii].argmax(1)
            correct=mp[ii]==yy[:,None]; pc=conf[ii[:,None],np.arange(16)[None,:],yy[:,None]]
            bad['correct_member_probability_mean']=(pc*correct).sum(1)/correct.sum(1).clip(1)
            bad['wrong_member_truth_probability_mean']=(pc*~correct).sum(1)/(~correct).sum(1).clip(1)
            residuals.append(bad)
            np.savez_compressed(DEST/f'fold{fold}_{arm}_frozen.npz',mean_probability=q,member_probability=conf,member_pred=mp,mean_logit=avgz)
            if arm=='R_decay':
                # Fixed frozen-state partition; unbiased whole-role original-row CE.
                # Not the actual moving-parameter optimization trajectory or causal proof.
                rng=np.random.RandomState(13601+fold); order=rng.permutation(used)
                groups=[order[s:s+256] for s in range(0,len(order),256)]
                accum=[torch.zeros_like(p) for p in model.parameters()]
                sum_norm_sq=0.; batch_info=[]
                for ids in groups:
                    model.zero_grad(set_to_none=True); z=batch(model,x,ids)
                    m=torch.as_tensor(c[ids],dtype=torch.float32,device='cuda')
                    pr=torch.as_tensor(pure[ids],device='cuda')
                    loss,_,_=objective(z,m,pr,len(fit),totals,len(groups),False);loss.backward()
                    normsq=sum(float(p.grad.double().square().sum()) for p in model.parameters() if p.grad is not None)
                    for a,p in zip(accum,model.parameters()):
                        if p.grad is not None:a.add_(p.grad/len(groups))
                    sum_norm_sq+=normsq
                    batch_info.append({'M_mass':int(c[ids,1].sum()),'S_mass':int(c[ids,2].sum()),'gradient_norm_sq':normsq,'objective':float(loss)})
                mean_normsq=sum(float(a.double().square().sum()) for a in accum)
                avg_normsq=sum_norm_sq/len(groups)
                gradients.append({'fold':fold,'arm':arm,'frozen_batches':len(groups),
                    'full_original_row_gradient_norm':mean_normsq**.5,
                    'mean_minibatch_gradient_norm_sq':avg_normsq,
                    'batch_gradient_variance_trace':max(0.,avg_normsq-mean_normsq),
                    'batch_second_moment_to_full_gradient_sq':avg_normsq/max(mean_normsq,1e-30),
                    'partition_seed':13601+fold,'batches':batch_info,
                    'limits':'Frozen-state original-row CE partition; no optimizer, no parameter update; full-gradient stationarity is not quality or capacity proof.'})
                model.zero_grad(set_to_none=True);del accum
            print(json.dumps({'fold':fold,'arm':arm,'TRAIN_pure_errors':len(bad),'new_updates':0}),flush=True)
            del model,state;gc.collect();torch.cuda.empty_cache()
    pd.DataFrame(outputs).to_json(DEST/'member_diagnostics.json',orient='records',indent=2)
    pd.concat(residuals,ignore_index=True).to_parquet(DEST/'TRAIN_pure_residuals.parquet',index=False)
    save(DEST/'mass_diagnostics.json',mass_reports);save(DEST/'frozen_gradient_diagnostics.json',gradients)
    require_run_seal(ROOT/'training/v135_train.py')
    save(DEST/'source_receipt.json',{'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(bound)},'new_fits':0,'new_updates':0})
    save(DEST/'diagnosis_receipt.json',{'status':'frozen_actual_model_diagnostics_completed','models_replayed':12,
        'new_fits':0,'new_updates':0,'model_promoted':False,'aggregation_selected':False,
        'output_sha256':{p.name:sha(p) for p in sorted(DEST.iterdir()) if p.is_file()},
        'limits':['TRAIN counts are original-role rows, not independent original records.',
                  'HELD is already observed development evidence; alternative aggregation scores are diagnosis only.',
                  'Gradient variance at frozen endpoints does not establish why particular rows or HELD failed.']})


if __name__=='__main__': main()
