"""One bounded loss comparison after diagnosing rare-source underfitting."""
import math
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from train_v62_heads import OUT,ROOT,Cache,Head,evaluate,group_metrics
from train_v61_neural import seed_all
from v61_common import read,save,sha
from v61_runtime import score,predictions


def weights_for(fit,source):
    weights=pd.Series(1.,index=fit.index)
    asa=int(fit.label.gt(0).sum())
    for label in [1,2]:
        mask=fit.label.eq(label);sub=fit[mask]
        if source:
            sizes=sub.groupby('group').size()
            weights.loc[mask]=(asa/2/len(sizes))/sub.group.map(sizes)
        else:weights.loc[mask]=asa/2/len(sub)
    assert np.isclose(weights.sum(),len(fit)) and (weights>0).all()
    assert (weights.loc[fit.label.eq(0)]==1).all()
    for label in [1,2]:
        mask=fit.label.eq(label);assert np.isclose(weights.loc[mask].sum(),asa/2)
        if source:
            totals=weights.loc[mask].groupby(fit.loc[mask,'group']).sum()
            assert np.allclose(totals,totals.iloc[0])
    return weights


def train(data,arm,weights):
    seed=20260916;seed_all(seed);target=OUT/arm;assert not target.exists();target.mkdir()
    model=Head(False).cuda();opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    epochs=12;steps=math.ceil(len(data.fit)/256)*epochs;step=0;curve=[];best=-np.inf;best_epoch=0;t=time.monotonic()
    for epoch in range(1,epochs+1):
        model.train();order=np.random.default_rng(seed+epoch).permutation(data.fit);total=0.
        for start in range(0,len(order),256):
            idx=order[start:start+256]
            ratio=min(1.,(step+1)/max(1,int(steps*.1))) if step<steps*.1 else .1+.9*.5*(1+math.cos(math.pi*(step-steps*.1)/(steps*.9)))
            for group in opt.param_groups:group['lr']=3e-4*ratio
            opt.zero_grad(set_to_none=True);z=model(**data.batch(idx))
            # Uniform row sampling plus globally normalized fixed weights estimates
            # the declared risk. Dividing by each minibatch's weight sum would differ.
            loss=(nn.functional.cross_entropy(z,data.y[idx],reduction='none')*weights[idx]).mean()
            assert torch.isfinite(loss)
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm)
            opt.step();step+=1;total+=float(loss.detach())*len(idx)
        if epoch%2:continue
        p=evaluate(model,data,data.dev);s=score(data.df.label.iloc[data.dev],p);g=group_metrics(data.df.iloc[data.dev],p)
        curve.append({'epoch':epoch,'online_fit_weighted_CE':total/len(order),'selection':s,'source_metrics':g})
        save(target/'curve.json',curve);value=g['M_S_class_balanced_source_recall']
        if value>best:
            best=value;best_epoch=epoch;torch.save(model.state_dict(),target/'model.pt')
            predictions(target/'selection.parquet',data.df,data.dev,p)
        print(f'{arm} epoch {epoch}/12 source-balanced recall {value:.6f}, row macro {s["ASA"]["macro_f1_M_S"]:.6f}',flush=True)
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
        'model_sha256':sha(target/'model.pt'),'no_external_or_old_outer_evaluation':True,
        'probabilities_not_calibrated_for_original_class_source_priors':True})
    del model,opt;torch.cuda.empty_cache()


def main():
    torch.set_num_threads(4);data=Cache();fit=data.df.iloc[data.fit]
    assert not (OUT/'objective_protocol.json').exists()
    save(OUT/'pre_objective_analysis.json',read(OUT/'capacity_analysis.json'))
    weights={};weight_report={}
    for arm in ['class_balanced','source_balanced']:
        w=weights_for(fit,arm=='source_balanced')
        wf=fit[['row_position','label','group']].copy();wf['weight']=w
        wf.to_parquet(OUT/(arm+'_fit_weights.parquet'),index=False)
        full=np.full(len(data.df),np.nan,dtype=np.float32);full[data.fit]=w
        assert np.isnan(full[data.dev]).all();weights[arm]=torch.tensor(full,device='cuda')
        weight_report[arm]={'min':float(w.min()),'max':float(w.max()),'sum':float(w.sum()),
            'classes':{str(c):float(w.loc[fit.label.eq(c)].sum()) for c in [0,1,2]},
            'file_sha256':sha(OUT/(arm+'_fit_weights.parquet'))}
    # Check actual weighted-risk algebra on deterministic per-row values, not only totals.
    loss=np.log1p(np.arange(len(fit)));toy=pd.Series(loss,index=fit.index)
    perclass=fit.assign(toy=toy).query('label > 0').groupby(['label','group']).toy.mean().groupby('label').mean()
    expected=(fit.label.gt(0).mean()*perclass.mean()+toy.loc[fit.label.eq(0)].sum()/len(fit))
    assert np.isclose(np.mean(loss*weights_for(fit,True).to_numpy()),expected)
    protocol={'created_stage':'2026-09-19, after inspecting original two head diagnostics and before weighted fits',
        'scope':'Adaptive development hypothesis test; not independent validation or production promotion',
        'baseline':'Existing meanmax unweighted CE, same initialization seed/order/architecture/optimizer/schedule',
        'new_arms':['class_balanced','source_balanced'],'only_changed':'Training loss row contribution',
        'class_formula':'ASA/(2*N_class); benign controls weight 1',
        'source_formula':'ASA/(2*G_class*N_class_source); benign controls weight 1',
        'source_is_training_group_only_not_model_input':True,'no_new_labels_or_rows':True,
        'no_dynamic_hard_example_or_duplicate_group_weighting':True,
        'weights_derived_only_from_fit':True,'weights':weight_report,
        'exact_weighted_risk_algebra_checked':True,
        'epochs':12,'seed':20260916,'batch':256,'lr':.0003,'weight_decay':.01,
        'selection':'M/S class-balanced mean source recall; fixed even epochs; earlier wins ties',
        'continuation_gate_vs_unweighted':{'balanced_source_gain':.01,'S_source_gain':.02,
            'M_source_min_delta':-.01,'row_M_S_macro_min_delta':-.01,
            'row_M_recall_min_delta':-.01,'row_S_recall_min_delta':-.01,
            'paired_source_interval_lower_bound_positive':True},
        'no_old_outer_predictions_or_tuning':True,
        'source_sha256':sha(__file__),'head_source_sha256':sha(ROOT/'training/train_v62_heads.py'),
        'cache_receipt_sha256':sha(OUT/'cache_receipt.json'),
        'pre_objective_analysis_sha256':sha(OUT/'pre_objective_analysis.json')}
    save(OUT/'objective_protocol.json',protocol)
    for arm in protocol['new_arms']:train(data,arm,weights[arm])


if __name__=='__main__':main()
