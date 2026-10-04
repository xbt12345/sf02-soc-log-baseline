"""V115 execution: fixed N1, nested K/U epoch selection; six trajectories max."""
import argparse
import json
import math
import time

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from v116_preflight import ROOT, DEST, VIEW, save, sha
from v104_phase_b import BATCH, DEVICE, K, SEED, SparseTabM, csr_tensor, predict_all
from v75_views import BYTE_FEATURES
from v107_evaluate import report_metric

EPOCHS = (1,2,5,10,15,20,25)


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def metrics(d, pred):
    result=report_metric(d.truth.to_numpy(),pred)
    means=[]
    for cls in (1,2):
        z=d[d.truth.eq(cls)].copy()
        z['correct']=np.asarray(pred)[d.truth.eq(cls).to_numpy()]==cls
        g=z.groupby('root').correct.mean()
        result['group_'+str(cls)]={'roots':len(g),'mean_recall':float(g.mean()) if len(g) else None,
                                 'zero_recall_roots':int(g.eq(0).sum())}
        if len(g):means.append(float(g.mean()))
    result['MS_root_macro_recall']=sum(means)/len(means) if len(means)==2 else None
    return result


def choose(records):
    by={q['epoch']:q for q in records}
    assert set(by)==set(EPOCHS)
    final=by[25]
    for q in records:
        for task in ('K','U'):
            for label in ('malicious','suspicious'):
                assert q['tasks'][task]['class'][label]['support'] > 0
                assert q['tasks'][task]['class'][label]['support']==final['tasks'][task]['class'][label]['support']
    eligible=[q for q in records if all(q['tasks'][task]['class'][label]['correct'] >=
             final['tasks'][task]['class'][label]['correct'] for task in ('K','U') for label in ('malicious','suspicious'))]
    def score(q):
        return (min(q['tasks'][t]['MS_equal_F1'] for t in ('K','U')),
                sum(q['tasks'][t]['MS_root_macro_recall'] for t in ('K','U'))/2,-q['epoch'])
    winner=max(eligible,key=score)
    return {'selected_epoch':winner['epoch'],'eligible_epochs':[q['epoch'] for q in eligible],
            'selection_key':list(score(winner)), 'only_inner_K_U_metrics_used':True,
            'epoch25_tasks':final['tasks'],'selected_tasks':winner['tasks']}


def register():
    pf=read(DEST/'preflight.json')
    assert pf['all_checks_passed'] and not (DEST/'training_registration.json').exists()
    assert sha(DEST/'inner_split_manifest.parquet')==pf['manifest_sha256']
    assert sha(DEST/'split_policy.json')==pf['policy_sha256']
    from v107_matched_training import check as old_check
    old_check()
    p=read(DEST/'split_policy.json')
    for rel,h in p['input_sha256'].items(): assert sha(ROOT/rel)==h,rel
    assert sha(ROOT/'training/v116_preflight.py')==p['source_sha256']
    paths=[VIEW,DEST/'inner_split_manifest.parquet',DEST/'preflight.json',DEST/'split_policy.json',
           ROOT/'training/v116_preflight.py',ROOT/'training/v116_train.py',ROOT/'training/v116_evaluate.py',
           ROOT/'training/v104_phase_b.py', ROOT/'training/v107_evaluate.py']
    save(DEST/'training_registration.json',{'status':'registered_before_any_fit','created_unix':time.time(),
        'classifier_fits_at_registration':0,'max_new_trajectories':6,'inner_fits':3,'outer_fits':3,
        'original_row_frequency_CE':True,'model_seed':SEED,'checkpoints':list(EPOCHS),
        'architecture':'unchanged N1 SparseTabM','optimizer':'AdamW','lr':.002,'weight_decay':.0003,
        'batch_size':BATCH,'K':K,'device':DEVICE,'torch':torch.__version__,
        'outer_labels_used_for_selection':False,'selection_rule':'four K/U class correct-count guards vs epoch25; maximize min task MS F1; tie root macro recall then earlier epoch',
        'issues':['Fold1 U contains 34 S rows from 15 roots; low row support may make checkpoint choice unstable.',
                  'Whole-root parameter holdout removes collateral rows from inner fit; they remain recorded and return in outer refit.',
                  'Outer folds are previously inspected development data, not new blind validation.'],
        'input_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in paths}})
    print(json.dumps({'stage':'registered','max_fits':6,'device':DEVICE}),flush=True)


def check():
    reg=read(DEST/'training_registration.json')
    for rel,h in reg['input_sha256'].items(): assert sha(ROOT/rel)==h,rel
    return reg


def lock_selection():
    check()
    assert not (DEST/'locked_selection.json').exists()
    assert not list(DEST.glob('outer_fold*'))
    out={}
    for fold in range(3):
        folder=DEST/f'inner_fold{fold}'
        fit=read(folder/'fit.json')
        assert fit['epochs']==25 and fit['scope']=='inner'
        history=read(folder/'checkpoints.json')
        assert sha(folder/'checkpoints.json')==fit['checkpoint_report_sha256']
        out[str(fold)]=choose(history)
        out[str(fold)]['checkpoint_report_sha256']=sha(folder/'checkpoints.json')
    save(DEST/'locked_selection.json',{'status':'locked_before_outer_fits','created_unix':time.time(),
        'folds':out,'new_fits_so_far':3,'outer_fits_so_far':0,'registration_sha256':sha(DEST/'training_registration.json')})
    print(json.dumps({'stage':'selection_locked','epochs':{k:v['selected_epoch'] for k,v in out.items()}},ensure_ascii=False),flush=True)


def fit(scope, fold):
    check()
    assert scope in ('inner','outer') and fold in (0,1,2)
    if scope=='outer':
        locked=read(DEST/'locked_selection.json')
        assert locked['status']=='locked_before_outer_fits'
        selected=locked['folds'][str(fold)]['selected_epoch']
    else:
        assert not (DEST/'locked_selection.json').exists()
        selected=None
    folder=DEST/f'{scope}_fold{fold}'
    assert not folder.exists()
    all_manifest=pd.read_parquet(DEST/'inner_split_manifest.parquet')
    d=all_manifest[all_manifest.outer_fold.eq(fold)].reset_index(drop=True)
    train=d.role.eq('fit').to_numpy() if scope=='inner' else d.fold.ne(fold).to_numpy()
    x=sparse.load_npz(VIEW)
    local=d.local.to_numpy(dtype=np.int64);y=d.truth.to_numpy(dtype=np.int8)
    counts=np.bincount(local[train]*3+y[train],minlength=x.shape[0]*3).reshape(-1,3)
    assert counts.sum()==train.sum()
    used=np.flatnonzero(counts.sum(1)); cc=counts[used].astype(np.float32); xt=x[used]
    inference_ids=np.unique(local[d.fold.ne(fold)]) if scope=='inner' else np.arange(x.shape[0])
    lookup=np.full(x.shape[0],-1,np.int64);lookup[inference_ids]=np.arange(len(inference_ids))
    assert (lookup[used]>=0).all()
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    model=SparseTabM().to(DEVICE)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0003)
    folder.mkdir();started=time.monotonic()
    save(folder/'started.json',{'status':'started','scope':scope,'fold':fold,'seed':SEED,'created_unix':time.time(),
        'train_rows':int(train.sum()),'train_class_counts':counts.sum(0).tolist(),'train_unique':len(used),
        'heldout_gradient_rows':0,'selected_epoch_before_fit':selected,
        'selection_sha256':sha(DEST/'locked_selection.json') if scope=='outer' else None,
        'registration_sha256':sha(DEST/'training_registration.json')})
    batches=math.ceil(len(used)/BATCH);normalizer=float(train.sum())/batches
    rng=np.random.default_rng(SEED+fold);history=[];checkpoints=[]
    for epoch in range(1,26):
        model.train();order=rng.permutation(len(used));numerator_total=0.
        for start in range(0,len(order),BATCH):
            b=order[start:start+BATCH];block=xt[b]
            fact=torch.as_tensor(block[:,BYTE_FEATURES:].toarray(),device=DEVICE,dtype=torch.float32)
            mass=torch.as_tensor(cc[b],device=DEVICE,dtype=torch.float32)
            z=model(csr_tensor(block,DEVICE),fact)
            numerator=-(torch.log_softmax(z,-1)*mass[:,None,:]).sum()/K
            loss=numerator/normalizer
            assert bool(torch.isfinite(loss).item())
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            numerator_total+=float(numerator.detach().item())
        history.append({'epoch':epoch,'online_row_CE':numerator_total/float(train.sum()),'seconds':time.monotonic()-started})
        keep=epoch in EPOCHS if scope=='inner' else epoch in {selected,25}
        if keep:
            probability=predict_all(model,'TabM',x[inference_ids],DEVICE)
            np.save(folder/f'epoch{epoch}_prob.npy',probability)
            np.save(folder/'inference_local_ids.npy',inference_ids)
            torch.save({'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                'scope':scope,'fold':fold,'epoch':epoch,'seed':SEED,'source_sha256':sha(__file__)},folder/f'epoch{epoch}_model.pt')
            train_pred=probability[lookup[used]].argmax(1)
            item={'epoch':epoch,'train_errors_by_class':{str(c):int((counts[used,c]*(train_pred!=c)).sum()) for c in (1,2)},
                  'model_sha256':sha(folder/f'epoch{epoch}_model.pt'),'prob_sha256':sha(folder/f'epoch{epoch}_prob.npy')}
            if scope=='inner':
                item['tasks']={}
                for task in ('K','U'):
                    q=d[d.role.eq(task)]
                    item['tasks'][task]=metrics(q,probability[lookup[q.local.to_numpy()]].argmax(1))
            checkpoints.append(item)
            save(folder/'checkpoints.json',checkpoints)
        save(folder/'progress.json',history)
        if epoch in EPOCHS:
            msg={'stage':'fit_progress','scope':scope,'fold':fold,'epoch':epoch,
                 'online_CE':history[-1]['online_row_CE'],'seconds':round(time.monotonic()-started,1)}
            if scope=='inner':
                msg['validation_correct']={t:{c:checkpoints[-1]['tasks'][t]['class'][c]['correct'] for c in ('malicious','suspicious')} for t in ('K','U')}
            print(json.dumps(msg,ensure_ascii=False),flush=True)
    save(folder/'fit.json',{'status':'fit_executed','scope':scope,'fold':fold,'epochs':25,
        'classifier_fits':1,'calibration_fits':0,'train_rows':int(train.sum()),'train_classes':counts.sum(0).tolist(),
        'heldout_gradient_rows':0,'seconds':time.monotonic()-started,'selected_epoch':selected,
        'checkpoint_report_sha256':sha(folder/'checkpoints.json'),'source_sha256':sha(__file__)})
    print(json.dumps({'stage':'fit_complete','scope':scope,'fold':fold,'seconds':round(time.monotonic()-started,1)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['register','inner','outer','lock']);p.add_argument('--fold',type=int)
    a=p.parse_args()
    if a.stage=='register':register()
    elif a.stage=='lock':lock_selection()
    else:fit(a.stage,a.fold)
