"""One matched SGD trajectory; checkpoint and averaging diagnostics, no target labels."""
import time
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from run_v75 import ROOT, BATCH, read, save, sha, load_sparse, new_model, metrics
from v79_execute import rows, guards

DEST=ROOT/'artifacts/v81_diagnosis_20260927/trajectory'
OLD=ROOT/'artifacts/v78_boundary_20260922'
LAST=ROOT/'artifacts/v79_execution_20260927'
CHECKPOINTS=[1,2,4,6,12,24]


def measure(y, pred, mask):
    return metrics(np.bincount(y[mask]*3+pred[mask],minlength=9).reshape(3,3))


def main():
    if DEST.exists():raise FileExistsError(DEST)
    DEST.mkdir()
    c=read(OLD/'contract.json');r=rows();y=r.label_index.to_numpy()
    selected=pd.read_parquet(OLD/'manifest.parquet',columns=['optimizer_diagnostic']).optimizer_diagnostic.to_numpy()
    fid=np.load(LAST/'row_feature_id.npy',mmap_mode='r');x=load_sparse(LAST/'X')
    fit=~r.fold.isin([0,2]).to_numpy();cal=r.fold.eq(2).to_numpy();held=r.fold.eq(0).to_numpy()
    assert selected.sum()==104544 and not (selected & ~fit).any()
    assert sha(ROOT/'training/v78_boundary.py')==c['source_sha256']
    baseline=joblib.load(OLD/'O_sgd.joblib')
    def predict(coef,intercept):return (x@coef.T+intercept).argmax(1).astype(np.int8)[fid]
    oldpred=predict(baseline.coef_,baseline.intercept_)
    half=(r.component.to_numpy().astype(np.int64)*2654435761)%2
    masks={'C':cal,'C_half0':cal&(half==0),'C_half1':cal&(half==1)}
    basem={name:measure(y,oldpred,m) for name,m in masks.items()}
    registration={'source_sha256':sha(__file__),'classifier_optimization_runs':1,'epochs':24,'checkpoints':CHECKPOINTS,
        'states':['averaged','standard_from_same_trajectory'],'roles':'same v78 fit/C/H; already inspected development',
        'sample_sha256':sha(OLD/'manifest.parquet'),'rows':int(selected.sum()),'alpha':c['alpha'],'eta0':c['eta0'],'seed':7801,
        'selection':'Strictly fewer errors and every class recall/precision/F1 nondecreasing, normal FP nonincreasing vs frozen six-epoch O_sgd on C and both fixed component hash halves. Then minimum C errors, earliest epoch, averaged state. None eligible means no candidate.',
        'H_read_for_selection':False,'target_read':False,'full_data_fit':False,
        'scope':'Mechanism test of optimization schedule, not the four-arm supervision/capacity training. Epoch snapshots are not independent fits.'}
    save(DEST/'registration.json',registration)
    model=new_model(c);rng=np.random.default_rng(7801);trace=[];start=time.monotonic()
    for ep in range(1,25):
        count=0
        for beg in rng.permutation(np.arange(0,len(r),BATCH)):
            ix=np.arange(beg,min(beg+BATCH,len(r)));rng.shuffle(ix);ix=ix[selected[ix]]
            if len(ix):model.partial_fit(x[fid[ix]],y[ix],classes=np.arange(3));count+=len(ix)
        assert count==int(selected.sum())
        if ep in CHECKPOINTS:
            for state,coef,intercept in [('averaged',model.coef_,model.intercept_),('standard',model._standard_coef,model._standard_intercept)]:
                name=f'epoch{ep:02}_{state}';p=predict(coef,intercept)
                score={k:measure(y,p,m) for k,m in masks.items()}
                gate={k:guards(basem[k],score[k]) for k in masks}
                ok=all(all(d.values()) for d in gate.values())
                joblib.dump({'coef':coef.T.copy(),'intercept':intercept.copy()},DEST/(name+'.joblib'))
                np.save(DEST/(name+'_pred.npy'),p)
                item={'name':name,'epoch':ep,'state':state,'fit':measure(y,p,fit),'C_scores':score,'guards':gate,'eligible':ok}
                trace.append(item)
                print({'name':name,'fit_errors':item['fit']['errors'],'C_errors':score['C']['errors'],'eligible':ok,'seconds':round(time.monotonic()-start,1)},flush=True)
                if ep==6 and state=='averaged':
                    diff=float(np.max(abs(coef-baseline.coef_)))
                    assert diff<1e-6 and np.array_equal(p,oldpred),(diff,int((p!=oldpred).sum()))
                    save(DEST/'baseline_reproduction.json',{'max_coefficient_delta':diff,'all_train_role_predictions_identical':True,'rows':len(r)})
            save(DEST/'trace.json',trace)
    eligible=[t for t in trace if t['eligible']]
    chosen=min(eligible,key=lambda t:(t['C_scores']['C']['errors'],t['epoch'],t['state']!='averaged')) if eligible else None
    # Persist selection before accessing H labels for any candidate.
    decision={'selected':None if chosen is None else chosen['name'],'eligible_count':len(eligible),'new_classifier_optimization_runs':1,'full_data_model':False}
    save(DEST/'selection.json',decision)
    evaluation={'baseline':measure(y,oldpred,held),'scope':'H used only after selection; failure cannot trigger alternate candidate choice.'}
    if chosen is not None:
        p=np.load(DEST/(chosen['name']+'_pred.npy'));m=measure(y,p,held)
        evaluation.update(candidate=m,guards=guards(evaluation['baseline'],m),candidate_name=chosen['name'])
    save(DEST/'evaluation.json',evaluation)
    save(DEST/'complete.json',{'classifier_optimization_runs':1,'checkpoint_states':len(trace),'calibration_fits':0,'seconds':time.monotonic()-start,'selection':decision,'target_scored':False,'model_promoted':False,'source_sha256':sha(__file__)})
    print({'completed':True,'selected':decision['selected'],'evaluation':evaluation},flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
