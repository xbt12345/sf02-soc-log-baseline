"""Complete the missing historical pressure comparison, with fixed R/I configs."""
import argparse
import hashlib
import importlib.util
import json
import shutil
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, f1_score, log_loss

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def module(p,n):
    spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def data(root,out):
    h=module(out/'v46_reference.py','v46_reference');e=module(out/'v44_reference.py','v44_reference')
    prep=root/'artifacts/v48_information_repair_r2_20260914'
    r=pd.read_parquet(prep/'rows.parquet');r=r[r.route=='asa'].sort_values('row_position').reset_index(drop=True)
    p=pd.read_parquet(prep/'projections.parquet')
    x=pd.DataFrame([e.encode(json.loads(p.iloc[int(i)].facts)) for i in r.projection_id],columns=e.FIELDS)
    assert len(r)==106953 and set(r.label_index)=={1,2}
    tr=r.inner_role!=2
    assert set(r.loc[tr,'body_group']).isdisjoint(r.loc[~tr,'body_group'])
    assert set(r.loc[tr,'union_group']).isdisjoint(r.loc[~tr,'union_group'])
    return h,r,x,tr.to_numpy()

def prepare(root,out):
    assert not out.exists(),'Never overwrite a run.'
    old=root/'artifacts/v46_interaction_20260914'
    assert old.exists()
    cfg=read(old/'configuration.json');rep=read(old/'report.json')
    paths=[root/'artifacts/v48_information_repair_r2_20260914'/n for n in ['rows.parquet','projections.parquet']]
    paths += [root/'artifacts/v51_fact_residual_20260914'/n/'evaluation.parquet' for n in ['pressure','fold_0','fold_1','fold_2']]
    paths += [old/'evaluation.parquet',old/'configuration.json',old/'report.json']
    paths += [root/'artifacts/v45_missingness_20260914/evaluation.parquet']
    for name,digest in rep['model_sha256'].items():
        assert sha(old/name)==digest
    assert sha(old/'probe_v46_interaction.py')==cfg['script_sha256']
    out.mkdir(parents=True)
    shutil.copyfile(old/'probe_v46_interaction.py',out/'v46_reference.py')
    shutil.copyfile(old/'v44_probe_reference.py',out/'v44_reference.py')
    shutil.copyfile(__file__,out/Path(__file__).name)
    h,r,x,tr=data(root,out)
    r.to_parquet(out/'split_manifest.parquet',index=False)
    config={'version':'v52-pressure-gap-fixed-methods-1','lr':cfg['lr'],'threshold':.5,
        'candidates':['R_without_interaction','I_with_protocol_roles'],
        'fit_policy':'Each of four views fits every ASA pressure-fit row once, with official labels and original multiplicity. No class reweighting, masks as labels, tuning, or early stopping.',
        'fit_rows':int(tr.sum()),'eval_rows':int((~tr).sum()),'new_fits_planned':8,
        'primary_comparison':'Reuse saved R/I original-fold scores and pair by row_position with current v51. No refits or threshold selection.',
        'predeclared_continuation_gates':['Pressure ASA total errors strictly below current v51','Pressure M and S errors separately no greater than current v51','Original-fold ASA total and each class errors no greater than current v51'],
        'decision':'Failure stops this fixed candidate from direct integration; success only warrants further development, never automatic production promotion.',
        'scenarios':'Original, hide source, hide destination, hide both. Report only originally observed fields as eligible stress rows; information deletion is not nuisance invariance.',
        'limitations':['All evaluations already inspected. New pressure execution is not a blind test.','ASA has no normal labels; this experiment cannot validate a normal-attack boundary.','No new telemetry/label evidence, no external data or unverified time/identity links.'],
        'bindings':{p.relative_to(root).as_posix():sha(p) for p in paths},
        'local_bindings':{p.name:sha(p) for p in out.iterdir() if p.is_file()}}
    save(out/'configuration.json',config)
    save(out/'preregistered.json',{'configuration_sha256':sha(out/'configuration.json'),'new_fits':0})
    print(json.dumps({'prepared':str(out),'fit_rows':int(tr.sum()),'eval_rows':int((~tr).sum()),'new_fits':0}),flush=True)

def stat(y,pred,q=None):
    z={'rows':len(y),'errors':int((pred!=y).sum()),'confusion_B_M_S':confusion_matrix(y,pred,labels=[0,1,2]).tolist(),
       'M_errors':int(((y==1)&(pred!=1)).sum()),'S_errors':int(((y==2)&(pred!=2)).sum())}
    if q is not None:z['conditional_log_loss']=float(log_loss(y==1,np.c_[1-q,q],labels=[False,True]))
    return z

def fit(root,out):
    c=read(out/'configuration.json');assert sha(out/'configuration.json')==read(out/'preregistered.json')['configuration_sha256']
    for p,d in c['bindings'].items():assert sha(root/p)==d,p
    for p,d in c['local_bindings'].items():assert sha(out/p)==d,p
    assert sha(__file__)==c['local_bindings'][Path(__file__).name]
    assert not (out/'started.json').exists()
    save(out/'started.json',{'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'configuration_sha256':sha(out/'configuration.json')})
    h,r,x,tr=data(root,out);ev=~tr;y=r.label_index.to_numpy();models={};timings=[];allrows=[]
    for name,interaction in [('R',False),('I',True)]:
        models[name]={}
        for view in h.VIEWS:
            t=time.monotonic();rec=h.records(x.loc[tr],view,interaction)
            model=make_pipeline(DictVectorizer(sparse=True),LogisticRegression(**c['lr']))
            model.fit(rec,y[tr]==1)
            assert model[-1].n_iter_.max()<c['lr']['max_iter']
            path=out/(name+'_'+view+'.joblib');joblib.dump(model,path);loaded=joblib.load(path)
            np.testing.assert_array_equal(model.predict_proba(h.records(x.loc[ev],view,interaction)),loaded.predict_proba(h.records(x.loc[ev],view,interaction)))
            # Independently enumerate exactly the fit-side vocabulary (no evaluation vocabulary fit).
            expected={k+'='+v for d in rec for k,v in d.items()}
            assert set(loaded[0].get_feature_names_out())==expected
            models[name][view]=loaded
            timings.append({'model':name,'view':view,'seconds':time.monotonic()-t,'iterations':int(model[-1].n_iter_.max()),'fit_rows':int(tr.sum()),'features':len(expected)})
            print(json.dumps(timings[-1]),flush=True)
    for scenario,cols in h.SCENARIOS.items():
        xx=x.loc[ev].copy();xx[cols]=h.MISSING;routes=h.routing(xx)
        d=r.loc[ev].copy().reset_index(drop=True);d['scenario']=scenario;d['view']=routes
        eligible=np.ones(len(d),bool) if not cols else np.zeros(len(d),bool)
        for col in cols:
            if col.endswith('_fixed'):eligible|=(x.loc[ev,col]!=h.MISSING).to_numpy()
        d['eligible']=eligible
        for name,interaction in [('R',False),('I',True)]:
            q=np.empty(len(d))
            for view in h.VIEWS:
                ix=routes==view
                if ix.any():q[ix]=models[name][view].predict_proba(h.records(xx.loc[ix],view,interaction))[:,1]
            d[name+'_qM']=q
        allrows.append(d)
    e=pd.concat(allrows,ignore_index=True);e.to_parquet(out/'pressure_evaluation.parquet',index=False)
    baseline=pd.read_parquet(root/'artifacts/v51_fact_residual_20260914/pressure/evaluation.parquet')
    b=baseline[baseline.route=='asa'].sort_values('row_position').reset_index(drop=True)
    natural=e[e.scenario=='original'].reset_index(drop=True)
    np.testing.assert_array_equal(b.row_position,natural.row_position);np.testing.assert_array_equal(b.label_index,natural.label_index)
    bp=b[['p_benign','p_malicious','p_suspicious']].to_numpy();bpred=bp.argmax(1)
    assert (bpred!=0).all()
    pressure={'v51':stat(b.label_index.to_numpy(),bpred,bp[:,1]/bp[:,1:].sum(1))}
    for name in ['R','I']:pressure[name]=stat(natural.label_index.to_numpy(),np.where(natural[name+'_qM']>=.5,1,2),natural[name+'_qM'].to_numpy())
    primary=pd.concat([pd.read_parquet(root/f'artifacts/v51_fact_residual_20260914/fold_{f}/evaluation.parquet') for f in range(3)]).sort_values('row_position').reset_index(drop=True)
    ap=primary[primary.route=='asa'].copy().reset_index(drop=True);ap['v51_pred']=ap[['p_benign','p_malicious','p_suspicious']].to_numpy().argmax(1)
    old=pd.read_parquet(root/'artifacts/v46_interaction_20260914/evaluation.parquet');old=old[old.scenario=='original'].sort_values('row_position').reset_index(drop=True)
    np.testing.assert_array_equal(ap.row_position,old.row_position);np.testing.assert_array_equal(ap.label_index,old.label_index)
    pair={'v51':stat(ap.label_index.to_numpy(),ap.v51_pred.to_numpy())};full={}
    for name in ['R','I']:
        ap[name+'_qM']=old[name+'_qM'].to_numpy();ap[name+'_pred']=np.where(ap[name+'_qM']>=.5,1,2)
        pair[name]=stat(ap.label_index.to_numpy(),ap[name+'_pred'].to_numpy(),ap[name+'_qM'].to_numpy())
        pair[name]['fixed']=int(((ap.v51_pred!=ap.label_index)&(ap[name+'_pred']==ap.label_index)).sum())
        pair[name]['regressed']=int(((ap.v51_pred==ap.label_index)&(ap[name+'_pred']!=ap.label_index)).sum())
        pred=primary[['p_benign','p_malicious','p_suspicious']].to_numpy().argmax(1)
        pred[primary.route=='asa']=ap[name+'_pred'].to_numpy()
        full[name]=dict(stat(primary.label_index.to_numpy(),pred),macro_f1=float(f1_score(primary.label_index,pred,labels=[0,1,2],average='macro')))
    ap.to_parquet(out/'paired_primary_ASA.parquet',index=False)
    stress={str(s):{name:stat(g.label_index.to_numpy(),np.where(g[name+'_qM']>=.5,1,2),g[name+'_qM'].to_numpy()) for name in ['R','I']} for s,g in e[e.eligible].groupby('scenario')}
    gates={}
    for name in ['R','I']:
        gates[name]={'pressure_total_improves':pressure[name]['errors']<pressure['v51']['errors'],
            'pressure_each_class_no_worse':all(pressure[name][k]<=pressure['v51'][k] for k in ['M_errors','S_errors']),
            'primary_total_no_worse':pair[name]['errors']<=pair['v51']['errors'],
            'primary_each_class_no_worse':all(pair[name][k]<=pair['v51'][k] for k in ['M_errors','S_errors'])}
    report={'status':'fixed_candidates_pressure_review_no_promotion','new_fits':8,'pressure':pressure,'primary_ASA':pair,
        'mechanical_full_primary_label_swap_not_deployment':full,'pressure_information_removal':stress,'continuation_gates':gates,
        'fit_log':timings,'fit_save_seconds':sum(t['seconds'] for t in timings),'limits':c['limitations']}
    save(out/'report.json',report)
    save(out/'complete.json',{'bindings':{p.name:sha(p) for p in out.iterdir() if p.is_file()},'model_promoted':False})
    print(json.dumps({k:report[k] for k in ['pressure','primary_ASA','continuation_gates','fit_save_seconds']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','fit']);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    globals()[a.mode](a.root.resolve(),a.out.resolve())
