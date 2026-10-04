"""One loss change, unchanged v48 input, original rows/labels/splits.

Preparation and gates are frozen before fitting. Primary comparisons add
paired v48 refits because v48 never ran the original three body folds.
"""
import argparse
import collections
import hashlib
import json
import shutil
import sys
import time
import unittest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy import sparse

BASE = 'artifacts/v48_information_repair_r2_20260914'
COLS = ['p_benign','p_malicious','p_suspicious']

def imports(out):
    sys.path.insert(0, str(out/'runtime'))
    import v48_input as adapter
    import v50_views as views
    from run_v48 import sha, read, save, raw_rows
    from run_v49 import data, metrics
    return adapter, views, sha, read, save, raw_rows, data, metrics

def prepare(root,out):
    if out.exists(): raise FileExistsError(out)
    old=root/'artifacts/v49_observed_encoding_20260914'
    sys.path.insert(0,str(old/'runtime'))
    from run_v48 import sha,read,save
    source=read(old/'configuration.json')
    for p,h in source['source_bindings'].items(): assert sha(root/p)==h,p
    for p,h in source['runtime_bindings'].items(): assert sha(old/'runtime'/p)==h,p
    out.mkdir(parents=True);(out/'runtime').mkdir()
    for p in (old/'runtime').glob('*.py'): shutil.copy2(p,out/'runtime'/p.name)
    for name in ['v50_views.py','test_v50_views.py','run_v50.py','verify_v50.py']:
        shutil.copy2(root/'training'/name,out/'runtime'/name)
    adapter,v,sha,read,save,raw_rows,data,metrics=imports(out)
    import test_v50_views
    t=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromModule(test_v50_views));assert t.wasSuccessful()
    rows,ids,texts,facts=data(root);fit=rows.inner_role.to_numpy()!=2
    views,ix,y,w,origin,eligible=v.training_views(facts,ids[fit],rows.label_index.to_numpy()[fit])
    selected=rows.loc[fit].iloc[np.flatnonzero(eligible)]
    support=[]
    for (route,label),g in selected.groupby(['route','label_index']):
        support.append(dict(route=route,label=int(label),rows=len(g),bodies=int(g.body_group.nunique())))
    # Reparse all fit-side invalid-credential records, plus first and last three
    # records of each other auth source/outcome, against actual official bytes.
    fit_indices=np.flatnonzero(fit)[eligible]
    targets=set(rows.iloc[[i for i in fit_indices if facts[ids[i]].get('credential_check')=='invalid']].row_position)
    for _,g in selected.groupby(['route','label_index']): targets.update(g.head(3).row_position);targets.update(g.tail(3).row_position)
    bypos=rows.set_index('row_position');active=np.unique(rows.projection_id);examples=[]
    for pos,raw in raw_rows(root/'data/official/train.parquet',list(targets)):
        parsed=adapter.prepare_message(raw);k=int(np.searchsorted(active,bypos.loc[pos,'projection_id']))
        assert parsed['facts']==facts[k] and parsed['text']==texts[k]
        examples.append({'row_position':pos,'raw_sha256':hashlib.sha256(raw.encode()).hexdigest(),
                         'full_facts':facts[k],'partial_facts':v.partial_auth(facts[k])})
    save(out/'semantic_audit.json',{'fit_auth_support':support,'eligible_rows':int(eligible.sum()),'unique_views':len(views),
        'raw_reparsed_rows':len(examples),'examples':examples,'core_fields':sorted(v.CORE),
        'total_weight':float(w.sum()),'normal_failure_counterexamples':int(((selected.label_index==0)&selected.row_position.isin(rows.iloc[[i for i in fit_indices if facts[ids[i]].get('auth_result')=='failure']].row_position)).sum()),
        'scope':'Partial observation supervision is a statistical augmentation hypothesis, NOT certified threat-label invariance. Full inputs, attempts and durations retained. Official labels are not incident ground truth.'})
    config={'version':v.VERSION,'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'candidate':'full_plus_partial_auth_context','C':.1,'partial_weight':.5,'complete_weight_for_eligible':.5,
        'core_fields':sorted(v.CORE),'labels':'unchanged official','decision':'three_class_argmax',
        'fit_only_views':True,'vocabulary_and_scaler':'original full fit rows only; pressure exactly frozen v48 encoders',
        'maximum_classifier_fits':7,'sequence':'one pressure fit; if all gates pass, three paired v48/control and v50 folds',
        'gates':{'pressure_errors_less_than':894,'pressure_auth_errors_max':8,'pressure_macro_f1_min':.8871606113729106,
            'normal_errors_no_increase':True,'each_route_errors_no_increase':True,'each_class_recall_no_drop':True,
            'native_flow_M_to_B_no_increase':True,'primary_paired_total_strict_improvement':True},
        'no_hyperparameter_or_threshold_search':True,'blind_or_external_validation':False,'automatic_promotion':False,
        'source_bindings':source['source_bindings'],
        'runtime_bindings':{p.name:sha(p) for p in (out/'runtime').glob('*.py')},
        'preparation_bindings':{'semantic_audit.json':sha(out/'semantic_audit.json')}}
    save(out/'configuration.json',config)
    print(json.dumps({'stage':'prepared','eligible_fit_rows':int(eligible.sum()),'partial_views':len(views),'raw_reparsed':len(examples),'tests':t.testsRun}),flush=True)

def check(root,out):
    adapter,v,sha,read,save,raw_rows,data,metrics=imports(out);c=read(out/'configuration.json')
    for p,h in c['source_bindings'].items():assert sha(root/p)==h,p
    for p,h in c['runtime_bindings'].items():assert sha(out/'runtime'/p)==h,p
    for p,h in c['preparation_bindings'].items():assert sha(out/p)==h,p
    assert Path(__file__).resolve()==(out/'runtime/run_v50.py').resolve()
    return adapter,v,sha,read,save,raw_rows,data,metrics,c

def quality(r,d,pressure=False):
    route={x['route']:x for x in r['routes']}
    f=d[d.route=='native_flow'];old=f[['b_benign','b_malicious','b_suspicious']].to_numpy().argmax(1);new=f[COLS].to_numpy().argmax(1)
    gates={'total_errors_decrease':r['after']['errors']<r['before']['errors'],
        'macro_f1_no_drop':r['after']['macro_f1']>=r['before']['macro_f1'],
        'normal_errors_no_increase':r['after']['normal_errors']<=r['before']['normal_errors'],
        'every_route_no_regression':all(x['after_errors']<=x['before_errors'] for x in r['routes']),
        'every_class_recall_no_drop':all(a>=b for a,b in zip(r['after']['class_recall'],r['before']['class_recall'])),
        'native_flow_M_to_B_no_increase':int(((f.label_index==1)&(new==0)).sum())<=int(((f.label_index==1)&(old==0)).sum())}
    if pressure:gates['auth_recovery']=route['authentication']['after_errors']<=8
    return gates

def one(root,out,c,name,fold=None,control=False):
    adapter,v,sha,read,save,raw_rows,data,metrics=imports(out)
    from run_v38_train import text_encoder
    folder=out/name
    if folder.exists():raise FileExistsError(folder)
    rows,ids,texts,facts=data(root);fit=rows.inner_role.to_numpy()!=2 if fold is None else rows.fold.to_numpy()!=fold;ev=~fit
    assert not set(rows.loc[fit,'body_group'])&set(rows.loc[ev,'body_group'])
    wide_overlap=set(rows.loc[fit,'union_group'])&set(rows.loc[ev,'union_group'])
    if fold is None:assert not wide_overlap
    if fold is None: reference_path=root/BASE/'pressure'
    elif control:reference_path=root/('artifacts/v39_local_r2_20260913/primary/fold_%s/SEMANTIC'%fold)
    else:reference_path=out/('control_fold_'+str(fold))
    ref=pd.read_parquet(reference_path/'evaluation.parquet').sort_values('row_position').reset_index(drop=True)
    d=rows.loc[ev].reset_index(drop=True)
    for k in ['row_position','label_index','body_group']:assert d[k].equals(ref[k]),k
    if fold is None or not control:
        base=joblib.load(reference_path/'model.joblib');te=base['text_encoder'];fe=base['fact_encoder']
    else:
        te=text_encoder(texts,ids,fit);fe=adapter.old.SemanticFacts().fit([facts[k] for k in np.unique(ids[fit])])
    x=sparse.hstack([te.transform(texts),fe.transform(facts)],format='csr');y=rows.label_index.to_numpy()
    if control:
        views=[];indices=ids[fit];targets=y[fit];weights=np.ones(int(fit.sum()));origin=np.arange(int(fit.sum()));eligible=np.zeros(int(fit.sum()),dtype=bool);xx=x
    else:
        views,indices,targets,weights,origin,eligible=v.training_views(facts,ids[fit],y[fit])
        vx=sparse.hstack([te.transform(['']*len(views)),fe.transform(views)],format='csr');xx=sparse.vstack([x,vx],format='csr')
    folder.mkdir();pd.DataFrame({'row_position':rows.loc[fit,'row_position'].to_numpy()[origin],
        'input_index':indices,'label_index':targets,'weight':weights,'partial_view':indices>=len(facts)}).to_parquet(folder/'training_view_manifest.parquet',index=False)
    save(folder/'binding.json',{'configuration_sha256':sha(out/'configuration.json'),'fit_rows':int(fit.sum()),'evaluation_rows':int(ev.sum()),
        'reference_model_sha256':sha(reference_path/'model.joblib'),'manifest_sha256':sha(folder/'training_view_manifest.parquet'),
        'partial_views':views,'eligible_fit_rows':int(eligible.sum()),'fold':fold,'control':control,
        'body_groups_cross_split':0,'historical_union_groups_cross_split':len(wide_overlap)})
    print(json.dumps({'stage':'fit_start','name':name,'full_fit_rows':int(fit.sum()),'partial_training_rows':int(eligible.sum())}),flush=True);start=time.perf_counter()
    model,opt=adapter.old.learning.fit_aggregated(xx,indices,targets,c['C'],row_weights=weights)
    assert opt['sum_weights']==int(fit.sum())
    bundle={'version':v.VERSION,'candidate':c['candidate'] if not control else 'v48_paired_control','text_encoder':te,'fact_encoder':fe,'model':model,
        'unseen_fit_fact_columns':[],'configuration_sha256':sha(out/'configuration.json'),'decision':'three_class_argmax','fold':fold}
    # Coverage stays tied to original full fit inputs, not artificial support.
    fx=fe.transform(facts);bundle['unseen_fit_fact_columns']=np.flatnonzero(np.asarray((fx[np.unique(ids[fit])]!=0).sum(axis=0)).ravel()==0).tolist()
    joblib.dump(bundle,folder/'model.joblib',compress=3);q=model.predict_proba(x)[ids[ev]]
    for i,k in enumerate(COLS):d[k]=q[:,i];d['b_'+k[2:]]=ref[k].to_numpy()
    d.to_parquet(folder/'evaluation.parquet',index=False);r=metrics(d);r.update(optimizer=opt,elapsed_seconds=time.perf_counter()-start)
    r['gates']=quality(r,d,fold is None);r['all_gates_passed']=all(r['gates'].values())
    if views:
        vp=model.predict_proba(xx[-len(views):]);oldp=base['model'].predict_proba(xx[-len(views):])
        save(folder/'partial_view_predictions.json',[dict(facts=f,before=a.tolist(),after=b.tolist()) for f,a,b in zip(views,oldp,vp)])
    save(folder/'report.json',r);save(folder/'complete.json',{p.name:sha(p) for p in folder.iterdir() if p.is_file()})
    print(json.dumps({'stage':'fit_complete','name':name,'before_errors':r['before']['errors'],'after_errors':r['after']['errors'],'fixes':r['fixes'],'regressions':r['regressions'],'gates':r['gates']}),flush=True)
    return r

def fit(root,out):
    adapter,v,sha,read,save,raw_rows,data,metrics,c=check(root,out)
    if (out/'execution.json').exists():raise FileExistsError('Already executed')
    if c.get('continuation_from'):
        r=read(out/'pressure/report.json')
        for p,h in read(out/'pressure/complete.json').items():assert sha(out/'pressure'/p)==h
    else:r=one(root,out,c,'pressure')
    names=['pressure'];primary=None
    if r['all_gates_passed']:
        for f in range(3):
            for control in [True,False]:
                name=('control_fold_' if control else 'fold_')+str(f);one(root,out,c,name,f,control);names.append(name)
        d=pd.concat([pd.read_parquet(out/('fold_'+str(f))/'evaluation.parquet') for f in range(3)],ignore_index=True)
        primary=metrics(d);primary['gates']=quality(primary,d);primary['all_gates_passed']=all(primary['gates'].values());save(out/'primary_report.json',primary)
    save(out/'execution.json',{'classifier_fits':len(names),'new_fits_in_this_directory':len(names)-int(bool(c.get('continuation_from'))),
        'models':names,'pressure_all_gates_passed':r['all_gates_passed'],
        'primary_executed':primary is not None,'primary_all_gates_passed':primary['all_gates_passed'] if primary else None,
        'classifier_promoted':False,'external_or_blind_validation':False})

def prepare_continuation(root,out):
    """Preserve the completed fit; fix an overbroad isolation assertion only."""
    if out.exists():raise FileExistsError(out)
    parent=root/'artifacts/v50_context_training_20260914'
    adapter,v,sha,read,save,raw_rows,data,metrics=imports(parent)
    c=read(parent/'configuration.json')
    for p,h in c['source_bindings'].items():assert sha(root/p)==h,p
    for p,h in c['runtime_bindings'].items():assert sha(parent/'runtime'/p)==h,p
    for p,h in read(parent/'pressure/complete.json').items():assert sha(parent/'pressure'/p)==h,p
    assert read(parent/'pressure/report.json')['all_gates_passed']
    assert not any(parent.glob('*fold_*'))
    out.mkdir(parents=True);shutil.copytree(parent/'runtime',out/'runtime');shutil.copytree(parent/'pressure',out/'pressure')
    shutil.copy2(root/'training/run_v50.py',out/'runtime/run_v50.py')
    shutil.copy2(root/'training/verify_v50.py',out/'runtime/verify_v50.py')
    shutil.copy2(parent/'semantic_audit.json',out/'semantic_audit.json')
    c['continuation_from']=str(parent.relative_to(root).as_posix())
    c['pressure_configuration_sha256']=sha(parent/'configuration.json')
    c['continuation_reason']='Original body folds permit historical union overlap; the overbroad assertion stopped before any fold fit. Pressure model copied exactly, not retrained. No split/quality gate change.'
    c['runtime_bindings']={p.name:sha(p) for p in (out/'runtime').glob('*.py')}
    for p in [parent/'configuration.json']+list((parent/'pressure').iterdir()):
        if p.is_file():c['source_bindings'][p.relative_to(root).as_posix()]=sha(p)
    save(out/'configuration.json',c)
    print(json.dumps({'stage':'continuation_prepared','existing_pressure_fits':1,'remaining_paired_fits':6}),flush=True)

def verify(root,out):
    from verify_v50 import main
    main(root,out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','prepare_continuation','fit','verify']);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();globals()[a.stage](a.root.resolve(),a.out.resolve())
