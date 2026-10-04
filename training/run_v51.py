"""A bounded four-fit comparison with immutable v48 paired references."""
import argparse
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

COLS=['p_benign','p_malicious','p_suspicious']

def imports(out):
    sys.path.insert(0,str(out/'runtime'))
    import v51_residual as v
    import v50_views
    from run_v48 import sha,read,save,raw_rows
    from run_v49 import data,metrics
    return v,v50_views,sha,read,save,raw_rows,data,metrics

def reference(root,fold):
    return root/('artifacts/v48_information_repair_r2_20260914/pressure' if fold is None else 'artifacts/v50_context_training_r2_20260914/control_fold_'+str(fold))

def prepare(root,out):
    if out.exists():raise FileExistsError(out)
    parent=root/'artifacts/v50_context_training_r2_20260914'
    sys.path.insert(0,str(parent/'runtime'));from run_v48 import sha,read,save
    old=read(parent/'configuration.json')
    for p,h in old['source_bindings'].items():assert sha(root/p)==h,p
    for p,h in old['runtime_bindings'].items():assert sha(parent/'runtime'/p)==h,p
    assert read(parent/'verification.json')['passed']
    bindings=dict(old['source_bindings'])
    for fold in [None,0,1,2]:
        folder=reference(root,fold)
        for p in folder.iterdir():
            if p.is_file():bindings[p.relative_to(root).as_posix()]=sha(p)
    out.mkdir(parents=True);shutil.copytree(parent/'runtime',out/'runtime')
    for name in ['v51_residual.py','test_v51_residual.py','run_v51.py']:shutil.copy2(root/'training'/name,out/'runtime'/name)
    v,views,sha,read,save,raw_rows,data,metrics=imports(out)
    import test_v50_views,test_v51_residual
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(m) for m in [test_v50_views,test_v51_residual]])
    t=unittest.TextTestRunner().run(suite);assert t.wasSuccessful()
    rows,ids,texts,facts=data(root);gate=v.applicable(facts);support=[]
    for fold in [None,0,1,2]:
        fit=rows.inner_role.to_numpy()!=2 if fold is None else rows.fold.to_numpy()!=fold
        g=rows.loc[fit&gate[ids]]
        support.append({'fold':fold,'eligible_fit_rows':len(g),'eligible_eval_rows':int((~fit&gate[ids]).sum()),
            'fit_sources':[{'route':r,'label':int(y),'rows':len(t),'body_groups':int(t.body_group.nunique())} for (r,y),t in g.groupby(['route','label_index'])]})
    save(out/'support_before_fit.json',support)
    c={'version':v.VERSION,'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'C':.1,
        'candidate':'frozen_v48_plus_fact_gated_linear_residual','gate':'v50 partial_auth eligibility; no route/source/product/time/prediction gate',
        'training_views':'same full/partial 0.5/0.5 as v50; each original row total weight 1; nonapplicable rows constant base loss',
        'residual_features':'original encoded complete input and bias, columns activated in applicable fit views only',
        'decision':'three_class_argmax','new_classifier_fits_max':4,'base_model_retraining':False,
        'optimizer':{'method':'L-BFGS-B','maxiter':1000,'maxls':40,'gtol':1e-9,'ftol':1e-12,'initialization':'zero','regularization':'L2 on residual weights and gated bias'},
        'gates':{'pressure_total_errors_strictly_lower':True,'pressure_auth_errors_max':0,'primary_total_errors_no_increase':True,
            'each_fold_and_route_no_error_increase':True,'each_class_recall_no_drop':True,'normal_errors_no_increase':True,
            'nonapplicable_probability_max_difference':0,'base_parameters_unchanged':True},
        'sequence':'one pressure fit then three original body folds if pressure gates pass; no parameter/threshold scan',
        'blind_or_external_test':False,'automatic_production_promotion':False,'source_bindings':bindings,
        'runtime_bindings':{p.name:sha(p) for p in (out/'runtime').glob('*.py')},
        'preparation_bindings':{'support_before_fit.json':sha(out/'support_before_fit.json')}}
    save(out/'configuration.json',c);print(json.dumps({'prepared':True,'tests':t.testsRun,'new_fits_max':4,'pressure_fit_auth_rows':support[0]['eligible_fit_rows']}),flush=True)

def check(root,out):
    v,views,sha,read,save,raw_rows,data,metrics=imports(out);c=read(out/'configuration.json')
    for p,h in c['source_bindings'].items():assert sha(root/p)==h,p
    for p,h in c['runtime_bindings'].items():assert sha(out/'runtime'/p)==h,p
    for p,h in c['preparation_bindings'].items():assert sha(out/p)==h,p
    assert Path(__file__).resolve()==(out/'runtime/run_v51.py').resolve()
    return v,views,sha,read,save,raw_rows,data,metrics,c

def gates(r,pressure=False):
    by={x['route']:x for x in r['routes']}
    g={'total_errors':r['after']['errors']<r['before']['errors'] if pressure else r['after']['errors']<=r['before']['errors'],
        'macro_f1_no_drop':r['after']['macro_f1']>=r['before']['macro_f1'],
        'normal_errors_no_increase':r['after']['normal_errors']<=r['before']['normal_errors'],
        'each_route_no_regression':all(t['after_errors']<=t['before_errors'] for t in r['routes']),
        'each_class_recall_no_drop':all(a>=b for a,b in zip(r['after']['class_recall'],r['before']['class_recall']))}
    if pressure:g['authentication_all_recovered']=by['authentication']['after_errors']==0
    return g

def one(root,out,c,fold=None):
    v,views,sha,read,save,raw_rows,data,metrics=imports(out);name='pressure' if fold is None else 'fold_'+str(fold);folder=out/name
    if folder.exists():raise FileExistsError(folder)
    rows,ids,texts,facts=data(root);fit=rows.inner_role.to_numpy()!=2 if fold is None else rows.fold.to_numpy()!=fold;ev=~fit
    assert not set(rows.loc[fit,'body_group'])&set(rows.loc[ev,'body_group'])
    wide=set(rows.loc[fit,'union_group'])&set(rows.loc[ev,'union_group'])
    if fold is None:assert not wide
    ref=reference(root,fold);base=joblib.load(ref/'model.joblib');prior=pd.read_parquet(ref/'evaluation.parquet').sort_values('row_position').reset_index(drop=True)
    d=rows.loc[ev].reset_index(drop=True)
    for k in ['row_position','label_index','body_group']:np.testing.assert_array_equal(d[k],prior[k])
    te=base['text_encoder'];fe=base['fact_encoder'];x=sparse.hstack([te.transform(texts),fe.transform(facts)],format='csr')
    partial,indices,targets,weights,origin,eligible=views.training_views(facts,ids[fit],rows.label_index.to_numpy()[fit])
    xx=sparse.vstack([x,sparse.hstack([te.transform(['']*len(partial)),fe.transform(partial)],format='csr')],format='csr')
    gate=v.applicable(facts);aug_gate=np.r_[gate,v.applicable(partial)]
    frozen_coef=base['model'].coef_.copy();frozen_intercept=base['model'].intercept_.copy()
    folder.mkdir();pd.DataFrame({'row_position':rows.loc[fit,'row_position'].to_numpy()[origin],'input_index':indices,'label_index':targets,
        'weight':weights,'partial':indices>=len(facts),'residual_active':aug_gate[indices]}).to_parquet(folder/'training_manifest.parquet',index=False)
    save(folder/'binding.json',{'configuration_sha256':sha(out/'configuration.json'),'fold':fold,'reference_model_sha256':sha(ref/'model.joblib'),
        'reference_predictions_sha256':sha(ref/'evaluation.parquet'),'fit_rows':int(fit.sum()),'evaluation_rows':int(ev.sum()),
        'fit_body_overlap':0,'historical_union_overlap':len(wide),'partial_views':partial,'manifest_sha256':sha(folder/'training_manifest.parquet')})
    print(json.dumps({'stage':'fit_start','model':name,'eligible_fit_rows':int(eligible.sum())}),flush=True);start=time.perf_counter()
    residual,opt=v.fit_residual(base['model'],xx,indices,targets,weights,aug_gate,c['C'])
    np.testing.assert_array_equal(base['model'].coef_,frozen_coef);np.testing.assert_array_equal(base['model'].intercept_,frozen_intercept)
    q=v.predict_matrix(base['model'],residual,x,gate);b=base['model'].predict_proba(x)
    outside_diff=float(abs(q[~gate]-b[~gate]).max());assert outside_diff==0
    np.testing.assert_allclose(b[ids[ev]],prior[COLS].to_numpy(),atol=1e-12,rtol=0)
    auth_support={}
    for value in ['success','failure']:
        matched=np.asarray([f.get('auth_result')==value for f in facts])[ids]&fit
        auth_support[value]=[int((matched&(rows.label_index.to_numpy()==k)).sum()) for k in range(3)]
    fx=fe.transform(facts);cold=np.flatnonzero(np.asarray((fx[np.unique(ids[fit])]!=0).sum(axis=0)).ravel()==0)
    bundle={'version':v.VERSION,'base':base,'residual':residual,'auth_support':auth_support,'original_fit_cold_fact_columns':cold.tolist(),
        'configuration_sha256':sha(out/'configuration.json'),'base_sha256':sha(ref/'model.joblib'),'fold':fold}
    joblib.dump(bundle,folder/'model.joblib',compress=3)
    for i,k in enumerate(COLS):d[k]=q[ids[ev],i];d['b_'+k[2:]]=prior[k].to_numpy()
    d['residual_applicable']=gate[ids[ev]];d.to_parquet(folder/'evaluation.parquet',index=False)
    r=metrics(d);r.update(optimizer=opt,elapsed_seconds=time.perf_counter()-start,nonapplicable_rows=int((~gate[ids[ev]]).sum()),nonapplicable_max_difference=outside_diff)
    r['gates']=gates(r,fold is None);r['gates']['nonapplicable_identical']=outside_diff==0;r['gates']['converged']=opt['converged'];r['all_gates_passed']=all(r['gates'].values())
    save(folder/'report.json',r);save(folder/'complete.json',{p.name:sha(p) for p in folder.iterdir() if p.is_file()})
    print(json.dumps({'stage':'fit_complete','model':name,'before_errors':r['before']['errors'],'after_errors':r['after']['errors'],'fixes':r['fixes'],'regressions':r['regressions'],'all_gates_passed':r['all_gates_passed']}),flush=True)
    return r

def fit(root,out):
    v,views,sha,read,save,raw_rows,data,metrics,c=check(root,out)
    if (out/'execution.json').exists():raise FileExistsError('Already executed')
    first=one(root,out,c);reports=[first];primary=None
    if first['all_gates_passed']:
        for f in range(3):reports.append(one(root,out,c,f))
        d=pd.concat([pd.read_parquet(out/('fold_'+str(f))/'evaluation.parquet') for f in range(3)],ignore_index=True)
        primary=metrics(d);primary['gates']=gates(primary);primary['gates']['all_folds_passed']=all(r['all_gates_passed'] for r in reports[1:]);primary['all_gates_passed']=all(primary['gates'].values());save(out/'primary_report.json',primary)
    save(out/'execution.json',{'new_classifier_fits':len(reports),'pressure_passed':first['all_gates_passed'],'primary_executed':primary is not None,
        'primary_passed':primary['all_gates_passed'] if primary else None,'development_gates_passed':bool(first['all_gates_passed'] and primary and primary['all_gates_passed']),
        'base_retrained':False,'production_promoted':False,'external_or_blind_validation':False})

def verify(root,out):
    v,views,sha,read,save,raw_rows,data,metrics,c=check(root,out)
    if (out/'verification.json').exists():raise FileExistsError('Existing verification')
    rows,ids,texts,facts=data(root);execution=read(out/'execution.json');names=['pressure']+(['fold_'+str(f) for f in range(3)] if execution['primary_executed'] else [])
    raw=list(raw_rows(root/'data/official/train.parquet',rows.loc[rows.route.isin(['native_flow','native_firewall','authentication']),'row_position']))
    import v48_input as adapter
    cases=[];variants=[];active=np.unique(rows.projection_id);bypos=rows.set_index('row_position')
    for pos,message in raw:
        parsed=adapter.prepare_record({'message_sanitized':message});k=int(np.searchsorted(active,bypos.loc[pos,'projection_id']));assert parsed['text']==texts[k] and parsed['facts']==facts[k]
        variant=message
        if parsed['route'] in ('native_flow','native_firewall'):
            m=adapter.NATIVE.fullmatch(message.strip());variant=message.strip()
            for key in sorted(adapter.IDENTITIES-{'priority'},key=lambda k:m.start(k),reverse=True):
                if m[key] is not None:a,b=m.span(key);variant=variant[:a]+'OPAQUE-identity'+variant[b:]
        cases.append({'message_sanitized':message});variants.append({'message_sanitized':variant,'timestamp':'2099','product_name':None,'label_binary':'unused','route':'arbitrary','src_ip':'203.0.113.5'})
    compact=np.searchsorted(active,bypos.loc[[pos for pos,_ in raw],'projection_id'].to_numpy());gate=v.applicable(facts);results=[]
    for name in names:
        folder=out/name
        for p,h in read(folder/'complete.json').items():assert sha(folder/p)==h,p
        b=joblib.load(folder/'model.joblib');fold=b['fold'];ref=reference(root,fold);base=joblib.load(ref/'model.joblib')
        assert b['base_sha256']==sha(ref/'model.joblib') and b['configuration_sha256']==sha(out/'configuration.json')
        np.testing.assert_array_equal(b['base']['model'].coef_,base['model'].coef_);np.testing.assert_array_equal(b['base']['model'].intercept_,base['model'].intercept_)
        fit=rows.inner_role.to_numpy()!=2 if fold is None else rows.fold.to_numpy()!=fold;ev=~fit
        partial,indices,labels,weights,origin,eligible=views.training_views(facts,ids[fit],rows.label_index.to_numpy()[fit])
        manifest=pd.read_parquet(folder/'training_manifest.parquet');assert (manifest.groupby('row_position').weight.sum()==1).all()
        for key,value in [('row_position',rows.loc[fit,'row_position'].to_numpy()[origin]),('input_index',indices),('label_index',labels),('weight',weights)]:np.testing.assert_array_equal(manifest[key],value)
        assert not set(manifest.row_position)&set(rows.loc[ev,'row_position']);assert not set(rows.loc[fit,'body_group'])&set(rows.loc[ev,'body_group'])
        te=b['base']['text_encoder'];fe=b['base']['fact_encoder'];assert te.fit_rows==int(fit.sum())
        x=sparse.hstack([te.transform(texts),fe.transform(facts)],format='csr');q=v.predict_matrix(base['model'],b['residual'],x,gate);d=pd.read_parquet(folder/'evaluation.parquet')
        np.testing.assert_array_equal(d.row_position,rows.loc[ev,'row_position']);np.testing.assert_array_equal(d.label_index,rows.loc[ev,'label_index'])
        diff=float(abs(q[ids[ev]]-d[COLS].to_numpy()).max());assert diff<1e-12
        original=base['model'].predict_proba(x);np.testing.assert_array_equal(q[~gate],original[~gate])
        prior=pd.read_parquet(ref/'evaluation.parquet').sort_values('row_position').reset_index(drop=True)
        np.testing.assert_array_equal(d[['b_benign','b_malicious','b_suspicious']].to_numpy(),prior[COLS].to_numpy())
        report=read(folder/'report.json')
        for key,value in metrics(d).items():assert report[key]==value,key
        qr,audit=v.classify_records(b,cases);qv,_=v.classify_records(b,variants)
        np.testing.assert_allclose(qr,q[compact],atol=1e-12,rtol=0);np.testing.assert_allclose(qr,qv,atol=1e-12,rtol=0)
        save(folder/'coverage_audit.json',[{'row_position':pos,**a,'probabilities_B_M_S':p.tolist()} for (pos,_),a,p in zip(raw,audit,qr) if bypos.loc[pos,'route']=='authentication'])
        results.append({'model':name,'replayed_rows':len(d),'max_difference':diff,'nonapplicable_probability_difference':0,'raw_rows':len(raw),'specified_variants':len(variants)})
        print('Verified '+name,flush=True)
    import test_v50_views,test_v51_residual
    t=unittest.TextTestRunner().run(unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(m) for m in [test_v50_views,test_v51_residual]]));assert t.wasSuccessful()
    save(out/'verification.json',{'passed':True,'models':results,'tests':t.testsRun,'development_gates_passed':execution['development_gates_passed'],
        'scope':'Paired probabilities, immutable base, original weight/label lineage and specified raw invariance; no external validation.'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','fit','verify']);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();globals()[a.stage](a.root.resolve(),a.out.resolve())
