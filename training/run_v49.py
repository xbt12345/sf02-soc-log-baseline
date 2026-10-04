"""Bounded observed-value experiment plus support audit. Original data only."""
import argparse
import collections
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

def runtime_import(path):
    sys.path.insert(0,str(path))
    import v49_encoding as v
    from run_v48 import sha,read,save,raw_rows
    return v,sha,read,save,raw_rows

def data(root):
    old=root/'artifacts/v48_information_repair_r2_20260914'
    rows=pd.read_parquet(old/'rows.parquet');allp=pd.read_parquet(old/'projections.parquet')
    active,ids=np.unique(rows.projection_id.to_numpy(),return_inverse=True)
    p=allp.iloc[active]
    return rows,ids,p.text.tolist(),[json.loads(f) for f in p.facts]

def prepare(root,out):
    if out.exists():raise FileExistsError(out)
    old=root/'artifacts/v48_information_repair_r2_20260914'
    sys.path.insert(0,str(old/'runtime'))
    from run_v48 import sha,read,save
    for p,h in read(old/'delivery.json')['bindings'].items():assert sha(root/p)==h,p
    out.mkdir(parents=True);rt=out/'runtime';rt.mkdir()
    for p in (old/'runtime').glob('*.py'):shutil.copy2(p,rt/p.name)
    for n in ['v49_encoding.py','test_v49_encoding.py','run_v49.py']:shutil.copy2(root/'training'/n,rt/n)
    v,sha,read,save,raw_rows=runtime_import(rt)
    import test_v49_encoding
    suite=unittest.defaultTestLoader.loadTestsFromModule(test_v49_encoding)
    result=unittest.TextTestRunner(verbosity=1).run(suite);assert result.wasSuccessful()
    rows,ids,texts,facts=data(root);fit=rows.inner_role.to_numpy()!=2;ev=~fit
    model=joblib.load(old/'pressure/model.joblib')
    fe=v.EvidenceFacts().fit([facts[k] for k in np.unique(ids[fit])]);fx=fe.transform(facts)
    oldx=model['fact_encoder'].transform(facts)
    nold=oldx.shape[1];columns=np.concatenate(list(fe.blocks.values()))
    nonfinite=np.setdiff1d(np.arange(nold),columns)
    assert (fx[:,:nold][:,nonfinite]!=oldx[:,nonfinite]).nnz==0
    observed_cells=0;missing_cells=0
    for j,k in enumerate(fe.keys):
        obs=np.array([v.finite_observation(f,k) is not None for f in facts])
        assert (fx[:,:nold][obs][:,fe.blocks[k]]!=oldx[obs][:,fe.blocks[k]]).nnz==0
        assert fx[:,:nold][~obs][:,fe.blocks[k]].nnz==0
        reconstructed=np.asarray(fx[:,:nold][:,fe.blocks[k]].dot(2**np.arange(len(fe.blocks[k])))).ravel()
        assert all(int(reconstructed[i])==v.finite_observation(facts[i],k) for i in np.flatnonzero(obs))
        observed_cells+=int(obs[ids].sum());missing_cells+=int((~obs[ids]).sum())
    # A negative control: the old linear function can be transferred EXACTLY
    # to this new basis. Thus this change adds no observable information.
    tx=model['text_encoder'].transform(texts);x=sparse.hstack([tx,fx],format='csr')
    weights=np.pad(model['model'].coef_,((0,0),(0,len(fe.keys))));intercept=model['model'].intercept_.copy()
    tdim=tx.shape[1]
    for j,k in enumerate(fe.keys):
        missing=v.FINITE[k][1];bit_pattern=np.array([(missing>>i)&1 for i in range(len(fe.blocks[k]))])
        delta=model['model'].coef_[:,tdim+fe.blocks[k]].dot(bit_pattern)
        intercept+=delta;weights[:,-len(fe.keys)+j]=-delta
    logits=x.dot(weights.T)+intercept;logits-=logits.max(axis=1,keepdims=True)
    remapped=np.exp(logits);remapped/=remapped.sum(axis=1,keepdims=True)
    original=model['model'].predict_proba(sparse.hstack([tx,oldx],format='csr'))
    maxdiff=float(np.max(abs(remapped-original)));assert maxdiff<1e-12
    oldcols=model['fact_encoder'].names();positive=(oldx!=0).tocsc()
    support=[]
    # Exhaustive same-column support for every fact active in authentication,
    # plus every evaluation-only feature identified in v48.
    auth=rows.route.to_numpy()=='authentication'
    wanted=set(np.asarray((oldx[np.unique(ids[auth])]!=0).sum(axis=0)).ravel().nonzero()[0])
    wanted.update(np.flatnonzero((np.asarray((oldx[np.unique(ids[fit])]!=0).sum(axis=0)).ravel()==0)&(np.asarray((oldx[np.unique(ids[ev])]!=0).sum(axis=0)).ravel()>0)))
    for col in sorted(wanted):
        present=np.asarray((positive[:,col]!=0).toarray()).ravel()[ids];tr=fit&present;te=ev&present
        groups=[]
        for name,g in rows.loc[tr].groupby(['route','label_index']):
            groups.append({'route':name[0],'label':int(name[1]),'rows':len(g),'bodies':int(g.body_group.nunique())})
        support.append({'feature':str(oldcols[col]),'fit_rows':int(tr.sum()),'evaluation_rows':int(te.sum()),
            'fit_class_rows_B_M_S':[int((tr&(rows.label_index.to_numpy()==c)).sum()) for c in range(3)],
            'fit_sources':groups})
    save(out/'fit_support.json',support)
    # Technical tokens are not silently replaced by generic substrings.
    cases=[];vocabulary=set(model['text_encoder'].names())
    for text in sorted(set(np.asarray(texts,dtype=object)[np.unique(ids[auth])])):
        split=text.replace('_',' ')
        cases.append({'original':text,'split':split,'original_nnz':model['text_encoder'].transform([text]).nnz,
            'split_nnz':model['text_encoder'].transform([split]).nnz,'matching_words':[w for w in split.split() if w in vocabulary],
            'complete_original_phrase_in_fit_vocabulary':text in vocabulary})
    save(out/'representation_audit.json',{'allowed_rows':len(rows),'active_projections':len(facts),
        'all_observed_finite_values_preserved':True,'all_nonfinite_features_unchanged':True,
        'observed_finite_cells':observed_cells,'unobserved_finite_cells':missing_cells,
        'old_function_transferred_without_fit_max_probability_difference':maxdiff,
        'new_information_added_by_reencoding':False,'availability_still_observable':True,
        'test_count':result.testsRun,'text_cases':cases,
        'interpretation':'A fixed-regularization comparison, not evidence that feature dependency or OOV semantics is solved.'})
    bindings={}
    for p in [old/'rows.parquet',old/'projections.parquet',old/'configuration.json',old/'pressure/model.joblib',old/'pressure/evaluation.parquet',old/'pressure/report.json']:
        bindings[p.relative_to(root).as_posix()]=sha(p)
    # Input and reference identity remains anchored to the official file and
    # original three folds; no historical current-entry documents are runtime inputs.
    bindings.update(read(old/'configuration.json')['source_bindings'])
    c={'version':v.VERSION,'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'C':0.1,
        'candidate':'observed_value_basis_only; unchanged v48 parser and text encoder',
        'row_weights':'original multiplicities; no class/repeat weighting','labels':'unchanged official',
        'decision':'three_class_argmax','maximum_classifier_fits':4,'hyperparameter_search':False,
        'gates':{'errors_less_than_v48':894,'macro_f1_at_least_v48':0.8871606113729106,'normal_errors_max':0,
            'authentication_errors_max':8,'native_flow_errors_max':48,'native_flow_M_to_B_max':0,
            'each_route_errors_no_increase_vs_v48':True,'all_class_recall_no_drop_vs_v48':True},
        'sequence':'one_pressure_fit; three_original_body_fold_fits_only_if_every_pressure_gate_passes',
        'blind_or_external_validation':False,'automatic_model_promotion':False,
        'source_bindings':bindings,'runtime_bindings':{p.name:sha(p) for p in rt.glob('*.py')},
        'preparation_bindings':{n:sha(out/n) for n in ['representation_audit.json','fit_support.json']}}
    save(out/'configuration.json',c)
    print(json.dumps({'stage':'prepared','identity_transfer_diff':maxdiff,'new_information_from_basis':False,'planned_pressure_fits':1}),flush=True)

def check(root,out):
    v,sha,read,save,raw_rows=runtime_import(out/'runtime');c=read(out/'configuration.json')
    for p,h in c['source_bindings'].items():assert sha(root/p)==h,p
    for p,h in c['runtime_bindings'].items():assert sha(out/'runtime'/p)==h,p
    for p,h in c['preparation_bindings'].items():assert sha(out/p)==h,p
    assert Path(__file__).resolve()==(out/'runtime/run_v49.py').resolve()
    return v,sha,read,save,raw_rows,c

def prepare_text(root,out):
    if out.exists():raise FileExistsError(out)
    old=root/'artifacts/v49_observed_encoding_20260914'
    sys.path.insert(0,str(old/'runtime'))
    from run_v48 import sha,read,save
    c=read(old/'configuration.json')
    for p,h in c['source_bindings'].items():assert sha(root/p)==h,p
    assert read(old/'execution.json')['classifier_fits']==1
    assert not read(old/'execution.json')['pressure_all_gates_passed']
    out.mkdir(parents=True);rt=out/'runtime';rt.mkdir()
    for p in (old/'runtime').glob('*.py'):shutil.copy2(p,rt/p.name)
    for n in ['run_v49.py','v49_text.py']:shutil.copy2(root/'training'/n,rt/n)
    sys.path.insert(0,str(rt))
    from v49_text import WordCharacter,MAX_FEATURES,CHARACTER_WEIGHT
    rows,ids,texts,facts=data(root);fit=rows.inner_role.to_numpy()!=2
    print(json.dumps({'stage':'fit_only_text_vocabulary','classifier_fits':0}),flush=True)
    encoder=WordCharacter().fit(texts,ids,fit)
    oldmodel=joblib.load(root/'artifacts/v48_information_repair_r2_20260914/pressure/model.joblib')
    assert np.array_equal(encoder.word.names(),oldmodel['text_encoder'].names())
    assert np.array_equal(encoder.word.idf,oldmodel['text_encoder'].idf)
    joblib.dump(encoder,out/'pressure_text_encoder.joblib',compress=3)
    auth=rows[rows.route=='authentication'];cases=[]
    for pid in np.unique(ids[rows.route.to_numpy()=='authentication']):
        text=texts[pid];x=encoder.transform([text]);w=encoder.word.transform([text])
        assert (x[:,:w.shape[1]]!=w).nnz==0
        active=x[:,w.shape[1]:].indices
        cases.append({'text':text,'word_nnz':w.nnz,'character_nnz':len(active),
            'character_features':[str(encoder.character.get_feature_names_out()[i]) for i in active],
            'security_semantics_verified':False})
    # Do not lose the original channel's polarity/order tokens while expanding.
    phrases=['allow','deny','authentication success','authentication failure','no response','response received','invalid passcode','valid passcode','cmd x && y','cmd x || y']
    union=encoder.transform(phrases);word=encoder.word.transform(phrases)
    assert (union[:,:word.shape[1]]!=word).nnz==0
    for a,b in [(0,1),(2,3),(4,5),(6,7)]:assert (union[a]!=union[b]).nnz>0
    save(out/'text_representation_audit.json',{'fit_rows':int(fit.sum()),'character_vocabulary_features':len(encoder.character.vocabulary_),
        'word_channel_names_and_idf_identical_to_v48':True,'word_channel_vectors_preserved_for_auth_and_counterexamples':True,
        'character_weight':CHARACTER_WEIGHT,'case_rows':len(auth),'cases':cases,
        'scope':'Different vectors and spelling overlap are software/coverage facts, not proof of learned semantic equivalence.'})
    c.update(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),candidate='word_character_only',
        fact_encoding='unchanged_v48_SemanticFacts',character_max_features=MAX_FEATURES,character_weight=CHARACTER_WEIGHT,
        character_ngram_range=[3,5],text_vocabulary_fit_rows_only=True,
        predecessor_observed_encoding_quality='rejected_no_classification_changes')
    c['source_bindings'][(old/'configuration.json').relative_to(root).as_posix()]=sha(old/'configuration.json')
    c['source_bindings'][(old/'pressure/report.json').relative_to(root).as_posix()]=sha(old/'pressure/report.json')
    c['runtime_bindings']={p.name:sha(p) for p in rt.glob('*.py')}
    c['preparation_bindings']={n:sha(out/n) for n in ['pressure_text_encoder.joblib','text_representation_audit.json']}
    save(out/'configuration.json',c)
    print(json.dumps({'stage':'text_candidate_prepared','classification_fits':0,'character_features':len(encoder.character.vocabulary_)}),flush=True)

def metrics(d):
    from run_v39_train import metric
    y=d.label_index.to_numpy();p=d[['p_benign','p_malicious','p_suspicious']].to_numpy();b=d[['b_benign','b_malicious','b_suspicious']].to_numpy()
    old=b.argmax(1);new=p.argmax(1);d=d.copy();d['fix']=(old!=y)&(new==y);d['regression']=(old==y)&(new!=y)
    route=[]
    for name,g in d.groupby('route'):
        idx=g.index.to_numpy();route.append({'route':name,'rows':len(g),'before_errors':int((old[idx]!=y[idx]).sum()),
          'after_errors':int((new[idx]!=y[idx]).sum()),'fixes':int(g.fix.sum()),'regressions':int(g.regression.sum())})
    groups=d.groupby(['route','body_group'])[['fix','regression']].sum();groups['net']=groups.fix-groups.regression
    return {'before':metric(y,b),'after':metric(y,p),'routes':route,'fixes':int(d.fix.sum()),'regressions':int(d.regression.sum()),
        'body_groups_improved':int((groups.net>0).sum()),'body_groups_worsened':int((groups.net<0).sum())}

def fit_one(root,out,c,name,fold=None):
    v,sha,read,save,raw_rows=runtime_import(out/'runtime')
    from run_v38_train import text_encoder
    folder=out/name
    if folder.exists():raise FileExistsError(folder)
    rows,ids,texts,facts=data(root);fit=rows.inner_role.to_numpy()!=2 if fold is None else rows.fold.to_numpy()!=fold;ev=~fit
    assert not set(rows.loc[fit,'body_group'])&set(rows.loc[ev,'body_group'])
    oldpath=root/('artifacts/v48_information_repair_r2_20260914/pressure' if fold is None else 'artifacts/v39_local_r2_20260913/primary/fold_%s/SEMANTIC'%fold)
    reference=pd.read_parquet(oldpath/'evaluation.parquet').sort_values('row_position').reset_index(drop=True)
    evaluation=rows.loc[ev].reset_index(drop=True)
    for k in ['row_position','label_index','body_group']:assert evaluation[k].equals(reference[k]),k
    folder.mkdir();save(folder/'binding.json',{'configuration_sha256':sha(out/'configuration.json'),'fit_rows':int(fit.sum()),'evaluation_rows':int(ev.sum()),'fold':fold})
    print(json.dumps({'stage':'fit_start','name':name,'rows':int(fit.sum())}),flush=True);start=time.perf_counter()
    if c['candidate']=='word_character_only':
        from v49_text import WordCharacter
        te=joblib.load(out/'pressure_text_encoder.joblib') if fold is None else WordCharacter().fit(texts,ids,fit)
        fe=v.input_adapter.old.SemanticFacts().fit([facts[k] for k in np.unique(ids[fit])])
    else:
        te=text_encoder(texts,ids,fit);fe=v.EvidenceFacts().fit([facts[k] for k in np.unique(ids[fit])])
    tx=te.transform(texts);fx=fe.transform(facts)
    x=sparse.hstack([tx,fx],format='csr')
    model,optimizer=v.input_adapter.old.learning.fit_aggregated(x,ids[fit],rows.label_index.to_numpy()[fit],c['C'])
    assert optimizer['sum_weights']==int(fit.sum())
    cold=np.flatnonzero(np.asarray((fx[np.unique(ids[fit])]!=0).sum(axis=0)).ravel()==0)
    bundle={'version':v.VERSION,'candidate':c['candidate'],'input_adapter_version':v.input_adapter.VERSION,'text_encoder':te,'fact_encoder':fe,'model':model,
        'unseen_fit_fact_columns':cold.tolist(),'configuration_sha256':sha(out/'configuration.json'),'decision':'three_class_argmax','fold':fold}
    joblib.dump(bundle,folder/'model.joblib',compress=3);q=model.predict_proba(x)[ids[ev]]
    for i,k in enumerate(['benign','malicious','suspicious']):evaluation['p_'+k]=q[:,i];evaluation['b_'+k]=reference['p_'+k].to_numpy()
    evaluation.to_parquet(folder/'evaluation.parquet',index=False);r=metrics(evaluation);r.update(optimizer=optimizer,elapsed_seconds=time.perf_counter()-start)
    if fold is None:
        by={x['route']:x for x in r['routes']};f=evaluation[evaluation.route=='native_flow'];g=c['gates']
        r['gates']={'errors_strictly_decrease':r['after']['errors']<g['errors_less_than_v48'],
            'macro_f1_no_drop':r['after']['macro_f1']>=g['macro_f1_at_least_v48'],
            'normal_errors_no_increase':r['after']['normal_errors']<=g['normal_errors_max'],
            'authentication_recovery':by['authentication']['after_errors']<=g['authentication_errors_max'],
            'native_flow_no_regression':by['native_flow']['after_errors']<=g['native_flow_errors_max'],
            'native_flow_no_M_to_B':int(((f.label_index==1)&(f[['p_benign','p_malicious','p_suspicious']].to_numpy().argmax(1)==0)).sum())<=g['native_flow_M_to_B_max'],
            'each_route_no_regression':all(x['after_errors']<=x['before_errors'] for x in r['routes']),
            'each_class_recall_no_drop':all(a>=b for a,b in zip(r['after']['class_recall'],r['before']['class_recall'])),
            'converged':bool(optimizer['converged'])}
        r['all_gates_passed']=all(r['gates'].values())
    save(folder/'report.json',r);save(folder/'complete.json',{n:sha(folder/n) for n in ['model.joblib','evaluation.parquet','report.json','binding.json']})
    print(json.dumps({'stage':'fit_complete','name':name,'before_errors':r['before']['errors'],'after_errors':r['after']['errors'],'macro_f1':r['after']['macro_f1'],'gates':r.get('gates')}),flush=True)
    return r

def fit(root,out):
    v,sha,read,save,raw_rows,c=check(root,out)
    if (out/'execution.json').exists():raise FileExistsError('Already executed')
    r=fit_one(root,out,c,'pressure');fits=1;primary=None
    if r['all_gates_passed']:
        for f in range(3):fit_one(root,out,c,'fold_'+str(f),f);fits+=1
        d=pd.concat([pd.read_parquet(out/('fold_'+str(f))/'evaluation.parquet') for f in range(3)],ignore_index=True)
        primary=metrics(d);save(out/'primary_report.json',primary)
    save(out/'execution.json',{'classifier_fits':fits,'pressure_all_gates_passed':r['all_gates_passed'],
        'primary_executed':primary is not None,'classifier_promoted':False,'external_validation':False})

def verify(root,out):
    v,sha,read,save,raw_rows,c=check(root,out)
    if (out/'verification.json').exists():raise FileExistsError('Existing verification')
    rows,ids,texts,facts=data(root);execution=read(out/'execution.json')
    folders=['pressure']+(['fold_'+str(f) for f in range(3)] if execution['primary_executed'] else [])
    selected=rows[rows.route.isin(['native_flow','native_firewall','authentication'])]
    native=list(raw_rows(root/'data/official/train.parquet',selected.row_position))
    active=np.unique(rows.projection_id);bypos=rows.set_index('row_position');rawcases=[];changes=[]
    for pos,raw in native:
        r=v.input_adapter.prepare_message(raw);cached=facts[int(np.searchsorted(active,bypos.loc[pos,'projection_id']))]
        assert r['facts']==cached
        rawcases.append({'message_sanitized':raw});variant=raw
        if r['route'] in ('native_flow','native_firewall'):
            m=v.input_adapter.NATIVE.fullmatch(raw.strip());variant=raw.strip()
            for key in sorted(v.input_adapter.IDENTITIES-{'priority'},key=lambda k:m.start(k),reverse=True):
                if m[key] is not None:
                    a,b=m.span(key);variant=variant[:a]+'OPAQUE-identity'+variant[b:]
        changes.append({'message_sanitized':variant,'timestamp':'2099','product_name':None,'label_binary':'unused'})
    import test_v49_encoding
    t=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromModule(test_v49_encoding));assert t.wasSuccessful()
    summary=[]
    for name in folders:
        folder=out/name
        for p,h in read(folder/'complete.json').items():assert sha(folder/p)==h,p
        bundle=joblib.load(folder/'model.joblib');ev=rows.inner_role.to_numpy()==2 if name=='pressure' else rows.fold.to_numpy()==int(name[-1]);fit=~ev
        assert bundle['text_encoder'].fit_rows==int(fit.sum())
        assert bundle['model'].C==c['C'] and bundle['model'].class_weight is None
        x=sparse.hstack([bundle['text_encoder'].transform(texts),bundle['fact_encoder'].transform(facts)],format='csr')
        q=bundle['model'].predict_proba(x);d=pd.read_parquet(folder/'evaluation.parquet')
        assert np.array_equal(d.row_position,rows.loc[ev,'row_position'])
        error=float(np.max(abs(q[ids[ev]]-d[['p_benign','p_malicious','p_suspicious']].to_numpy())));assert error<1e-10
        report=read(folder/'report.json');recomputed=metrics(d)
        for key in recomputed:assert recomputed[key]==report[key],key
        qr,audit=v.classify_records(bundle,rawcases);qv,_=v.classify_records(bundle,changes)
        compact=np.searchsorted(active,bypos.loc[[pos for pos,raw in native],'projection_id'].to_numpy())
        np.testing.assert_allclose(qr,q[compact],atol=1e-12,rtol=0);np.testing.assert_allclose(qr,qv,atol=1e-12,rtol=0)
        z=sum(a['nonempty_text_encoded_as_zero'] for a in audit)
        details=[]
        for (pos,raw),a,p in zip(native,audit,qr):
            if bypos.loc[pos,'route']=='authentication':
                details.append({'row_position':pos,'label_index':int(bypos.loc[pos,'label_index']),
                    'text_vector_zero':a['nonempty_text_encoded_as_zero'],'cold_fact_columns':a['facts_without_fit_activation'],'p_B_M_S':p.tolist()})
        save(folder/'authentication_coverage.json',details)
        summary.append({'model':name,'probability_replay_rows':len(d),'max_abs_difference':error,
            'real_raw_rows':len(native),'specified_variants':len(changes),'zero_text_warnings':z})
    save(out/'verification.json',{'passed':True,'replays':summary,'encoding_tests':t.testsRun,
        'quality_gates_passed':execution['pressure_all_gates_passed'],'fresh_blind_or_external_validation':False,
        'configuration_sha256':sha(out/'configuration.json')})
    print(json.dumps({'verified_models':len(summary),'quality_passed':execution['pressure_all_gates_passed']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','prepare_text','fit','verify']);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();globals()[a.stage](a.root.resolve(),a.out.resolve())
