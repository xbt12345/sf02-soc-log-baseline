"""Fresh-process full probability replay and raw-input boundary verification."""
import argparse
import collections
import json
import sys
import unittest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
import v48_input as v
from run_v48 import sha,read,save,raw_rows,compare

def main(root,out):
    if (out/'verification.json').exists():raise FileExistsError('Do not overwrite verification')
    c=read(out/'configuration.json')
    for name,h in c['source_bindings'].items():assert sha(root/name)==h,name
    for name,h in c['runtime_bindings'].items():assert sha(out/'runtime'/name)==h,name
    for name,h in c['prepared_files'].items():assert sha(out/name)==h,name
    import test_v48_input
    result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromModule(test_v48_input))
    assert result.wasSuccessful()
    rows=pd.read_parquet(out/'rows.parquet');p=pd.read_parquet(out/'projections.parquet')
    original=pd.read_parquet(root/'artifacts/v39_local_r2_20260913/prepared/rows.parquet')
    original=original[original.inner_role>=0].sort_values('row_position').reset_index(drop=True)
    for key in original.columns:
        if key!='projection_id':assert rows[key].equals(original[key]),key
    assert rows.old_projection_id.equals(original.projection_id)
    changes=pd.read_parquet(out/'input_changes.parquet')
    assert set(changes.row_position)==set(rows.loc[rows.projection_id!=rows.old_projection_id,'row_position'])
    oldp=pd.read_parquet(root/'artifacts/v39_local_r2_20260913/prepared/projections.parquet')
    for item in changes.drop_duplicates(['old_projection_id','projection_id']).itertuples():
        a=json.loads(oldp.iloc[item.old_projection_id].facts);b=json.loads(p.iloc[item.projection_id].facts)
        assert p.iloc[item.projection_id].text==oldp.iloc[item.old_projection_id].text
        assert 'action' not in a and 'outcome' not in a
        assert b.pop('action')=='deny' and b.pop('outcome')=='blocked' and a==b
    # Source-independent counts of the observed code, not the adapter's own audit.
    selected=rows[rows.route.isin(['native_flow','native_firewall'])]
    native=list(raw_rows(root/'data/official/train.parquet',selected.row_position))
    seen=[];variants=[];positions=[];raw_audits={}
    bypos=rows.set_index('row_position');state=collections.Counter()
    import re
    for pos,raw in native:
        r=v.prepare_message(raw);row=bypos.loc[pos];cached=p.iloc[int(row.projection_id)]
        assert r['text']==cached.text and v.old.canonical(r['facts'])==v.old.canonical(json.loads(cached.facts))
        seen.append({'message_sanitized':raw});positions.append(pos);raw_audits[pos]=r['information_audit']['raw_sha256']
        m=v.NATIVE.fullmatch(raw.strip());changed=raw.strip()
        for key in sorted(v.IDENTITIES-{'priority'},key=lambda k:m.start(k),reverse=True):
            if m[key] is not None:
                a,b=m.span(key);changed=changed[:a]+'OPAQUE-identity'+changed[b:]
        other=v.prepare_message(changed)
        assert (other['text'],other['facts'])==(r['text'],r['facts'])
        variants.append({'message_sanitized':changed,'timestamp':'2099','product_name':'different',
                         'label_binary':'unused','src_ip':'unrelated'})
        match=re.search(r'\bpattern:\s*([01])\s+(?:all|dst\s+(?:\d{1,3}\.){3}\d{1,3})\s*$',raw)
        if match and row.old_projection_id!=row.projection_id:
            assert match[1]=='1';state['independently_confirmed_restored_pattern_1']+=1
        for f in r['information_audit']['fields']:
            a,b=f['span'];assert raw[a:b]==f['raw']
    assert state['independently_confirmed_restored_pattern_1']==3188
    for record in changes.itertuples():assert raw_audits[record.row_position]==record.raw_sha256
    run=read(out/'execution.json');folders=['pressure']+(['fold_'+str(f) for f in range(3)] if run['primary_regression_executed'] else [])
    replay=[];coverage=[]
    active,ids=np.unique(rows.projection_id.to_numpy(),return_inverse=True)
    unique=p.iloc[active];texts=unique.text.tolist();facts=[json.loads(s) for s in unique.facts]
    for name in folders:
        folder=out/name
        for file,h in read(folder/'complete.json').items():assert sha(folder/file)==h
        b=joblib.load(folder/'model.joblib');fit=rows.inner_role.to_numpy()!=2 if name=='pressure' else rows.fold.to_numpy()!=int(name[-1]);ev=~fit
        assert b['configuration_sha256']==sha(out/'configuration.json')
        assert b['text_encoder'].fit_rows==int(fit.sum())
        assert b['model'].C==0.1 and b['model'].class_weight is None
        assert not set(rows.loc[fit,'body_group'])&set(rows.loc[ev,'body_group'])
        tx=b['text_encoder'].transform(texts);fx=b['fact_encoder'].transform(facts)
        x=sparse.hstack([tx,fx],format='csr');prob=b['model'].predict_proba(x)
        saved=pd.read_parquet(folder/'evaluation.parquet').sort_values('row_position').reset_index(drop=True)
        assert np.array_equal(saved.row_position,rows.loc[ev,'row_position'])
        difference=float(np.max(abs(prob[ids[ev]]-saved[['p_benign','p_malicious','p_suspicious']].to_numpy())))
        assert difference<1e-10
        reconstructed,bad=compare(saved);report=read(folder/'report.json')
        for key in ['before','after','fixes','regressions','routes']:assert reconstructed[key]==report[key],key
        # Each frozen baseline is reloaded and replayed on its original inputs.
        olddir=root/'artifacts/v39_local_r2_20260913'/('old_protocol_stress' if name=='pressure' else 'primary/fold_'+name[-1]+'/SEMANTIC')
        baseline=joblib.load(olddir/'model.joblib')
        oldids=rows.old_projection_id.to_numpy()[ev];op=oldp.iloc[np.unique(oldids)]
        opids=np.searchsorted(np.unique(oldids),oldids)
        ox=sparse.hstack([baseline['text_encoder'].transform(op.text.tolist()),baseline['fact_encoder'].transform([json.loads(s) for s in op.facts])],format='csr')
        baseline_diff=float(np.max(abs(baseline['model'].predict_proba(ox)[opids]-saved[['b_benign','b_malicious','b_suspicious']].to_numpy())))
        assert baseline_diff<1e-10
        q,audit=v.classify_records(b,seen);q2,_=v.classify_records(b,variants)
        np.testing.assert_allclose(q,q2,rtol=0,atol=1e-12)
        compact=np.searchsorted(active,bypos.loc[positions,'projection_id'].to_numpy())
        np.testing.assert_allclose(q,prob[compact],rtol=0,atol=1e-12)
        # Coverage is measured after the actual trained encoders, not from JSON names.
        nonempty=np.array([bool(t) for t in texts]);zero=(tx.getnnz(axis=1)==0)
        missing_text=nonempty[ids[ev]]&zero[ids[ev]]
        fitmax=np.asarray(abs(fx[np.unique(ids[fit])]).max(axis=0).toarray()).ravel()
        cold=np.flatnonzero(fitmax==0)
        evalcounts=np.bincount(ids[ev],minlength=len(texts))
        counts=np.asarray((fx[:,cold]!=0).T.dot(evalcounts)).ravel()
        coldfeatures=[{'feature':str(b['fact_encoder'].names()[k]),'evaluation_rows':int(n)} for k,n in zip(cold,counts) if n]
        coverage.append({'model':name,'nonempty_text_encoded_as_zero_evaluation_rows':int(missing_text.sum()),
            'nonempty_text_encoded_as_zero_by_route':saved.loc[missing_text].groupby('route').size().to_dict(),
            'fact_dimensions_present_only_at_evaluation':coldfeatures,
            'caveat':'A nonzero vector does not prove all meanings are preserved; evaluation-only columns have no fitted signal.'})
        replay.append({'model':name,'evaluation_rows':len(saved),'new_model_max_abs_diff':difference,'baseline_max_abs_diff':baseline_diff,
                       'real_native_raw_predictions':len(seen),'native_wrapper_variant_predictions':len(variants)})
        print(json.dumps({'stage':'verified','model':name,'evaluation_rows':len(saved)}),flush=True)
    save(out/'encoding_coverage.json',coverage)
    save(out/'verification.json',{'all_checks_passed':True,'model_replays':replay,'input_changes_verified':len(changes),
        'official_labels_and_folds_unchanged':True,'boundary_tests':result.testsRun,'semantic_pattern_independent_count':dict(state),
        'same_training_and_inference_adapter':True,'external_validation':False,'deployment_ready':False,
        'configuration_sha256':sha(out/'configuration.json'),'verifier_sha256':sha(Path(__file__))})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.root.resolve(),a.out.resolve())
