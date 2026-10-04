"""Independent v50 verifier; frozen v49 inference intentionally rejects v50."""
import argparse
import sys
import unittest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy import sparse

def main(root,out):
    sys.path.insert(0,str(out/'runtime'))
    import v48_input as adapter
    import v50_views as v
    import test_v50_views
    from run_v48 import read,save,sha,raw_rows
    from run_v49 import data,metrics
    c=read(out/'configuration.json')
    for p,h in c['source_bindings'].items():assert sha(root/p)==h,p
    for p,h in c['runtime_bindings'].items():assert sha(out/'runtime'/p)==h,p
    for p,h in c['preparation_bindings'].items():assert sha(out/p)==h,p
    if (out/'verification.json').exists():raise FileExistsError('Verification already exists')
    tests=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromModule(test_v50_views));assert tests.wasSuccessful()
    rows,ids,texts,facts=data(root);execution=read(out/'execution.json');results=[]
    raw=list(raw_rows(root/'data/official/train.parquet',rows.loc[rows.route.isin(['native_flow','native_firewall','authentication']),'row_position']))
    bypos=rows.set_index('row_position');active=np.unique(rows.projection_id);records=[];variants=[]
    for pos,message in raw:
        parsed=adapter.prepare_record({'message_sanitized':message});variant=message
        k=int(np.searchsorted(active,bypos.loc[pos,'projection_id']))
        assert parsed['facts']==facts[k] and parsed['text']==texts[k]
        if parsed['route'] in ('native_flow','native_firewall'):
            m=adapter.NATIVE.fullmatch(message.strip());variant=message.strip()
            for key in sorted(adapter.IDENTITIES-{'priority'},key=lambda k:m.start(k),reverse=True):
                if m[key] is not None:
                    a,b=m.span(key);variant=variant[:a]+'OPAQUE-identity'+variant[b:]
        records.append(parsed)
        variants.append(adapter.prepare_record({'message_sanitized':variant,'timestamp':'2099','product_name':None,'label_binary':'unused','src_ip':'203.0.113.5'}))
    def matrix(b,rr):return sparse.hstack([b['text_encoder'].transform([r['text'] for r in rr]),b['fact_encoder'].transform([r['facts'] for r in rr])],format='csr')
    compact=np.searchsorted(active,bypos.loc[[p for p,_ in raw],'projection_id'].to_numpy())
    for name in execution['models']:
        folder=out/name
        for p,h in read(folder/'complete.json').items():assert sha(folder/p)==h,p
        bind=read(folder/'binding.json');fold=bind['fold'];fit=rows.inner_role.to_numpy()!=2 if fold is None else rows.fold.to_numpy()!=fold;ev=~fit
        assert not set(rows.loc[fit,'body_group'])&set(rows.loc[ev,'body_group'])
        if fold is None:assert not set(rows.loc[fit,'union_group'])&set(rows.loc[ev,'union_group'])
        m=pd.read_parquet(folder/'training_view_manifest.parquet');tr=rows.loc[fit].set_index('row_position')
        assert set(m.row_position)==set(tr.index)
        np.testing.assert_array_equal(m.label_index,tr.loc[m.row_position,'label_index']);assert (m.groupby('row_position').weight.sum()==1).all()
        if not bind['control']:
            views,indices,targets,weights,origin,eligible=v.training_views(facts,ids[fit],rows.label_index.to_numpy()[fit])
            for col,value in [('input_index',indices),('label_index',targets),('weight',weights)]:np.testing.assert_array_equal(m[col],value)
            assert views==bind['partial_views']
            np.testing.assert_array_equal(m.row_position,rows.loc[fit,'row_position'].to_numpy()[origin])
        b=joblib.load(folder/'model.joblib');assert b['version']==v.VERSION
        expected=c['pressure_configuration_sha256'] if fold is None else sha(out/'configuration.json')
        assert b['configuration_sha256']==expected==bind['configuration_sha256']
        assert b['text_encoder'].fit_rows==int(fit.sum()) and b['model'].C==c['C'] and b['model'].class_weight is None
        x=sparse.hstack([b['text_encoder'].transform(texts),b['fact_encoder'].transform(facts)],format='csr');q=b['model'].predict_proba(x)
        d=pd.read_parquet(folder/'evaluation.parquet');np.testing.assert_array_equal(d.row_position,rows.loc[ev,'row_position']);np.testing.assert_array_equal(d.label_index,rows.loc[ev,'label_index'])
        diff=float(abs(q[ids[ev]]-d[['p_benign','p_malicious','p_suspicious']].to_numpy()).max());assert diff<1e-12
        report=read(folder/'report.json')
        for k,value in metrics(d).items():assert report[k]==value,k
        qr=b['model'].predict_proba(matrix(b,records));qv=b['model'].predict_proba(matrix(b,variants))
        np.testing.assert_allclose(qr,q[compact],atol=1e-12,rtol=0);np.testing.assert_allclose(qr,qv,atol=1e-12,rtol=0)
        results.append({'model':name,'probability_replay_rows':len(d),'max_difference':diff,'fit_rows_weight_and_labels_verified':int(fit.sum()),'raw_rows':len(raw),'specified_variants':len(variants)})
        print('Verified '+name,flush=True)
    save(out/'verification.json',{'passed':True,'models':results,'unit_tests':tests.testsRun,'source_sha256':sha(Path(__file__)),
        'quality_pressure_passed':execution['pressure_all_gates_passed'],'quality_primary_passed':execution['primary_all_gates_passed'],
        'scope':'Replay, source identity, original labels/weights and specified raw invariance. Not external or blind validation.'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.root.resolve(),a.out.resolve())
