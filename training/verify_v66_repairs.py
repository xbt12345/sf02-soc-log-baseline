"""Postcondition checks of actual predictions, split repair and old artifact preservation."""
import json
import numpy as np
import pandas as pd
import joblib
from train_v65_rank_heads import ROOT,load_fit
from v61_common import read,save,sha
from v66_selection import select_both
from audit_v66_endpoint_mapping import bijection_mask


def main():
    out=ROOT/'artifacts/v66_issue_resolution_20260920';v65=ROOT/'artifacts/v65_rank_heads_20260920'
    old=read(ROOT/'evidence/2026-09-20/v65_rank_heads/delivery.json')
    for path,digest in old['artifact_sha256'].items():assert sha(ROOT/path)==digest,path
    f,_=load_fit();codes=pd.Index(pd.read_parquet(v65/'unique_text.parquet').text).get_indexer(f.text)
    X=np.load(v65/'upstream_text_features.npy')[codes];y=f.label.eq(0).to_numpy()
    replay=read(out/'selection_replay.json')
    for fold in range(3):
        r=v65/f'fold{fold}';actual=select_both({arm:read(r/arm/'curve.json') for arm in ['H0','H1']},read(r/'references_before_H1.json')['references'])
        assert actual==replay[str(fold)]
    original=pd.read_parquet(v65/'H0_guarded_oof.parquet');base=original[['p_B','p_M','p_S']].to_numpy().argmax(1)
    models=0;maxdiff=0.
    for arm in ['unweighted_gate','balanced_gate']:
        d=pd.read_parquet(out/'normal_gate'/f'{arm}_oof.parquet');pred=base.copy()
        for fold in range(3):
            train=np.flatnonzero(f.fold.ne(fold));val=np.flatnonzero(f.fold.eq(fold))
            model=joblib.load(out/'normal_gate'/f'{arm}_fold{fold}.joblib')
            np.testing.assert_allclose(model.named_steps['scale'].mean_,X[train].mean(0,dtype=np.float64),rtol=1e-12,atol=1e-12)
            p=model.predict_proba(X[val])[:,1];np.testing.assert_allclose(p,d.normal_score.iloc[val],rtol=1e-12,atol=1e-14)
            maxdiff=max(maxdiff,float(abs(p-d.normal_score.iloc[val].to_numpy()).max()))
            assert np.array_equal(p>=.5,d.normal_score.iloc[val].to_numpy()>=.5)
            pred[val[p>=.5]]=0;models+=1
        assert np.array_equal(pred,d.pred)
        assert (pred[y]==0).all() and np.array_equal(pred[~y],base[~y])
    m=pd.read_parquet(out/'supported_fold_manifest.parquet')
    pd.testing.assert_frame_equal(f[['row_position','label','group','body_group']],m[['row_position','label','group','body_group']])
    assert m.groupby('group').fold.nunique().max()==1 and m.groupby('body_group').fold.nunique().max()==1
    d=pd.read_parquet(out/'supported_normal_gate/oof.parquet')
    for fold in range(3):
        train=np.flatnonzero(m.fold.ne(fold));val=np.flatnonzero(m.fold.eq(fold));assert y[val].any()
        model=joblib.load(out/'supported_normal_gate'/f'fold{fold}.joblib')
        np.testing.assert_allclose(model.named_steps['scale'].mean_,X[train].mean(0,dtype=np.float64),rtol=1e-12,atol=1e-12)
        p=model.predict_proba(X[val])[:,1];np.testing.assert_allclose(p,d.normal_score.iloc[val],rtol=1e-12,atol=1e-14);models+=1
        maxdiff=max(maxdiff,float(abs(p-d.normal_score.iloc[val].to_numpy()).max()))
        assert np.array_equal(p>=.5,y[val])
    mapping=pd.read_parquet(out/'endpoint_mapping_private.parquet')
    for side in ['src','dst']:
        subset=mapping[mapping[side+'_ip'].fillna('').ne('')]
        expected,_,_=bijection_mask(subset,side+'_ip','body_'+side)
        assert np.array_equal(expected,subset[side+'_mapping_eligible'])
    for name,script in [('audit.json','audit_v66_failures.py'),('context_coverage.json','audit_v66_context_coverage.py'),
                         ('endpoint_mapping.json','audit_v66_endpoint_mapping.py'),('context_bridge.json','audit_v66_context_bridge.py'),
                         ('supported_folds.json','prepare_v66_supported_folds.py')]:
        assert read(out/name)['source_sha256']==sha(ROOT/'training'/script)
    for path in [out/'normal_gate/preregistered.json',out/'supported_normal_gate/preregistered.json']:
        source='probe_v66_normal_gate.py' if path.parent.name=='normal_gate' else 'replicate_v66_normal_gate.py'
        assert read(path)['source_sha256']==sha(ROOT/'training'/source)
    result={'verification_passed':True,'quality_acceptance':False,'previous_v65_bound_files_unchanged':len(old['artifact_sha256']),
      'same_selector_and_normal_guards_replayed_folds':3,'normal_models_restored':models,
      'max_probability_difference':maxdiff,'rows_per_three_model_gate':len(f),
      'new_manifest_no_group_crossing':True,'normal_controls_errors_after_repair':0,'additional_threat_to_B':0,
      'scope':'Actual normal-gate reload and prediction replay, fit-only scaling, split and endpoint-quarantine checks. No new M/S model or external quality acceptance.',
      'source_sha256':sha(__file__)}
    save(out/'verification.json',result);print(result)


if __name__=='__main__':main()
