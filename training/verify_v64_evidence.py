"""Recompute selection inputs from every saved prediction, then verify new artifacts."""
import json
import unittest
import warnings
import numpy as np
import pandas as pd
from audit_v64_evidence import ROOT,DATA,PREV,pure_rule
from v61_common import read,save,sha
from train_v63_factorial import metric_summary
from v64_selection import flatten,select


def main():
    out=ROOT/'artifacts/v64_evidence_20260920'
    report=read(out/'summary.json');protocol=read(out/'protocol.json')
    assert sha(ROOT/'training/audit_v64_evidence.py')==protocol['source_sha256']
    assert sha(ROOT/'training/v64_selection.py')==protocol['selection_source_sha256']
    for name,digest in protocol['inputs'].items():assert sha(ROOT/name)==digest
    for name,digest in report['outputs'].items():assert sha(out/name)==digest
    frame=pd.read_parquet(DATA/'records.parquet',filters=[('role','==','selection')]).sort_values('row_position').reset_index(drop=True)
    replay=read(out/'selection_replay.json');checked=0
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='The y_pred values do not sum to one')
        for arm in ['A0','A1','P0','P1']:
            curve=read(PREV/arm/'curve.json')
            for item in curve:
                d=pd.read_parquet(PREV/arm/item['predictions_file']).sort_values('row_position').reset_index(drop=True)
                assert d[['row_position','label','group']].equals(frame[['row_position','label','group']])
                prob=d[['p_B','p_M','p_S']].to_numpy()
                assert np.isfinite(prob).all() and (prob>=0).all() and np.max(np.abs(prob.sum(1)-1))<2e-6
                actual=flatten(metric_summary(frame,prob));expected=flatten(item['metrics'])
                np.testing.assert_allclose(list(actual.values()),list(expected.values()),rtol=0,atol=1e-12)
                checked+=1
            assert select(curve,replay['references'])==replay['arms'][arm]
    p=pd.read_parquet(out/'context_probe.parquet')
    assert p[['row_position','label','group']].equals(frame[['row_position','label','group']])
    table=pd.read_parquet(out/'context_shape_support.parquet').set_index('context_shape')
    rules={}
    for c,col,other in [(1,'M_sources','S_sources'),(2,'S_sources','M_sources')]:
        rules.update({k:c for k in table.index[(table[col]>=3)&table[other].eq(0)]})
    override=p.context_shape.map(rules)
    actual=np.where(override.notna(),override.fillna(1),p.baseline_prediction).astype(int)
    np.testing.assert_array_equal(actual,p.prediction)
    assert p.rule_applied.tolist()==override.notna().tolist()
    recomputed=metric_summary(frame,np.eye(3)[actual])
    assert recomputed==report['context_probe']['candidate']
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT/'training'),pattern='test_v64*.py'))
    assert result.wasSuccessful()
    save(out/'verification.json',{'all_checks_passed':True,'quality_acceptance':False,
         'implementation_tests':result.testsRun,'old_checkpoints_recomputed_from_predictions':checked,
         'rows_per_checkpoint':len(frame),'new_context_probe_decisions_replayed':len(p),
         'new_context_probe_changed_decisions':int(p.prediction.ne(p.baseline_prediction).sum()),
         'source_sha256':sha(__file__),'scope':'Saved-prediction metric replay and rule replay; no neural weight replay or new blind evaluation in v6.4.'})
    print('V64_VERIFICATION_COMPLETE',checked,flush=True)


if __name__=='__main__':main()
