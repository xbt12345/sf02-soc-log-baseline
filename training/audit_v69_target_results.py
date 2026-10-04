"""Post-training input and fixed-target audit only; no model fitting/selection."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from run_v69_support_control import ARMS, ROOT
from v61_common import read, save, sha


def main(out):
    trace=pd.read_parquet(out/'raw_input_trace.parquet'); checks={}
    for side in ['src','dst']:
        raw=trace['parsed_'+side+'_port'].astype(str); observed=trace[side+'_port_observed'].astype(str)
        numeric=raw.str.fullmatch('[0-9]+') & pd.to_numeric(raw,errors='coerce').between(0,65535)
        assert not (numeric & raw.ne(observed)).any()
        assert not (~numeric & observed.ne('__MISSING__')).any()
        checks[side]={'numeric_raw_ports':int(numeric.sum()),'numeric_values_preserved':True,
                      'opaque_not_guessed':True,'raw_missing_or_opaque':int((~numeric).sum())}
    targets=pd.read_csv(out/'target_578_results.csv')
    old=pd.read_parquet(ROOT/'artifacts/v67_delivery_tables_20260920/target_578_ledger.parquet')
    assert np.array_equal(targets.row_position,old.row_position)
    for column in ['evidence_bucket','normalized_text','raw_sha256']:
        targets[column]=old[column].to_numpy()
    actual=pd.read_parquet(out/'target_actual_model_inputs.parquet')
    result={'input_port_checks':checks,'arms':{},'script_sha256':sha(__file__),
            'scope':'Primary preregistered 1% calibration-budget predictions, original targets only; descriptive not causal assignment',
            'any_union_NOT_A_MODEL':int(targets[[a+'_pred' for a in ARMS]].eq(2).any(axis=1).sum())}
    evidence=[]
    for arm in ARMS:
        subset=actual[actual.arm.eq(arm)].set_index('row_position').loc[targets.row_position]
        m=subset.same_actual_matrix_train_M_sources.to_numpy();s=subset.same_actual_matrix_train_S_sources.to_numpy()
        correct=targets[arm+'_pred'].eq(2).to_numpy()
        targets[arm+'_same_matrix_M_sources']=m;targets[arm+'_same_matrix_S_sources']=s
        buckets=[]
        for bucket in old.evidence_bucket.unique():
            mask=old.evidence_bucket.eq(bucket).to_numpy()
            buckets.append({'bucket':bucket,'rows':int(mask.sum()),'repaired':int((mask&correct).sum()),
                            'remaining':int((mask&~correct).sum())})
        result['arms'][arm]={'target_repaired':int(correct.sum()),'remaining':int((~correct).sum()),
            'repaired_source_symbols':int(targets.loc[correct,'group'].nunique()),
            'raw_texts_repaired':int(targets.loc[correct,'raw_sha256'].nunique()),'buckets':buckets,
            'same_actual_matrix_as_training_M':int((m>0).sum()),'same_actual_matrix_as_training_S':int((s>0).sum()),
            'same_actual_matrix_as_both':int(((m>0)&(s>0)).sum()),
            'repaired_by_fold':{str(k):int((correct&targets.fold.eq(k).to_numpy()).sum()) for k in range(3)}}
        for (fold,group),d in targets.assign(correct=correct).groupby(['fold','group']):
            evidence.append({'arm':arm,'fold':int(fold),'group':int(group),'rows':len(d),'correct':int(d.correct.sum())})
    targets.to_csv(out/'target_578_evidence.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(evidence).to_csv(out/'target_source_breakdown.csv',index=False,encoding='utf-8-sig')
    save(out/'input_and_target_audit.json',result);print(result,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();main(a.out.resolve())
