"""Extend the actual convex-family counterexample to all original ASA rows."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiment_review import ROOT,read,sha,check_bindings

OUT=ROOT/'artifacts/v158_full_bank_capacity_case_replay_20261001'
BANK=ROOT/'artifacts/v158_legal_fusion_bank_v2_20261001'
INDEPENDENT=ROOT/'artifacts/v158_independent_fusion_result_audit_v2_20261001'
CASE=ROOT/'training/review_policy/v158_full_bank_capacity_cases.json'

def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    sources=[Path(__file__).resolve(),INDEPENDENT/'audit.json',INDEPENDENT/'all_original_ASA_initial_endpoint_and_capacity.parquet',
        ROOT/'artifacts/v158_fusion_trial_20261001/ASA_prediction_ledger.parquet']
    sources.extend(BANK/f'fold{f}/deployment_probabilities.npy' for f in range(3))
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sources}
    if OUT.exists():check_bindings(read(OUT/'pre_saved_array_bindings.json')['source_sha256'])
    else:
        OUT.mkdir();save(OUT/'pre_saved_array_bindings.json',dict(status='bound_before_full_bank_saved_arithmetic',source_sha256=bindings))
    frame=pd.read_parquet(sources[3]);ind=pd.read_parquet(sources[2]);parent=read(sources[1])
    assert np.array_equal(frame.row_position,ind.row_position) and np.array_equal(frame.truth,ind.truth)
    assert len(frame)==112807 and frame.row_position.is_unique
    strict=np.zeros(len(frame),bool);anycorrect=np.zeros(len(frame),bool);best=np.zeros(len(frame),np.float64)
    for f in range(3):
        mask=frame.fold.eq(f).to_numpy();part=frame[mask];q=np.load(BANK/f'fold{f}/deployment_probabilities.npy')[part.local]
        y=part.truth.to_numpy();other=3-y;assert np.isin(y,[1,2]).all()
        margin=q[np.arange(len(q))[:,None],np.arange(17)[None,:],y[:,None]]-q[np.arange(len(q))[:,None],np.arange(17)[None,:],other[:,None]]
        strict[mask]=(margin<0).all(1);best[mask]=margin.max(1);anycorrect[mask]=(q.argmax(2)==y[:,None]).any(1)
    assert np.array_equal(strict,ind.strict_convex_wrong_margin)
    assert np.array_equal(anycorrect,ind.some_expert_correct)
    assert np.array_equal(best,ind.best_true_vs_other_expert_margin)
    assert np.all(frame.loc[strict,'pred_B'].ne(frame.loc[strict,'truth']))
    result={}
    for c in [1,2]:
        mask=frame.truth.eq(c).to_numpy();wrong=frame.pred_B.ne(frame.truth).to_numpy()
        result[str(c)]=dict(original_rows=int(mask.sum()),all17_strict_wrong_margin_rows=int((mask&strict).sum()),
            errors_with_some_correct_expert=int((mask&wrong&anycorrect).sum()),errors_without_correct_expert=int((mask&wrong&~anycorrect).sum()))
    assert result==parent['convex_capacity_diagnostic']
    assert result['1']==dict(original_rows=78748,all17_strict_wrong_margin_rows=148,errors_with_some_correct_expert=266,errors_without_correct_expert=148)
    assert result['2']==dict(original_rows=34059,all17_strict_wrong_margin_rows=979,errors_with_some_correct_expert=893,errors_without_correct_expert=979)
    check_bindings(bindings)
    case=dict(version='V158_actual_whole_bank_convex_capacity_case',source_sha256=bindings,
        fixed_bank_scope='All112807 original ASA rows with own-outer-fold deployed17 probabilities.',
        observed=result,proof='For each strict case and every expert, p_true-p_other<0; every nonnegative normalized combination has negative true-versus-other margin.',
        required_action='Do not claim additional iterations, new scoring conditions, or convex-weight tuning can repair these fixed-bank cases; do not remove them from quality denominators.',
        limits='Model-family representability only; not raw-input indistinguishability, causal identification, or authorization for new fits.')
    if CASE.exists():assert read(CASE)==case
    else:save(CASE,case)
    receipt=dict(status='actual_full_ASA_convex_family_capacity_counterexample_replayed',official_classifier_calls=0,new_features=0,
        new_gradients=0,new_fits=0,new_updates=0,classes=result,strict_cases=1127,quality_acceptance=False,model_promoted=False,
        source_bindings_sha256=sha(OUT/'pre_saved_array_bindings.json'),case_sha256=sha(CASE))
    if (OUT/'replay.json').exists():assert read(OUT/'replay.json')==receipt
    else:
        part=frame.assign(strict_convex_wrong_margin=strict,some_expert_correct=anycorrect,best_true_vs_other_expert_margin=best)
        part.to_parquet(OUT/'all_original_ASA_capacity_cases.parquet',index=False);save(OUT/'replay.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
