"""All25 local floor predictions versus actual classifier outputs, no model calls."""
import json,traceback
from pathlib import Path
import numpy as np
import pandas as pd
from experiment_review import ROOT,read,sha,check_bindings
from v160_independent_saved_direction_certificate import dot
import v168_decision_floor_diagnostic_v2 as entry

OUT=ROOT/'artifacts/v168_saved_actual_floor_geometry_review_20261002'

def main():
    trial=entry.OUT;folder=trial/'role1';assert not OUT.exists() and (folder/'diagnostic.json').exists();seal=read(trial/'run_seal.json');check_bindings(seal['source_sha256']);diag=read(folder/'diagnostic.json');assert diag['exception'] is None;goldfile=ROOT/'data/official/train.parquet';gold=pd.read_parquet(goldfile,columns=['label_binary']).label_binary.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy();paths={Path(__file__).resolve(),goldfile,ROOT/'training/v160_independent_saved_direction_certificate.py',ROOT/'training/v168_decision_floor_diagnostic_v2.py'}|{p for p in trial.rglob('*') if p.is_file()};bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)};OUT.mkdir();entry.save(OUT/'pre_bindings.json',dict(source_sha256=bindings,official_calls=0));functions=[];blocked_rows=None
    correction=folder/'correction0';certificate=read(correction/'original_unit_restoration_review.json');refs=read(correction/'active_normal_references.json');assert len(refs)==25;proposal=folder/'probe0'
    if diag['finite_proposals']:
        blocked=pd.read_parquet(proposal/'actual_blocking_original_rows.parquet');assert np.array_equal(blocked.truth,gold[blocked.row_position]);ctx=entry.load_context(1);bad=entry.blocker_identities(ctx,blocked);blocked_rows=len(blocked);c=np.load(correction/'actual_trial_margins.npy');floors=np.load(correction/'prospective_decision_floors.npy');e=np.load(correction/'correction.npy')
        for i,(identity,ref) in enumerate(refs.items()):
            meta=ref['metadata'];scope,local,truth,rival=[meta[k] for k in ['scope','local','truth','rival']];q=np.load(proposal/f'{scope}_q.npy')[local];lp=np.load(proposal/f'{scope}_logq.npy')[local];g=np.load(ROOT/ref['gradient']);change,error=dot(g,e);numeric=certificate['inequality_reviews'][i];assert numeric['linear_recovery']==change;prediction=float(c[i])+change;actual=float(lp[truth]-lp[rival]);qgap=float(q[truth]-q[rival]);pred=int(q.argmax());functions.append(dict(identity=identity,scope=scope,local=local,truth=truth,rival=rival,prospective_local_floor=float(floors[i]),trial_log_margin=float(c[i]),predicted_log_margin=prediction,actual_log_margin=actual,actual_probability_margin=qgap,actual_log_margin_hex=actual.hex(),actual_probability_margin_hex=qgap.hex(),actual_probability_values=q.tolist(),actual_log_probability_values=lp.tolist(),actual_argmax=pred,actual_truth_correct=pred==truth,actual_exact_q_tie=bool(q[truth]==q[rival]),actual_exact_logq_tie=bool(lp[truth]==lp[rival]),original_unit_inequality_passed=numeric['passed'],original_unit_residual_limit=numeric['residual_limit'],direction_dot_arithmetic_error=error,is_actual_protected_blocking_identity=identity in bad))
    check_bindings(bindings);entry.save(OUT/'review.json',dict(status='V168_all_measured_function_local_floor_predictions_and_actual_argmax_geometry_reviewed',actual_status=diag['status'],actual_candidate_accepted=diag['actual_finite_accepted'],functions=functions,complete_measured_functions=25,actual_finite_proposals=diag['finite_proposals'],actual_protected_blocking_original_rows=blocked_rows,previous_actual_tie_function_geometry=[f for f in functions if f['scope']=='OOF' and f['local']==21985 and f['truth']==2 and f['rival']==1],uncovered_actual_blocking_identities=sorted(bad-set(refs)) if diag['finite_proposals'] else [],local_floor_does_not_override_original_classifier_acceptance=True,official_heads=0,official_features=0,official_derivatives=0,fits=0,permanent_updates=0,no_new_correction_or_training_permission=True,source_sha256=bindings));print(json.dumps(dict(status='V168_saved_floor_geometry_review_passed',actual_accepted=diag['actual_finite_accepted'],official_calls=0)))

if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():entry.save(OUT/'failure.json',dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),official_calls=0))
        raise
