"""Saved remaining wrong argmax versus zero local margin, no model calls."""
import json,traceback
from pathlib import Path
import numpy as np
import pandas as pd
from experiment_review import ROOT,read,sha,check_bindings
from v160_independent_saved_direction_certificate import dot
import v167_trial_point_restoration_diagnostic as entry

TRIAL=entry.OUT
OUT=ROOT/'artifacts/v167_saved_remaining_argmax_geometry_review_20261002'

def main():
    assert not OUT.exists() and all((TRIAL/f'role{r}/diagnostic.json').exists() for r in range(3));OUT.mkdir();seal=read(TRIAL/'run_seal.json');check_bindings(seal['source_sha256']);goldfile=ROOT/'data/official/train.parquet';gold=pd.read_parquet(goldfile,columns=['label_binary']).label_binary.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy();paths={Path(__file__).resolve(),ROOT/'training/v160_independent_saved_direction_certificate.py',ROOT/'training/v167_trial_point_measurement.py',goldfile}|{p for p in TRIAL.rglob('*') if p.is_file()};bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)};save=lambda p,v:entry.save(p,v);save(OUT/'pre_bindings.json',dict(source_sha256=bindings,official_calls=0));roles=[]
    for role in range(3):
        folder=TRIAL/f'role{role}';diag=read(folder/'diagnostic.json');ctx=entry.load_context(role);original_refs=read(entry.CONTROL/f'role{role}/joint_restoration/active_normal_references.json');stage_results=[]
        for stage,correction in enumerate(sorted(folder.glob('correction*'),key=lambda p:int(p.name[10:]))):
            proposal=folder/f'probe{stage}';assert proposal.is_dir();blocked=pd.read_parquet(proposal/'actual_blocking_original_rows.parquet');assert np.array_equal(blocked.truth,gold[blocked.row_position]);bad=entry.blocker_identities(ctx,blocked);refs=read(correction/'active_normal_references.json');assert bad<=set(refs)==set(original_refs);c=np.load(correction/'actual_trial_margins.npy');e=np.load(correction/'correction.npy');certificate=read(correction/'original_unit_restoration_review.json');records=[]
            for i,(identity,ref) in enumerate(refs.items()):
                if identity not in bad:continue
                meta=ref['metadata'];scope,local,truth,rival=[meta[k] for k in ['scope','local','truth','rival']];q=np.load(proposal/f'{scope}_q.npy')[local];lp=np.load(proposal/f'{scope}_logq.npy')[local];gradient=np.load(ROOT/ref['gradient']);change,error=dot(gradient,e);proof=certificate['inequality_reviews'][i];assert proof['passed'] and proof['linear_recovery']==change;prediction=float(c[i])+change;actual=float(lp[truth]-lp[rival]);qgap=float(q[truth]-q[rival]);pred=int(q.argmax());assert pred!=truth;original_group=blocked[(blocked.scope==scope)&blocked.local.eq(local)&blocked.truth.eq(truth)&blocked.rival.eq(rival)];assert len(original_group)>0
                records.append(dict(identity=identity,local=local,scope=scope,truth=truth,rival=rival,original_blocking_rows=len(original_group),predicted_log_probability_margin=prediction,actual_log_probability_margin=actual,actual_probability_margin=qgap,log_probability_margin_hex=float(actual).hex(),probability_margin_hex=qgap.hex(),probabilities=[float(v) for v in q],log_probabilities=[float(v) for v in lp],exact_probability_tie=bool(q[truth]==q[rival]),exact_log_probability_tie=bool(lp[truth]==lp[rival]),true_class_has_larger_index_than_rival=truth>rival,actual_argmax=pred,original_unit_inequality_passed=True,original_unit_residual_limit=proof['residual_limit'],actual_argmax_failure_not_ignored_by_zero_margin=True))
            stage_results.append(dict(stage=stage,blocking_original_rows=len(blocked),blocking_measured_functions=len(records),all_actual_functions_still_covered=True,all_actual_blocker_geometry=records,zero_local_margin_not_classifier_safety=True))
        roles.append(dict(role=role,status=diag['status'],actual_finite_accepted=diag['actual_finite_accepted'],stages=stage_results))
    check_bindings(bindings);summary=dict(status='V167_remaining_actual_argmax_failures_and_zero_margin_geometry_saved_exactly',roles=roles,official_heads=0,official_features=0,official_derivatives=0,fits=0,permanent_updates=0,no_new_floor_tolerance_or_third_correction_permission=True,source_sha256=bindings);save(OUT/'review.json',summary);print(json.dumps(dict(status='V167_saved_argmax_geometry_review_passed',role1=roles[1],official_calls=0)))

if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():entry.save(OUT/'failure.json',dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),official_calls=0))
        raise
