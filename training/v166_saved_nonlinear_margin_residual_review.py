"""Exact saved original-unit linear/finite margin residuals, zero head calls."""
import json,traceback
from pathlib import Path
import numpy as np
from experiment_review import ROOT,read,sha,check_bindings
from v160_independent_saved_direction_certificate import dot,norm
from v159_float64_repeat_policy_v2 import EPS,STEP_EPS

TRIAL=ROOT/'artifacts/v166_coverage_first_diagnostic_20261002'
CONTROL=ROOT/'artifacts/v165_fixed_endpoint_decision_floor_diagnostic_20261002'
OUT=ROOT/'artifacts/v166_saved_nonlinear_margin_residual_review_20261002'

def save(path,value):path.write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))

def main():
    assert not OUT.exists();assert all((TRIAL/f'role{r}/diagnostic.json').is_file() for r in range(3));OUT.mkdir();seal=read(TRIAL/'run_seal.json');check_bindings(seal['source_sha256']);plan=read(ROOT/seal['plan_path']);paths={Path(__file__).resolve(),ROOT/'training/v160_independent_saved_direction_certificate.py',ROOT/'training/v159_float64_repeat_policy_v2.py'}|{p for p in TRIAL.rglob('*') if p.is_file()};blocker_review=ROOT/'artifacts/v166_saved_residual_blocker_identity_review_20261002/review.json';observed=read(blocker_review);check_bindings(observed['source_sha256']);paths.add(blocker_review)
    for spec in plan['roles']:
        folder=TRIAL/f"role{spec['role']}";refs=read(folder/'joint_restoration/active_normal_references.json');paths.update(ROOT/v['gradient'] for v in refs.values());paths.update(CONTROL/f"role{spec['role']}/treatment/{scope}_logq.npy" for scope in ['OOF','deployment'])
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)};save(OUT/'pre_bindings.json',dict(source_sha256=bindings,official_calls=0));roles=[]
    for spec in plan['roles']:
        role=spec['role'];folder=TRIAL/f'role{role}';joint=folder/'joint_restoration';refs=read(joint/'active_normal_references.json');certificate=read(joint/'original_unit_restoration_review.json');assert read(folder/'diagnostic.json')['candidate'] is not None;u=np.load(joint/'current_displacement.npy');e=np.load(joint/'correction.npy');v=np.load(joint/'displacement.npy');assert np.array_equal(v,np.load(folder/'treatment/direction.npy'));assert np.array_equal(v,u+e);old_norm=norm(u);new_norm=norm(v);correction_norm=norm(e)
        saved_b=np.load(joint/'actual_origin_margins.npy');saved_c=np.load(joint/'actual_current_margins.npy');logs={scope:np.load(folder/f'baseline/{scope}_logq.npy') for scope in ['OOF','deployment']};actual={scope:np.load(folder/f'treatment/{scope}_logq.npy') for scope in logs};control={scope:np.load(CONTROL/f'role{role}/treatment/{scope}_logq.npy') for scope in logs};rows=[]
        bad_ids=set(observed['roles'][role]['known_bad_input_identities'])
        for i,(identity,ref) in enumerate(refs.items()):
            meta=ref['metadata'];scope,local,truth,rival=[meta[k] for k in ['scope','local','truth','rival']];g=np.load(ROOT/ref['gradient']);b=float(logs[scope][local,truth]-logs[scope][local,rival]);c=float(control[scope][local,truth]-control[scope][local,rival]);d=float(actual[scope][local,truth]-actual[scope][local,rival]);assert b==saved_b[i] and c==saved_c[i]
            slope_u,error_u=dot(g,u);slope_e,error_e=dot(g,e);slope_v,error_v=dot(g,v);review=certificate['inequality_reviews'][i];assert review['kind']=='protected_margin' and review['linear_recovery']==slope_e and review['target']==-c and review['passed'];predicted_control=c+slope_e;predicted_origin=b+slope_v;control_residual=c-(b+slope_u);origin_residual=d-predicted_origin;correction_residual=d-predicted_control
            resolution=STEP_EPS*EPS*max(abs(c),norm(g)*correction_norm);assert predicted_control>=-max(error_e,resolution);assert abs((origin_residual-control_residual)-correction_residual)<=64*EPS*max(1,abs(d),abs(predicted_control),abs(predicted_origin))
            rows.append(dict(identity=identity,scope=scope,local=local,truth=truth,rival=rival,original_margin=b,V165_actual_margin=c,V166_actual_margin=d,origin_gradient_dot_control=slope_u,origin_gradient_dot_correction=slope_e,origin_gradient_dot_V166_displacement=slope_v,original_unit_dot_errors=dict(control=error_u,correction=error_e,displacement=error_v),predicted_control_anchored_margin=predicted_control,predicted_origin_anchored_margin=predicted_origin,V165_origin_linearization_residual=control_residual,V166_origin_linearization_residual=origin_residual,V166_correction_linearization_residual=correction_residual,original_unit_certificate_passed=review['passed'],actual_margin_is_negative=d<0,actual_protected_blocker=identity in bad_ids,original_scale_resolution=resolution,finite_negative_resolved_beyond_arithmetic=bool(d<-max(error_e,resolution))))
        failures=[r for r in rows if r['actual_protected_blocker']];assert len(failures)==observed['roles'][role]['previously_measured_blocking_functions'];assert all(r['predicted_control_anchored_margin']>=-max(r['original_unit_dot_errors']['correction'],r['original_scale_resolution']) and r['V166_actual_margin']<0 and r['finite_negative_resolved_beyond_arithmetic'] for r in failures)
        summary=dict(role=role,full_parameter_count=1060832,measured_functions=len(rows),actual_blocking_measured_functions=len(failures),all_original_unit_certificates_reproduced=True,all_actual_known_blockers_negative_beyond_original_arithmetic=True,control_displacement_norm=old_norm,V166_displacement_norm=new_norm,V166_correction_norm=correction_norm,known_failure_rows=failures,all_function_margin_residuals=rows,reason='origin derivatives do not certify actual finite nonlinear margins',no_Maratos_or_global_infeasibility_claim=True,no_new_capacity_or_training_authority=True);save(OUT/f'role{role}.json',summary);roles.append(summary)
    check_bindings(bindings);save(OUT/'review.json',dict(status='V166_saved_original_unit_nonlinear_margin_residuals_verified',roles=roles,official_heads=0,official_features=0,official_derivatives=0,fits=0,permanent_updates=0,all_certified_predicted_margins_do_not_replace_true_finite_guards=True,next_mechanism_evidence='known constraint finite nonlinear residual after complete coverage',source_sha256=bindings));print(json.dumps(dict(status='V166_nonlinear_residual_review_passed',roles=[dict(role=r['role'],known_failures=r['actual_blocking_measured_functions'],bad_actual_margins=[v['V166_actual_margin'] for v in r['known_failure_rows']],predicted_margins=[v['predicted_control_anchored_margin'] for v in r['known_failure_rows']]) for r in roles],official_calls=0)))

if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():save(OUT/'failure.json',dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),official_calls=0))
        raise
