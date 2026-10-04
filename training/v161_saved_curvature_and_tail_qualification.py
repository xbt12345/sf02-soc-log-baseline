"""Saved nonlinear boundary/domain diagnosis and zero-gradient tail checks."""
import ast,json,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment_review import ROOT,read,sha,check_bindings
from v160_independent_saved_direction_certificate import certificate
from v159_float64_repeat_policy_v2 import finite_step_review
import v161_cached_direction_backtrack_tail as entry
OUT=ROOT/'artifacts/v161_saved_curvature_and_tail_qualification_20261002'

def main():
    assert not OUT.exists();OUT.mkdir();folder=entry.DIAG/'role2'
    files={Path(__file__).resolve(),ROOT/'training/v161_cached_direction_backtrack_tail.py',ROOT/'training/v161_fixed_error_endpoint_diagnostic_v2.py',ROOT/'training/v159_float64_repeat_policy_v2.py',ROOT/'training/v160_independent_saved_direction_certificate.py'}|{p for p in folder.rglob('*') if p.is_file()}
    files.add(ROOT/'artifacts/v161_saved_fixed_error_result_audit_20261002/audit.json')
    binding={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)};entry.save(OUT/'pre_bindings.json',dict(source_sha256=binding,official_calls=0))
    b=pd.read_parquet(folder/'baseline_error_class1_repeat0/OOF_original_rows.parquet');last=pd.read_parquet(folder/'round0/probe19/actual_blocking_original_rows.parquet')
    assert len(last)==2 and len(last.local.unique())==1 and last.truth.eq(2).all() and last.scope.eq('OOF').all()
    records=read(folder/'round0/normal_records.json');assert len(records)==1 and next(iter(records.values()))['local']==int(last.local.iloc[0])
    a=np.load(folder/'round0/raw_margin_normals.npy');d=np.load(folder/'round0/polished_direction.npy');gs=[np.load(folder/f'baseline_error_class{c}_repeat0/complete_fixed_error_target_gradient.npy') for c in [1,2]]
    cert=certificate(*gs,a,d);assert cert['eligible_for_finite_trial_only'] and cert['protection'][0]['status']=='boundary_unresolved'
    local=int(last.local.iloc[0]);row=b[b.local.eq(local)].iloc[0];m0=float(row.logp2-row.logp1);assert m0>0
    observations=[]
    for j in range(2,20):
        lp=np.load(folder/f'round0/probe{j}/OOF_logq.npy');m=float(lp[local,2]-lp[local,1]);h=2.**(-j);curvature=(m-m0)/h**2
        assert m<0 and -.28<curvature<-.26
        observations.append(dict(backtrack=j,step=h,actual_margin=m,actual_quadratic_secant=curvature))
    # Synthetic quadratic fixture with measured scale: predicts a tail test,
    # never substitutes for the actual model's finite guard.
    coefficient=observations[-1]['actual_quadratic_secant'];toy_m19=m0+coefficient*(2.**-19)**2;toy_m20=m0+coefficient*(2.**-20)**2
    assert toy_m19<0<toy_m20
    rv=np.load(folder/'baseline_error_class1_repeat0/fixed_pure_error_contribution.npy');slopes=np.array(read(folder/'round0/polished_certificate.json')['class_slopes']);h=2.**-20
    toy_risk=rv+h*slopes;finite=finite_step_review(rv,toy_risk,slopes,59770,32550,'B',h,True);assert finite['accepted']
    dummy=torch.nn.Linear(1,1,dtype=torch.float64);dummy.opinions=torch.nn.Identity();counter=entry.Counter(dummy,1,OUT/'synthetic_counter.jsonl',0)
    try:counter.gradient_before(1,[0,1,1])
    except RuntimeError:pass
    else:raise AssertionError('Zero gradient tail budget bypassed')
    assert counter.gradient_attempts==0;counter.close()
    tree=ast.parse((ROOT/'training/v161_cached_direction_backtrack_tail.py').read_text(encoding='utf-8'));source=ast.unparse(tree)
    assert 'for j in range(20, 40)' in source and 'Counter(model, 484' in source
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ['solve_direction','polish','add_normals','measure'] for n in ast.walk(tree))
    check_bindings(binding)
    entry.save(OUT/'qualification.json',dict(status='saved_actual_nonlinear_boundary_original40_tail_zero_gradient_entry_qualified',role=2,blocking_original_rows=last.row_position.tolist(),one_actual_input_function=True,base_margin=m0,observations=observations,independent_saved_direction_certificate=cert,synthetic_quadratic_margin_at19=toy_m19,synthetic_quadratic_margin_at20=toy_m20,synthetic_finite_gate=finite,actual_tail_not_yet_tested=True,official_heads=0,official_features=0,official_gradients=0,official_fits=0,permanent_updates=0,source_sha256=binding))
    print(json.dumps(dict(status='V161_saved_curvature_and_tail_qualification_passed',official_calls=0)))

if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():entry.save(OUT/'failure.json',dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),official_calls=0))
        raise
