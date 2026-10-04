"""Independent counterexamples for the new inequality function, CPU only."""
import json
import math
import traceback
from pathlib import Path
import numpy as np
from v163_one_sided_joint_restoration import propose
from v161_independent_all_finite_results_review import save
from v160_independent_fixed_diagnostic_review import sha

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/v163_independent_inequality_regression_review_20261002'
PASS='one_sided_joint_restoration_requires_full_actual_finite_guard'

def embed(v,width):
    a=np.zeros(width);a[-len(v):]=v;return a

def fixture(width):
    # The second correct margin is already safer; a true minimum correction
    # must not force its coordinate to move backwards to its old confidence.
    u=embed([1.,0.,0.],width)
    a=np.stack([embed([0.,1.,0.],width),embed([0.,0.,1.],width)])
    b=np.array([.1,.1]);c=np.array([-.2,.7])
    gm=embed([-1.,.02,.01],width);gs=embed([-.7,.01,.02],width)
    r=propose(u,a,b,c,gm,gs);assert r['status']==PASS
    assert np.allclose(r['correction'][-3:],[0.,.3,0.],rtol=0.,atol=8*np.finfo(float).eps)
    assert all(v['resolved_negative'] for v in r['class_reviews'])
    assert math.fsum(a[1]*r['correction'])>=-8*np.finfo(float).eps
    # Same physical constraint duplicated: reduced row rank is supported.
    redundant=propose(u,np.vstack([a,a[0]]),np.r_[b,b[0]],np.r_[c,c[0]],gm,gs)
    assert redundant['status']==PASS and redundant['normal_rank']<5
    assert np.max(np.abs(redundant['displacement']-r['displacement']))<=8*np.finfo(float).eps
    # Joint class constraints are active: repairing a margin while making the
    # full M target non-decreasing must stop, even if the local solver succeeds.
    incompatible=propose(embed([0.,0.,-1.],width),a[:1],np.array([.1]),np.array([-.2]),
                         embed([0.,1.,0.],width),embed([0.,0.,1.],width))
    assert incompatible['status']=='local_inequality_or_common_descent_unqualified_stop'
    assert not incompatible['finite_step_authority'] and incompatible['no_global_infeasibility_claim']
    # A restoration normal which contains no derivative information is an
    # explicit limitation, never silently replaced with an arbitrary floor.
    zero=propose(u,np.zeros_like(a[:1]),b[:1],c[:1],gm,gs)
    assert zero['status']=='zero_actual_constraint_normal_stop'
    return dict(width=width,already_safer_correct_margin_not_forced_back=True,
                redundant_physical_constraints_preserve_candidate=True,
                margin_repair_against_joint_class_descent_rejected=True,
                zero_physical_normal_explicit_stop=True,actual_SOC_quality_proven=False)

def main():
    assert not OUT.exists();OUT.mkdir()
    sources=[Path(__file__).resolve(),ROOT/'training/v163_one_sided_joint_restoration.py',
             ROOT/'training/v160_independent_saved_direction_certificate.py',ROOT/'training/v159_float64_repeat_policy_v2.py']
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sources}
    save(OUT/'pre_bindings.json',dict(source_sha256=bindings,official_calls=0))
    cases=[fixture(n) for n in [3,1060832]]
    assert all(sha(ROOT/k)==v for k,v in bindings.items())
    result=dict(status='new_one_sided_function_independent_confidence_redundancy_and_joint_descent_counterexamples_passed',
                cases=cases,source_sha256=bindings,official_calls=0,new_fits=0,permanent_updates=0,quality_acceptance=False)
    save(OUT/'review.json',result);print(json.dumps({k:result[k] for k in ['status','official_calls','new_fits']}))

if __name__=='__main__':
    try:main()
    except Exception as error:
        OUT.mkdir(exist_ok=True)
        save(OUT/'failure.json',dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),official_calls=0))
        raise
