"""Synthetic geometry only; no SOC data/model/feature or official gradients."""
from pathlib import Path
import json
import hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/v159_synthetic_guard_direction_counterexample_20261002'


def main():
    assert not OUT.exists()
    theta=np.zeros(2);centers=[np.array([1.,-1.]),np.array([1.,-2.])]
    gm,gs=[theta-c for c in centers]
    mm,ms,ss=[float(a@b) for a,b in [(gm,gm),(gm,gs),(gs,gs)]]
    alpha=1. if ms>=mm else (0. if ms>=ss else (ss-ms)/((mm-ms)+(ss-ms)))
    proposed=-(alpha*gm+(1-alpha)*gs);proposed/=np.abs(proposed).max()
    # The already-correct decision has margin -theta_x >=0 (tie selects its
    # lower class index). Its tangent constraint is d_x <=0.
    protected=lambda t: bool(t[0]<=0.)
    risks=lambda t:np.array([.5*np.sum((t-c)**2) for c in centers])
    assert all(g@proposed<0 for g in [gm,gs])
    original_steps=[2.**(-i) for i in range(41)]
    failures=[not protected(theta+step*proposed) for step in original_steps]
    assert all(failures)
    # Exact Euclidean projection of d onto the known halfspace. This adds
    # the active protection normal; it cannot be a convex combination of
    # the two original negative gradients, whose x coordinate is always1.
    projected=proposed.copy();projected[0]=min(projected[0],0.)
    assert protected(theta+.5*projected) and all(g@projected<0 for g in [gm,gs])
    before,after=risks(theta),risks(theta+.5*projected)
    assert np.all(after<before) and np.all(after<=before+1e-4*.5*np.array([gm@projected,gs@projected]))
    result=dict(status='synthetic_counterexample_fixed_common_descent_can_fail_guard_despite_existing_safe_descent',
                class_gradients=[gm.tolist(),gs.tolist()],common_min_norm_alpha=alpha,
                original_direction=proposed.tolist(),original_class_slopes=[float(g@proposed) for g in [gm,gs]],
                scalar_steps_tested=original_steps,all_original_direction_steps_rejected=all(failures),
                projected_direction=projected.tolist(),protected_decision_preserved=True,
                risks_before=before.tolist(),risks_after=after.tolist(),finite_step=.5,
                scope='This is an explicit 2D synthetic counterexample, not evidence that the current SOC head has a feasible projected direction.',
                actual_SOC_guard_normals_not_measured=True,official_model_queries=0,official_gradients=0,official_fits=0,
                next_mechanism_supported_for_qualification_only=True,training_permission=False,quality_acceptance=False,
                source_sha256={Path(__file__).resolve().relative_to(ROOT).as_posix():hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    OUT.mkdir();(OUT/'counterexample.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['status','original_direction','projected_direction','risks_before','risks_after','scope','quality_acceptance']}))


if __name__=='__main__':main()
