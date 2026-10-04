"""Pure numeric dual sign and boundary tests; not an official training solver."""
import json,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/v160_synthetic_constrained_direction_review_20261002'

def toy_direction(gm,gs,normals):
    gm,gs,a=np.asarray(gm,float),np.asarray(gs,float),np.asarray(normals,float)
    if a.size==0:a=np.empty((0,len(gm)))
    delta=gm-gs
    def values(z):
        v=gs+z[0]*delta-z[1:]@a
        return .5*(v@v),np.r_[delta@v,-a@v]
    result=minimize(lambda z:values(z)[0],np.r_[.5,np.zeros(len(a))],jac=lambda z:values(z)[1],
      bounds=[(0,1)]+[(0,None)]*len(a),method='SLSQP',options=dict(ftol=1e-14,maxiter=200))
    assert result.success
    v=gs+result.x[0]*delta-result.x[1:]@a;d=-v
    return dict(direction=d.tolist(),class_slopes=[float(g@d) for g in [gm,gs]],
      protected_margin_slopes=(a@d).tolist(),alpha=float(result.x[0]),multipliers=result.x[1:].tolist(),
      norm=float(np.linalg.norm(d)),optimizer_success=bool(result.success),objective=float(result.fun))

def main():
    assert not OUT.exists()
    positive=toy_direction([-1,1],[-1,2],[[-1,0]])
    d=np.array(positive['direction']);assert np.allclose(d,[0,-1],atol=1e-10,rtol=0)
    assert max(positive['class_slopes'])<-1e-3 and min(positive['protected_margin_slopes'])>=-1e-10
    unprotected=toy_direction([-1,1],[-1,2],[])
    assert unprotected['direction'][0]>0
    impossible=toy_direction([-1,0],[0,-1],[[-1,0],[0,-1]])
    assert impossible['norm']<1e-8
    conflict=toy_direction([-1,1],[-1,-1],[[-1,0]])
    assert conflict['norm']<1e-8
    # A tangent constraint can pass while a nonlinear finite guard fails.
    step=.5;trial=step*d;curved_guard_margin=-trial[0]-10*trial[1]**2
    assert curved_guard_margin<0
    report=dict(status='synthetic_correct_constraint_dual_sign_safe_direction_and_infeasible_cases_passed',
      positive=positive,unprotected=unprotected,no_common_safe_descent=impossible,opposed_classes_with_guard=conflict,
      nonlinear_counterexample=dict(local_linear_constraint_passed=True,finite_guard_margin=float(curved_guard_margin),finite_guard_passed=False),
      original_problem='min 0.5||d||^2+t subject to gm.d<=t, gs.d<=t, margin_normals@d>=0',
      dual='min 0.5||alpha*gm+(1-alpha)*gs-margin_normals.T@mu||^2, alpha in[0,1], mu>=0',
      official_heads=0,official_features=0,official_gradients=0,official_fits=0,official_updates=0,
      scope='2D synthetic optimizer only; toy numeric tolerance is not an official qualification policy. Actual SOC guard normals, conditioning, finite steps and gains have not been tested.',
      source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    OUT.mkdir();(OUT/'qualification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['status','positive','no_common_safe_descent','nonlinear_counterexample','official_fits']}))
if __name__=='__main__':main()
