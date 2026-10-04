"""Fixed float64 repeat envelope; never a relaxation of learning acceptance.

Eight eps at each parameter segment's own scale, and relative segment L2.
No absolute floor for gradients. Same-point risks/log probabilities use eight
eps at max(1, magnitude); probability decisions must still match exactly.
Finite-step acceptance reserves twice that envelope, tightening Armijo.
"""
import numpy as np
from v159_class_direction import class_direction

EPS=float(np.finfo(np.float64).eps)
REPEAT_EPS=8
STEP_EPS=2*REPEAT_EPS
SEGMENTS=[('observation_weight',0,1060592),('opinion_weight',1060592,1060768),('bias',1060768,1060784),('output_weight',1060784,1060832)]

def repeat_values(a,b,kind='risk'):
    a,b=np.asarray(a,np.float64),np.asarray(b,np.float64)
    if a.shape!=b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        return dict(passed=False,reason='shape_or_nonfinite')
    scale=np.maximum(1.,np.maximum(np.abs(a),np.abs(b)))
    gap=np.abs(a-b);limits=REPEAT_EPS*EPS*scale
    decisions=True
    if kind=='probability':
        if a.ndim!=2 or a.shape[1]!=3 or np.any(a<0) or np.any(b<0):return dict(passed=False,reason='invalid_probability')
        decisions=bool(np.array_equal(a.argmax(1),b.argmax(1)))
    return dict(passed=bool(np.all(gap<=limits) and decisions),maximum_absolute=float(gap.max()),maximum_scaled_eps=float((gap/(EPS*scale)).max()),argmax_exact=decisions,kind=kind)

def repeat_gradient(a,b):
    a,b=np.asarray(a,np.float64),np.asarray(b,np.float64)
    if a.shape!=(1060832,) or b.shape!=a.shape or not np.isfinite(a).all() or not np.isfinite(b).all():return dict(passed=False,reason='complete_shape_or_nonfinite')
    result=[]
    for name,start,end in SEGMENTS:
        x,y=a[start:end],b[start:end];d=x-y
        scale=max(float(np.abs(x).max()),float(np.abs(y).max()))
        norm=max(float(np.linalg.norm(x)),float(np.linalg.norm(y)))
        maximum=float(np.abs(d).max());dnorm=float(np.linalg.norm(d))
        limit=REPEAT_EPS*EPS*scale
        # Numerically unresolved near-zero components may differ; a resolved
        # sign reversal cannot be hidden by the much larger output segment.
        meaningful=(np.maximum(np.abs(x),np.abs(y))>limit)
        sign_changes=int(np.count_nonzero(meaningful&(np.sign(x)!=np.sign(y))))
        passed=maximum<=limit and dnorm<=REPEAT_EPS*EPS*norm and sign_changes==0
        result.append(dict(segment=name,passed=bool(passed),maximum_absolute=maximum,own_scale=scale,maximum_scaled_eps=maximum/(EPS*scale) if scale else 0.,relative_L2=dnorm/norm if norm else 0.,resolved_sign_changes=sign_changes,exact_zero=scale==0))
    return dict(passed=all(r['passed'] for r in result),segments=result)

def repeat_direction(g0,g1,mass_m,mass_s,arm):
    a,b=[class_direction(*g,mass_m,mass_s,arm) for g in [g0,g1]]
    if a['status']!=b['status']:return dict(passed=False,reason='direction_status_changed')
    reports={k:repeat_values(a[k],b[k],k) for k in ['direction','M_weight','class_slopes']}
    signs=bool(np.array_equal(np.sign(a['class_slopes']),np.sign(b['class_slopes'])))
    return dict(passed=all(r['passed'] for r in reports.values()) and signs,status=a['status'],descent_signs_exact=signs,comparisons=reports)

def finite_step_review(before,after,slopes,mass_m,mass_s,arm,step,classification_guard):
    before,after,slopes=[np.asarray(v,np.float64) for v in [before,after,slopes]]
    if any(v.shape!=(2,) or not np.isfinite(v).all() for v in [before,after,slopes]) or not np.isfinite(step) or step<=0 or arm not in ['A','B'] or mass_m<=0 or mass_s<=0:
        return dict(accepted=False,reason='invalid_finite_proposal')
    if not classification_guard:return dict(accepted=False,reason='classification_guard')
    weights=np.array([mass_m,mass_s],np.float64)/float(mass_m+mass_s)
    if arm=='A':
        b,a,s=np.asarray([float(weights@before)]),np.asarray([float(weights@after)]),np.asarray([float(weights@slopes)])
    else:b,a,s=before,after,slopes
    if not np.all(s<0):return dict(accepted=False,reason='no_strict_descent')
    resolution=STEP_EPS*EPS*np.maximum(1.,np.maximum(np.abs(b),np.abs(a)))
    bounds=b+1e-4*step*s
    actual_drop=b-a;armijo_slack=bounds-a
    accepted=bool(np.all(actual_drop>resolution) and np.all(armijo_slack>resolution))
    reason='finite_descent_above_resolution' if accepted else ('below_numeric_resolution' if np.any(actual_drop<=resolution) else 'armijo_margin')
    return dict(accepted=accepted,reason=reason,actual_drop=actual_drop.tolist(),resolution=resolution.tolist(),Armijo_bounds=bounds.tolist(),Armijo_slack=armijo_slack.tolist(),risk_tolerance_added_to_acceptance=False)

def finite_armijo(before,after,slopes,mass_m,mass_s,arm,step,classification_guard):
    return finite_step_review(before,after,slopes,mass_m,mass_s,arm,step,classification_guard)['accepted']
