"""Certify slopes from saved vectors only, without any model/gradient call.

This does not grant training or classify a finite step as safe. A boundary
normal is unresolved at zero; every original finite guard must still run.
"""
import hashlib,json,math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/v160_independent_saved_direction_certificate_v2_20261002'
EPS=float(np.finfo(np.float64).eps)
SUBNORMAL=float(np.nextafter(np.float64(0),np.float64(1)))

def norm(vector):
    maximum=float(np.abs(vector).max())
    if maximum==0:return 0.
    result=maximum*math.sqrt(math.fsum((vector/maximum)**2))
    if not math.isfinite(result):raise ValueError('Unresolved overflowed norm')
    return result

def dot(vector,direction):
    products=vector*direction
    if not np.isfinite(products).all():raise ValueError('Nonfinite dot products')
    value=math.fsum(products);absolute=math.fsum(np.abs(products))
    # Accounts for rounded input products plus fsum's final rounding and
    # underflow. It does not bound an upstream autograd measurement error.
    error=EPS*absolute+2*EPS*abs(value)+len(vector)*float(SUBNORMAL)
    return value,error

def certificate(gm,gs,margin_normals,direction):
    gm,gs,d=[np.asarray(a,dtype=np.float64) for a in [gm,gs,direction]]
    normals=np.asarray(margin_normals,dtype=np.float64)
    if gm.ndim!=1 or len(gm)==0 or gs.shape!=gm.shape or d.shape!=gm.shape:raise ValueError('Vector shape mismatch')
    if normals.ndim!=2 or normals.shape[1]!=len(gm):raise ValueError('Normal shape mismatch')
    if not all(np.isfinite(a).all() for a in [gm,gs,d,normals]):raise ValueError('Nonfinite input')
    nd=norm(d);classes=[];protection=[]
    for name,g in [('M',gm),('S',gs)]:
        slope,arithmetic_error=dot(g,d);resolution=max(arithmetic_error,16*EPS*norm(g)*nd)
        classes.append(dict(class_name=name,slope=slope,arithmetic_error=arithmetic_error,
          resolution=resolution,resolved_negative=slope < -resolution))
    for index,a in enumerate(normals):
        slope,error=dot(a,d)
        status='resolved_negative' if slope < -error else ('resolved_positive' if slope > error else 'boundary_unresolved')
        protection.append(dict(index=index,margin_slope=slope,arithmetic_error=error,status=status))
    eligible=nd>0 and all(c['resolved_negative'] for c in classes) and not any(c['status']=='resolved_negative' for c in protection)
    return dict(direction_norm=nd,classes=classes,protection=protection,
      eligible_for_finite_trial_only=eligible,finite_step_safety_proven=False,quality_acceptance=False,
      scope='Numerical directional signs of supplied saved vectors only; no proof of upstream gradient repeatability, QP optimality, nonlinear protection, actual learning or transfer.')

def main():
    assert not OUT.exists();reports=[]
    for width in [2,1060832]:
        def embed(values):
            x=np.zeros(width);x[-2:]=values;return x
        for scale in [1.,1e-6,1e6]:
            gm=embed([-1,1])*scale;gs=embed([-1,2])*scale;a=embed([-1,0])[None,:]*scale
            raw=certificate(gm,gs,a,embed([1,-1.5])*scale)
            projected=certificate(gm,gs,a,embed([0,-1]))
            zero=certificate(gm,gs,a,np.zeros(width))
            sacrificed=certificate(gm,gs,a,embed([0,1]))
            assert not raw['eligible_for_finite_trial_only'] and raw['protection'][0]['status']=='resolved_negative'
            assert projected['eligible_for_finite_trial_only'] and not projected['finite_step_safety_proven']
            assert not zero['eligible_for_finite_trial_only'] and not sacrificed['eligible_for_finite_trial_only']
            reports.append(dict(width=width,scale=scale,unsafe_solver_direction=raw,safe_tangent_candidate=projected,
              zero_direction_rejected=True,class_sacrifice_rejected=True))
    report=dict(status='independent_saved_vector_certification_rejects_false_success_and_zero_direction',reports=reports,
      official_heads=0,official_features=0,official_gradients=0,official_fits=0,official_updates=0,
      synthetic_only=True,numerical_policy_scope='Computational sign checks only. Formal measurement repeat policy and call bounds must be separately bound.',
      source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    OUT.mkdir();(OUT/'qualification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=report['status'],vector_cases=len(reports),official_heads=0,official_fits=0)))
if __name__=='__main__':main()
