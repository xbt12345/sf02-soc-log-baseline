"""Expose optimizer-success/constraint mismatch using synthetic gradients only."""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/v160_synthetic_solver_scale_counterexample_20261002'

def solve(gm,gs,a):
    delta=gm-gs
    def values(z):
        v=gs+z[0]*delta-z[1:]@a
        return .5*(v@v),np.r_[delta@v,-a@v]
    opt=minimize(lambda z:values(z)[0],np.r_[.5,np.zeros(len(a))],jac=lambda z:values(z)[1],method='SLSQP',
      bounds=[(0,1)]+[(0,None)]*len(a),options=dict(ftol=1e-14,maxiter=200))
    v=gs+opt.x[0]*delta-opt.x[1:]@a;direction=-v
    return direction,dict(success=bool(opt.success),message=str(opt.message),iterations=int(opt.nit),
      multipliers=opt.x.tolist(),class_slopes=[float(g@direction) for g in [gm,gs]],
      protected_margin_slopes=(a@direction).tolist(),direction_norm=float(np.linalg.norm(direction)))

def main():
    assert not OUT.exists();reports=[];arrays={}
    for width in [2,1060832]:
        def embed(v):
            z=np.zeros(width);z[-2:]=v;return z
        base=[embed([-1,1]),embed([-1,2]),embed([-1,0])[None,:]]
        for scale in [1.,1e-6,1e6]:
            gm,gs,a=[z*scale for z in base]
            direct,direct_report=solve(gm,gs,a)
            # One common scaling preserves class tradeoffs and the cone;
            # it is not separate normalization of the two class losses.
            common_scale=max(np.linalg.norm(gm),np.linalg.norm(gs),np.linalg.norm(a))
            unit,unit_report=solve(gm/common_scale,gs/common_scale,a/common_scale)
            norm=float(np.max(np.abs(unit)));assert norm>0
            normalized=unit/norm
            assert max(gm@normalized,gs@normalized)<0
            assert float((a@normalized)[0])>=-1e-10*scale
            assert np.allclose(normalized[-2:],[0,-1],rtol=0,atol=1e-10)
            violation=bool((a@direct<0).any())
            reports.append(dict(width=width,scale=scale,unscaled=direct_report,
              optimizer_success_with_unsafe_direction=direct_report['success'] and violation,
              commonly_scaled_solver=unit_report,normalized_safe_direction_tail=normalized[-2:].tolist(),
              common_scaling_factor=float(common_scale)))
            if width==2:
                arrays[f'scale{scale}_raw_direction']=direct
                arrays[f'scale{scale}_normalized_safe_direction']=normalized
    observed=[r for r in reports if r['scale']==1e-6]
    assert len(observed)==2 and all(r['optimizer_success_with_unsafe_direction'] for r in observed)
    report=dict(status='synthetic_optimizer_success_does_not_certify_guard_common_scale_preserves_cone',reports=reports,
      synthetic_only=True,official_heads=0,official_features=0,official_gradients=0,official_fits=0,
      scope='Full parameter count synthetic vectors, not official gradient data. Common scaling and toy bounds alone do not establish a valid official numerical policy, nonlinear safety or classification gains.',
      source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    OUT.mkdir()
    for name,array in arrays.items():np.save(OUT/(name+'.npy'),array)
    (OUT/'qualification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=report['status'],observed_false_success=[{k:r[k] for k in ['width','scale','unscaled','normalized_safe_direction_tail']} for r in observed],official_fits=0)))
if __name__=='__main__':main()
