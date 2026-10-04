"""Create new numeric-only versions; preserve already executed v1 sources."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
old=ROOT/'training/v160_active_margin_direction.py'
new=ROOT/'training/v160_active_margin_direction_v2.py'
assert not new.exists()
s=old.read_text(encoding='utf-8')
s=s.replace('import numpy as np','import numpy as np\nfrom v160_independent_saved_direction_certificate import certificate, norm')
s=s.replace("scale=max(float(np.max(np.abs(gm))),float(np.max(np.abs(gs))),float(np.max(np.abs(a))) if a.size else 0.)", "scale=max(float(np.max(np.abs(gm))),float(np.max(np.abs(gs))))")
s=s.replace('m,s,n=gm/scale,gs/scale,a/scale', '''m,s=gm/scale,gs/scale
    # Positive row scaling preserves the constraint cone; class scale is common.
    normal_scales=np.max(np.abs(a),axis=1) if len(a) else np.empty(0)
    if np.any(normal_scales==0):raise ValueError('Zero normal contains no constraint')
    n=a/normal_scales[:,None] if len(a) else a.copy()''')
s=s.replace("solved=True;records.append", """if np.any(np.abs(gradient[free])>threshold[free]):
            records.append(dict(iteration=iteration,event='free_stationarity_unresolved',rank=int(rank),condition=condition));break
        solved=True;records.append""")
s=s.replace("alpha=float(z[0]);mu=z[1:];raw=-(alpha*gm+(1-alpha)*gs-a.T@mu);raw_scale=float(np.max(np.abs(raw)))", """alpha=float(z[0]);mu_scaled=z[1:]
    mu=mu_scaled*scale/normal_scales if len(a) else mu_scaled
    raw_scaled=-(alpha*m+(1-alpha)*s-n.T@mu_scaled)
    cancellation_bound=CERT_EPS*EPS*variables*(abs(alpha)*norm(m)+abs(1-alpha)*norm(s)+sum(abs(float(u))*norm(row) for u,row in zip(mu_scaled,n)))
    reconstructed_norm=norm(raw_scaled)
    if reconstructed_norm<=cancellation_bound:
        return dict(status='direction_reconstruction_unresolved_stop',direction=np.zeros_like(gm),alpha=alpha,multipliers=mu.tolist(),common_scale=scale,normal_positive_scales=normal_scales.tolist(),reconstructed_scaled_norm=reconstructed_norm,reconstruction_resolution=cancellation_bound,iterations=len(records),solver_trace=records,finite_update_qualified=False,no_global_infeasibility_claim=True)
    raw=raw_scaled*scale;raw_scale=float(np.max(np.abs(raw)))""")
s=s.replace("passed=solved and all", "independent=certificate(gm,gs,a,direction)\n    passed=solved and all")
s=s.replace("if passed and cert['resolved_common_descent']", "if passed and cert['resolved_common_descent'] and independent['eligible_for_finite_trial_only']")
s=s.replace("certificate=cert,iterations=", "certificate=cert,independent_saved_vector_certificate=independent,normal_positive_scales=normal_scales.tolist(),reconstructed_scaled_norm=reconstructed_norm,reconstruction_resolution=cancellation_bound,iterations=")
new.write_text(s,encoding='utf-8')
oldq=ROOT/'training/v160_active_direction_numeric_qualification.py'
newq=ROOT/'training/v160_active_direction_numeric_qualification_v2.py'
assert not newq.exists()
s=oldq.read_text(encoding='utf-8').replace('from v160_active_margin_direction import','from v160_active_margin_direction_v2 import').replace('v160_active_direction_numeric_qualification_20261002','v160_active_direction_numeric_qualification_v2_20261002').replace("ROOT/'training/v160_active_margin_direction.py'","ROOT/'training/v160_active_margin_direction_v2.py',ROOT/'training/v160_independent_saved_direction_certificate.py',ROOT/'training/v160_active_margin_direction.py',ROOT/'artifacts/v160_active_direction_numeric_qualification_20261002/failure.json'")
s=s.replace("assert all(r['status']=='no_resolved_safe_common_descent' for r in impossible)","assert all(r['status'] in ['no_resolved_safe_common_descent','direction_reconstruction_unresolved_stop'] for r in impossible)")
s=s.replace("redundant=solve_direction", """normal_rescaling=[]
    for ns in [1e-24,1.,1e24]:
        r=solve_direction(gm,gs,n*ns);normal_rescaling.append(short(r))
        assert r['status']=='local_QP_certified_requires_actual_finite_guard' and np.allclose(r['direction'],[0.,-1.],atol=1e-14,rtol=0)
    near=solve_direction(gm,gs,np.array([[-1.,0.],[-1.,1e-10]]))
    if near['status']=='local_QP_certified_requires_actual_finite_guard':
        assert near['independent_saved_vector_certificate']['eligible_for_finite_trial_only']
    else:assert near['status'] in ['local_QP_certificate_failed_stop','direction_reconstruction_unresolved_stop','no_resolved_safe_common_descent']
    redundant=solve_direction""")
s=s.replace('no_common_safe_descent_stops=True,', 'no_common_safe_descent_stops=True,cancellation_residue_not_normalized=True,positive_normal_row_rescaling_preserves_cone=True,near_dependent_normal_never_false_certified=True,')
s=s.replace('redundant=short(redundant),primal=', 'redundant=short(redundant),normal_rescaling=normal_rescaling,near_dependent=short(near),primal=')
newq.write_text(s,encoding='utf-8')
print('Created new versions; no official data calls.')
