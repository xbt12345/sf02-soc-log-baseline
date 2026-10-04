"""One bound-constrained quadratic dual, common scale and explicit certificates.

Primal min .5||d||²+t: gm.d<=t, gs.d<=t, normals.d>=0.
Dual min .5||alpha*gm+(1-alpha)*gs-normals.T@mu||²,
alpha in[0,1], mu>=0. No diagonal damping, per-class normalization or restart.
This local result never authorizes a finite parameter update by itself.
"""
import numpy as np
from v160_independent_saved_direction_certificate import certificate, norm
from v159_mgda_synthetic_qualification import exact_two_gradient_direction
from v159_float64_repeat_policy_v2 import EPS,STEP_EPS

MAX_QP_ITERATIONS=256
CERT_EPS=64

def solve_direction(gm,gs,normals):
    gm,gs,a=[np.asarray(v,np.float64) for v in [gm,gs,normals]]
    if gm.ndim!=1 or gs.shape!=gm.shape or not gm.size or not np.isfinite(gm).all() or not np.isfinite(gs).all():raise ValueError('Complete matched finite gradients required')
    if a.size==0:a=np.empty((0,gm.size))
    if a.ndim!=2 or a.shape[1]!=gm.size or not np.isfinite(a).all():raise ValueError('Complete finite margin normals required')
    scale=max(float(np.max(np.abs(gm))),float(np.max(np.abs(gs))))
    if scale==0:return dict(status='zero_inputs_no_automatic_fallback',direction=np.zeros_like(gm),alpha=.5,multipliers=[],common_scale=0.,finite_update_qualified=False)
    m,s=gm/scale,gs/scale
    # Positive row scaling preserves the constraint cone; class scale is common.
    normal_scales=np.max(np.abs(a),axis=1) if len(a) else np.empty(0)
    if np.any(normal_scales==0):raise ValueError('Zero normal contains no constraint')
    n=a/normal_scales[:,None] if len(a) else a.copy()
    basis=np.vstack([m-s,-n]);h=basis@s;matrix=basis@basis.T;matrix=(matrix+matrix.T)*.5
    variables=len(basis);eig=np.linalg.eigvalsh(matrix);matrix_scale=float(np.max(np.abs(eig))) if eig.size else 0.
    if eig.min(initial=0)<-CERT_EPS*EPS*variables*matrix_scale:return dict(status='non_PSD_gram_unresolved',direction=np.zeros_like(gm),finite_update_qualified=False)
    z=np.zeros(variables);z[0]=exact_two_gradient_direction(m,s)[0]
    upper=np.r_[1.,np.full(variables-1,np.inf)];active=np.where(z==0,-1,0);active[0]=1 if z[0]==1 else active[0]
    solved=False;records=[]
    for iteration in range(MAX_QP_ITERATIONS):
        free=np.flatnonzero(active==0);bound=np.flatnonzero(active!=0)
        target=z.copy();rank=0;condition=None
        if len(free):
            rhs=-h[free]-matrix[np.ix_(free,bound)]@z[bound]
            values,_,rank,singular=np.linalg.lstsq(matrix[np.ix_(free,free)],rhs,rcond=None);target[free]=values
            condition=float(singular[0]/singular[rank-1]) if rank else None
        p=target-z;ratio=1.;hit=None;hit_kind=0
        for j in free:
            fraction=None;kind=0
            if target[j]<0 and p[j]<0:fraction=-z[j]/p[j];kind=-1
            elif target[j]>upper[j] and p[j]>0:fraction=(upper[j]-z[j])/p[j];kind=1
            if fraction is not None and fraction<ratio:ratio=max(0.,float(fraction));hit=int(j);hit_kind=kind
        z=z+ratio*p
        if hit is not None:
            z[hit]=0. if hit_kind==-1 else upper[hit];active[hit]=hit_kind
            records.append(dict(iteration=iteration,event='bound_hit',variable=hit,rank=int(rank),condition=condition));continue
        gradient=matrix@z+h
        # Algorithm stopping only; final certificate uses original vectors.
        threshold=CERT_EPS*EPS*variables*np.maximum(np.abs(h)+np.abs(matrix)@np.abs(z),np.finfo(float).tiny)
        violations=np.where((active==-1)&(gradient<-threshold),-gradient,np.where((active==1)&(gradient>threshold),gradient,0.))
        if np.any(violations>0):
            release=int(np.argmax(violations));active[release]=0;records.append(dict(iteration=iteration,event='release_bound',variable=release,rank=int(rank),condition=condition));continue
        if np.any(np.abs(gradient[free])>threshold[free]):
            records.append(dict(iteration=iteration,event='free_stationarity_unresolved',rank=int(rank),condition=condition));break
        solved=True;records.append(dict(iteration=iteration,event='stationarity_candidate',rank=int(rank),condition=condition));break
    alpha=float(z[0]);mu_scaled=z[1:]
    mu=mu_scaled*scale/normal_scales if len(a) else mu_scaled
    raw_scaled=-(alpha*m+(1-alpha)*s-n.T@mu_scaled)
    cancellation_bound=CERT_EPS*EPS*variables*(abs(alpha)*norm(m)+abs(1-alpha)*norm(s)+sum(abs(float(u))*norm(row) for u,row in zip(mu_scaled,n)))
    reconstructed_norm=norm(raw_scaled)
    if reconstructed_norm<=cancellation_bound:
        return dict(status='direction_reconstruction_unresolved_stop',direction=np.zeros_like(gm),alpha=alpha,multipliers=mu.tolist(),common_scale=scale,normal_positive_scales=normal_scales.tolist(),reconstructed_scaled_norm=reconstructed_norm,reconstruction_resolution=cancellation_bound,iterations=len(records),solver_trace=records,finite_update_qualified=False,no_global_infeasibility_claim=True)
    raw=raw_scaled*scale;raw_scale=float(np.max(np.abs(raw)))
    direction=raw/raw_scale if raw_scale else np.zeros_like(raw)
    slopes=np.array([gm@direction,gs@direction]);margin_slopes=a@direction
    norm_d=float(np.linalg.norm(direction));norm_raw=float(np.linalg.norm(raw));class_norms=np.array([np.linalg.norm(gm),np.linalg.norm(gs)]);normal_norms=np.linalg.norm(a,axis=1)
    dot_limits=CERT_EPS*EPS*variables*normal_norms*norm_d
    raw_slopes=np.array([gm@raw,gs@raw]);t=float(raw_slopes.max());gap=float(norm_raw**2+t)
    gap_limit=CERT_EPS*EPS*variables*max(norm_raw**2,float(np.abs(raw_slopes).max()))
    raw_margins=a@raw;complementarity=mu*raw_margins
    comp_limits=CERT_EPS*EPS*variables*np.abs(mu)*normal_norms*norm_raw
    class_complementarity=np.array([alpha,(1-alpha)])*(raw_slopes-t)
    class_comp_limit=CERT_EPS*EPS*variables*np.maximum(class_norms*norm_raw,np.finfo(float).tiny)
    cert=dict(optimizer_stationarity_candidate=solved,dual_bounds=bool(0<=alpha<=1 and np.all(mu>=0)),normalized_margin_primal_feasible=bool(np.all(margin_slopes>=-dot_limits)),duality_gap=gap,duality_gap_limit=gap_limit,duality_gap_pass=abs(gap)<=gap_limit,margin_complementarity_pass=bool(np.all(np.abs(complementarity)<=comp_limits)),class_complementarity_pass=bool(np.all(np.abs(class_complementarity)<=class_comp_limit)),resolved_common_descent=bool(np.all(slopes<-STEP_EPS*EPS*class_norms*norm_d)),no_absolute_gradient_floor=True)
    independent=certificate(gm,gs,a,direction)
    passed=solved and all(cert[k] for k in ['dual_bounds','normalized_margin_primal_feasible','duality_gap_pass','margin_complementarity_pass','class_complementarity_pass'])
    status=('local_QP_certificate_failed_stop' if not passed else ('no_resolved_safe_common_descent' if not cert['resolved_common_descent'] else ('independent_direction_sign_failed_stop' if not independent['eligible_for_finite_trial_only'] else 'local_QP_certified_requires_actual_finite_guard')))
    return dict(status=status,direction=direction,alpha=alpha,multipliers=mu.tolist(),common_scale=scale,raw_infinity_norm=raw_scale,class_slopes=slopes.tolist(),protected_margin_slopes=margin_slopes.tolist(),protected_margin_resolution=dot_limits.tolist(),certificate=cert,independent_saved_vector_certificate=independent,normal_positive_scales=normal_scales.tolist(),reconstructed_scaled_norm=reconstructed_norm,reconstruction_resolution=cancellation_bound,iterations=len(records),solver_trace=records,finite_update_qualified=False)
