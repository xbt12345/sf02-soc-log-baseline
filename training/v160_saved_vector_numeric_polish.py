"""Fixed Decimal80 active-set reconstruction, unchanged float64 certificates.

This arithmetic module does not call a model or supply finite-step authority.
"""
from decimal import Decimal,localcontext
import numpy as np
from v159_float64_repeat_policy_v2 import EPS,STEP_EPS
from v160_active_margin_direction_v5 import CERT_EPS
from v160_independent_saved_direction_certificate import certificate,norm

DECIMAL_PRECISION=80
MAX_DECIMAL_DOT_TERMS=20000000
MAX_INWARD_CORRECTIONS=3
INWARD_ERROR_MULTIPLIER=16

def original_float64_review(gm,gs,a,alpha,mu,raw):
    variables=1+len(a);raw_scale=float(np.max(np.abs(raw)))
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
    cert=dict(dual_bounds=bool(0<=alpha<=1 and np.all(mu>=0)),normalized_margin_primal_feasible=bool(np.all(margin_slopes>=-dot_limits)),duality_gap=gap,duality_gap_limit=gap_limit,duality_gap_pass=abs(gap)<=gap_limit,margin_complementarity_pass=bool(np.all(np.abs(complementarity)<=comp_limits)),class_complementarity_pass=bool(np.all(np.abs(class_complementarity)<=class_comp_limit)),resolved_common_descent=bool(np.all(slopes<-STEP_EPS*EPS*class_norms*norm_d)),no_absolute_gradient_floor=True)
    independent=certificate(gm,gs,a,direction)
    passed=all(cert[k] for k in ['dual_bounds','normalized_margin_primal_feasible','duality_gap_pass','margin_complementarity_pass','class_complementarity_pass','resolved_common_descent']) and independent['eligible_for_finite_trial_only']
    return dict(passed=passed,direction=direction,raw=raw,certificate=cert,independent_saved_vector_certificate=independent,class_slopes=slopes.tolist(),raw_infinity_norm=raw_scale)

def eliminate(matrix,rhs):
    n=len(rhs);a=[list(row)+[rhs[i]] for i,row in enumerate(matrix)]
    for column in range(n):
        pivot=max(range(column,n),key=lambda row:abs(a[row][column]))
        if a[pivot][column]==0:raise ArithmeticError('Exact active-system rank deficiency')
        a[column],a[pivot]=a[pivot],a[column]
        for row in range(column+1,n):
            ratio=a[row][column]/a[column][column]
            for j in range(column+1,n+1):a[row][j]-=ratio*a[column][j]
            a[row][column]=Decimal(0)
    result=[Decimal(0)]*n
    for row in range(n-1,-1,-1):result[row]=(a[row][-1]-sum(a[row][j]*result[j] for j in range(row+1,n)))/a[row][row]
    return result

def polish(gm,gs,normals,seed_result):
    gm,gs,a=[np.asarray(v,np.float64) for v in [gm,gs,normals]]
    if gm.ndim!=1 or gs.shape!=gm.shape or a.ndim!=2 or a.shape[1]!=len(gm) or len(a)>24 or not all(np.isfinite(v).all() for v in [gm,gs,a]):raise ValueError('Bounded complete finite vectors required')
    alpha=float(seed_result['alpha']);mu=np.asarray(seed_result['multipliers'],np.float64)
    if mu.shape!=(len(a),) or not 0<=alpha<=1 or np.any(mu<0):return dict(status='invalid_seed_dual_bounds_stop',finite_update_qualified=False)
    columns=1+len(a);union=np.flatnonzero((gm!=0)|(gs!=0)|np.any(a!=0,axis=0))
    estimated_terms=len(union)*(columns*(columns+1)//2+columns)
    if estimated_terms>MAX_DECIMAL_DOT_TERMS:return dict(status='decimal_dot_term_budget_stop',estimated_dot_terms=estimated_terms,finite_update_qualified=False)
    with localcontext() as ctx:
        ctx.prec=DECIMAL_PRECISION
        m=[Decimal.from_float(float(v)) for v in gm[union]];s=[Decimal.from_float(float(v)) for v in gs[union]]
        normals_decimal=[[Decimal.from_float(float(v)) for v in row[union]] for row in a]
        basis=[[x-y for x,y in zip(m,s)]]+[[-v for v in row] for row in normals_decimal]
        def dot(x,y):return sum((v*w for v,w in zip(x,y)),Decimal(0))
        h=[dot(row,s) for row in basis];gram=[[Decimal(0) for _ in basis] for _ in basis]
        for i in range(columns):
            for j in range(i+1):gram[i][j]=gram[j][i]=dot(basis[i],basis[j])
        z=[Decimal.from_float(alpha)]+[Decimal.from_float(float(v)) for v in mu]
        free=[i for i,v in enumerate(z) if (0<v<1 if i==0 else v>0)];bound=[i for i in range(columns) if i not in free]
        if free:
            rhs=[-h[i]-sum(gram[i][j]*z[j] for j in bound) for i in free]
            try:solution=eliminate([[gram[i][j] for j in free] for i in free],rhs)
            except ArithmeticError:return dict(status='decimal_active_rank_deficiency_stop',estimated_dot_terms=estimated_terms,finite_update_qualified=False)
            for i,v in zip(free,solution):z[i]=v
        if not 0<=z[0]<=1 or any(v<0 for v in z[1:]):return dict(status='decimal_fixed_active_set_bounds_failed_stop',finite_update_qualified=False)
        raw=np.zeros_like(gm)
        for j,coordinate in enumerate(union):raw[coordinate]=float(-(z[0]*m[j]+(1-z[0])*s[j]-sum(z[i+1]*row[j] for i,row in enumerate(normals_decimal))))
        alpha=float(z[0]);mu=np.array([float(v) for v in z[1:]])
        decimal_coefficients=[str(v) for v in z]
    reconstruction=-(alpha*gm+(1-alpha)*gs-a.T@mu)
    reconstruction_limit=CERT_EPS*EPS*columns*(abs(alpha)*norm(gm)+abs(1-alpha)*norm(gs)+sum(abs(float(v))*norm(row) for v,row in zip(mu,a)))
    reconstruction_difference=norm(raw-reconstruction)
    if reconstruction_difference>reconstruction_limit:return dict(status='original_dual_reconstruction_failed_stop',reconstruction_difference=reconstruction_difference,reconstruction_limit=reconstruction_limit,finite_update_qualified=False)
    if norm(raw)<=reconstruction_limit:return dict(status='direction_reconstruction_unresolved_stop',finite_update_qualified=False,no_global_infeasibility_claim=True)
    reviewed=original_float64_review(gm,gs,a,alpha,mu,raw);trace=[];original_raw=raw.copy()
    # Correct only a resolved negative computational margin, never a class loss.
    if not reviewed['independent_saved_vector_certificate']['eligible_for_finite_trial_only'] and len(a):
        scales=np.max(np.abs(a),axis=1)
        if np.any(scales==0):return dict(status='zero_constraint_normal_stop',finite_update_qualified=False)
        normalized=a/scales[:,None]
        for iteration in range(MAX_INWARD_CORRECTIONS):
            protection=reviewed['independent_saved_vector_certificate']['protection']
            if not any(p['status']=='resolved_negative' for p in protection):break
            d=reviewed['direction'];target=np.array([INWARD_ERROR_MULTIPLIER*p['arithmetic_error'] for p in protection]);current=np.array([p['margin_slope'] for p in protection])
            correction,_,rank,singular=np.linalg.lstsq(normalized,(target-current)/scales,rcond=EPS*max(1,len(a)))
            new_direction=d+correction;new_raw=new_direction*reviewed['raw_infinity_norm']
            change=norm(new_raw-original_raw)
            if change>reconstruction_limit:return dict(status='inward_reconstruction_budget_stop',correction_norm=change,reconstruction_limit=reconstruction_limit,finite_update_qualified=False)
            reviewed=original_float64_review(gm,gs,a,alpha,mu,new_raw)
            trace.append(dict(iteration=iteration,rank=int(rank),singular_values=singular.tolist(),target_margin=target.tolist(),raw_change_norm=change,original_certificate=reviewed['certificate'],independent_sign_certificate=reviewed['independent_saved_vector_certificate']))
    status='numeric_polish_certified_for_finite_probe_only' if reviewed['passed'] else 'numeric_polish_original_certificate_failed_stop'
    return dict(status=status,alpha=alpha,multipliers=mu.tolist(),decimal_coefficients=decimal_coefficients,decimal_precision=DECIMAL_PRECISION,nonzero_union_coordinates=len(union),estimated_dot_terms=estimated_terms,reconstruction_difference=reconstruction_difference,reconstruction_limit=reconstruction_limit,inward_trace=trace,finite_update_qualified=False,quality_acceptance=False,**reviewed)
