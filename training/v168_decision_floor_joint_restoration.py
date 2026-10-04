"""Prospective argmax-aware local targets using the unchanged V166 core."""
import numpy as np
from v159_float64_repeat_policy_v2 import EPS,STEP_EPS
from v166_solver_trace import observed_propose

def decision_floors(records,trial_logs):
    floors=[]
    if not 0<len(records)<=25:raise ValueError('One through25 complete functions required')
    for identity,meta in records.items():
        if meta['input_identity']!=identity:raise ValueError('Complete function identity mismatch')
        scope,local,truth,rival=[meta[k] for k in ['scope','local','truth','rival']]
        if not isinstance(truth,(int,np.integer)) or not isinstance(rival,(int,np.integer)) or not 0<=truth<3 or not 0<=rival<3 or truth==rival:raise ValueError('Distinct original class indices required')
        row=np.asarray(trial_logs[scope][local],np.float64)
        if row.shape!=(3,) or not np.isfinite(row).all():raise ValueError('Finite actual trial log probabilities required')
        floors.append(STEP_EPS*EPS*max(1.,abs(float(row[truth])),abs(float(row[rival]))) if truth>rival else 0.)
    return np.asarray(floors,np.float64)

def shortfall(margins,floors):
    c,t=np.asarray(margins,np.float64),np.asarray(floors,np.float64)
    if c.shape!=t.shape or c.ndim!=1 or not c.size or not np.isfinite(c).all() or not np.isfinite(t).all() or np.any(t<0):raise ValueError('Matched finite local targets required')
    return np.maximum(t-c,0.)

def propose(callback,displacement,normals,origin_margins,trial_margins,gm,gs,floors):
    c,t=np.asarray(trial_margins,np.float64),np.asarray(floors,np.float64)
    deficit=shortfall(c,t)
    # This is exactly A.e >= tau-c. The core's zero RHS identity preserves
    # original arithmetic, iteration caps and strict original class signs.
    result=observed_propose(callback,displacement,normals,origin_margins,c-t,gm,gs)
    result.update(local_protection_target='prospective_existing_argmax_decision_floor',actual_trial_margins=c.tolist(),prospective_decision_floors=t.tolist(),actual_trial_floor_shortfalls=deficit.tolist(),truth_index_losing_tie_is_only_positive_floor_trigger=True,local_floor_not_actual_acceptance_tolerance=True,actual_argmax_all_original_rows_and_retention_guards_still_required=True,finite_step_authority=False,no_new_fitting_permission=True)
    return result
