"""Full current working functions, trial-point Jacobians and original class slopes."""
import sys
import numpy as np
import v169_working_joint_restoration as solver
from v159_float64_repeat_policy_v2 import EPS,STEP_EPS

def floors(records,logs):
    if not 0<len(records)<=64:raise ValueError('Current local resource bound64')
    output=[]
    for identity,meta in records.items():
        if meta['input_identity']!=identity:raise ValueError('Function identity mismatch')
        scope,local,truth,rival=[meta[k] for k in ['scope','local','truth','rival']]
        if not isinstance(truth,(int,np.integer)) or not isinstance(rival,(int,np.integer)) or not 0<=truth<3 or not 0<=rival<3 or truth==rival:raise ValueError('Original class identities required')
        row=np.asarray(logs[scope][local],np.float64)
        if row.shape!=(3,) or not np.isfinite(row).all():raise ValueError('Actual finite trial logs required')
        output.append(STEP_EPS*EPS*max(1.,abs(float(row[truth])),abs(float(row[rival]))) if truth>rival else 0.)
    return np.asarray(output,np.float64)

def propose(callback,u,a,b,c,gm,gs,tau):
    c,t=np.asarray(c,np.float64),np.asarray(tau,np.float64)
    if c.shape!=t.shape or c.ndim!=1 or not c.size or not np.isfinite(t).all() or np.any(t<0):raise ValueError('Matched nonnegative original decision floors')
    previous=sys.getprofile();code=solver.minimize.__code__
    def observer(frame,event,value):
        if frame.f_code is code and event in ['call','return']:callback(event,value)
        if previous is not None:previous(frame,event,value)
    sys.setprofile(observer)
    try:result=solver.propose(u,a,b,c-t,gm,gs)
    finally:sys.setprofile(previous)
    result.update(actual_trial_margins=c.tolist(),decision_floors=t.tolist(),finite_step_authority=False,full_guard_ledger_required=True)
    return result
