"""Decision-only local restoration; unchanged original-unit joint solver.

Zero margin is a necessary local decision floor, not an argmax guarantee.
Actual full-row guards still decide every candidate, including ties.
"""
import numpy as np
from v163_one_sided_joint_restoration import propose as joint_propose

def propose(displacement,normals,origin_margins,current_margins,gm,gs):
    origin=np.asarray(origin_margins,np.float64)
    if origin.ndim!=1 or not np.isfinite(origin).all() or np.any(origin<0):
        raise ValueError('Matched nonnegative actual origin margins required')
    result=joint_propose(displacement,normals,np.zeros_like(origin),current_margins,gm,gs)
    result['local_protection_target']='exact_zero_decision_floor'
    result['actual_origin_margins_for_identity_only']=origin.tolist()
    result['actual_argmax_all_original_rows_and_retention_guards_still_required']=True
    result['confidence_restoration_not_required_by_this_local_policy']=True
    return result
