"""Commit only verified actual proposals and retain all newly repaired rows."""
import numpy as np
import torch
from v159_boundary_train_v4 import assign,restore,stats,tensor_hash
from v161_fixed_error_endpoint_diagnostic_v2 import store_scope,joint_check,save

def repaired_protection(ctx,previous_q,candidate_q):
    rows=ctx['OOF_rows'];truth=rows.truth.to_numpy();local=rows.local.to_numpy();previous=previous_q[local].argmax(1);candidate=candidate_q[local].argmax(1)
    old=rows.protected_correct.to_numpy(bool);assert not np.any(old&(candidate!=truth))
    repaired=(previous!=truth)&(candidate==truth)
    return old|repaired,rows.loc[repaired].assign(previous_pred=previous[repaired],accepted_pred=candidate[repaired])

def commit(model,base,ctx,folder,direction,proof,previous_q,qo,lo,qd,ld,state_index,fault=None):
    assert proof['accepted'] and proof['actual_parameter_change'] and proof['classification_guard']
    old_hash=tensor_hash(model.state_dict());assert all(torch.equal(p,b) for p,b in zip(model.parameters(),base))
    assert stats(ctx,qo,'OOF')['protected_regressions']==0 and stats(ctx,qd,'deployment')['mastered'] and joint_check(ctx,qd,ld)['passed']
    old_mask=ctx['OOF_rows'].protected_correct.to_numpy(bool).copy();mask,repairs=repaired_protection(ctx,previous_q,qo);committed=False
    try:
        assign(model,base,direction,1.)
        if fault is not None:fault('after_assignment')
        assert tensor_hash(model.state_dict())==proof['probe_parameter_sha256'] and tensor_hash(model.state_dict())!=old_hash
        torch.save(dict(state=model.state_dict(),state_index=state_index,parameter_sha256=tensor_hash(model.state_dict()),supervised_training_scope=True),folder/'checkpoint.pt')
        repairs.to_parquet(folder/'newly_repaired_original_rows.parquet',index=False)
        ctx['OOF_rows']['protected_correct']=mask
        ctx['OOF_rows'].loc[mask,['row_position','local','truth','root','pure_current_input']].to_parquet(folder/'cumulative_correct_protection.parquet',index=False)
        for scope,q,lp in [('OOF',qo,lo),('deployment',qd,ld)]:store_scope(folder,ctx,scope,q,lp)
        if fault is not None:fault('before_commit_receipt')
        receipt=dict(status='actual_finite_proposal_committed_supervised_training_state',state_index=state_index,previous_parameter_sha256=old_hash,parameter_sha256=tensor_hash(model.state_dict()),newly_repaired_original_rows=len(repairs),newly_repaired_mixed_original_rows=int((~repairs.pure_current_input).sum()),cumulative_protected_original_rows=int(mask.sum()),all_old_correct_and_cumulative_real_repairs_retained=True,no_new_training_targets=True,quality_acceptance=False)
        save(folder/'commit.json',receipt);committed=True
        return receipt
    finally:
        if not committed:restore(model,base);ctx['OOF_rows']['protected_correct']=old_mask;assert tensor_hash(model.state_dict())==old_hash
