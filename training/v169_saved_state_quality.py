"""Full original row and fixed initial S-cohort diagnostics from already measured q."""
import numpy as np

def review(reference,q,logq,cohort,prior,output_weight,coefficient):
    if reference.row_position.duplicated().any() or cohort.row_position.duplicated().any():raise ValueError('Unique original rows required')
    local=reference.local.to_numpy();truth=reference.truth.to_numpy();pred=q[local].argmax(1);old=reference.pred.to_numpy()
    classes={}
    for cls in [0,1,2]:
        actual=truth==cls;chosen=pred==cls;correct=actual&chosen;mass=int(actual.sum());guess=int(chosen.sum())
        classes[str(cls)]=dict(original_rows=mass,errors=int((actual&~chosen).sum()),precision=float(correct.sum()/guess) if guess else None,recall=float(correct.sum()/mass) if mass else None,repairs_vs_V164=int((actual&(old!=truth)&(pred==truth)).sum()),new_errors_vs_V164=int((actual&(old==truth)&(pred!=truth)).sum()))
    actual_S=reference.loc[reference.truth.eq(2)].sort_values('row_position');initial_S=cohort.sort_values('row_position')
    if not np.array_equal(actual_S[['row_position','local','truth']].to_numpy(),initial_S[['row_position','local','truth']].to_numpy()):raise ValueError('Fixed complete all-S cohort/controls mismatch')
    idx=initial_S.local.to_numpy();s_pred=q[idx].argmax(1);low=initial_S.initial_low_prior.to_numpy(bool);initial_error=initial_S.initial_error.to_numpy(bool)
    rival=logq[idx].copy();rival[:,2]=-np.inf;rival=rival.argmax(1);n=np.arange(len(idx));margin=logq[idx,2]-logq[idx,rival]
    fixed_prior=prior[idx,2]-prior[idx,rival];scaled_prior=coefficient*fixed_prior;residual=margin-scaled_prior
    def quantiles(v):return {str(p):float(np.quantile(v,p)) for p in [0.,.25,.5,.75,1.]} if len(v) else None
    slices={}
    for name,mask in [('initial_low_prior',low),('initial_other_prior',~low),('initial_other_prior_correct_controls',~low&~initial_error)]:
        wrong=s_pred!=2
        slices[name]=dict(original_rows=int(mask.sum()),unique_current_inputs=int(np.unique(idx[mask]).size),errors=int((mask&wrong).sum()),repairs_of_initial_errors=int((mask&initial_error&~wrong).sum()),regressions_of_initial_correct=int((mask&~initial_error&wrong).sum()),actual_margin=quantiles(margin[mask]),scaled_fixed_prior_margin=quantiles(scaled_prior[mask]),learned_residual_margin=quantiles(residual[mask]))
    bound={f'{a}_vs_{b}':float(np.abs(output_weight[:,a]-output_weight[:,b]).sum()) for a in [0,1,2] for b in [0,1,2] if a!=b}
    return dict(classes=classes,fixed_all_S_slices=slices,prior_coefficient=float(coefficient),current_frozen_readout_pair_bounds=bound,readout_bound_not_full_trainable_model_impossibility=True,zero_support_class_quality_not_estimated=True)
