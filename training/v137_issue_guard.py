"""Single-issue budget review and original-row repair protection; no training."""
from pathlib import Path
import copy
import numpy as np
import pandas as pd
from v135_runtime import ROOT,OUT,read,save,sha,require_run_seal
from v136_verify_design import validate as parent_validate

PLAN=ROOT/'training/review_policy/v137_single_issue_plan.json'
DEST=ROOT/'artifacts/v137_single_issue_design_20260930'

def need(value,message):
    if not value:raise ValueError(message)

def validate_plan(p):
    need(p['status']=='designed_not_trained' and p['new_fits']==p['new_updates']==0,'Do not invent training')
    issue=p['primary_issue'];r=p['retention'];r2=p['round2']
    need(issue['id']=='TRAIN-PURE-READOUT' and issue['maximum_experiment_rounds']==2,'Changed target or unbounded rounds')
    need(not issue['scope_may_expand_inside_round'] and not issue['round_counter_may_reset_by_version_rename'],'Issue scope or budget evasion')
    need(issue['goal_pure_M_errors']==issue['goal_pure_S_errors']==0,'Relaxing acceptance')
    parent=read(ROOT/p['round1']['parent_plan_path']);parent_validate(parent)
    need(sha(ROOT/p['round1']['parent_plan_path'])==p['round1']['parent_plan_sha256'],'Parent plan changed')
    need(p['round1']['primary']==parent['primary'] and p['round1']['mastery']==parent['mastery'],'Unreviewed method or population change')
    need(p['round1']['registered_fits_max']==r2['fits_max']==6 and p['problem_budget']['primary_fits_max']==12,'Excess fits')
    need(not r2['automatic'] and r2['needs_new_bound_runtime_and_specific_round1_evidence'],'Blind second round')
    need(r2['probe_inconclusive_disallows_second_round'] and r2['new_target_or_trainable_scope_disallows_second_round'],'Inconclusive probe used as license')
    need(not r2['HELD_labels_in_constraints'] and r2['mixed_labels_not_rewritten'] and r2['all_original_true_label_CE_retained'],'Label leakage or rewrite')
    need(r['accepted_repair_original_rows_and_truth_immutable'] and r['accepted_repair_new_errors_max']==0,'Solved cases may regress')
    need(r['aggregate_counts_cannot_hide_repair_regression'] and r['unaccepted_experiment_gains_are_not_closed_repairs'],'Net gain hides regression')
    need(r['HELD_repair_rows_readonly_never_gradient_or_threshold_targets'],'Protected held labels used to fit')
    need(r['conflicting_guard_population_must_block_adoption_not_drop_rows'],'Protection contradiction hidden')
    need(r['training_repair_without_transfer_is_scoped_branch_only'] and r['adoption_requires_original_full_quality'],'Training success falsely promoted')
    need(not p['runtime_readiness']['trainer_implemented'] and not p['runtime_readiness']['run_seal_created'],'False readiness')

def protect_original_rows(reference, baseline, candidate, protected, keys=('row_position',)):
    """Read-only acceptance gate. Truth only comes from separately supplied reference.

    `keys` must include training-role identity when one original row occurs in
    multiple roles. Protected rows are immutable prior acceptance identities,
    not recomputed from the candidate's convenient input groups.
    """
    keys=list(keys)
    for frame,name in [(reference,'reference'),(baseline,'baseline'),(candidate,'candidate'),(protected,'protected')]:
        need(all(k in frame for k in keys),'Missing role/row identity '+name)
        need(not frame.duplicated(keys).any(),'Duplicate original role rows '+name)
    need('truth' in reference and 'pred' in baseline and 'pred' in candidate,'Missing independent truth or decisions')
    ref=reference[keys+['truth']].set_index(keys).sort_index()
    before=baseline[keys+['pred']].set_index(keys).sort_index();after=candidate[keys+['pred']].set_index(keys).sort_index()
    need(ref.index.equals(before.index) and ref.index.equals(after.index),'Incomplete scored population')
    need(ref.truth.isin([0,1,2]).all() and before.pred.isin([0,1,2]).all() and after.pred.isin([0,1,2]).all(),'Invalid class')
    ids=protected[keys].set_index(keys).sort_index().index
    need(ids.isin(ref.index).all(),'Unknown accepted repair rows')
    good=before.pred.eq(ref.truth);correct=after.pred.eq(ref.truth)
    need(bool(good.reindex(ids).all()),'Protected baseline does not establish the repair')
    regressions=~correct.reindex(ids)
    whole_new=good&~correct;whole_fixed=~good&correct
    return {'repair_protection_passed':not bool(regressions.any()),'protected_original_role_rows':len(ids),
        'new_errors_on_protected_original_rows':int(regressions.sum()),
        'all_reference_correct_regressions':int(whole_new.sum()),'all_reference_errors_repaired':int(whole_fixed.sum()),
        'not_task_quality_or_runtime_permission':True}

def adversaries(p):
    edits=[('bundle another target',lambda z:z['primary_issue'].update(scope_may_expand_inside_round=True)),
        ('third round by rename',lambda z:z['primary_issue'].update(round_counter_may_reset_by_version_rename=True)),
        ('blind automatic second round',lambda z:z['round2'].update(automatic=True)),
        ('uncertain probe licensed fitting',lambda z:z['round2'].update(probe_inconclusive_disallows_second_round=False)),
        ('repair row regression forgiven',lambda z:z['retention'].update(accepted_repair_new_errors_max=1)),
        ('net gain covers old new errors',lambda z:z['retention'].update(aggregate_counts_cannot_hide_repair_regression=False)),
        ('held anchors fed into fit',lambda z:z['retention'].update(HELD_repair_rows_readonly_never_gradient_or_threshold_targets=False)),
        ('delete contradictory protected row',lambda z:z['retention'].update(conflicting_guard_population_must_block_adoption_not_drop_rows=False)),
        ('train zero mistaken for task solved',lambda z:z['retention'].update(adoption_requires_original_full_quality=False)),
        ('unaccepted history called repaired',lambda z:z['retention'].update(unaccepted_experiment_gains_are_not_closed_repairs=False))]
    results=[]
    for name,edit in edits:
        z=copy.deepcopy(p);edit(z)
        try:validate_plan(z)
        except ValueError:results.append({'case':name,'rejected':True})
        else:raise AssertionError('Unsafe design accepted '+name)
    return results

def main():
    p=read(PLAN);validate_plan(p);require_run_seal(ROOT/'training/v135_train.py')
    for rel,h in p['evidence_sha256'].items():need(sha(ROOT/rel)==h,'Evidence changed '+rel)
    registry=read(ROOT/p['retention']['registry_path'])
    need(not registry['classified_case_repair_sets'],'Do not fabricate accepted classification repairs')
    for e in registry['engineering_contracts']:need(sha(ROOT/e['evidence'])==e['evidence_sha256'],'Engineering evidence changed')
    rows=pd.read_parquet(DEST/'retention_collision_original_rows.parquet')
    ref=rows[['row_position','truth']];before=rows[['row_position','pred_A0']].rename(columns={'pred_A0':'pred'});after=rows[['row_position','pred_R_decay']].rename(columns={'pred_R_decay':'pred'})
    # Hypothetical protection of all A0-good rows tests a real counterexample;
    # it does not register these empirical hits as accepted repairs.
    guard=rows.loc[rows.pred_A0.eq(rows.truth),['row_position']]
    gate=protect_original_rows(ref,before,after,guard)
    need(not gate['repair_protection_passed'] and gate['new_errors_on_protected_original_rows']==2,'Actual retention regression was ignored')
    target=DEST/'plan_preflight.json'
    if target.exists():raise FileExistsError(target)
    save(target,{'status':'single_issue_design_preflight_passed','new_fits':0,'new_updates':0,'model_promoted':False,
        'negative_design_replays':adversaries(p),'actual_counterexample_retention_gate':gate,
        'old_training_seal_intact':True,'plan_sha256':sha(PLAN),'source_sha256':sha(Path(__file__)),
        'scope':'Preflight and hypothetical gate on real data, not accepted repair registration or new training runtime.'})
    print({'negative_designs_rejected':10,'real_protected_regressions_rejected':2,'new_fits':0,'new_updates':0},flush=True)

if __name__=='__main__':main()
