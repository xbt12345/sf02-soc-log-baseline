"""Actual final-tie point, complete inputs, source history and separate repair goals."""
import ast,copy,json,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment_review import ROOT,read,sha,check_bindings
from v160_margin_normal import input_identity
from v168_decision_floor_execution_review import prospective_plan,review_plan
from v168_decision_floor_joint_restoration import decision_floors,shortfall
import v168_decision_floor_diagnostic as entry

OUT=ROOT/'artifacts/v168_decision_floor_identity_qualification_20261002'

def main():
    assert not OUT.exists();OUT.mkdir();plan=prospective_plan();review_plan(plan)
    core_path=ROOT/'artifacts/v168_decision_floor_core_qualification_20261002/qualification.json';core=read(core_path);check_bindings(core['source_sha256']);assert core['actual_CPU_QP_solves']==5 and core['official_heads']==core['official_derivatives']==0
    replay=ROOT/'artifacts/v167_observed_runtime_failure_replay_20261002/replay.json';actual=read(replay);check_bindings(actual['source_sha256']);assert len(actual['cases'])==6
    active=[Path(__file__).resolve(),Path(entry.__file__).resolve(),ROOT/'training/v168_decision_floor_joint_restoration.py',ROOT/'training/v168_decision_floor_execution_review.py'];sources=set(active)|{core_path,replay,ROOT/'training/review_policy/v168_decision_aware_floor_draft.json'};sources.update(ROOT/p for p in core['source_sha256']);sources.update(ROOT/p for p in actual['source_sha256'])
    inherited=[];registry=ROOT/plan['failure_constraints']
    while True:
        value=read(registry);sources.add(registry);inherited.append(dict(path=registry.relative_to(ROOT).as_posix(),sha256=sha(registry)))
        if not value.get('inherit_applicable_constraints'):break
        registry=ROOT/value['inherit_applicable_constraints']
    assert inherited[1]['path']=='training/review_policy/v166_observed_runtime_boundaries_v2.json'
    for folder in [entry.OLD,ROOT/'artifacts/v168_decision_floor_plan_20261002']:
        sources.update(p for p in folder.rglob('*') if p.is_file())
    for path in active:ast.parse(path.read_text(encoding='utf-8-sig'))
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(sources)};entry.save(OUT/'pre_bindings.json',dict(source_sha256=bindings,official_calls=0));ctx=entry.load_context(1);state=torch.load(entry.PRIOR/'role1/endpoint.pt',map_location='cpu',weights_only=True)['state'];records,gradients,u,proof,goal,tie_rows=entry.cached_inputs(plan,state,ctx);model=entry.CurrentInputBoundary().cpu();model.load_state_dict(state);base=tuple(p.detach().clone() for p in model.parameters());origin=entry.tensor_hash(state);old_mask=ctx['OOF_rows'].protected_correct.copy();entry.assign(model,base,u,1.)
    try:
        trial=entry.tensor_hash(model.state_dict());assert trial==plan['actual_failed_trial_parameter_sha256']!=origin
        for identity,meta in records.items():
            scope,local,truth,rival=[meta[k] for k in ['scope','local','truth','rival']];ids=ctx['ids'] if scope=='OOF' else np.arange(22546);pos=int(np.searchsorted(ids,local));assert ids[pos]==local;chunk=ids[pos//2048*2048:pos//2048*2048+2048];assert input_identity(1,scope,chunk,ctx['x'][chunk],np.asarray(ctx[scope][chunk],np.float64),pos%2048,truth,rival)==identity
        folder=ROOT/plan['actual_failed_trial_path'];logs={scope:np.load(folder/f'{scope}_logq.npy') for scope in ['OOF','deployment']};q=np.load(folder/'OOF_q.npy');floors=decision_floors(records,logs);c=entry.margins(records,list(records),logs);bad_ids=entry.blocker_identities(ctx,tie_rows);assert len(bad_ids)==1;index=list(records).index(next(iter(bad_ids)));meta=records[next(iter(bad_ids))];assert meta['local']==21985 and meta['truth']==2 and meta['rival']==1 and c[index]==0 and floors[index]>0 and shortfall(c,floors)[index]>0 and q[21985,1]==q[21985,2] and int(q[21985].argmax())==1
        stale=read(entry.OLD/'role1/correction1/parameter_point_roles.json')['margin_Jacobian_parameter_sha256'];assert stale!=trial
    finally:entry.restore(model,base)
    assert entry.tensor_hash(model.state_dict())==origin and all(torch.equal(p,v) for p,v in zip(model.parameters(),base)) and np.array_equal(ctx['OOF_rows'].protected_correct,old_mask)
    indexed=ctx['OOF_rows'].set_index('row_position');assert not indexed.loc[goal.row_position].protected_correct.any()
    candidate=pd.read_parquet(folder/'OOF_original_rows.parquet');guard=entry.extra_guard(candidate,goal,tie_rows,plan['prospective_full_class_error_caps']);assert guard['all_four_observed_M_candidate_repairs_retained'] and not guard['all_two_actual_S_tie_rows_now_correct'] and not guard['passed']
    oracle=candidate.copy();oracle.loc[oracle.row_position.isin(tie_rows.row_position),'pred']=2;positive=entry.extra_guard(oracle,goal,tie_rows,plan['prospective_full_class_error_caps']);assert positive['passed']
    lost=oracle.copy();lost.loc[lost.row_position.eq(int(goal.row_position.iloc[0])),'pred']=2;assert not entry.extra_guard(lost,goal,tie_rows,plan['prospective_full_class_error_caps'])['passed']
    wrong_identity=oracle.copy();wrong_identity.loc[wrong_identity.row_position.eq(int(goal.row_position.iloc[0])),'truth']=2;refused=False
    try:entry.extra_guard(wrong_identity,goal,tie_rows,plan['prospective_full_class_error_caps'])
    except AssertionError:refused=True
    assert refused
    safe=entry.safe_references(plan);assert [r['role'] for r in safe]==[0,2]
    for field,value in [('complete_functions',26),('execute_only_actual_failed_role',0),('new_fit_permission',True),('one_correction_only_no_automatic_second_or_floor_seed_step_change',False)]:
        changed=copy.deepcopy(plan);changed[field]=value;refused=False
        try:review_plan(changed)
        except AssertionError:refused=True
        assert refused
    changed=copy.deepcopy(plan);changed['new_caps']['QP_solves']=2;refused=False
    try:review_plan(changed)
    except AssertionError:refused=True
    assert refused
    check_bindings(bindings);entry.save(OUT/'qualification.json',dict(status='V168_actual_final_tie_complete_inputs_separate_four_M_goals_safe_references_and_caps_qualified',origin_parameter_sha256=origin,actual_final_tie_parameter_sha256=trial,old_Jacobian_parameter_sha256_is_different=stale,complete_functions=25,all_original_full_input_identities_verified=True,exact_actual_final_tie_detected_by_positive_floor_shortfall=True,four_M_candidate_goals_not_in_origin_correct_mask=True,actual_candidate_extra_guard_rejected=guard,positive_extra_guard_is_explicit_oracle_not_actual_model=positive,wrong_goal_identity_and_lost_repair_refused=True,safe_unchanged_references=safe,applicable_failure_registry_chain=inherited,all_six_V167_failure_cases_replayed=True,actual_full_CPU_parameter_assignment_and_restore_passed=True,official_heads=0,official_features=0,official_derivatives=0,fits=0,permanent_updates=0,entry_lifecycle_and_physical_seal_still_required=True,execution_authority=False,source_sha256=bindings));print(json.dumps(dict(status='V168_final_tie_identity_qualified',complete_functions=25,official_calls=0)))

if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():entry.save(OUT/'failure.json',dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),official_calls=0))
        raise
