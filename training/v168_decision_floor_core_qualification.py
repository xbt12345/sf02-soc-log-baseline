"""Actual full-width CPU floor numerics, exact-tie fixtures and saved regressions."""
import gc,json,traceback
from pathlib import Path
import numpy as np
from experiment_review import ROOT,read,sha,check_bindings
from v166_coverage_core_qualification_v3 import memory
from v166_coverage_joint_restoration import propose as original_propose
from v168_decision_floor_joint_restoration import decision_floors,shortfall,propose
from v159_float64_repeat_policy_v2 import EPS,STEP_EPS,finite_step_review

OUT=ROOT/'artifacts/v168_decision_floor_core_qualification_20261002'

def save(path,value):path.write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))

def main():
    assert not OUT.exists();OUT.mkdir();draft=ROOT/'training/review_policy/v168_decision_aware_floor_draft.json';check_bindings(read(draft)['source_sha256']);assert not read(draft)['execution_authority']
    sources={Path(__file__).resolve(),draft,ROOT/'training/v168_decision_floor_joint_restoration.py',ROOT/'training/v166_coverage_joint_restoration.py',ROOT/'training/v166_solver_trace.py',ROOT/'training/v159_float64_repeat_policy_v2.py',ROOT/'training/v160_independent_saved_direction_certificate.py'}
    audit=ROOT/'artifacts/v167_independent_actual_trial_point_review_v2_20261002/pre_review_bindings.json';check_bindings(read(audit)['source_sha256']);sources.add(audit)
    trial=ROOT/'artifacts/v167_trial_point_restoration_diagnostic_20261002/role1';spec=read(ROOT/'training/review_policy/v167_trial_point_restoration_contract.json')['roles'][1];point=ROOT/f"artifacts/v164_short_supervised_trajectory_20261002/role1/parameter_point{spec['parameter_point']}";gradient_paths=[point/f'class{cls}_repeat0/complete_fixed_error_target_gradient.npy' for cls in [1,2]];sources.update(gradient_paths)
    for stage in [0,1]:
        folder=trial/f'correction{stage}';refs=read(folder/'active_normal_references.json');sources.update(ROOT/ref['gradient'] for ref in refs.values());sources.update(folder/name for name in ['active_normal_references.json','current_displacement.npy','actual_origin_margins.npy','actual_trial_margins.npy','displacement.npy','correction.npy','original_unit_restoration_review.json'])
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(sources)};save(OUT/'pre_bindings.json',dict(source_sha256=bindings,official_calls=0));before=memory();save(OUT/'initial_memory.json',before);qps=nit=0
    def event(kind,value):
        nonlocal qps,nit
        if kind=='call':qps+=1
        elif value is not None:nit+=int(value.nit)
    regressions=[];gm,gs=[np.load(p) for p in gradient_paths]
    for stage in [0,1]:
        folder=trial/f'correction{stage}';refs=read(folder/'active_normal_references.json');a=np.stack([np.load(ROOT/ref['gradient']) for ref in refs.values()]);u,b,c=[np.load(folder/name) for name in ['current_displacement.npy','actual_origin_margins.npy','actual_trial_margins.npy']]
        result=propose(event,u,a,b,c,gm,gs,np.zeros(25))
        for name in ['displacement','correction']:assert np.array_equal(result[name],np.load(folder/f'{name}.npy'))
        original=read(folder/'original_unit_restoration_review.json');assert result['inequality_reviews']==original['inequality_reviews'] and result['class_reviews']==original['class_reviews'] and result['optimizer_iterations']==original['optimizer_iterations']
        regressions.append(dict(stage=stage,zero_floor_original_saved_displacement_and_correction_bit_exact=True,original_inequalities_and_class_signs_exact=True));del a,result;gc.collect()
    width=1060832;u=np.zeros(width);u[25]=-1.;a=np.zeros((25,width));a[np.arange(25),np.arange(25)]=1.;gm=np.zeros(width);gm[25]=1.;gs=2*gm;b=np.ones(25);c=np.zeros(25)
    records={str(i):dict(input_identity=str(i),scope='OOF',local=i,truth=2 if i%2==0 else 1,rival=1 if i%2==0 else 2) for i in range(25)};logs=dict(OOF=np.tile([-30.,-.7,-.7],(25,1)));floors=decision_floors(records,logs);assert np.array_equal(floors[::2],np.full(13,STEP_EPS*EPS)) and np.all(floors[1::2]==0)
    assert np.max(shortfall(c,floors))>0 and max(0.,-c.min())==0
    first=propose(event,u,a,b,c,gm,gs,floors);second=propose(event,u,a,b,c,gm,gs,floors)
    assert first['status']==second['status']=='one_sided_joint_restoration_requires_full_actual_finite_guard'
    assert np.array_equal(first['displacement'],second['displacement']) and np.array_equal(first['correction'],second['correction'])
    recovered=c+a@first['correction'];assert np.all(recovered[::2]>0) and np.all(recovered[::2]>=floors[::2])
    # Apply a literal affine synthetic logit function at the trial point;
    # all three logits and original first-index argmax participate.
    logits=np.tile([-30.,-.7,-.7],(25,1))
    for i,meta in enumerate(records.values()):logits[i,meta['truth']]+=recovered[i]
    assert np.array_equal(logits.argmax(1),[m['truth'] for m in records.values()])
    save(OUT/'full_width_exact_tie_math.json',{k:v for k,v in first.items() if k not in ['displacement','correction']});np.save(OUT/'full_width_exact_tie_correction.npy',first['correction'])
    swapped={k:dict(v,truth=v['rival'],rival=v['truth']) for k,v in records.items()};swapped_floor=decision_floors(swapped,logs);assert np.all(swapped_floor[::2]==0) and np.all(swapped_floor[1::2]>0)
    swapped_result=propose(event,u,a,b,c,gm,gs,swapped_floor);assert swapped_result['status']==first['status'];recovered_swap=c+a@swapped_result['correction'];assert np.all(recovered_swap[1::2]>0)
    zero=original_propose(u,a,b,c,gm,gs);assert zero['status']=='no_actual_negative_protected_margin_to_restore_stop'
    bad_finite=finite_step_review([.3,.4],[.299,.399],[r['linear_change'] for r in first['class_reviews']],100,100,'B',1.,False);assert not bad_finite['accepted']
    refused=[]
    for name,args in [('negative_floor',(c,-np.ones(25))),('nonfinite_floor',(c,np.full(25,np.nan))),('mismatched_floor',(c,np.zeros(24)))]:
        try:shortfall(*args)
        except ValueError:refused.append(name)
    assert len(refused)==3
    oversized=False
    try:propose(event,u,np.vstack([a,a[:1]]),np.r_[b,b[0]],np.r_[c,c[0]],gm,gs,np.r_[floors,floors[0]])
    except ValueError:oversized=True
    assert oversized and qps==5
    after=memory();save(OUT/'final_memory.json',after);check_bindings(bindings)
    save(OUT/'qualification.json',dict(status='V168_full_width_decision_floor_exact_tie_repeat_class_index_and_saved_regressions_qualified',saved_zero_floor_regressions=regressions,full_parameter_count=width,complete_functions=25,actual_CPU_QP_solves=qps,actual_CPU_optimizer_iterations=nit,full_width_synthetic_affine_argmax_ties_repaired=True,same_point_full_width_correction_repeat_bit_exact=True,class_index_relation_permutation_passed=True,zero_negative_margin_does_not_cover_losing_tie=True,negative_nonfinite_and_mismatched_floors_rejected=refused,oversized26_function_request_refused=True,original_finite_guard_still_authoritative=True,official_heads=0,official_features=0,official_derivatives=0,fits=0,permanent_updates=0,CPU_fixture_not_actual_SOC_nonlinear_gain=True,physical_memory_observed_before=before,physical_memory_observed_after=after,entry_lifecycle_and_physical_seal_still_required=True,execution_authority=False,source_sha256=bindings));print(json.dumps(dict(status='V168_decision_floor_CPU_core_qualified',CPU_QPs=qps,official_calls=0)))

if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():save(OUT/'failure.json',dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),official_calls=0))
        raise
