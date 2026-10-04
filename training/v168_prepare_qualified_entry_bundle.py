"""Reviewable evidence bundle before independent root review or runtime seal."""
import ast,json
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings
from v168_decision_floor_execution_review import prospective_plan,review_plan
import v168_decision_floor_diagnostic_v2 as entry

OUT=ROOT/'artifacts/v168_decision_floor_qualification_bundle_20261002'

def main():
    assert not OUT.exists();plan=prospective_plan();review_plan(plan);paths={Path(__file__).resolve(),Path(entry.__file__).resolve(),ROOT/'training/v168_decision_floor_execution_review.py'};qualifications=[]
    for name in ['v168_decision_floor_core_qualification','v168_decision_floor_identity_qualification_v2','v168_decision_floor_lifecycle_qualification_v2','v168_decision_floor_adversarial_qualification']:
        folder=ROOT/f'artifacts/{name}_20261002';q=read(folder/'qualification.json');check_bindings(q['source_sha256']);assert 'qualified' in q['status'] and all(q[k]==0 for k in ['official_heads','official_features','official_derivatives','fits','permanent_updates']);paths.update(ROOT/p for p in q['source_sha256']);paths.update(p for p in folder.rglob('*') if p.is_file());paths.add(ROOT/f'artifacts/{name}_original_console_20261002.txt');qualifications.append(dict(path=(folder/'qualification.json').relative_to(ROOT).as_posix(),sha256=sha(folder/'qualification.json'),status=q['status']))
    for p in [Path(entry.__file__).resolve(),ROOT/'training/v168_decision_floor_execution_review.py',ROOT/'training/v168_decision_floor_joint_restoration.py',Path(__file__).resolve()]:ast.parse(p.read_text(encoding='utf-8-sig'))
    replay=ROOT/'artifacts/v167_observed_runtime_failure_replay_20261002/replay.json';check_bindings(read(replay)['source_sha256']);assert len(read(replay)['cases'])==6;paths.add(replay)
    registry=ROOT/plan['failure_constraints'];chain=[]
    while True:
        value=read(registry);paths.add(registry);chain.append(dict(path=registry.relative_to(ROOT).as_posix(),sha256=sha(registry)))
        if not value.get('inherit_applicable_constraints'):break
        registry=ROOT/value['inherit_applicable_constraints']
    assert chain[1]['path']=='training/review_policy/v166_observed_runtime_boundaries_v2.json';assert not entry.OUT.exists() and not entry.PLAN.exists()
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)};check_bindings(bindings);OUT.mkdir();entry.save(OUT/'bundle.json',dict(status='V168_entry_and_four_qualifications_ready_for_independent_preseal_root_review',entry='training/v168_decision_floor_diagnostic_v2.py',qualifications=qualifications,new_caps=plan['new_caps'],prior_actual_costs=plan['prior_actual_costs'],future_cumulative_caps=plan['future_cumulative_caps'],failure_registry_chain=chain,explicit_six_V167_actions=plan['failure_case_actions'],actual_V167_failure_replay_bound=True,actual_CPU_core_QPs=5,qualification_heads_and_QP_oracles_not_actual_SOC_quality=True,original_OOF_unobserved_NaN_reference_failure_and_independent_v2_preserved=True,lifecycle_v1_syntax_failure_and_independent_v2_preserved=True,independent_root_entry_review_still_required=True,physical_seal_still_required=True,official_heads=0,official_features=0,official_derivatives=0,fits=0,permanent_updates=0,execution_authority=False,source_sha256=bindings));print(json.dumps(dict(status='V168_qualification_bundle_ready',sources=len(bindings),official_calls=0)))

if __name__=='__main__':main()
