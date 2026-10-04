"""Preserve and replay actual V167 failures without calling the model."""
import json,traceback
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings

OUT=ROOT/'artifacts/v167_observed_runtime_failure_replay_20261002'
REGISTRY=ROOT/'training/review_policy/v167_observed_runtime_boundaries.json'
TRIAL=ROOT/'artifacts/v167_trial_point_restoration_diagnostic_20261002'
QUALITY=ROOT/'artifacts/v167_saved_trial_point_quality_review_20261002/review.json'
GEOMETRY=ROOT/'artifacts/v167_saved_remaining_argmax_geometry_review_20261002/review.json'
AUDIT=ROOT/'artifacts/v167_independent_actual_trial_point_review_v2_20261002/review.json'
ROOT_GEOMETRY=ROOT/'artifacts/v167_root_decision_tie_and_floor_review_20261002/review.json'

def save(path,value):path.write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))

def main():
    assert not OUT.exists();OUT.mkdir()
    registry=read(REGISTRY);quality=read(QUALITY);geometry=read(GEOMETRY);audit=read(AUDIT);root_geometry=read(ROOT_GEOMETRY)
    sources={Path(__file__).resolve(),REGISTRY,QUALITY,GEOMETRY,AUDIT,ROOT_GEOMETRY,ROOT/'training/experiment_review.py',ROOT/registry['inherit_applicable_constraints']}
    for p in [QUALITY,GEOMETRY,AUDIT.parent/'pre_review_bindings.json',ROOT_GEOMETRY.parent/'pre_bindings.json']:
        sources.add(p);check_bindings(read(p)['source_sha256'])
    for r in range(3):sources.add(TRIAL/f'role{r}/diagnostic.json')
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(sources)};save(OUT/'pre_bindings.json',dict(source_sha256=bindings,official_calls=0))
    stages=geometry['roles'][1]['stages'];assert len(stages)==2
    a,b=[s['all_actual_blocker_geometry'][0] for s in stages]
    assert all(s['blocking_original_rows']==2 and s['blocking_measured_functions']==1 and s['all_actual_functions_still_covered'] for s in stages)
    assert a['identity']==b['identity'] and a['local']==b['local']==21985 and a['truth']==b['truth']==2 and b['rival']==1
    assert a['actual_log_probability_margin']<0 and a['predicted_log_probability_margin']>=0 and a['original_unit_inequality_passed']
    assert b['actual_probability_margin']==b['actual_log_probability_margin']==b['predicted_log_probability_margin']==0 and b['exact_probability_tie'] and b['exact_log_probability_tie']
    assert b['actual_argmax']==1 and b['probabilities'][1]==b['probabilities'][2]==0.49999999999448597 and b['original_unit_inequality_passed']
    assert root_geometry['status']=='V167_exact_actual_argmax_tie_and_saved_trial_geometry_independently_confirmed' and root_geometry['official_heads']==root_geometry['official_derivatives']==0 and root_geometry['no_new_actual_candidate_or_classification_gain_claim']
    candidates=quality['roles'][1]['all_finite_candidates'];assert len(candidates)==2
    for c in candidates:
        assert not c['actual_candidate_accepted'] and c['pure_errors_before']==1952 and c['pure_errors_after']==1950
        assert c['paired_vs_V164']['M']['repairs_vs_previous_accepted']==4 and c['paired_vs_V164']['S']['new_errors_vs_previous_accepted']==2
        assert c['paired_vs_V166']['M']['new_errors_vs_previous_accepted']==12 and c['paired_vs_V166']['S']['repairs_vs_previous_accepted']==8
    diag=read(TRIAL/'role1/diagnostic.json');assert diag['joint_restoration_attempts']==diag['actual_QP_solves']==2 and diag['optimizer_iterations']==4 and not diag['actual_finite_accepted']
    assert [r['actual_finite_accepted'] for r in quality['roles']]==[True,False,True] and all(quality['roles'][r]['unchanged_V166_control_replay_not_new_gain'] for r in [0,2])
    assert not audit['all_three_actual_finite_guards_passed'] and not audit['supports_new_short_training_registration']
    assert quality['actual_new_heads']==audit['new_heads']==296 and quality['actual_new_complete_margin_derivatives']==audit['new_complete_margin_derivatives']==100 and quality['actual_QP_solves']==2
    for r in range(3):
        d=read(TRIAL/f'role{r}/diagnostic.json');assert d['exception'] is None and d['all_parameters_restored'] and d['origin_parameter_sha256']==d['restored_parameter_sha256'] and d['new_fits']==d['permanent_updates']==0
        assert quality['roles'][r]['all_restored_full_tensors_exact'] and quality['roles'][r]['all_restored_original_argmax_exact']
    cases={r['id']:dict(passed=True,evidence=r['evidence']) for r in registry['cases']}
    check_bindings(bindings);save(OUT/'replay.json',dict(status='V167_six_actual_runtime_constraints_replayed',cases=cases,exact_tie_geometry=b,official_heads=0,official_features=0,official_derivatives=0,optimizer_calls=0,fits=0,permanent_updates=0,actual_cost_not_reset=dict(heads=19080,derivatives=718),no_new_execution_authority=True,classification_mastery=False,root_goal_complete=False,source_sha256=bindings));print(json.dumps(dict(status='V167_actual_runtime_failures_replayed',cases=len(cases),official_calls=0)))

if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():save(OUT/'failure.json',dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),official_calls=0))
        raise
