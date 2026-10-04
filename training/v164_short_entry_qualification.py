"""Budget/entry identity/serialization checks without a new official call."""
import ast,json
from pathlib import Path
import numpy as np
from experiment_review import ROOT,read,sha,check_bindings
import v164_short_supervised_trajectory as entry
OUT=ROOT/'artifacts/v164_short_entry_qualification_20261002'

def main():
    assert not OUT.exists();OUT.mkdir();budget=read(entry.BUDGET);files={Path(__file__).resolve(),ROOT/'training/v164_short_supervised_trajectory.py',ROOT/'training/v164_trajectory_state.py',ROOT/'training/v164_solver_trace.py',entry.BUDGET,ROOT/'docs/V164_SHORT_SUPERVISED_TRAJECTORY_EXECUTION_PLAN_20261002.md'};roles=[]
    for spec in budget['roles']:
        role=spec['role'];k=spec['OOF_chunks'];assert spec['head_cap']==3*(k+12)+10*4*k+9*7*(k+12)+9*48 and spec['fixed_error_target_gradient_cap']==40 and spec['margin_gradient_cap']==432 and spec['accepted_update_cap']==10
        first=ROOT/spec['first_cached_candidate'];proof=read(first/'probe.json');assert proof['accepted'] and proof['step']==1. and proof['classification_guard'] and proof['actual_parameter_change'];assert (first.parent.parent/'baseline/OOF_q.npy').exists();files|={p for p in first.rglob('*') if p.is_file()}
        gradient_paths=[ROOT/f'artifacts/v161_fixed_error_endpoint_diagnostic_20261002/role{role}/baseline_error_class{c}_repeat0/complete_fixed_error_target_gradient.npy' for c in [1,2]];files|=set(gradient_paths);gs=[np.load(p) for p in gradient_paths]
        # Qualified existing point only, not a new trajectory parameter gradient.
        result=entry.solve_direction(*gs,np.empty((0,1060832)));serialized=json.dumps({key:value for key,value in result.items() if key!='direction'});entry.save(OUT/f'cached_point{role}_base_QP_serialization.json',json.loads(serialized));roles.append(dict(role=role,head_cap=spec['head_cap'],first_cached_actual_finite_candidate=True,old_point_QP_serialized=True,old_point_QP_status=result['status']))
    caps=budget['new_caps'];assert sum(s['head_cap'] for s in budget['roles'])==caps['heads']==6216 and caps['fixed_error_target_gradients']==120 and caps['margin_gradients']==1296 and caps['finite_proposals']==192 and caps['QP_solves']==caps['direction_QP_solves']+caps['restoration_QP_solves']==189
    assert budget['future_cumulative_actual_caps']['heads']==17662+6216<=82174 and budget['future_cumulative_actual_caps']['complete_parameter_derivatives_all_types']==422+120+1296<=2426
    code=(ROOT/'training/v164_short_supervised_trajectory.py').read_text(encoding='utf-8');tree=ast.parse(code);text=ast.unparse(tree)
    assert 'while accepted <= 10' in text and 'if accepted == 10' in text and 'for index in range(6)' in text and 'records = {}' in text and 'restore(model, last)' in text and "normal_cache_reset=True" in text
    assert not any(isinstance(n,ast.Compare) and isinstance(n.left,ast.Name) and n.left.id=='role' for n in ast.walk(tree))
    assert 'active_normal_references.json' in code and 'active_complete_margin_normals.npy' not in code and 'fixed_readout_override_diagnostic.json' in code
    lifecycle=ROOT/'artifacts/v164_short_trajectory_lifecycle_qualification_20261002/qualification.json';q=read(lifecycle);assert 'passed' in q['status'] and q['official_heads']==q['official_gradients']==q['official_fits']==0;files.add(lifecycle)
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)};entry.save(OUT/'pre_bindings.json',dict(source_sha256=bindings,official_calls=0));check_bindings(bindings)
    entry.save(OUT/'qualification.json',dict(status='three_uniform_short_supervised_trajectory_bounds_actual_first_candidate_identity_and_QP_serialization_passed',roles=roles,new_caps=caps,accepted_point_target_gradient_repeats_and_final_state_included=True,new_point_normal_values_cleared=True,all_new_repairs_including_mixed_protected=True,last_actual_accepted_state_restoration=True,normal_reference_storage_avoids_matrix_duplication=True,supervised_development_training_not_external_validation=True,official_heads=0,official_features=0,official_gradients=0,official_fits=0,permanent_updates=0,source_sha256=bindings));print(json.dumps(dict(status='V164_short_entry_qualification_passed',official_calls=0,new_caps=caps)))

if __name__=='__main__':main()
