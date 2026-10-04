"""Preserve v1, bind explicit review adapter and v2 real entry before calls."""
import ast,copy,json
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings
from experiment_review_v159 import PROTOCOL,review_plan
from v159_boundary_runtime_v2 import PLAN,save,review

OUT=ROOT/'artifacts/v159_execution_protocol_contract_v2_20261002'

def main():
    assert not PLAN.exists() and not OUT.exists();OUT.mkdir()
    previous=ROOT/'training/review_policy/v159_boundary_execution_contract.json';p=copy.deepcopy(read(previous))
    p.update(protocol=PROTOCOL,risk_registry='training/review_policy/history_cases.json',activation_entries=['training/v159_boundary_train_v2.py','training/v159_boundary_evaluate_v2.py'],
        previous_pre_execution_contract=previous.relative_to(ROOT).as_posix(),experiment_review_adapter='training/experiment_review_v159.py',historical_experiment_review_unchanged=True)
    paths=[Path(__file__).resolve(),previous]+[ROOT/k for k in p['source_sha256']]+[ROOT/s for s in [
        'training/experiment_review_v159.py','training/v159_boundary_runtime_v2.py','training/v159_boundary_train_v2.py','training/v159_boundary_evaluate_v2.py',
        'training/v159_protocol_synthetic_qualification.py','artifacts/v159_protocol_synthetic_qualification_20261002/qualification.json',
        'training/v159_saved_CSR_identity_audit.py','artifacts/v159_saved_CSR_identity_audit_20261002/audit.json',
        'training/v159_independent_combined_guard_audit.py','artifacts/v159_independent_combined_guard_audit_20261002/audit.json',
        'artifacts/v159_boundary_input_preparation_20261002/qualification.json','artifacts/v159_boundary_input_preparation_20261002/pre_array_bindings.json',
        'docs/V159_VERSIONED_REVIEW_PROTOCOL_SUPPLEMENT.md']]
    for f in range(3):paths.extend((ROOT/f'artifacts/v159_boundary_input_preparation_20261002/fold{f}').glob('*'))
    for path in paths:
        if path.suffix=='.py':ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
    p['source_sha256']={q.relative_to(ROOT).as_posix():sha(q) for q in sorted(set(paths))}
    save(OUT/'pre_contract_bindings.json',dict(status='actual_protocol_entries_CSR_guard_sources_bound_before_official_functions',source_sha256=p['source_sha256']))
    check_bindings(read(ROOT/'artifacts/v159_protocol_synthetic_qualification_20261002/qualification.json')['source_sha256'])
    r=review_plan(p);assert r['plan_review_passed'];save(PLAN,p);review()
    save(OUT/'qualification.json',dict(status='versioned_review_protocol_and_real_v2_entries_bound_pending_independent_review_actual_seal_and_preflight',protocol=PROTOCOL,
        contract_path=PLAN.relative_to(ROOT).as_posix(),contract_sha256=sha(PLAN),candidate_module=p['candidate_module'],review=r,
        official_classifier_calls=0,official_features_calls=0,official_gradients=0,official_fits=0,official_updates=0,official_runtime_sealed=False,official_zero_step_replay=False,quality_acceptance=False,
        total_future_caps=p['total_future_caps'],source_sha256={Path(__file__).resolve().relative_to(ROOT).as_posix():sha(__file__),PLAN.relative_to(ROOT).as_posix():sha(PLAN)}))
    print(json.dumps(dict(status='new_review_protocol_contract_bound_no_official_calls',entries=p['activation_entries'],review_passed=r['plan_review_passed'],official_calls=0),ensure_ascii=False))

if __name__=='__main__':main()
