"""Read-only concrete source/contract review; never execute official models."""
import ast
import json
import hashlib
from pathlib import Path

from experiment_review_v159_v2 import review_plan, check_bindings

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/v159_independent_entry_activation_review_20261002'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def function(p, name):
    tree = ast.parse(p.read_text(encoding='utf-8-sig'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(nodes) == 1
    return ast.dump(nodes[0], include_attributes=False)


def main():
    assert not OUT.exists()
    OUT.mkdir()
    plan_path = ROOT / 'training/review_policy/v159_boundary_execution_contract_v3.json'
    p = json.loads(plan_path.read_text(encoding='utf-8'))
    review = review_plan(p)
    assert review['plan_review_passed']
    q_path = ROOT / 'artifacts/v159_protocol_synthetic_qualification_v2_20261002/qualification.json'
    q = json.loads(q_path.read_text(encoding='utf-8'))
    check_bindings(q['source_sha256'])
    assert all(q['cases'].values()) and q['official_classifier_calls'] == q['official_gradients'] == q['official_fits'] == 0
    assert p['total_future_caps']['fits'] == 6
    assert p['total_future_caps']['classifier_forward_chunks'] == p['total_future_caps']['opinion_feature_blocks'] == 78072
    assert p['total_future_caps']['full_class_gradients'] == 2412
    assert p['total_future_caps']['evaluation_classifier_forward_chunks'] == 720
    assert 'data/official/train.parquet' in p['source_sha256']
    assert p['source_sha256']['data/official/train.parquet'] == '6b6d5e23caebfd1c4f6b70c9e58c27f437bca7f0cd26497eefa3e4908f2cb742'
    original = ROOT / 'training/v159_boundary_train.py'
    actual = ROOT / 'training/v159_boundary_train_v3.py'
    same = {n: function(original, n) == function(actual, n) for n in ('context', 'inputs', 'probabilities', 'risk', 'stats', 'rows', 'assign', 'restore')}
    assert all(same.values())
    e = (ROOT / 'training/v159_boundary_evaluate_v3.py').read_text(encoding='utf-8')
    assert "for item in r['last5']:" in e and "ov,oq,olp,_=risk(model,ctx,'OOF',ctx['ids'])" in e
    assert "actual_OOF_classifier_replay=True" in e
    runtime = (ROOT / 'training/v159_boundary_runtime_v3.py').read_text(encoding='utf-8')
    assert 'from experiment_review_v159_v2 import review_plan,seal_run,require_run_seal,require_checkpoint' in runtime
    combined_path = ROOT / 'artifacts/v159_independent_combined_guard_audit_20261002/audit.json'
    combined = json.loads(combined_path.read_text(encoding='utf-8'))
    check_bindings(combined['source_sha256'])
    assert all(z['targets_compatible'] for z in combined['roles'])
    paths = [Path(__file__).resolve(), plan_path, q_path, combined_path, actual,
             ROOT / 'training/v159_boundary_evaluate_v3.py', ROOT / 'training/v159_boundary_runtime_v3.py',
             ROOT / 'training/experiment_review_v159_v2.py']
    out = dict(status='concrete_entry_eligible_for_prospective_seal_and_real_preflight_not_quality_acceptance',
               protocol=p['protocol'], plan_review=review, protocol_synthetic_cases=q['cases'],
               actual_gradient_input_and_guard_functions_unchanged_from_reviewed_entry=same,
               last_five_actual_OOF_and_deployment_replay_required=True,
               combined_input_and_retention_targets_compatible=True,
               source_sha256={f.relative_to(ROOT).as_posix(): sha(f) for f in paths},
               official_classifier_calls=0, official_feature_calls=0, official_gradients=0, official_fits=0, official_updates=0,
               permission_scope='Worker may seal/register and execute 336 head/feature chunks and 12 preflight class gradients; no parameter update until real preflight passes.',
               quality_acceptance=False, model_promoted=False)
    (OUT / 'audit.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in out.items() if k not in ('source_sha256', 'protocol_synthetic_cases')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
