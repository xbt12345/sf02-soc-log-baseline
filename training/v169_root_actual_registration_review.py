"""Verify actual registration receipts and resources without model calls."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/v169_root_actual_registration_review_20261002'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()


def main():
    assert not OUT.exists()
    run = ROOT/'artifacts/v169_prior_pair_training'
    registration_path = run/'registration.json'
    seal_path = run/'run_seal.json'
    registration, seal = read(registration_path), read(seal_path)
    plan_path = ROOT/seal['plan_path']; plan = read(plan_path)
    root_path = ROOT/seal['root_review_path']; root = read(root_path)
    bundle_path = ROOT/plan['dependency_bundle']; bundle = read(bundle_path)
    assert registration['status'] == 'V169_physically_sealed_before_official_calls'
    assert registration['run_seal_sha256'] == sha(seal_path)
    assert seal['plan_sha256'] == sha(plan_path)
    assert seal['root_review_sha256'] == sha(root_path)
    assert plan['independent_preseal_review_sha256'] == sha(root_path)
    assert plan['dependency_bundle_sha256'] == sha(bundle_path)
    assert root['supports_physical_seal'] and not root['execution_authority']
    assert plan['execution_authority'] and plan['new_fit_permission']
    assert registration['new_caps'] == plan['new_caps'] == root['prospective_caps']
    assert registration['resources'] == plan['resources'] == root['resources'] == bundle['resources']
    assert all(v == seal['source_sha256'][k] for k, v in bundle['source_sha256'].items())
    assert registration['physical_sources'] == len(seal['source_sha256']) == 18971
    entry_path = ROOT/seal['trainer_path']
    assert sha(entry_path) == registration['entry_sha256'] == root['reviewed_entry_sha256']
    assert entry_path.name == 'v169_prior_pair_training_entry_v14.py'
    snapshot, resources = plan['preseal_resource_snapshot'], plan['resources']
    fields = [('disk', 'free_disk_bytes', 'minimum_free_disk_start_bytes'),
              ('RAM', 'free_RAM_bytes', 'minimum_free_RAM_bytes'),
              ('commit', 'free_commit_bytes', 'minimum_free_commit_bytes'),
              ('GPU', 'free_GPU_bytes', 'minimum_free_GPU_bytes')]
    actual = {}
    for kind, observed, required in fields:
        assert snapshot[observed] >= resources[required]
        actual[kind] = dict(available_bytes=snapshot[observed],
                           minimum_bytes=resources[required],
                           headroom_bytes=snapshot[observed]-resources[required])
    assert all(registration[k] == root[k] == 0 for k in
               ['official_heads', 'official_features', 'official_derivatives', 'fits', 'permanent_updates'])
    for name in [seal['plan_path'], seal['root_review_path'], plan['dependency_bundle'], seal['trainer_path']]:
        assert seal['source_sha256'][name] == sha(ROOT/name)
    files = [Path(__file__).resolve(), registration_path, seal_path, plan_path,
             root_path, bundle_path, entry_path]
    result = dict(status='V169_actual_physical_registration_and_prospective_scope_independently_verified',
        all_checks_passed=True, actual_seal_exists=True, actual_resources_at_seal=actual,
        inherited_bundle_bindings_preserved=18967, total_seal_bindings=18971,
        entry_sha256=registration['entry_sha256'], registered_caps=plan['new_caps'],
        actual_training_progress_not_inferred_from_registration=True,
        no_model_quality_or_promotion_claim=True, execution_authority=False,
        official_heads=0, official_features=0, official_derivatives=0, fits=0, permanent_updates=0,
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files})
    OUT.mkdir()
    (OUT/'review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'source_sha256'},ensure_ascii=False))


if __name__ == '__main__':
    main()
