"""Check that all old scope guards are covered by the fixed A initialization."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/v155_independent_guard_containment_20261001'
TRAIN = ROOT / 'artifacts/v153_independent_training_transfer_gap_20261001/both_arms_legal_TRAIN_outer_control.parquet'
REGISTRIES = [
    ('V138', 'artifacts/v138_single_issue_round1_20260930/scoped_training_capabilities.json', 'verified_training_scopes'),
    ('V140', 'artifacts/v140_ensemble_training_round2_20261001/additional_verified_TRAIN_scopes.json', 'scopes'),
    ('V142', 'artifacts/v142_second_layer_training_20261001/verified_TRAIN_mastery_registry.json', 'scopes')]


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    assert not OUT.exists()
    d = pd.read_parquet(TRAIN)
    d = d[d.arm.eq('A')].copy()
    assert len(d) == 225614 and not d.duplicated(['training_role', 'row_position']).any()
    assert d.fold.ne(d.training_role).all()
    good = d[d.pred.eq(d.truth)].set_index(['training_role', 'row_position'])
    assert len(good) == 225558
    results, bindings = [], {str(TRAIN.relative_to(ROOT)): sha(TRAIN)}
    for version, name, field in REGISTRIES:
        registry = ROOT / name
        bindings[name] = sha(registry)
        obj = json.loads(registry.read_text(encoding='utf-8'))
        for item in obj[field]:
            path = ROOT / item['guard']
            assert sha(path) == item['guard_sha256']
            bindings[item['guard']] = sha(path)
            g = pd.read_parquet(path)
            assert not g.duplicated(['training_role', 'row_position']).any()
            idx = pd.MultiIndex.from_frame(g[['training_role', 'row_position']])
            assert idx.isin(good.index).all(), 'Protected scope not covered by actual own-role initial correct predicate'
            actual = good.reindex(idx)
            assert np.array_equal(g.truth.to_numpy(), actual.truth.to_numpy())
            assert np.array_equal(g.truth.to_numpy(), actual.pred.to_numpy())
            results.append({'version': version, 'scope_id': item['id'], 'rows': len(g),
                            'training_roles': sorted(int(v) for v in g.training_role.unique()),
                            'contained_in_actual_V146_A_own_role_correct': True})
    OUT.mkdir()
    audit = {'status': 'all_old_correct_guard_scopes_contained_in_actual_initial_A_correct_rows',
             'own_model_forwards': 0, 'own_gradients': 0, 'own_fits': 0, 'own_updates': 0,
             'original_TRAIN_role_rows': 225614, 'initial_correct_TRAIN_role_rows': 225558,
             'scopes': results, 'source_sha256': {**bindings, str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))},
             'scope': 'For these exact data/role/model identities, maintaining every initial-correct TRAIN row implies scope retention. Future full joined retention must still execute; no source-out or task quality guarantee.'}
    (OUT / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in audit.items() if k not in ['source_sha256']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
