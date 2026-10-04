"""Exact identities of registered, frozen cache tensors; no learned probes.

Distinct cache bytes are not proof of field decodability or portable evidence.
Numerical variants within a canonical input remain one underlying observation.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/v149_independent_first_cache_identity_20261001'


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def hashes(arrays):
    result = []
    for i in range(len(arrays[0])):
        h = hashlib.sha256()
        for a in arrays:
            value = np.asarray(a[i]).copy()
            value[value == 0] = 0  # Signed zero carries no field information.
            h.update(value.dtype.str.encode())
            h.update(value.shape.__repr__().encode())
            h.update(value.tobytes())
        result.append(h.hexdigest())
    # Reject hypothetical digest collisions rather than treating them as equal inputs.
    buckets = {}
    for i, key in enumerate(result):
        if key in buckets:
            assert all(np.array_equal(a[i], a[buckets[key]]) for a in arrays)
        else:
            buckets[key] = i
    return np.array(result, dtype=object)


def profile(frame, key):
    ct = frame.groupby([key, 'truth']).size().unstack(fill_value=0).reindex(columns=[1, 2], fill_value=0)
    mixed = ct[(ct > 0).sum(1) > 1]
    canonical_mass = frame.groupby(key).canonical_key.nunique()
    return dict(original_rows=len(frame), unique_numeric_states=len(ct), mixed_numeric_states=len(mixed),
        original_rows_in_mixed_numeric_states=int(mixed.sum().sum()),
        empirical_label_collision_minimum_errors=int((mixed.sum(1) - mixed.max(1)).sum()),
        states_merging_different_underlying_CSR_inputs=int(canonical_mass.gt(1).sum()),
        original_rows_in_those_merges=int(frame[key].isin(canonical_mass[canonical_mass.gt(1)].index).sum()))


def main():
    assert not OUT.exists(), 'Preserve completed evidence.'
    identity_path = ROOT / 'artifacts/v147_independent_observability_20261001/observable_identity_ledger.parquet'
    fields_path = ROOT / 'artifacts/v149_independent_input_fidelity_20261001/all_ASA_input_decoding_ledger.parquet'
    seal_path = ROOT / 'artifacts/v146_guarded_pair_training_20261001/run_seal.json'
    cache = ROOT / 'artifacts/v138_single_issue_round1_20260930'
    prior = ROOT / 'artifacts/v141_representation_evidence_20261001'
    d = pd.read_parquet(identity_path)
    fields = pd.read_parquet(fields_path)
    assert np.array_equal(d.row_position, fields.row_position)
    assert np.array_equal(d.truth, fields.truth)
    seal = json.loads(seal_path.read_text(encoding='utf-8'))
    bindings = seal['source_sha256']
    facts_path = cache / 'facts.npy'
    assert sha(facts_path) == bindings[facts_path.relative_to(ROOT).as_posix()]
    facts = np.load(facts_path, mmap_mode='r')
    assert facts.shape == (22546, 495) and np.isfinite(facts).all()
    sources = [Path(__file__), identity_path, fields_path, seal_path, facts_path]
    OUT.mkdir()
    records = []
    for fold in range(3):
        h1_path, h2_path = prior / f'fold{fold}_h1.npy', cache / f'fold{fold}_hidden.npy'
        for p in [h1_path, h2_path]:
            assert sha(p) == bindings[p.relative_to(ROOT).as_posix()]
        h1, h2 = np.load(h1_path, mmap_mode='r'), np.load(h2_path, mmap_mode='r')
        assert h1.shape == h2.shape == (22546, 16, 128)
        assert np.isfinite(h1).all() and np.isfinite(h2).all()
        sources.extend([h1_path, h2_path])
        f = d.copy()
        f['h1'] = hashes([h1])[d.local.to_numpy()]
        f['h1_and_facts'] = hashes([h1, facts])[d.local.to_numpy()]
        f['actual_second_entry_and_direct_facts'] = hashes([h1, h2, facts])[d.local.to_numpy()]
        info = dict(fold=fold, all_cache_dependencies_match_completed_V146_seal=True,
            views={k:{r:profile(f.loc[mask], k) for r,mask in
                [('all_observed',np.ones(len(d),dtype=bool)),('legal_TRAIN',~d.fold.eq(fold)),('source_HELD',d.fold.eq(fold))]}
                for k in ['canonical_key','h1','h1_and_facts','actual_second_entry_and_direct_facts']})
        canonical_states = f.groupby('canonical_key').h1.nunique()
        info['canonical_inputs_with_multiple_h1_byte_states'] = int(canonical_states.gt(1).sum())
        field_conflicts = []
        for name in [k for k in fields.columns if k.startswith('decoded_')]:
            vals = fields[name].map(lambda a: '<unobserved>' if pd.isna(a) else str(a))
            conflicts = f.assign(field_value=vals).groupby('h1').field_value.nunique()
            field_conflicts.append(dict(field=name, h1_states_with_multiple_decoded_field_values=int(conflicts.gt(1).sum())))
        info['exact_first_cache_field_collisions'] = field_conflicts
        hard = fields.known_578_cohort & d.fold.eq(fold)
        info['known_578_HELD_h1_merged_with_other_numeric_inputs'] = int(
            f.loc[hard, 'h1'].isin(f.groupby('h1').canonical_key.nunique().loc[lambda a:a.gt(1)].index).sum())
        records.append(info)
        f[['row_position','local','root','fold','truth','canonical_key','h1','h1_and_facts',
            'actual_second_entry_and_direct_facts']].to_parquet(OUT / f'fold{fold}_cache_identity_ledger.parquet',index=False)
        print(json.dumps(dict(fold=fold, all_h1=info['views']['h1']['all_observed'],
            current_available=info['views']['actual_second_entry_and_direct_facts']['all_observed'],
            canonical_inputs_with_multiple_h1_byte_states=info['canonical_inputs_with_multiple_h1_byte_states'],
            hard_HELD_merge_rows=info['known_578_HELD_h1_merged_with_other_numeric_inputs'],
            first_cache_field_collisions=field_conflicts),ensure_ascii=False),flush=True)
    result = dict(status='registered_cache_numeric_identity_recount_not_information_learning_test',
        latest_actual_training='V146', new_model_forwards=0, new_gradients=0,new_fits=0,new_updates=0,
        roles=records, identity_rule='Exact finite stored values, dtype and shape; signed zeros canonicalized.', limits=[
            'The first layer and cache tensors remain unchanged by V142/V146; this audit does not replay their current second-layer features.',
            'Different states do not prove a field can be accurately decoded or that the classifier uses it correctly.',
            'Canonical input byte variants from numerical/cache effects are not new independent behavior observations.',
            'Exact collision floors are empirical diagnostics for recorded states/populations, not Bayes bounds or source-transfer quality.',
            'No model gradients, new learned decoder, classification target, candidate selection or weighting is performed.'
        ],source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sources})
    (OUT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
