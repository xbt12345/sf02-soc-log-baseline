"""Read-only V150 recount against official rows and independent coordinate decoding.

No model/decoder imports, forwards, gradient computation, fitting or updates.
Saved decoder predictions are audited, not independently regenerated.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'artifacts/v150_field_readout_diagnostic_20261001'
OUT = ROOT / 'artifacts/v150_independent_result_review_20261001'
FIELDS = ('action', 'outcome', 'transport_protocol', 'src_role', 'dst_role',
          'src_port_range', 'dst_port_range', 'icmp_message', 'icmp_unreachable',
          'src_port_fixed', 'dst_port_fixed', 'icmp_type', 'icmp_code')
LAYERS = ('H1', 'V146_B_H2')
ROLES = ('readout_FIT', 'inner_HELD', 'outer_HELD')


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def objects(s):
    return s.astype(object).where(s.notna(), None).to_numpy()


def observed(f, name):
    v, proto = f.get(name), f.get('transport_protocol')
    if name in ('src_port_fixed', 'dst_port_fixed'):
        return proto in ('tcp', 'udp') and type(v) is int and 0 <= v <= 65535
    if name in ('src_port_range', 'dst_port_range'):
        return observed(f, name.replace('_range', '_fixed')) and v in ('system', 'user', 'dynamic')
    if name in ('icmp_type', 'icmp_code'):
        return proto == 'icmp' and type(v) is int and 0 <= v <= 255
    if name.startswith('icmp_'):
        return proto == 'icmp' and isinstance(v, str) and v not in ('', 'unknown')
    if name.endswith('_role'):
        return v in ('inside', 'outside', 'dmz')
    return isinstance(v, str) and v not in ('', 'unknown')


def main():
    assert not OUT.exists(), 'Do not overwrite completed independent evidence'
    input_audit = ROOT / 'artifacts/v149_independent_input_fidelity_20261001/audit.json'
    direct_path = input_audit.parent / 'all_ASA_input_decoding_ledger.parquet'
    trace_path = ROOT / 'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet'
    identity_path = ROOT / 'artifacts/v147_independent_observability_20261001/observable_identity_ledger.parquet'
    support_path = ROOT / 'artifacts/v147_independent_conditional_support_20261001/all_known_port_conditional_support.parquet'
    official_path = ROOT / 'data/official/train.parquet'
    direct = pd.read_parquet(direct_path).sort_values('row_position').reset_index(drop=True)
    trace = pd.read_parquet(trace_path).sort_values('row_position').reset_index(drop=True)
    identity = pd.read_parquet(identity_path).sort_values('row_position').reset_index(drop=True)
    support = pd.read_parquet(support_path).set_index('row_position')
    for ref in (trace, identity):
        for k in ('row_position', 'local', 'root', 'fold', 'truth'):
            assert np.array_equal(direct[k], ref[k])
    assert len(direct) == 112807 and not direct.row_position.duplicated().any()
    original = trace.set_index('row_position'); official_seen = offset = 0
    for batch in pq.ParquetFile(official_path).iter_batches(batch_size=32768, columns=['message_sanitized', 'label_binary']):
        data = batch.to_pandas()
        pos = direct.row_position[(direct.row_position >= offset) & (direct.row_position < offset + len(data))].to_numpy()
        if len(pos):
            got = data.iloc[pos - offset]; want = original.loc[pos]
            assert np.array_equal(got.message_sanitized, want.raw_message)
            assert np.array_equal(got.label_binary.map({'benign': 0, 'malicious': 1, 'suspicious': 2}), want.truth)
            official_seen += len(pos)
        offset += len(data)
    assert official_seen == 112807 and offset == 2056871
    previous = read(input_audit)
    assert sha(official_path) == previous['source_sha256']['data/official/train.parquet']
    facts = trace.facts_json.map(json.loads).tolist()
    expected = {f: objects(direct['decoded_' + f]) for f in FIELDS}
    masks = {f: np.array([observed(v, f) for v in facts]) for f in FIELDS}
    proto = expected['transport_protocol']
    inner = np.array([int(hashlib.sha256(('V150-field-readout:' + str(int(r))).encode()).hexdigest(), 16) % 5 == 0 for r in direct.root])
    ids = identity[['local', 'canonical_key']].drop_duplicates().sort_values('local')
    mapping = ids.groupby('canonical_key').local.min()
    reps = ids.set_index('local').canonical_key.map(mapping).reindex(range(22546)).to_numpy()
    assert np.array_equal(reps, np.load(SRC / 'canonical_min_local.npy'))
    seal = read(SRC / 'run_seal.json'); plan = ROOT / 'training/review_policy/v150_field_readout_plan.json'
    assert seal['status'] == 'sealed_before_diagnostic_fit' and seal['plan_sha256'] == sha(plan)
    # Check the actual prospective trainer/decoder/plan. Other runtime bindings remain worker receipts.
    for p in (plan, ROOT / 'training/v150_field_decoder.py', ROOT / 'training/v150_field_readout_diagnostic.py'):
        assert sha(p) == seal['source_sha256'][p.relative_to(ROOT).as_posix()]
    aggregate, scope, receipts, bundles = [], [], [], {z: [] for z in LAYERS}
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in
              (Path(__file__), input_audit, direct_path, trace_path, identity_path, support_path, official_path, plan, SRC / 'run_seal.json')}
    for fold in range(3):
        role = np.where(direct.fold.to_numpy() == fold, 'outer_HELD', np.where(inner, 'inner_HELD', 'readout_FIT'))
        fit = role == 'readout_FIT'; fitkeys = set(identity.canonical_key[fit])
        canonical_seen = identity.canonical_key.isin(fitkeys).to_numpy()
        roots = {z: set(direct.root[role == z]) for z in ROLES}
        assert not any(roots[a] & roots[b] for i, a in enumerate(ROLES) for b in ROLES[i + 1:])
        reference_prob = ROOT / f'artifacts/v146_guarded_pair_training_20261001/fold{fold}_B/sealed_all_prob.npy'
        classification = np.load(reference_prob)[direct.local.to_numpy()].argmax(1)
        hashes[reference_prob.relative_to(ROOT).as_posix()] = sha(reference_prob)
        for layer in LAYERS:
            folder = SRC / f'fold{fold}_{layer}'; receipt = read(folder / 'fit.json'); started = read(folder / 'started.json')
            rows_path = folder / 'all_original_field_ledger.parquet'; decoder_path = folder / 'decoder.pt'
            assert sha(rows_path) == receipt['rows_sha256'] and sha(decoder_path) == receipt['decoder_sha256']
            assert receipt['seal_sha256'] == sha(SRC / 'run_seal.json') == started['seal_sha256']
            assert receipt['initial_classifier_sha256'] == started['classifier_state_before']
            assert receipt['status'] == 'diagnostic_decoder_fit_executed'
            assert receipt['diagnostic_decoder_fits'] == receipt['normal_equation_solves'] == 1
            assert receipt['classifier_fits'] == receipt['classifier_gradients'] == receipt['classifier_updates'] == 0
            assert receipt['classifier_state_unchanged'] and receipt['ridge'] == 0.0001
            assert receipt['solve']['normal_equation_relative_residual'] < 1e-8
            r = pd.read_parquet(rows_path).sort_values('row_position').reset_index(drop=True)
            assert len(r) == 112807 and not r.row_position.duplicated().any()
            for k in ('row_position', 'local', 'root', 'fold', 'truth'):
                assert np.array_equal(r[k], direct[k])
            assert np.array_equal(r.canonical_key, identity.canonical_key)
            assert np.array_equal(r.readout_role, role) and np.array_equal(r.canonical_seen_by_decoder, canonical_seen)
            assert np.array_equal(r.classifier_pred_frozen, classification)
            exacts = {}
            for field in FIELDS:
                want = expected[field]; got = objects(r[field + '_decoded']); known = masks[field]
                exact = got == want; exacts[field] = exact
                assert np.array_equal(exact, r[field + '_exact']) and np.array_equal(known, r[field + '_observed'])
                counts = Counter(str(v) for v in want[fit]); modekey = sorted(counts, key=lambda z: (-counts[z], z))[0]
                mode = next(v for v in want[fit] if str(v) == modekey); mode_exact = want == mode
                assert np.array_equal(mode_exact, r[field + '_mode_exact'])
                applicable = np.isin(proto, ('tcp', 'udp')) if field.startswith(('src_port', 'dst_port')) else proto == 'icmp' if field.startswith('icmp_') else np.ones(len(r), bool)
                for name in ROLES:
                    pop = role == name
                    for stratum, mask in {'observed': known, 'unknown_applicable': applicable & ~known,
                                          'not_applicable': ~applicable, 'observed_zero': known & (want == 0)}.items():
                        take = pop & mask
                        aggregate.append(dict(fold=fold, layer=layer, role=name, field=field, stratum=stratum,
                                              original_rows=int(take.sum()), errors=int((take & ~exact).sum()),
                                              mode_errors=int((take & ~mode_exact).sum())))
            all_observed = np.logical_and.reduce([~masks[f] | exacts[f] for f in FIELDS])
            all_states = np.logical_and.reduce([exacts[f] for f in FIELDS])
            ports_exact = masks['src_port_fixed'] & masks['dst_port_fixed'] & exacts['src_port_fixed'] & exacts['dst_port_fixed']
            wrong = classification != direct.truth.to_numpy()
            hard = direct.known_578_cohort.to_numpy(); outer = role == 'outer_HELD'
            summary = direct[['row_position', 'local', 'root', 'fold', 'truth']].copy()
            summary['classifier_pred_frozen'] = classification; summary['known_578_cohort'] = hard
            summary['all_observed_exact'] = all_observed; summary['all_states_exact'] = all_states
            summary['complete_ports_exact'] = ports_exact; summary['canonical_seen_by_decoder'] = canonical_seen
            summary = summary.join(support[['same_class_rows', 'same_class_roots', 'opposite_class_rows']], on='row_position')
            for name in ROLES:
                pop = role == name
                for tag, mask in {'all_original': np.ones(len(r), bool), 'known_578': hard,
                                  'wrong_M': wrong & direct.truth.eq(1).to_numpy(), 'wrong_S': wrong & direct.truth.eq(2).to_numpy(),
                                  'correct_M': ~wrong & direct.truth.eq(1).to_numpy(), 'correct_S': ~wrong & direct.truth.eq(2).to_numpy(),
                                  'canonical_unseen_decoder': ~canonical_seen}.items():
                    take = pop & mask
                    scope.append(dict(fold=fold, layer=layer, role=name, stratum=tag, original_rows=int(take.sum()),
                                      classifier_errors=int((take & wrong).sum()), all_observed_exact=int((take & all_observed).sum()),
                                      all_states_exact=int((take & all_states).sum()), complete_ports_exact=int((take & ports_exact).sum()),
                                      fields_exact_but_classifier_wrong=int((take & all_observed & wrong).sum())))
            bundles[layer].append(summary[outer])
            receipts.append(dict(fold=fold, layer=layer, diagnostic_fits=1, classifier_fits=0,
                                 fit_rows=int(fit.sum()), inner_rows=int((role == 'inner_HELD').sum()), outer_rows=int(outer.sum()),
                                 inner_canonical_seen=int((canonical_seen & (role == 'inner_HELD')).sum()),
                                 outer_canonical_seen=int((canonical_seen & outer).sum()),
                                 frozen_state_unchanged_receipt=True, feature_forward_calls=receipt['feature_model_forward_calls'],
                                 parameters=receipt['decoder_coefficient_values']))
            for p in (folder / 'fit.json', folder / 'started.json', rows_path, decoder_path):
                hashes[p.relative_to(ROOT).as_posix()] = sha(p)
            print(json.dumps(dict(stage='independent_recount', fold=fold, layer=layer, official_rows_checked=len(r))), flush=True)
    OUT.mkdir()
    agg = pd.DataFrame(aggregate); controls = pd.DataFrame(scope)
    agg.to_parquet(OUT / 'field_state_recount.parquet', index=False)
    controls.to_parquet(OUT / 'classifier_control_recount.parquet', index=False)
    profiles = []
    for layer, parts in bundles.items():
        whole = pd.concat(parts).sort_values('row_position').reset_index(drop=True)
        assert len(whole) == 112807 and not whole.row_position.duplicated().any()
        whole.to_parquet(OUT / f'{layer}_all_outer_control_ledger.parquet', index=False)
        hard = whole[whole.known_578_cohort]
        assert len(hard) == 578 and hard.all_observed_exact.all() and hard.all_states_exact.all()
        assert int(hard.classifier_pred_frozen.ne(2).sum()) == 576
        for (same, opposite), g in hard.groupby([hard.same_class_rows.gt(0), hard.opposite_class_rows.gt(0)]):
            profiles.append(dict(layer=layer, same_class_present=bool(same), opposite_class_present=bool(opposite),
                                 original_rows=len(g), classifier_errors=int(g.classifier_pred_frozen.ne(2).sum()),
                                 all_observed_exact=int(g.all_observed_exact.sum())))
    original_metrics = read(SRC / 'result_audit.json')['all_observed_field_metrics']
    primary = agg[agg.stratum.eq('observed')].groupby(['layer', 'role', 'field'])[['original_rows', 'errors', 'mode_errors']].sum().reset_index()
    comparison = pd.DataFrame(original_metrics).set_index(['layer', 'role', 'field']).sort_index()
    assert primary.set_index(['layer', 'role', 'field']).sort_index().equals(comparison)
    result = dict(status='independent_original_row_and_field_recount_passed_not_classifier_gain',
                  latest_actual_classifier='V146', official_rows=offset, original_ASA_rows=len(direct),
                  own_model_forwards=0, own_gradients=0, own_fits=0, own_updates=0,
                  executed_diagnostic_fits=6, executed_classifier_fits=0,
                  decoder_prediction_replayed=False, classifier_state_independently_replayed=False,
                  official_truth_and_body_rechecked=True, original_frequencies_retained=True,
                  independent_coordinate_targets_exact=True, split_and_fit_mode_rechecked=True,
                  canonical_fixed_minimum_representative_rechecked=True, worker_primary_metrics_match=True,
                  diagnostic_receipts=receipts, known_578_conditional_support=profiles,
                  outer_field_metrics=primary[primary.role.eq('outer_HELD')].to_dict('records'),
                  limits=['Recounts saved decoded values; neither decoder predictions nor classifier forward outputs freshly regenerated.',
                          'Frozen-state identity is supported by sealed execution source and worker receipts; no independent model replay.',
                          'Linear readability is not information sufficiency, decision use, or proof of missing information.',
                          'Backbone has seen inner-held threat labels; outer development has been repeatedly inspected.',
                          'Same-key support excludes exact source-port value; neither a full behavior identity nor a legal label-copy rule.',
                          'No extra supervision, new classifier training, automatic model promotion, or new blind acceptance.'],
                  source_sha256=hashes)
    (OUT / 'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('status', 'executed_diagnostic_fits', 'executed_classifier_fits',
                      'known_578_conditional_support')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
