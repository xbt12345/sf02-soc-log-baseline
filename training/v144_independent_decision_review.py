"""Read-only recount of registered derivatives, not a model/gradient execution.

Recover the matched CE-only direction from the two existing JVP directions.
The algebra uses the registered 0.1 gradient norm ratio, not a tuned coefficient.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / 'artifacts/v144_aux_gradient_evidence_20261001'
OUT = ROOT / 'artifacts/v144_independent_decision_review_20261001'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def pairing_key(f):
    if f.get('transport_protocol') not in ('tcp', 'udp'):
        return None
    if not all(0 <= f.get(k, 65536) <= 65535
               for k in ('src_port_fixed', 'dst_port_fixed')):
        return None
    value = {k: v for k, v in f.items() if k != 'src_port_fixed'}
    value.update(src_port_observed=True, dst_port_observed=True)
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def main():
    if OUT.exists():
        raise FileExistsError('Do not overwrite a completed independent review')
    report = read(PROBE / 'probe.json')
    registration = read(PROBE / 'pre_registered_probe.json')
    assert report['new_fits'] == report['new_updates'] == 0
    assert registration['new_fits_max'] == registration['new_parameter_updates_max'] == 0
    for rel, expected in report['evidence_bindings'].items():
        assert sha(ROOT / rel) == expected, rel
    rows_path = PROBE / 'legal_TRAIN_margin_differentials.parquet'
    assert sha(rows_path) == report['output_sha256']
    rows = pd.read_parquet(rows_path)
    official = ROOT / 'data/official/train.parquet'
    y = pd.read_parquet(official, columns=['label_binary']).label_binary.map(
        {'benign': 0, 'malicious': 1, 'suspicious': 2}).to_numpy()
    assert len(y) == 2056871 and np.array_equal(rows.truth, y[rows.row_position])
    assert not rows.duplicated(['training_role', 'row_position']).any()
    trace_path = ROOT / 'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet'
    trace = pd.read_parquet(trace_path, columns=['row_position', 'local', 'root', 'fold', 'facts_json'])
    assert not trace.row_position.duplicated().any()
    source = trace.set_index('row_position')
    assert np.array_equal(rows.root, source.loc[rows.row_position, 'root'])
    assert np.array_equal(rows.local, source.loc[rows.row_position, 'local'])
    assert np.all(rows.training_role.to_numpy() != source.loc[rows.row_position, 'fold'].to_numpy())
    facts = source.facts_json.map(json.loads)
    source['pair_key'] = facts.map(pairing_key)
    source['body_source_port'] = facts.map(lambda f: f.get('src_port_fixed', 65536))
    results = []
    summaries = []
    ratio = report['auxiliary_gradient_ratio']
    assert ratio == registration['auxiliary_gradient_ratio'] == 0.1
    for f in report['folds']:
        role = f['fold']
        r = rows[rows.training_role.eq(role)].copy()
        cn, an = f['full_CE_gradient_norm'], f['auxiliary_gradient_norm']
        cosine = f['CE_auxiliary_gradient_cosine']
        coefficient = f['fixed_initial_auxiliary_coefficient']
        assert np.isclose(coefficient * an / cn, ratio, rtol=1e-12, atol=1e-14)
        # Unit-combined direction norm differs from unit-CE direction.
        # Convert both to the same CE-normalized displacement before comparison.
        scale = float(np.sqrt(1 + ratio ** 2 + 2 * ratio * cosine))
        r['CE_only_unit_margin_derivative'] = (
            scale * r.combined_margin_derivative - ratio * r.aux_margin_derivative)
        r['combined_CE_normalized_margin_derivative'] = scale * r.combined_margin_derivative
        r['increment_from_auxiliary'] = ratio * r.aux_margin_derivative
        assert np.allclose(r.combined_CE_normalized_margin_derivative,
                           r.CE_only_unit_margin_derivative + r.increment_from_auxiliary,
                           rtol=1e-12, atol=1e-15)
        assert not (set(r.root) & set(trace[trace.fold.eq(role)].root))
        assert len(r) == f['capacity']['retained_classification_original_rows']
        for direction in f['margin_differentials']:
            for c in direction['classes']:
                mask = r.truth.eq(c['truth']) & r.pure
                v = r.loc[mask, direction['direction'] + '_margin_derivative']
                assert len(v) == c['pure_original_rows']
                assert int(v.gt(0).sum()) == c['positive_pure_margin_slopes']
                assert int(v.lt(0).sum()) == c['negative_pure_margin_slopes']
                assert np.isclose(v.mean(), c['mean_pure_original_margin_derivative'],
                                  rtol=1e-12, atol=1e-15)
        a = r[['row_position', 'truth', 'root', 'canonical_key', 'pure']].join(
            source[['pair_key', 'body_source_port']], on='row_position')
        eligible = a[a.pure & a.pair_key.notna()]
        valid = []
        for key, group in eligible.groupby('pair_key'):
            roots = group.groupby('truth').root.nunique()
            inputs = group.groupby('truth').canonical_key.nunique()
            if all(roots.get(c, 0) >= 2 and inputs.get(c, 0) >= 2 for c in (1, 2)):
                valid.append(key)
        selected = eligible[eligible.pair_key.isin(valid)]
        cap = f['capacity']
        assert len(valid) == cap['keys'] and len(selected) == cap['selected_original_rows']
        assert int(selected.truth.eq(2).sum()) == cap['original_S']
        assert int(selected.truth.eq(1).sum()) == cap['original_M']
        for cl in (1, 2):
            clrows = r[r.truth.eq(cl) & r.pure]
            ce_means = {}
            for name in ('CE_only_unit_margin_derivative',
                         'combined_CE_normalized_margin_derivative',
                         'increment_from_auxiliary'):
                v = clrows[name]
                ce_means[name] = dict(mean=float(v.mean()), increasing=int(v.gt(0).sum()),
                                      decreasing=int(v.lt(0).sum()), original_rows=len(v))
            aux = next(c for d in f['margin_differentials'] if d['direction'] == 'aux'
                       for c in d['classes'] if c['truth'] == cl)
            combined = next(c for d in f['margin_differentials'] if d['direction'] == 'combined'
                            for c in d['classes'] if c['truth'] == cl)
            ce_deriv = (scale * combined['full_original_member_CE_derivative']
                        - ratio * aux['full_original_member_CE_derivative'])
            results.append(dict(training_role=role, truth=cl,
                                CE_only_class_member_CE_derivative=ce_deriv,
                                combined_class_member_CE_derivative_at_equal_CE_displacement=(
                                    scale * combined['full_original_member_CE_derivative']),
                                auxiliary_increment_class_member_CE_derivative=(
                                    ratio * aux['full_original_member_CE_derivative']),
                                pure_original_margin_derivatives=ce_means))
        summaries.append(dict(training_role=role, selected_rows=len(selected),
                              S_rows=cap['original_S'], M_rows=cap['original_M'],
                              selected_system_body_source_port_rows=int(selected.body_source_port.lt(1024).sum()),
                              original_S_training_rows=int(r.truth.eq(2).sum()),
                              S_pair_row_fraction=cap['original_S'] / int(r.truth.eq(2).sum())))
    pure = rows[rows.pure]
    assert len(pure) == 225202 and int(pure.pred.ne(pure.truth).sum()) == 0
    assert int(rows.pred.eq(rows.truth).sum()) == 225558
    OUT.mkdir()
    output = dict(status='independent_derivative_and_pair_capacity_recount', latest_actual_training='V142',
                  new_fits=0, new_gradients=0, new_updates=0, original_rows_checked=len(rows),
                  roles=summaries, matched_CE_only_recovered_algebraically=results,
                  decision='Do not authorize a full classifier retraining from this derivative evidence alone.',
                  limitations=[
                      'Existing JVP values were recounted, not recomputed from models.',
                      'CE-only direction is an exact local linear reconstruction, not an executed finite update.',
                      'Negative margin slopes are risks, not actual new classification errors.',
                      'New auxiliary geometry does not create fine behavior/category support absent in official TRAIN.',
                      'A pooled CE decrease cannot prove simultaneous per-class or source-held improvement.'
                  ], source_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in
                                    [Path(__file__), PROBE/'probe.json', PROBE/'pre_registered_probe.json',
                                     rows_path, trace_path, official]})
    (OUT/'review.json').write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
