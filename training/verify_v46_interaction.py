"""Read frozen candidates, replay real predictions and diagnose supported effects."""
import argparse
import collections
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import confusion_matrix, f1_score, log_loss


def main(root, out):
    assert not (out/'verification.json').exists()
    spec = importlib.util.spec_from_file_location('v46', out/'probe_v46_interaction.py')
    v = importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
    old, m = v.helpers(out); c = v.read(out/'configuration.json'); rep = v.read(out/'report.json'); pre = v.read(out/'preregistered.json')
    assert m.sha(out/'configuration.json') == pre['configuration_sha256']
    assert m.sha(out/'probe_v46_interaction.py') == pre['script_sha256']
    for p, h in c['source_bindings'].items(): assert m.sha(root/p) == h, p
    for p, h in c['reference_sha256'].items(): assert m.sha(out/p) == h
    r, x, b, ab = m.data(root); e = pd.read_parquet(out/'evaluation.parquet')
    controls = pd.read_parquet(root/'artifacts/v45_missingness_20260914/evaluation.parquet')
    assert np.array_equal(e.row_position, controls.row_position)
    assert np.array_equal(e.scenario, controls.scenario)
    assert np.array_equal(e.R_qM, controls.R_qM)
    assert np.array_equal(e.eligible_stress_row, controls.eligible_stress_row)
    models = {}; replayed = 0; vocabulary = []
    for fold in range(3):
        models[fold] = {}; ix = r.fold.to_numpy() == fold
        for view in v.VIEWS:
            p = out/('fold_%s/%s.joblib'%(fold, view)); assert m.sha(p) == rep['model_sha256'][p.relative_to(out).as_posix()]
            model = joblib.load(p); models[fold][view] = model
            names = model[0].get_feature_names_out()
            assert {n.split('=', 1)[0] for n in names} <= (set(v.FIELDS)-set(v.VIEWS[view]))|{v.COMBO}
            assert not any(v.MISSING in n for n in names)
            fit = v.records(x.loc[~ix], view)
            observed_combos = {z[v.COMBO] for z in fit if v.COMBO in z}
            learned = {n.split('=', 1)[1] for n in names if n.startswith(v.COMBO+'=')}
            assert learned == observed_combos
            # Novel combinations must be all-zero in the interaction block, not
            # mapped to a similar fitted tuple or inferred from evaluation labels.
            test = dict(fit[0]); test[v.COMBO] = '["NEVER_SEEN_PROTOCOL","X","Y"]'
            matrix = model[0].transform([test]); columns = np.flatnonzero(np.array([n.startswith(v.COMBO+'=') for n in names]))
            assert matrix[:, columns].nnz == 0
            vocabulary.append({'fold': fold, 'view': view, 'fit_only_interaction_categories': len(learned), 'unknown_combo_encoding_nonzero': 0})
        for scenario, cols in v.SCENARIOS.items():
            xx = x.loc[ix].copy(); xx[cols] = v.MISSING
            d = e[(e.fold == fold)&(e.scenario == scenario)].sort_values('row_position')
            assert np.array_equal(d.row_position, r.loc[ix, 'row_position'])
            assert np.array_equal(d.label_index, r.loc[ix, 'label_index'])
            q = v.predict(models[fold], xx)
            assert np.array_equal(q, d.I_qM.to_numpy()); replayed += len(d)
            assert np.isfinite(q).all() and ((q >= 0)&(q <= 1)).all()
            for view in v.VIEWS:
                use = v.routing(xx) == view
                fit_keys = set(v.key_array(x.loc[~ix], view))
                seen = [k in fit_keys for k in v.key_array(xx.loc[use], view)]
                assert np.array_equal(seen, d.loc[use, 'seen_remaining_view_key'])
    floors = {}; effects = {}; probabilities = {}; behavior = []; changed_positions = set()
    original_index = pd.Index(r.row_position)
    for scenario, cols in v.SCENARIOS.items():
        d = e[(e.scenario == scenario)&e.eligible_stress_row].copy()
        xx = x.copy(); xx[cols] = v.MISSING
        keys = np.array([json.dumps(z, separators=(',', ':')) for z in xx.to_numpy().tolist()])
        idx = original_index.get_indexer(d.row_position); d['key'] = keys[idx]
        counts = d.groupby(['fold', 'key', 'label_index']).size().unstack(fill_value=0).reindex(columns=[1, 2], fill_value=0)
        mixedkeys = set(counts.index[(counts > 0).all(axis=1)])
        mixed = np.array([(f, k) in mixedkeys for f, k in zip(d.fold, d.key)])
        y = d.label_index.to_numpy(); pr = np.where(d.R_qM >= .5, 1, 2); pi = np.where(d.I_qM >= .5, 1, 2)
        wrong_r = pr != y; wrong_i = pi != y
        floors[scenario] = {'rows': len(d), 'finite_sample_same_input_minimum_errors': int(counts.min(axis=1).sum()),
            'R_errors': int(wrong_r.sum()), 'I_errors': int(wrong_i.sum()),
            'I_mixed_key_errors': int((wrong_i&mixed).sum()), 'I_pure_key_errors': int((wrong_i&~mixed).sum())}
        assert d.assign(q=d.I_qM).groupby(['fold', 'key']).q.nunique().max() == 1
        effects[scenario] = {'fixes': int((wrong_r&~wrong_i).sum()), 'regressions': int((~wrong_r&wrong_i).sum()),
            'changed_decisions': int((pr != pi).sum()), 'changed_body_groups': int(d.loc[pr != pi, 'body_group'].nunique()),
            'by_class': {str(k): {'fixes': int(((y == k)&wrong_r&~wrong_i).sum()),
                'regressions': int(((y == k)&~wrong_r&wrong_i).sum())} for k in [1, 2]}}
        # Sample a real representative of each changed projection for raw-path checks.
        changed_positions.update(d.loc[pr != pi].drop_duplicates('projection_id').row_position.astype(int))
        probabilities[scenario] = {}
        for name in ['R', 'I']:
            q = d[name+'_qM'].to_numpy(); pred = np.where(q >= .5, 1, 2)
            expected = rep['scenario_metrics'][scenario][name]
            assert expected['errors'] == int((pred != y).sum())
            assert expected['confusion_M_S'] == confusion_matrix(y, pred, labels=[1, 2]).tolist()
            assert abs(expected['brier']-np.mean((q-(y == 1))**2)) < 1e-12
            assert abs(expected['conditional_log_loss']-log_loss(y == 1, np.c_[1-q, q], labels=[False, True])) < 1e-12
            cf = np.maximum(q, 1-q); points = []
            for threshold in [.5, .6, .7, .8, .9, .95, .99]:
                use = cf >= threshold
                points.append({'min_score_confidence': threshold, 'rows': int(use.sum()), 'coverage': float(use.mean()),
                    'error_rate': float((pred[use] != y[use]).mean()) if use.any() else None,
                    'mean_score_confidence': float(cf[use].mean()) if use.any() else None})
            probabilities[scenario][name] = points
        # All behavior tuples reported, never used as patch rules or relabeling.
        tuple_keys = [json.dumps(z, separators=(',', ':')) for z in xx.iloc[idx][v.COMBO_FIELDS].to_numpy().tolist()]
        d['behavior'] = tuple_keys
        for key, z in d.groupby('behavior'):
            stats = {'scenario': scenario, 'tuple': json.loads(key), 'rows': len(z), 'bodies': int(z.body_group.nunique())}
            for name in ['R', 'I']:
                q = z[name+'_qM'].to_numpy(); yy = z.label_index.to_numpy()
                stats[name] = {'errors': int((np.where(q >= .5, 1, 2) != yy).sum()),
                    'log_loss': float(log_loss(yy == 1, np.c_[1-q, q], labels=[False, True])),
                    'brier': float(np.mean((q-(yy == 1))**2))}
            behavior.append(stats)
    # Revisit the hypothesis-generating example without treating its labels as a target.
    diagnostic = x.copy(); diagnostic[v.VIEWS['no_ports']] = v.MISSING
    chosen = (diagnostic.transport_protocol == 'tcp')&(diagnostic.src_role == 'dmz')&(diagnostic.dst_role == 'outside')
    example = []
    for fold in range(3):
        use = chosen&(r.fold != fold)
        cf = v.records(diagnostic.loc[chosen].iloc[:1], 'no_ports')
        control = joblib.load(root/('artifacts/v45_missingness_20260914/fold_%s/no_ports.joblib'%fold))
        example.append({'fold': fold, 'fit_counts_M_S': [int((use&(r.label_index == k)).sum()) for k in [1, 2]],
            'R_qM': float(control.predict_proba(v.records(diagnostic.loc[chosen].iloc[:1], 'no_ports', False))[0, 1]),
            'I_qM': float(models[fold]['no_ports'].predict_proba(cf)[0, 1])})
    # Check frozen real parser closure, not a separately mocked feature builder.
    frozen = root/'artifacts/v42_local_r1_20260914/frozen_training_runtime'
    receipt = v.read(frozen.parent/'prepared/complete.json')
    for name, h in receipt['runtime_sources'].items(): assert m.sha(frozen/name) == h
    sys.path.insert(0, str(frozen)); import v39_core as core
    from audit_v37_prepared import variants
    official = root/'data/official/train.parquet'
    assert m.sha(official) == '6b6d5e23caebfd1c4f6b70c9e58c27f437bca7f0cd26497eefa3e4908f2cb742'
    selected = set(r.drop_duplicates('projection_id').iloc[::16].row_position.astype(int))|changed_positions
    positions = np.array(sorted(selected)); raws = {}; offset = 0
    for batch in pq.ParquetFile(official).iter_batches(batch_size=8192, columns=['message_sanitized'], use_threads=False):
        wanted = positions[np.searchsorted(positions, offset):np.searchsorted(positions, offset+len(batch))]
        for pos in wanted: raws[int(pos)] = batch.column(0)[int(pos)-offset].as_py() or ''
        offset += len(batch)
    original = []; modified = []; parents = []; types = collections.Counter()
    for index, (pos, raw) in enumerate(sorted(raws.items())):
        base = core.prepare_record({'message_sanitized': raw}); enc = m.encode(base['facts'])
        assert enc == x.iloc[original_index.get_loc(pos)].tolist(); original.append(enc)
        changes = [('metadata', {'message_sanitized': raw, 'timestamp': '2099-01-01T00:00:00Z', 'product_name': None,
            'vendor_name': 'unseen', 'src_ip': '192.0.2.9', 'dst_ip': '203.0.113.99', 'username': 'new', 'label_binary': 'benign'})]
        changes += [(n, {'message_sanitized': text}) for n, text in variants(raw, 'asa')]
        text = re.sub(r'(?<!\w)(?:\d{1,3}\.){3}\d{1,3}(?!\w)', '203.0.113.99', raw)
        text = re.sub(r'dmz[-_]\d+', 'dmz-999', text)
        changes.append(('address_interface', {'message_sanitized': text}))
        text = re.sub(r'(?:CRED|USER|ORG|HOST|IP|EMAIL)-\d+', 'CRED-999999', raw)
        if text != raw: changes.append(('redaction_identity', {'message_sanitized': text}))
        for name, record in changes:
            ans = core.prepare_record(record); values = m.encode(ans['facts'])
            assert ans['text'] == base['text'] and values == enc, (pos, name)
            modified.append(values); parents.append(index); types[name] += 1
    a = pd.DataFrame(original, columns=v.FIELDS); z = pd.DataFrame(modified, columns=v.FIELDS); parent = np.array(parents)
    rawchecks = []
    for fold, mm in models.items():
        for scenario, cols in v.SCENARIOS.items():
            aa = a.copy(); zz = z.copy(); aa[cols] = v.MISSING; zz[cols] = v.MISSING
            delta = float(np.max(np.abs(v.predict(mm, aa)[parent]-v.predict(mm, zz))))
            assert delta == 0
            rawchecks.append({'fold': fold, 'scenario': scenario, 'variants': len(z), 'max_probability_difference': delta})
    # A protected offline overlay retains every normal boundary and non-ASA score.
    orig = e[e.scenario == 'original'].sort_values('row_position')
    bp = b[['p_benign', 'p_malicious', 'p_suspicious']].to_numpy(); p = bp.copy(); inside = b.route.to_numpy() == 'asa'
    assert np.array_equal(b.loc[inside, 'row_position'], orig.row_position)
    q = orig.I_qM.to_numpy(); part = bp[inside].copy(); mass = part[:, 1]+part[:, 2]
    part[:, 1] = mass*q; part[:, 2] = mass*(1-q)
    fallback = (bp[inside].argmax(1) == 0)|(part.argmax(1) == 0); part[fallback] = bp[inside][fallback]; p[inside] = part
    assert np.array_equal(p[~inside], bp[~inside]) and np.array_equal(p[:, 0], bp[:, 0])
    assert np.array_equal(p.argmax(1) == 0, bp.argmax(1) == 0)
    yy = b.label_index.to_numpy()
    protected = {'rows': len(b), 'outside_rows': int((~inside).sum()), 'outside_probability_changes': 0, 'normal_boundary_changes': 0,
        'fallback_rows': int(fallback.sum()), 'B_errors': int((bp.argmax(1) != yy).sum()),
        'I_errors': int((p.argmax(1) != yy).sum()), 'I_macro_f1': float(f1_score(yy, p.argmax(1), labels=[0, 1, 2], average='macro')),
        'scope': 'Offline reconstruction only. No production API changes or submission.'}
    m.save(out/'diagnostics.json', {'floors': floors, 'paired_effects': effects, 'confidence_points': probabilities,
        'behavior_tuples': behavior, 'hypothesis_example': example,
        'limits': 'Evaluation-label-aware analysis only. Sample lower bounds are not population Bayes risk or guaranteed learnable improvement. No rule/label changes.'})
    shutil.copyfile(__file__, out/Path(__file__).name)
    m.save(out/'verification.json', {'all_implementation_checks_passed': True,
        'primary_quality_passed': rep['all_primary_quality_gates_pass'], 'replayed_probability_rows': replayed,
        'source_and_model_hashes_verified': True, 'fit_only_vocabulary': vocabulary,
        'raw_originals': len(a), 'raw_variants': len(z), 'raw_variant_types': dict(types), 'raw_checks': rawchecks,
        'protected_overlay': protected, 'verifier_sha256': m.sha(__file__),
        'new_fits': 0, 'production_changed': False, 'platform_used': False, 'external_transfer_validated': False})
    print(json.dumps({'verified': True, 'quality_pass': rep['all_primary_quality_gates_pass'],
        'replayed': replayed, 'raw_rows': len(a), 'variants': len(z), 'effects': effects,
        'floors': floors, 'example': example, 'protected': protected}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); p.add_argument('--out', required=True)
    a = p.parse_args(); main(Path(a.root).resolve(), Path(a.out).resolve())
