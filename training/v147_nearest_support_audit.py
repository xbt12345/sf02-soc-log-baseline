"""Nearest factual same-class TRAIN support, retaining every correct control.

No model forward, gradient, fitting, labels synthesized, or input changes. A
distance is diagnostic; differing concrete facts do not become equivalent.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from v135_runtime import load_data, fit_context
from v143_pair_geometry_probe import observed_key

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'artifacts/v146_guarded_pair_training_20261001'
OUT = ROOT / 'artifacts/v147_nearest_support_audit_20261001'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert not OUT.exists(), 'Preserve completed evidence'
    actual = json.loads((RUN / 'final_delivery.json').read_text(encoding='utf-8'))
    assert actual['latest_actual'] == 'V146' and not actual['matched_effect_passed']
    _, d = load_data()
    trace_path = ROOT / 'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet'
    trace = pd.read_parquet(trace_path, columns=['row_position', 'facts_json'])
    assert np.array_equal(trace.row_position, d.row_position)
    facts = trace.facts_json.map(json.loads)
    d['key'] = facts.map(observed_key)
    # Exact observation states, ranges, roles and all other fields are preserved.
    # Only an observed exact source-port value is omitted, as in the old probe.
    fields = sorted(set().union(*(f.keys() for f in facts)) - {'src_port_fixed'})
    values = pd.DataFrame({name: facts.map(lambda f: json.dumps(f[name], sort_keys=True) if name in f else '<absent>') for name in fields})
    codes = np.stack([pd.factorize(values[name], sort=True)[0] for name in fields], axis=1)
    pred_path = RUN / 'ASA_prediction_ledger.parquet'
    pred = pd.read_parquet(pred_path)
    assert np.array_equal(d.row_position, pred.row_position) and np.array_equal(d.truth, pred.truth)
    raw_path = ROOT / 'data/official/train.parquet'
    y = pd.read_parquet(raw_path, columns=['label_binary']).label_binary.map({'benign': 0, 'malicious': 1, 'suspicious': 2}).to_numpy(np.int8)
    assert len(y) == 2056871 and np.array_equal(d.truth, y[d.row_position])
    summaries, contrasts, groups = [], [], []
    for fold in range(3):
        train, _, pure, _, _ = fit_context(d, fold)
        train = train[train.key.notna()].copy()
        train['pure_numeric_input'] = pure[train.local].astype(bool)
        held = d[d.fold.eq(fold) & d.key.notna()].copy()
        assert not (set(train.root) & set(held.root))
        for cl in [1, 2]:
            eligible = train[train.truth.eq(cl) & train.pure_numeric_input]
            available = eligible.groupby('key').agg(original_rows=('truth', 'size'), roots=('root', 'nunique'))
            pos = available[available.roots.ge(2)]
            cells = pd.read_parquet(RUN / f'fold{fold}_pair_reference.parquet')
            registered_keys = set(cells.pair_key)
            q = held[held.truth.eq(cl)].copy()
            q['same_key_rows'] = q.key.map(available.original_rows).fillna(0).astype(int)
            q['same_key_roots'] = q.key.map(available.roots).fillna(0).astype(int)
            q['old_registered_key'] = q.key.isin(registered_keys)
            candidates = eligible.drop_duplicates('key')
            query = q.drop_duplicates('key')
            nearest = {}
            train_codes = codes[candidates.index]
            for i in query.index:
                if not len(candidates):
                    nearest[d.key.iloc[i]] = (None, [], 0, 0)
                    continue
                mismatch = train_codes != codes[i]
                distances = mismatch.sum(1)
                minimum = int(distances.min())
                take = np.flatnonzero(distances == minimum)
                patterns = sorted(set(tuple(fields[j] for j in np.flatnonzero(mismatch[k])) for k in take))
                ties = eligible[eligible.key.isin(candidates.iloc[take].key)]
                nearest[d.key.iloc[i]] = (minimum, patterns, int(ties.root.nunique()), len(ties))
            q['nearest_differing_fields'] = q.key.map(lambda k: nearest[k][0])
            q['nearest_mismatch_patterns'] = q.key.map(lambda k: json.dumps(nearest[k][1], ensure_ascii=False))
            q['nearest_roots'] = q.key.map(lambda k: nearest[k][2])
            q['nearest_original_rows'] = q.key.map(lambda k: nearest[k][3])
            for name in ['pred_A0', 'pred_V142', 'pred_A', 'pred_B']:
                q[name] = pred.loc[q.index, name]
            q['known_578_cohort'] = q.truth.eq(2) & q.pred_V142.ne(q.truth)
            for (distance, patterns), g in q.groupby(['nearest_differing_fields', 'nearest_mismatch_patterns'], dropna=False):
                groups.append(dict(fold=fold, truth=cl, differing_fields=None if pd.isna(distance) else int(distance), mismatch_patterns=patterns,
                    rows=len(g), roots=int(g.root.nunique()), V142_errors=int(g.pred_V142.ne(g.truth).sum()),
                    A_errors=int(g.pred_A.ne(g.truth).sum()), B_errors=int(g.pred_B.ne(g.truth).sum())))
            summaries.append(dict(fold=fold, truth=cl, legal_known_port_original_rows=len(train[train.truth.eq(cl)]),
                legal_pure_original_rows=len(eligible), legal_pure_keys=len(available), multi_root_positive_keys=len(pos),
                multi_root_positive_original_rows=int(pos.original_rows.sum()), held_known_port_rows=len(q),
                held_any_same_key_rows=int(q.same_key_rows.gt(0).sum()), held_multiroot_same_key_rows=int(q.same_key_roots.ge(2).sum()),
                held_registered_old_key_rows=int(q.old_registered_key.sum()), held_V142_errors=int(q.pred_V142.ne(q.truth).sum()),
                V142_errors_any_same_key=int((q.pred_V142.ne(q.truth) & q.same_key_rows.gt(0)).sum()),
                V142_errors_multiroot_same_key=int((q.pred_V142.ne(q.truth) & q.same_key_roots.ge(2)).sum())))
            contrasts.append(q)
        print(json.dumps(dict(stage='same_class_factual_support_audited', fold=fold, summary=summaries[-2:]), ensure_ascii=False), flush=True)
    allq = pd.concat(contrasts, ignore_index=True)
    assert len(allq) == int(d.key.notna().sum()) and int(allq.known_578_cohort.sum()) == 578
    OUT.mkdir()
    allq.to_parquet(OUT / 'all_known_port_query_support.parquet', index=False)
    pd.DataFrame(groups).to_parquet(OUT / 'nearest_field_patterns.parquet', index=False)
    report = dict(status='whole_known_port_correct_and_error_controls_factual_support_only', latest_actual_training='V146',
        new_classifier_fits=0, new_gradients=0, new_parameter_updates=0, new_model_forwards=0,
        official_rows=len(y), ASA_rows=len(d), known_port_query_rows=len(allq), fields=fields,
        summaries=summaries, nearest_patterns=groups,
        original_frequency_preserved=True, HELD_used_for_postfit_diagnosis_only=True, next_method_selected=False,
        limits=['Same class, same observed body context, original frequency, distinct TRAIN roots and actual numeric purity are separate facts.',
                'Nearest differing fields are not authorized relaxations, new labels, or a class rule.',
                'Unknown and absent states remain distinct; known ports do not make every other field known.',
                'Unknown-port rows remain in the complete V146 population; this audit is a declared factual subset, not a new main score.',
                'A wider positive population does not prove coverage of the difficult behavior or classification transfer.'],
        source_sha256={str(p.relative_to(ROOT)).replace(chr(92), '/'):sha(p) for p in [Path(__file__), trace_path, raw_path, pred_path, RUN/'final_delivery.json', ROOT/'training/v135_runtime.py', ROOT/'training/v143_pair_geometry_probe.py']})
    (OUT / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=report['status'], known_port_rows=len(allq), summaries=summaries), ensure_ascii=False))


if __name__ == '__main__':
    main()
