"""Retrospective V121 audit, with no fits, optimizer updates or promotion.

Checkpoint averaging is an explicitly post-hoc diagnostic, not model selection.
Parameter/fact matches are observable projections, not proved attack semantics.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PREV = ROOT / 'artifacts/v121_paired_batch_training_20260929'
DEST = ROOT / 'artifacts/v122_evidence_review_20260929'
OFFICIAL = ROOT / 'data/official/train.parquet'
TRACE = ROOT / 'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet'
SUPPORT = ROOT / 'artifacts/v116_nested_selection_20260929/inner_split_manifest.parquet'
ROWS = ROOT / 'artifacts/v75_four_arm_20260921_r2/rows.parquet'
EPOCHS = (1, 2, 5, 10, 15, 20, 25)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def errors(y, prediction):
    return {str(k): int(((prediction != y) & (y == k)).sum()) for k in (1, 2)}


def main():
    if DEST.exists():
        raise FileExistsError('Preserve existing review evidence.')
    bound = read(PREV / 'delivery.json')['artifact_sha256']
    for rel, expected in bound.items():
        assert sha(ROOT / rel) == expected, rel
    sources = {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in
               [Path(__file__), PREV/'delivery.json', TRACE, SUPPORT, OFFICIAL, ROWS]}
    d = pd.read_parquet(PREV / 'expert_ASA_predictions.parquet')
    full = pd.read_parquet(PREV / 'full_prediction_ledger.parquet')
    trace = pd.read_parquet(TRACE)
    official = pd.read_parquet(OFFICIAL, columns=['event_id', 'label_binary'])
    truth = official.label_binary.map({'benign': 0, 'malicious': 1, 'suspicious': 2})
    assert truth.notna().all() and len(full) == len(official) == 2056871
    assert np.array_equal(full.row_position, np.arange(len(official)))
    row_map = pd.read_parquet(ROWS, columns=['row_position', 'event_id']).sort_values('row_position')
    assert np.array_equal(row_map.row_position, np.arange(len(official)))
    assert np.array_equal(row_map.event_id.astype(str), official.event_id.astype(str))
    for col in ['row_position', 'local', 'root', 'fold', 'truth']:
        assert np.array_equal(d[col], trace[col]), col
    assert len(d) == 112807 and not d.row_position.duplicated().any()
    assert np.array_equal(d.truth, truth.iloc[d.row_position])
    assert d.groupby('root').fold.nunique().max() == 1
    y = d.truth.to_numpy()
    curves, averages, persistent, group_stats, fit_curves = [], [], {}, [], []
    for arm in ('A', 'B'):
        probabilities = {e: np.zeros((len(d), 3), dtype=np.float32) for e in EPOCHS}
        for fold in range(3):
            folder = PREV / f'fold{fold}_{arm}'
            check = read(folder / 'checkpoints.json')
            sources[str((folder/'checkpoints.json').relative_to(ROOT)).replace('\\', '/')] = sha(folder/'checkpoints.json')
            assert [v['epoch'] for v in check] == list(EPOCHS)
            q = d.fold.eq(fold).to_numpy()
            for c in check:
                e = c['epoch']; path = folder / f'epoch{e}_prob.npy'
                assert sha(path) == c['prob_sha256']
                sources[str(path.relative_to(ROOT)).replace('\\', '/')] = c['prob_sha256']
                v = np.load(path)
                assert v.shape == (22546, 3) and np.isfinite(v).all()
                assert np.allclose(v.sum(1), 1, atol=2e-6)
                probabilities[e][q] = v[d.loc[q, 'local'].to_numpy()]
                for label in (1, 2):
                    f = c['fit_by_class'][str(label)]
                    fit_curves.append({'arm': arm, 'fold': fold, 'epoch': e, 'truth': label,
                                       'support': f['support'], 'errors': f['support']-f['correct'],
                                       'ensemble_CE': f['ensemble_CE'], 'member_mean_CE': f['member_mean_CE']})
        predictions = np.stack([probabilities[e].argmax(1) for e in EPOCHS])
        assert np.array_equal(predictions[-1], d[f'expert_pred_{arm}'])
        always_wrong = (predictions != y).all(0)
        d[f'{arm}_all_seven_wrong'] = always_wrong
        persistent[arm] = {'all_seven_wrong': errors(y, np.where(always_wrong, 3-y, y)),
                           'final_errors': errors(y, predictions[-1])}
        for i, e in enumerate(EPOCHS):
            curves.append({'arm': arm, 'epoch': e, 'errors': errors(y, predictions[i])})
        for window in ((20, 25), (15, 20, 25)):
            pred = np.mean([probabilities[e] for e in window], axis=0).argmax(1)
            averages.append({'arm': arm, 'epochs': list(window), 'errors': errors(y, pred),
                             'scope': 'posthoc diagnostic probability average; not SWA weights, no promotion'})
        d[f'{arm}_correct'] = d[f'expert_pred_{arm}'].eq(d.truth)
        g = d.groupby(['root', 'truth'])[f'{arm}_correct'].agg(['size', 'mean', 'sum']).reset_index()
        for label in (1, 2):
            q = g[g.truth.eq(label)]
            group_stats.append({'arm': arm, 'truth': label, 'roots': len(q),
                                'zero_recall_roots': int(q['sum'].eq(0).sum()),
                                'group_macro_recall': float(q['mean'].mean()),
                                'original_row_recall': float(q['sum'].sum()/q['size'].sum())})
    facts = trace.facts_json.map(json.loads)
    d['facts_json'] = trace.facts_json
    d['full_fact_fit_M_rows'] = 0
    d['full_fact_fit_S_rows'] = 0
    d['full_fact_fit_M_roots'] = 0
    d['full_fact_fit_S_roots'] = 0
    for fold in range(3):
        fit = d[d.fold.ne(fold)]
        for label, name in [(1, 'M'), (2, 'S')]:
            table = fit[fit.truth.eq(label)].groupby('facts_json').agg(rows=('root','size'), roots=('root','nunique'))
            for stat in ('rows', 'roots'):
                d.loc[d.fold.eq(fold), f'full_fact_fit_{name}_{stat}'] = d.loc[d.fold.eq(fold), 'facts_json'].map(table[stat]).fillna(0).astype(int)
    support_rows = []
    for label in (1, 2):
        for persistent_only in (False, True):
            q = d[d.truth.eq(label) & (d.A_all_seven_wrong if persistent_only else True)]
            name = 'M' if label == 1 else 'S'; other = 'S' if label == 1 else 'M'
            same = q[f'full_fact_fit_{name}_rows'].gt(0)
            opp = q[f'full_fact_fit_{other}_rows'].gt(0)
            support_rows.append({'truth': label, 'all_seven_wrong_only': persistent_only, 'rows': len(q),
                'same_only': int((same & ~opp).sum()), 'opposite_only': int((opp & ~same).sum()),
                'both': int((same & opp).sum()), 'neither': int((~same & ~opp).sum())})
    udp = facts.map(lambda a: a.get('transport_protocol') == 'udp' and a.get('dst_port_fixed') == 514
                    and a.get('src_role') == 'outside' and a.get('dst_role') == 'dmz')
    udp_table = d[udp].groupby(['fold', 'root', 'truth']).agg(rows=('local', 'size'),
        unique_inputs=('local', 'nunique'), A_correct=('A_correct', 'sum'), B_correct=('B_correct', 'sum')).reset_index()
    full_metrics = []
    for arm in ('A', 'B'):
        pred = full[f'final_pred_{arm}'].to_numpy(); yt = truth.to_numpy()
        for label in (0, 1, 2):
            tp = int(((yt == label) & (pred == label)).sum()); n = int((yt == label).sum())
            fp = int(((yt != label) & (pred == label)).sum())
            full_metrics.append({'arm':arm,'truth':label,'support':n,'correct':tp,'missed':n-tp,'false_called':fp,
                                 'recall':tp/n,'precision':tp/(tp+fp),'f1':2*tp/(n+tp+fp)})
    DEST.mkdir()
    d.to_parquet(DEST/'row_diagnosis.parquet',index=False)
    pd.DataFrame(fit_curves).to_csv(DEST/'fit_curves.csv',index=False)
    pd.DataFrame(curves).to_json(DEST/'OOF_curves.json',orient='records',indent=2)
    udp_table.to_csv(DEST/'udp514_source_support.csv',index=False)
    result = {'status':'v122_retrospective_review_no_training','classifier_fits_new':0,'optimizer_steps':0,
        'quality_acceptance':False,'model_promoted':False,'v121_delivery_files_verified':len(bound),
        'full_metrics':full_metrics,'group_metrics':group_stats,'curves':curves,'probability_average_probes':averages,
        'persistent_errors':persistent,'exact_parsed_fact_support':support_rows,'udp514':udp_table.to_dict('records'),
        'source_sha256':sources,
        'limitations':['Observed development folds only; repeated analysis is not independent confirmation.',
            'Identical parsed facts are not identical complete input or proved identical security context.',
            'Checkpoint probes do not test SWA/SWAD training or prove all averaging ineffective.',
            'Groups are isolation components, not verified organizations.'],
        'diagnostic_issues':['Preliminary console groupby.apply excluded truth group key; corrected using explicit boolean correctness before aggregation.',
            'Full prediction ledger has row_position but no event_id. Initial pre-write check failed; now independently binds the full row-position/event-id mapping against official data. No model was affected.']}
    save(DEST/'review.json', result)
    save(DEST/'outputs.json', {p.name: sha(p) for p in DEST.iterdir() if p.is_file()})
    print(json.dumps({k:result[k] for k in ['status','v121_delivery_files_verified','group_metrics','persistent_errors','exact_parsed_fact_support','udp514']},ensure_ascii=False))


if __name__ == '__main__':
    main()
