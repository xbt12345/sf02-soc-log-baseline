"""V114: frozen 2x2, ranking and raw-event evidence audit; no fitting."""
import json
import re
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from scipy import sparse
from sklearn.metrics import roc_auc_score, average_precision_score
from run_v75 import ROOT, save, sha
from v104_phase_b import DEVICE, PREP, SEED, SparseTabM, predict_all
from v113_case_train import DEST as PREVIOUS, OLD, VIEW, check

DEST = ROOT/'artifacts/v114_mechanism_review_20260929_r2'


def metric(y, p):
    pred = p.argmax(1)
    score = p[:, 2]/(p[:, 1]+p[:, 2])
    return {'errors': int((pred != y).sum()),
            'M_errors': int(((y == 1)&(pred != 1)).sum()),
            'S_errors': int(((y == 2)&(pred != 2)).sum()),
            'S_AUROC': float(roc_auc_score(y == 2, score)),
            'S_AP': float(average_precision_score(y == 2, score))}


def oracle_frontier(y, score, budget):
    # Exhaustive score-tie blocks, retrospective diagnostic, NEVER a selected rule.
    t = pd.DataFrame({'score': score, 'M': y == 1, 'S': y == 2})
    g = t.groupby('score', sort=True)[['M', 'S']].sum().iloc[::-1]
    cm = g.M.cumsum().to_numpy(); cs = g.S.cumsum().to_numpy()
    eligible = np.flatnonzero(cm <= budget)
    correct = int(cs[eligible].max()) if len(eligible) else 0
    return {'M_error_budget': budget, 'maximum_S_correct': correct,
            'minimum_S_errors': int((y == 2).sum())-correct,
            'scope': 'label-informed oracle upper bound for one global scalar threshold; not calibration or transfer evidence'}


def main():
    check()
    assert not DEST.exists()
    previous_ver = json.loads((PREVIOUS/'verification.json').read_text(encoding='utf-8'))
    assert previous_ver['all_checks_passed']
    for p, h in previous_ver['artifact_sha256'].items():
        assert sha(ROOT/p) == h, p
    d = pd.read_parquet(PREVIOUS/'OOF_ASA_decisions.parquet')
    trace_path = ROOT/'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet'
    raw = pd.read_parquet(trace_path)
    assert np.array_equal(d.row_position, raw.row_position)
    assert np.array_equal(d.truth, raw.truth)
    official = ROOT/'data/official/train.parquet'
    # Compare every ASA raw message and label with the actual official parquet.
    positions = d.row_position.to_numpy(); offset = 0; matched = 0
    for batch in pq.ParquetFile(official).iter_batches(batch_size=65536, columns=['message_sanitized', 'label_binary']):
        idx = np.flatnonzero((positions >= offset)&(positions < offset+batch.num_rows))
        if len(idx):
            b = batch.to_pandas().iloc[positions[idx]-offset]
            assert np.array_equal(b.message_sanitized.to_numpy(), raw.raw_message.to_numpy()[idx])
            label = b.label_binary.map({'benign': 0, 'malicious': 1, 'suspicious': 2}).to_numpy()
            assert np.array_equal(label, d.truth.to_numpy()[idx])
            matched += len(idx)
        offset += batch.num_rows
    assert matched == 112807 and offset == 2056871
    DEST.mkdir()
    x = {'N1': sparse.load_npz(PREP/'N1_ASA.npz'), 'case': sparse.load_npz(VIEW)}
    y = d.truth.to_numpy(); local = d.local.to_numpy(); fold = d.fold.to_numpy()
    scores = {m+'__'+v: np.empty((len(d), 3), np.float32) for m in ('A', 'B') for v in x}
    replay = []
    for k in range(3):
        mask = fold == k
        for m, folder, own in [('A', OLD/f'fold{k}_N1_TabM25_seed{SEED}', 'N1'),
                               ('B', PREVIOUS/f'fold{k}_case_TabM25_seed{SEED}', 'case')]:
            model = SparseTabM().to(DEVICE)
            ckpt = torch.load(folder/'model.pt', map_location='cpu', weights_only=True)
            model.load_state_dict(ckpt['state_dict'])
            for view, matrix in x.items():
                p = predict_all(model, 'TabM', matrix, DEVICE)
                if view == own:
                    saved = np.load(folder/'ASA_input_prob.npy')
                    assert np.array_equal(p.argmax(1), saved.argmax(1))
                    assert np.allclose(p, saved, atol=2e-6, rtol=2e-6)
                scores[m+'__'+view][mask] = p[local[mask]]
            replay.append({'fold': k, 'model': m, 'model_sha256': sha(folder/'model.pt')})
            del model
            if DEVICE == 'cuda': torch.cuda.empty_cache()
        print(json.dumps({'frozen_replay_fold': k, 'fits': 0}), flush=True)
    assert np.array_equal(scores['A__N1'].argmax(1), d.A_N1_prediction)
    assert np.array_equal(scores['B__case'].argmax(1), d.B_case_prediction)
    np.savez_compressed(DEST/'frozen_cross_input_probabilities.npz', **scores)
    metrics = {name: {'all': metric(y, p), 'folds': {str(k): metric(y[fold == k], p[fold == k]) for k in range(3)}} for name, p in scores.items()}
    for name, p in scores.items():
        d[name+'_pred'] = p.argmax(1)
    a = d.A_N1_prediction.to_numpy(); b = d.B_case_prediction.to_numpy()
    oracle = {name: oracle_frontier(y, p[:, 2]/(p[:, 1]+p[:, 2]), 318) for name, p in scores.items()}
    # Ordinary nonparametric resampling of complete existing isolation roots.
    root_delta = d.assign(delta=(a != y).astype(int)-(b != y).astype(int)).groupby('root').delta.sum()
    active = root_delta[root_delta != 0]
    probabilities = [1/len(root_delta)]*len(active)+[(len(root_delta)-len(active))/len(root_delta)]
    weights = np.random.default_rng(114).multinomial(len(root_delta), probabilities, size=20000)
    boot = weights[:, :-1]@active.to_numpy()
    bootstrap = {'root_count': len(root_delta), 'replicates': len(boot),
                 'error_reduction_point': int(root_delta.sum()),
                 'percentile_95_interval': np.quantile(boot, [.025, .975]).tolist(),
                 'scope': 'retrospective root-cluster sampling uncertainty, not correction for repeated development or real-world domain uncertainty'}
    d = d.merge(raw[['row_position', 'raw_message', 'facts_json']], on='row_position', validate='one_to_one')
    d['protocol'] = d.raw_message.str.extract(r'\bDeny (\w+)')[0].fillna('unparsed')
    facts = d.facts_json.map(json.loads)
    d['fact_no_srcport'] = facts.map(lambda f: json.dumps({k: v for k, v in f.items() if not k.startswith('src_port')}, sort_keys=True))
    for k in ('icmp_type', 'icmp_code'):
        d[k] = facts.map(lambda f: f.get(k, -1))
    # Coverage keys use already parsed factual content, not model scores.
    support = []; eligible_cells = []
    for k in range(3):
        train = d[d.fold != k]; test = d[d.fold == k]
        for field in ('behavior', 'fact_no_srcport'):
            counts = train.groupby([field, 'truth']).agg(n=('local', 'size'), roots=('root', 'nunique'))
            for c in (1, 2):
                subset = test[test.truth == c].copy()
                for label, cls in [('same', c), ('opposite', 3-c)]:
                    cc = counts.xs(cls, level='truth')
                    subset[label+'_rows'] = subset[field].map(cc.n).fillna(0).astype(int)
                    subset[label+'_roots'] = subset[field].map(cc.roots).fillna(0).astype(int)
                subset['key_complete'] = subset[field].notna()
                for state, mask in [('all', np.ones(len(subset), bool)), ('B_error', subset.B_case_prediction != c),
                                    ('B_regression', subset.regressed)]:
                    z = subset[mask]
                    support.append({'fold': k, 'truth': c, 'key': field, 'state': state, 'rows': len(z),
                        'incomplete_key_rows': int((~z.key_complete).sum()),
                        'zero_same_class_roots': int((z.key_complete&z.same_roots.eq(0)).sum()),
                        'zero_both_class_roots': int((z.key_complete&z.same_roots.eq(0)&z.opposite_roots.eq(0)).sum()),
                        'opposite_only_roots': int((z.key_complete&z.same_roots.eq(0)&z.opposite_roots.gt(0)).sum()),
                        'same_class_at_least_3_roots': int(z.same_roots.ge(3).sum()),
                        'both_classes_at_least_3_roots': int((z.same_roots.ge(3)&z.opposite_roots.ge(3)).sum())})
                if field == 'fact_no_srcport' and c == 2:
                    eligible = subset[subset.same_roots.ge(3)&subset.opposite_roots.ge(3)&subset.B_case_prediction.ne(c)]
                    for key, g in eligible.groupby(field):
                        f = json.loads(key)
                        eligible_cells.append({'fold': k, 'facts': f, 'error_rows': len(g),
                            'M_training_roots': int(g.opposite_roots.iloc[0]), 'S_training_roots': int(g.same_roots.iloc[0]),
                            'known_exact_protocol_parameter': f.get('dst_port_fixed',65536) < 65536
                                if f['transport_protocol'] != 'icmp' else 'icmp_type' in f and 'icmp_code' in f})
    table = pd.DataFrame(support); table.to_csv(DEST/'training_only_support.csv', index=False)
    protocol = d.assign(A_wrong=a != y, B_wrong=b != y).groupby(['protocol', 'truth']).agg(
        rows=('local', 'size'), roots=('root', 'nunique'), unique_inputs=('local', 'nunique'),
        A_errors=('A_wrong', 'sum'), B_errors=('B_wrong', 'sum')).reset_index()
    protocol.to_csv(DEST/'protocol_errors.csv', index=False)
    icmp = d[d.protocol == 'icmp'].assign(A_wrong=lambda q: q.A_N1_prediction != q.truth, B_wrong=lambda q: q.B_case_prediction != q.truth)
    icmp.groupby(['fold', 'truth', 'icmp_type', 'icmp_code', 'interface_literal_pair']).agg(
        rows=('local', 'size'), roots=('root', 'nunique'), A_errors=('A_wrong', 'sum'), B_errors=('B_wrong', 'sum')).reset_index().to_csv(DEST/'icmp_support.csv', index=False)
    unique = d.groupby(['local', 'root', 'truth']).agg(rows=('row_position', 'size'),
        fold=('fold', 'first'), raw_example=('raw_message', 'first'), facts=('facts_json', 'first'),
        A=('A_N1_prediction', 'first'), B=('B_case_prediction', 'first'),
        A_prob=('A_S_probability', 'first'), B_prob=('B_S_probability', 'first')).reset_index()
    unique[(unique.A != unique.truth)|(unique.B != unique.truth)].to_parquet(DEST/'all_error_input_cases.parquet', index=False)
    d[['row_position', 'local', 'root', 'fold', 'truth']+[n+'_pred' for n in scores]].to_parquet(DEST/'crossed_decisions.parquet', index=False)
    report = {'status': 'frozen_diagnostics_no_fit', 'new_classifier_fits': 0, 'new_calibration_fits': 0,
              'official_raw_messages_and_labels_rechecked': matched,
              'source_sha256': sha(__file__), 'official_sha256': sha(official),
              'decision_sha256': sha(PREVIOUS/'OOF_ASA_decisions.parquet'), 'trace_sha256': sha(trace_path),
              'metrics_2x2': metrics, 'oracle_not_a_candidate': oracle, 'cluster_bootstrap': bootstrap,
              'protocol_errors': protocol.to_dict('records'), 'model_replays': replay,
              'M_regression_rows': int(((y == 1)&(a == 1)&(b != 1)).sum()),
              'M_regression_unique_inputs': int(d[(d.truth == 1)&d.regressed].local.nunique()),
              'S_repair_unique_inputs': int(d[(d.truth == 2)&d.repaired].local.nunique()),
              'apparent_supported_S_error_cells': eligible_cells,
              'missing_destination_is_not_a_shared_behavior': True,
              'class_source_concentration': {str(c): {
                  'rows': int((y == c).sum()), 'roots': int(d[y == c].root.nunique()),
                  'inverse_Herfindahl_mass_equivalent_roots': float(1/(d[y == c].groupby('root').size().div((y == c).sum())**2).sum()),
                  'note': 'row-mass concentration index, NOT an estimate of independent sample size'} for c in (1,2)},
              'protocol_specific_threshold_oracles': {protocol: {name: oracle_frontier(
                  y[d.protocol.eq(protocol)], p[d.protocol.eq(protocol),2]/p[d.protocol.eq(protocol),1:].sum(1),
                  int(((y==1)&(a!=1)&d.protocol.eq(protocol)).sum())) for name,p in scores.items() if name in ('A__N1','B__case')}
                  for protocol in ('tcp','udp','icmp')},
              'limits': ['Cross-input runs are inference interventions, not retrained candidates.',
                         'Behavior-key zero support does not prove full raw input unidentifiability.',
                         'Outer folds are repeatedly inspected developmental data, never fresh blind evidence.']}
    save(DEST/'diagnosis.json', report)
    save(DEST/'verification.json', {'all_checks_passed': True, 'classifier_fits': 0,
        'checks': {'V113_bound_artifacts_unchanged': True, 'official_all_ASA_raw_and_labels_equal': True,
                   'six_saved_models_own_input_replay': True, 'own_input_decisions_match_V113': True},
        'scope': 'Identity, official raw/labels and frozen inference replay; no new model quality acceptance.',
        'artifact_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in DEST.iterdir() if p.is_file()}})
    print(json.dumps({'metrics': {k: v['all'] for k, v in metrics.items()}, 'bootstrap': bootstrap, 'oracle': oracle}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
