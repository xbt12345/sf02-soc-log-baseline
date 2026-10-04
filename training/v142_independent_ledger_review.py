"""Independent official-truth recount and legal-TRAIN support audit; no fitting."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'artifacts/v142_second_layer_training_20261001'
OUT = ROOT / 'artifacts/v142_independent_review_20261001'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def pairs(y, a, b):
    return {str(c): {
        'support': int((y == c).sum()),
        'before_errors': int(((y == c) & (a != y)).sum()),
        'after_errors': int(((y == c) & (b != y)).sum()),
        'repairs': int(((y == c) & (a != y) & (b == y)).sum()),
        'new_errors': int(((y == c) & (a == y) & (b != y)).sum()),
    } for c in range(3)}


def classes(y, pred):
    result = {}
    for c in range(3):
        support = int((y == c).sum())
        correct = int(((y == c) & (pred == c)).sum())
        called = int((pred == c).sum())
        result[str(c)] = dict(support=support, correct=correct,
                             missed=support-correct, false_called=called-correct,
                             precision=correct/called if called else None,
                             recall=correct/support if support else None,
                             f1=2*correct/(support+called) if support+called else None)
    return result


def main():
    if OUT.exists():
        raise FileExistsError('Preserve the completed independent audit')
    delivery = read(RUN/'final_delivery.json')
    quality = read(RUN/'quality.json')
    official = ROOT/'data/official/train.parquet'
    raw = pd.read_parquet(official, columns=['label_binary']).label_binary
    assert raw.isin(['benign', 'malicious', 'suspicious']).all()
    truth = raw.map({'benign': 0, 'malicious': 1, 'suspicious': 2}).to_numpy(np.int8)
    assert len(truth) == 2056871
    d = pd.read_parquet(ROOT/'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet',
                        columns=['row_position', 'local', 'root', 'fold', 'truth'])
    keys = pd.read_parquet(ROOT/'artifacts/v130_learning_review_20260930_r2/training_role_error_ledger.parquet',
                           columns=['local', 'canonical_key']).drop_duplicates()
    assert not keys.local.duplicated().any()
    d['canonical_key'] = d.local.map(keys.set_index('local').canonical_key)
    assert len(d) == 112807 and not d.row_position.duplicated().any()
    assert np.array_equal(d.truth, truth[d.row_position]) and d.canonical_key.notna().all()

    folds, allroles = [], []
    for f in range(3):
        folder = RUN/f'fold{f}_S2'
        receipt = read(folder/'fit.json')
        for name, field in [('endpoint.pt','model_sha256'), ('sealed_all_prob.npy','probability_sha256'),
                            ('endpoint_original_rows.parquet','rows_sha256'), ('progress.json','progress_sha256'),
                            ('closures.jsonl','closures_sha256')]:
            assert sha(folder/name) == receipt[field], name
        rows = pd.read_parquet(folder/'endpoint_original_rows.parquet')
        expected = d[d.fold.ne(f)]
        assert np.array_equal(rows.row_position, expected.row_position)
        assert np.array_equal(rows.truth, truth[rows.row_position])
        assert np.array_equal(rows.canonical_key, expected.canonical_key)
        assert not rows.row_position.duplicated().any() and rows.training_role.eq(f).all()
        counts = expected.groupby('canonical_key').truth.nunique()
        pure = rows.canonical_key.map(counts).eq(1).to_numpy()
        assert np.array_equal(rows.pure_TRAIN_input, pure)
        prob = np.load(folder/'sealed_all_prob.npy')
        assert np.isfinite(prob).all() and np.allclose(prob.sum(1),1,rtol=0,atol=1e-12)
        assert np.array_equal(prob[rows.local].argmax(1),rows.pred)
        assert np.array_equal(prob[rows.local],rows[['p0','p1','p2']].to_numpy())
        y,a,b = (rows[k].to_numpy() for k in ['truth','pred_start','pred'])
        pure_pairs = pairs(y[pure],a[pure],b[pure])
        conflicts = expected.groupby(['canonical_key','truth']).size().unstack(fill_value=0)
        floor = int((conflicts.sum(1)-conflicts.max(1)).sum())
        assert int((y != b).sum()) == floor == [22,6,28][f]
        history = read(folder/'progress.json')
        assert len(history) == receipt['accepted_updates']
        assert len({z['parameter_sha256'] for z in history[-5:]}) == 5
        assert all(z['stats']['pure_M_errors'] == z['stats']['pure_S_errors'] == z['stats']['old_correct_pure_regressions'] == 0
                   and z['stats']['M_errors'] == 0 and z['stats']['S_errors'] == floor for z in history[-5:])
        logs = [json.loads(z) for z in (folder/'closures.jsonl').read_text().splitlines()]
        assert len(logs) == receipt['full_gradient_evaluations'] == 200
        mass = np.bincount(y,minlength=3).tolist()
        assert all(z['original_class_mass_seen'] == mass and z['finite'] for z in logs)
        assert int(((a == y) & (b != y) & pure).sum()) == 0
        folds.append(dict(fold=f, original_role_rows=len(rows), pure_role_rows=int(pure.sum()),
                          mixed_empirical_floor=floor, pure_pairs=pure_pairs,
                          full_gradient_evaluations=len(logs), accepted_updates=len(history),
                          receipt_window_recount_passed=True))
        allroles.append(rows)
    role_rows = pd.concat(allroles)
    guards = []
    for file, key in [(ROOT/'artifacts/v138_single_issue_round1_20260930/scoped_training_capabilities.json','verified_training_scopes'),
                      (ROOT/'artifacts/v140_ensemble_training_round2_20261001/additional_verified_TRAIN_scopes.json','scopes')]:
        for scope in read(file)[key]:
            assert sha(ROOT/scope['guard']) == scope['guard_sha256']
            guard = pd.read_parquet(ROOT/scope['guard'])
            assert np.array_equal(guard.truth,truth[guard.row_position])
            selected = role_rows[role_rows.training_role.eq(int(guard.training_role.iloc[0]))]
            actual = guard[['row_position','truth']].merge(selected[['row_position','pred']],on='row_position',validate='one_to_one')
            assert len(actual) == len(guard)
            bad = int(actual.truth.ne(actual.pred).sum())
            assert bad == 0
            guards.append(dict(scope_id=scope['id'],role_original_rows=len(guard),new_errors=bad))

    full = pd.read_parquet(RUN/'full_prediction_ledger.parquet')
    assert np.array_equal(full.row_position,np.arange(len(truth)))
    metrics = {p:classes(truth,full[p].to_numpy()) for p in ['pred_A0','pred_C','pred_S2']}
    for cl in range(3):
        for field,val in metrics['pred_S2'][str(cl)].items():
            reported = quality['task']['full_task']['B'][str(cl)][field]
            assert abs(val-reported) < 1e-12
    full_pairs = pairs(truth,full.pred_A0.to_numpy(),full.pred_S2.to_numpy())
    assert full_pairs == {k:{f:v for f,v in fields.items() if f in full_pairs[k]} for k,fields in delivery['original_row_pairs']['full_vs_A0'].items()}
    asa = pd.read_parquet(RUN/'ASA_prediction_ledger.parquet')
    for col in ['row_position','root','fold','truth','canonical_key']:
        assert np.array_equal(asa[col],d[col])
    assert np.array_equal(asa.pred_S2,full.pred_S2.to_numpy()[asa.row_position])
    support = pd.read_parquet(ROOT/'artifacts/v123_targeted_plan_20260929/support_ladder.parquet',
                              columns=['row_position','exact_key','destination_key','behavior_key','diagnostic_bucket','protocol'])
    evidence = asa.merge(support,on='row_position',validate='one_to_one')
    # All support is recomputed from the legal complement of each held source fold.
    for level,column in [('numeric','canonical_key'),('fact','exact_key'),('destination','destination_key'),('behavior','behavior_key')]:
        for cl in [1,2]:
            evidence[f'{level}_{cl}_rows'] = 0
            evidence[f'{level}_{cl}_roots'] = 0
        for f in range(3):
            held = evidence.fold.eq(f)
            train = evidence[~held]
            assert not (set(train.root) & set(evidence.loc[held,'root']))
            for cl in [1,2]:
                same = train[train.truth.eq(cl)].groupby(column).agg(rows=('row_position','size'),roots=('root','nunique'))
                for what in ['rows','roots']:
                    evidence.loc[held,f'{level}_{cl}_{what}'] = evidence.loc[held,column].map(same[what]).fillna(0).astype(int)
        same_rows = np.where(evidence.truth.eq(1),evidence[f'{level}_1_rows'],evidence[f'{level}_2_rows'])
        same_roots = np.where(evidence.truth.eq(1),evidence[f'{level}_1_roots'],evidence[f'{level}_2_roots'])
        other_rows = np.where(evidence.truth.eq(1),evidence[f'{level}_2_rows'],evidence[f'{level}_1_rows'])
        evidence[f'{level}_support_bucket'] = np.select(
            [same_rows == 0, same_roots == 1],['no_same_class_TRAIN','one_same_class_source'],default='multiple_same_class_sources')
        evidence[f'{level}_opposite_only'] = (same_rows == 0) & (other_rows > 0)
    evidence['before_wrong'] = evidence.pred_C.ne(evidence.truth)
    evidence['after_wrong'] = evidence.pred_S2.ne(evidence.truth)
    evidence['repair'] = evidence.before_wrong & ~evidence.after_wrong
    evidence['new_error'] = ~evidence.before_wrong & evidence.after_wrong
    summaries = {}
    for level in ['numeric','fact','destination','behavior']:
        s = evidence.groupby(['truth',f'{level}_support_bucket']).agg(
            support=('row_position','size'),sources=('root','nunique'),before_errors=('before_wrong','sum'),
            after_errors=('after_wrong','sum'),repairs=('repair','sum'),new_errors=('new_error','sum'),
            opposite_only=(f'{level}_opposite_only','sum')).reset_index()
        summaries[level] = s.to_dict('records')
    by_source = evidence.groupby(['fold','root','truth']).agg(support=('row_position','size'),
        before_errors=('before_wrong','sum'),after_errors=('after_wrong','sum'),
        repairs=('repair','sum'),new_errors=('new_error','sum')).reset_index()
    report = dict(scope='Independent official-truth ledger recount and legal-TRAIN support diagnosis; no new forward, fit, gradient or model selection.',
        latest_actual='V142',new_classifier_fits=0,new_parameter_updates=0,
        official_rows=len(truth),official_sha256=sha(official),run_seal_sha256=sha(RUN/'run_seal.json'),
        delivery_sha256=sha(RUN/'final_delivery.json'),source_sha256=sha(__file__),
        all_checks_passed=True,training_folds=folds,guard_recount=guards,
        pure_role_rows=int(role_rows.pure_TRAIN_input.sum()),
        pure_independent_official_rows=int(role_rows.loc[role_rows.pure_TRAIN_input,'row_position'].nunique()),
        all_full_class_metrics=metrics,full_pairs_vs_A0=full_pairs,
        ASA_pairs_vs_C=pairs(asa.truth.to_numpy(),asa.pred_C.to_numpy(),asa.pred_S2.to_numpy()),
        ASA_pairs_vs_A0=pairs(asa.truth.to_numpy(),asa.pred_A0.to_numpy(),asa.pred_S2.to_numpy()),
        support_summaries=summaries,task_quality_passed=delivery['quality_acceptance'],
        limits=['All source folds have been inspected; this is not blind or external acceptance.',
                'Support buckets describe observed coverage, not proof of causal error attribution.',
                'No same-class exact-input support need not imply missing same-class behavior support.',
                'Same numeric input with opposing training labels can be ambiguous despite TRAIN-pure mastery.',
                'Actual endpoint/window forward replays remain the executor verification; this audit does not duplicate them.'])
    OUT.mkdir()
    evidence.to_parquet(OUT/'legal_TRAIN_support_and_error_ledger.parquet',index=False)
    by_source.sort_values('after_errors',ascending=False).to_csv(OUT/'source_error_transitions.csv',index=False)
    (OUT/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['latest_actual','all_checks_passed','pure_role_rows','pure_independent_official_rows','ASA_pairs_vs_C','task_quality_passed','support_summaries']},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
