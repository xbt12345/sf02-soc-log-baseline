"""Controlled official-support experiment. Prepare -> fit once -> evaluate -> replay.

No outer-answer selection. Common source-held-out calibration, no post-calibration
refit. Historical development only; does not certify whole-SOC or external quality.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from threadpoolctl import threadpool_limits

from prepare_v61 import ASA, CTX_NAMES, endpoint, normalize
from run_v67_targeted import design, counts
from v61_common import FIELDS, sha, read, save

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'artifacts/v61_source_factorial_20260914_r2'
OLD = ROOT / 'artifacts/v67_targeted_20260920'
ARMS = ['A_original_S', 'B_small_sources', 'C_all_sources']
BUDGETS = [.001, .005, .01, .02, .05]
PARAMS = dict(learning_rate=.05, max_iter=500, max_leaf_nodes=15,
              min_samples_leaf=10, l2_regularization=1., early_stopping=False,
              random_state=20260920)


def load():
    conf = read(DATA / 'configuration.json')
    for name in ['records.parquet', 'context.npz', 'private_join_audit.parquet']:
        assert sha(DATA / name) == conf['data_bindings'][name]
    f = pd.read_parquet(DATA / 'records.parquet')
    m = pd.read_parquet(OLD / 'manifest.parquet')
    f['fold'] = f.row_position.map(m.set_index('row_position').fold).fillna(-1).astype(int)
    with np.load(DATA / 'context.npz') as z:
        context = {'stats': z['stats'].copy()}
        ne = z['neighbors']; valid = ne >= 0
        assert ((f.group.to_numpy()[ne.clip(min=0)] == f.group.to_numpy()[:, None]) | ~valid).all()
    assert f.groupby('group').fold.nunique().max() == 1
    assert f.groupby('body_group').fold.nunique().max() == 1
    return f, context, m


def hard(f):
    return (f.label.isin([1, 2]) & f.action.eq('deny') & f.outcome.eq('blocked') &
            f.src_role.eq('outside') & f.dst_role.eq('dmz') & f.transport_protocol.isin(['tcp', 'udp'])).to_numpy()


def roles(f, fold):
    h = hard(f)
    hashed = f.group.map(lambda g: int(hashlib.sha256(('v69-cal-20260921:' + str(g)).encode()).hexdigest()[:16], 16) % 5 == 0).to_numpy()
    cal = h & f.fold.ne(fold).to_numpy() & hashed & f.group.ne(3225).to_numpy()
    fit = h & f.fold.ne(fold).to_numpy() & ~cal
    train = {
        'A_original_S': fit & (f.label.eq(1).to_numpy() | f.fold.ge(0).to_numpy()),
        'B_small_sources': fit & ~(f.label.eq(2).to_numpy() & f.group.eq(3225).to_numpy()),
        'C_all_sources': fit}
    va = h & f.fold.eq(fold).to_numpy()
    for arm, mask in train.items():
        for x, y in [(mask, cal), (mask, va), (cal, va)]:
            for col in ['group', 'body_group']:
                assert not set(f.loc[x, col]) & set(f.loc[y, col])
        assert set(f.loc[mask, 'label']) == {1, 2}
    assert np.array_equal(train['A_original_S'] & f.label.eq(1), train['C_all_sources'] & f.label.eq(1))
    return train, cal, va


def threshold(d, score, budget):
    """Lowest real cutoff satisfying empirical M row AND source loss; whole ties."""
    ix = d.label.eq(1).to_numpy()
    assert ix.any(), 'No malicious calibration observations'
    v = d.loc[ix]; s = np.asarray(score)[ix]
    assert np.isfinite(s).all()
    ct = v.groupby('group').size()
    limits = []
    for w in [np.full(len(v), 1 / len(v)), 1 / len(ct) / v.group.map(ct).to_numpy()]:
        block = pd.DataFrame({'score': s, 'w': w}).groupby('score').w.sum().sort_index(ascending=False)
        cross = np.flatnonzero(block.cumsum().to_numpy() > budget + 1e-12)
        limits.append(float(np.nextafter(block.index[cross[0]], np.inf)) if len(cross) else 0.)
    return max(limits)


def rates(f, pred):
    ans = {}
    for label, name in [(1, 'M'), (2, 'S')]:
        d = f[f.label.eq(label)].copy()
        d['ok'] = np.asarray(pred)[f.label.eq(label)] == label
        ans[name] = {'rows': len(d), 'source_symbols': int(d.group.nunique()),
                     'row_recall': float(d.ok.mean()), 'source_recall': float(d.groupby('group').ok.mean().mean())}
    return ans


def ranking(f, score):
    if f.label.nunique() != 2:
        return None
    ct = f.groupby(['label', 'group']).label.transform('size')
    n = f.groupby('label').group.nunique()
    w = 1 / ct.to_numpy() / f.label.map(n).to_numpy()
    return {'source_AUC': float(roc_auc_score(f.label.eq(2), score, sample_weight=w)),
            'source_standardized_pAUC_1pct': float(roc_auc_score(f.label.eq(2), score, sample_weight=w, max_fpr=.01))}


def prepare(out):
    assert not out.exists(), 'Never replace a run'
    out.mkdir(parents=True)
    f, c, m = load()
    keys = f[FIELDS].astype(str).agg('|'.join, axis=1)
    f['single_key_audit'] = keys
    ledger = pd.read_parquet(ROOT / 'artifacts/v67_delivery_tables_20260920/target_578_ledger.parquet')
    pairs = pd.read_csv(ROOT / 'artifacts/v67_final_evidence_20260920/ten_cross_fold_counterexamples.csv')
    assert len(ledger) == 578
    wanted = set(ledger.row_position) | set(pairs.M_training_row_position)
    # Add deterministic correct M/S controls from each observed branch, not error-only inspection.
    joined = m.merge(f[['row_position', 'src_port_range', 'dst_port_range']], on='row_position', validate='one_to_one')
    controls = joined[joined.label.gt(0) & joined.baseline_pred_with_normal_gate.eq(joined.label)].sort_values('row_position').groupby(['behavior', 'label', 'empty_context', 'src_port_range', 'dst_port_range']).head(1)
    wanted |= set(controls.row_position)
    raw = {}; offset = 0
    for b in pq.ParquetFile(ROOT / 'data/official/train.parquet').iter_batches(batch_size=65536):
        local = sorted(wanted & set(range(offset, offset + len(b))))
        if local:
            table = b.take(np.array(local) - offset).to_pylist()
            raw.update(zip(local, table))
        offset += len(b)
    assert set(raw) == wanted
    meta = pd.read_parquet(DATA / 'private_join_audit.parquet')
    fp = f.set_index('row_position'); mp = meta.set_index(f.row_position)
    trace = []
    labelmap = {'benign': 0, 'malicious': 1, 'suspicious': 2}
    for position in sorted(wanted):
        row = raw[position]; ref = fp.loc[position]
        text, parsed = normalize(row['message_sanitized'], ref.to_dict())
        assert text == ref.text and parsed['raw_hash'] == mp.loc[position, 'raw_hash']
        assert row['event_id'] == ref.event_id and labelmap[row['label_binary']] == ref.label
        r = {'row_position': position, 'event_id': ref.event_id, 'label': int(ref.label), 'group': int(ref.group),
             'is_target': position in set(ledger.row_position), 'raw_message': row['message_sanitized'],
             'raw_sha256': parsed['raw_hash'], 'normalized_text': text,
             'parsed_src_port': parsed['source_port'], 'parsed_dst_port': parsed['destination_port'],
             'src_role': ref.src_role, 'dst_role': ref.dst_role,
             'src_port_observed': ref.src_port_fixed, 'dst_port_observed': ref.dst_port_fixed,
             'numeric_src_survives': str(ref.src_port_fixed).isdigit(),
             'numeric_dst_survives': str(ref.dst_port_fixed).isdigit(),
             'raw_to_normalized_matches': True, 'original_label_kept': True,
             'removed_fields': 'absolute header/time; literal identity; literal zone suffix; policy identity; rule hashes',
             'unknown_contract': 'No M/S operational rule or anonymization fidelity contract; do not infer intent from identity.'}
        trace.append(r)
    pd.DataFrame(trace).to_parquet(out / 'raw_input_trace.parquet', index=False)
    cards = []
    for _, pair in pairs.drop_duplicates('exact_model_input_sha256').iterrows():
        rows = [raw[int(pair.S_row_position)], raw[int(pair.M_training_row_position)]]
        card = {'S_row_position': int(pair.S_row_position), 'M_row_position': int(pair.M_training_row_position),
                'current_input_sha256': pair.exact_model_input_sha256, 'labels_changed': False,
                'conclusion': 'Current-view collision; no newly established transferable discriminator.'}
        for name, row in zip(['S', 'M'], rows):
            match = ASA.fullmatch(row['message_sanitized'].strip()); assert match
            card[name] = {'raw_message': row['message_sanitized'], 'header': match['header'],
                          'src_zone': endpoint(match['src'])[0], 'dst_zone': endpoint(match['dst'])[0],
                          'policy': match['acl'], 'rule_hashes': match['hashes']}
        cards.append(card)
    save(out / 'five_collision_cards.json', cards)
    support_rows = []; role_summary = []
    for fold in range(3):
        train, cal, va = roles(f, fold)
        r = f[['row_position', 'label', 'group', 'body_group', 'fold']].copy()
        r['calibration'] = cal; r['evaluation'] = va
        for arm in ARMS: r[arm] = train[arm]
        r.to_parquet(out / f'fold{fold}_roles.parquet', index=False)
        for name, mask in [('calibration', cal), ('evaluation', va)] + list(train.items()):
            for (proto, label), d in f[mask].groupby(['transport_protocol', 'label']):
                role_summary.append({'fold': fold, 'role': name, 'protocol': proto, 'label': int(label), 'rows': len(d), 'sources': int(d.group.nunique())})
        d = f.copy(); d['empty_context'] = c['stats'][:, 0] == 0
        d['origin'] = np.where(f.fold.ge(0), 'original', np.where(f.group.eq(3225), 'large_auxiliary', 'other_auxiliary'))
        for role, mask in [('training_pool', train['C_all_sources']), ('calibration', cal), ('evaluation', va)]:
            table = d[mask].groupby(['origin', 'transport_protocol', 'src_port_range', 'dst_port_range', 'empty_context', 'label']).agg(rows=('row_position', 'size'), sources=('group', 'nunique'), single_views=('single_key_audit', 'nunique')).reset_index()
            table['fold'] = fold; table['role'] = role; support_rows.append(table)
        small = train['B_small_sources'] & ~train['A_original_S']
        ms = set(keys[train['A_original_S'] & f.label.eq(1).to_numpy()])
        old_s = set(keys[train['A_original_S'] & f.label.eq(2).to_numpy()])
        new = keys[small & keys.isin(ms).to_numpy() & ~keys.isin(old_s).to_numpy()].nunique()
        assert new > 0, 'No additional S support for existing M single-event views'
        role_summary.append({'fold': fold, 'role': 'support_gate', 'new_S_single_views_already_seen_as_M': int(new),
                             'new_small_S_rows': int(small.sum()), 'new_small_S_sources': int(f.loc[small, 'group'].nunique()),
                             'claim': 'New independent-source label support exists; not proof of semantic distinguishability.'})
    pd.concat(support_rows, ignore_index=True).to_csv(out / 'behavior_support.csv', index=False, encoding='utf-8-sig')
    save(out / 'preflight.json', {'raw_trace_rows': len(trace), 'all_578_raw_roundtrips_passed': True,
         'collision_cards': len(cards), 'new_semantic_discriminator_proven': False, 'support_gate_passed': True,
         'source_context_scope': 'Frozen label-free full-source corpus context; no verified session/rate or physical identity.',
         'roles': role_summary, 'data_roles': {'all_current_rows': 'Previously inspected official development; no new blind set',
         'original_59640': 'Same historical source outer folds', 'auxiliary_39758': 'Historical v61 selection/evaluation roles already retired',
         'valid_input_and_private_answer': 'Not read or used this round'},
         'five_pairs_note': '5 distinct input collisions, 10 target rows; no labels changed; no identity restored'})
    helpers = ['run_v69_support_control.py', 'run_v67_targeted.py', 'prepare_v61.py', 'v61_common.py']
    registration = {'version': 'v69-controlled-support-1', 'parameters': PARAMS, 'feature_arm': 'context_numeric',
         'feature_reason': 'Preselected using v68 historical ranking audit; numeric observation plus existing 16 context statistics; no feature search.',
         'arms': ARMS, 'fits_planned': 9, 'class_weight': None, 'post_calibration_refit': False,
         'fixed_calibration': 'SHA256(v69-cal-20260921:source_group) modulo 5 = 0; group3225 reserved for support intervention; common across arms; no split retry',
         'target_used_in_training_or_calibration': False, 'checkpoint_or_hyperparameter_search': False,
         'thresholds': 'Separate protocols; lowest threshold meeting BOTH empirical M row/source loss; whole score ties; no S label objective',
         'budgets_predeclared': BUDGETS, 'primary_reference_budget': .01,
         'hypothesis_comparisons': [['B_small_sources', 'A_original_S'], ['C_all_sources', 'B_small_sources']],
         'mechanism_evidence': 'At 1% calibration budget: paired-source bootstrap S recall gain lower bound>0, gain after removing best-benefit evaluation source>0, mean six-cell source pAUC@1% gain>0. Descriptive adaptive development, not confirmatory inference.',
         'further_validation_gate': 'Mechanism evidence AND every outer fold/protocol empirical M row/source error<=1%; not deployment acceptance.',
         'bootstrap': {'seed': 20260921, 'samples': 2000, 'unit': 'source symbol', 'scope': 'descriptive, not physical independence or adaptive-selection corrected'},
         'deployment_acceptance': 'Requires separate full-SOC/fresh appropriate validation; this run cannot certify it',
         'source_sha256': {n: sha(ROOT / 'training' / n) for n in helpers},
         'inputs_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [DATA/'records.parquet', DATA/'context.npz', DATA/'private_join_audit.parquet', OLD/'manifest.parquet', ROOT/'data/official/train.parquet']},
         'prepared_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()},
         'scope': 'Official data only; same source-symbol held-out adaptive development; no external/general deployment claim'}
    save(out / 'preregistered.json', registration)
    print('PREFLIGHT_AND_REGISTRATION_COMPLETE', len(trace), 'raw rows; 9 fits registered', flush=True)


def train_run(out):
    reg = read(out/'preregistered.json')
    for name, digest in reg['source_sha256'].items(): assert sha(ROOT/'training'/name) == digest
    for name, digest in reg['inputs_sha256'].items(): assert sha(ROOT/name) == digest
    for name, digest in reg['prepared_sha256'].items(): assert sha(out/name) == digest
    assert not (out/'fold0').exists(), 'Do not silently retrain an existing run'
    f, c, m = load(); start = time.monotonic(); fits = 0
    with threadpool_limits(limits=4):
        for fold in range(3):
            r = pd.read_parquet(out/f'fold{fold}_roles.parquet')
            assert np.array_equal(r.row_position, f.row_position)
            cal = r.calibration.to_numpy(); va = r.evaluation.to_numpy()
            for arm in ARMS:
                sub = out/f'fold{fold}'/arm; sub.mkdir(parents=True)
                fit = r[arm].to_numpy(); ix = np.flatnonzero(fit)
                x, encoder, names, cat = design(f, c, 'context_numeric', ix)
                model = HistGradientBoostingClassifier(**PARAMS, categorical_features=cat)
                t = time.monotonic(); model.fit(x[fit], f.loc[fit, 'label'].eq(2)); fits += 1
                assert int(model.n_iter_) == PARAMS['max_iter']
                artifact = {'model': model, 'encoder': encoder, 'names': names,
                            'fit_positions': f.loc[fit, 'row_position'].to_numpy(),
                            'calibration_positions': f.loc[cal, 'row_position'].to_numpy(),
                            'preregistered_sha256': sha(out/'preregistered.json')}
                joblib.dump(artifact, sub/'model.joblib')
                for role, mask in [('fit', fit), ('calibration', cal), ('evaluation', va)]:
                    scores = model.predict_proba(x[mask])[:, 1]
                    assert np.isfinite(scores).all()
                    d = f.loc[mask, ['row_position', 'label', 'group', 'transport_protocol']].copy()
                    d['score'] = scores; d.to_parquet(sub/(role+'.parquet'), index=False)
                cd = pd.read_parquet(sub/'calibration.parquet'); cuts = {}
                for budget in BUDGETS:
                    cuts[str(budget)] = {}
                    for proto in ['tcp', 'udp']:
                        d = cd[cd.transport_protocol.eq(proto)]
                        cut = threshold(d, d.score, budget)
                        rr = rates(d, np.where(d.score.ge(cut), 2, 1))
                        assert 1-rr['M']['row_recall'] <= budget+1e-12 and 1-rr['M']['source_recall'] <= budget+1e-12
                        cuts[str(budget)][proto] = {'threshold': cut, 'calibration': rr}
                save(sub/'thresholds.json', cuts)
                save(sub/'fit_receipt.json', {'fitted_iterations': int(model.n_iter_), 'fit_rows': int(fit.sum()),
                    'fit_sources': int(f.loc[fit, 'group'].nunique()), 'S_training_fraction': float(f.loc[fit, 'label'].eq(2).mean()),
                    'seconds': time.monotonic()-t, 'model_sha256': sha(sub/'model.joblib'),
                    'calibration_after_freeze': True, 'post_calibration_fit_calls': 0})
                print('FIT_COMPLETE', fold, arm, 'rows', int(fit.sum()), 'seconds', round(time.monotonic()-t, 2), flush=True)
    save(out/'training_complete.json', {'actual_new_fits': fits, 'seconds': time.monotonic()-start,
         'platform_used': False, 'device': 'CPU, 4 thread limit', 'outer_label_selection': False})


def paired_gain(frame, a, b):
    s = frame[frame.label.eq(2)].copy()
    s['a'] = (np.asarray(a)[frame.label.eq(2)] == 2).astype(float)
    s['b'] = (np.asarray(b)[frame.label.eq(2)] == 2).astype(float)
    g = s.groupby('group')[['a', 'b']].mean(); delta = (g.a-g.b).to_numpy()
    rng = np.random.default_rng(20260921)
    boot = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(2000)])
    return {'source_recall_gain': float(delta.mean()), 'descriptive_bootstrap_95pct': np.quantile(boot, [.025, .975]).tolist(),
            'gain_after_removing_best_source': float(np.delete(delta, np.argmax(delta)).mean()),
            'sources_improved': int((delta>0).sum()), 'sources_worsened': int((delta<0).sum()),
            'source_count': len(delta)}


def analyze(out):
    assert read(out/'training_complete.json')['actual_new_fits'] == 9
    f, c, m = load(); pos = pd.Index(f.row_position).get_indexer(m.row_position)
    original = f.iloc[pos].reset_index(drop=True); h = hard(original)
    base = m.baseline_pred_with_normal_gate.to_numpy(); target = m.target_578.to_numpy()
    result = {'new_fits': 9, 'quality_acceptance': False, 'scope': 'Adaptive source-symbol development only; all 59640 original rows with frozen nonhard predictions',
              'historical_baseline_only': counts(original, base, base, target), 'arms': {}, 'hypothesis_comparisons': {}}
    primary = {}; ranks = {}; sources = []; curves = []
    ledger = m[target].copy()
    for arm in ARMS:
        scores = np.full(len(m), np.nan); cuts_by_fold = {}
        train_rates = []
        for fold in range(3):
            sub = out/f'fold{fold}'/arm
            d = pd.read_parquet(sub/'evaluation.parquet'); idx = pd.Index(m.row_position).get_indexer(d.row_position)
            assert (idx>=0).all(); scores[idx] = d.score
            cuts_by_fold[fold] = read(sub/'thresholds.json')
            fd = pd.read_parquet(sub/'fit.parquet')
            train_rates.append({'fold': fold, 'default_threshold_05': rates(fd, np.where(fd.score.ge(.5), 2, 1)),
                                'ranking': ranking(fd, fd.score)})
        assert np.isfinite(scores[h]).all() and np.isnan(scores[~h]).all()
        cell_ranks = []
        for fold in range(3):
            for proto in ['tcp', 'udp']:
                ix = h & original.fold.eq(fold).to_numpy() & original.transport_protocol.eq(proto).to_numpy()
                cell_ranks.append({'fold': fold, 'protocol': proto, **ranking(original[ix], scores[ix])})
        ranks[arm] = cell_ranks
        effects = {}
        for budget in BUDGETS:
            pred = base.copy(); cells = []
            for fold in range(3):
                for proto in ['tcp', 'udp']:
                    ix = h & original.fold.eq(fold).to_numpy() & original.transport_protocol.eq(proto).to_numpy()
                    cut = cuts_by_fold[fold][str(budget)][proto]['threshold']
                    pred[ix] = np.where(scores[ix] >= cut, 2, 1)
                    rr = rates(original[ix], pred[ix]); cells.append({'fold': fold, 'protocol': proto, **rr})
                    curves.append({'arm': arm, 'budget': budget, 'fold': fold, 'protocol': proto,
                                   'M_row_error': 1-rr['M']['row_recall'], 'M_source_error': 1-rr['M']['source_recall'],
                                   'S_source_recall': rr['S']['source_recall'], 'S_row_recall': rr['S']['row_recall']})
            effects[str(budget)] = {'historical_effects': counts(original, pred, base, target),
                'all_row_classification': classification_report(original.label, pred, labels=[0,1,2], output_dict=True, zero_division=0),
                'confusion_B_M_S': confusion_matrix(original.label, pred, labels=[0,1,2]).tolist(),
                'hard_rates': rates(original[h], pred[h]), 'cells': cells}
            o = m[['row_position', 'label', 'group', 'fold']].copy(); o['hard_score'] = scores; o['pred'] = pred
            o.to_parquet(out/f'{arm}_budget{budget}_oof.parquet', index=False)
            if budget == .01:
                primary[arm] = pred; ledger[arm+'_pred'] = pred[target]
                for group, d in original[h].assign(pred=pred[h], baseline=base[h]).groupby('group'):
                    sources.append({'arm': arm, 'group': int(group), 'rows': len(d), 'labels': '|'.join(map(str, sorted(d.label.unique()))),
                                    'correct': int(d.pred.eq(d.label).sum()), 'baseline_correct': int(d.baseline.eq(d.label).sum())})
        result['arms'][arm] = {'fit_diagnostics': train_rates, 'ranking_per_fold_protocol': cell_ranks, 'budgets': effects}
    for candidate, reference in [('B_small_sources','A_original_S'), ('C_all_sources','B_small_sources')]:
        pair = paired_gain(original[h], primary[candidate][h], primary[reference][h])
        pair['mean_six_cell_pAUC_gain'] = float(np.mean([x['source_standardized_pAUC_1pct']-y['source_standardized_pAUC_1pct'] for x,y in zip(ranks[candidate], ranks[reference])]))
        pair['mechanism_evidence'] = bool(pair['descriptive_bootstrap_95pct'][0]>0 and pair['gain_after_removing_best_source']>0 and pair['mean_six_cell_pAUC_gain']>0)
        cells = result['arms'][candidate]['budgets']['0.01']['cells']
        pair['outer_M_budget_met'] = all(max(1-x['M']['row_recall'],1-x['M']['source_recall'])<=.01+1e-12 for x in cells)
        pair['eligible_for_further_validation'] = pair['mechanism_evidence'] and pair['outer_M_budget_met']
        pair['matched_effects'] = counts(original, primary[candidate], primary[reference], target)
        result['hypothesis_comparisons'][candidate+'_vs_'+reference] = pair
    ledger['accepted_repair'] = False
    ledger.to_csv(out/'target_578_results.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(sources).to_csv(out/'source_effects.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(curves).to_csv(out/'fixed_budget_curves.csv',index=False,encoding='utf-8-sig')
    save(out/'analysis.json',result)
    print(json.dumps({'comparisons': result['hypothesis_comparisons'], 'primary_effects': {a:result['arms'][a]['budgets']['0.01']['historical_effects'] for a in ARMS}},ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['prepare','train','analyze']);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(); {'prepare':prepare,'train':train_run,'analyze':analyze}[args.phase](args.out.resolve())
