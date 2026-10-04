"""Read-only model diagnosis on inspected v39 development; never fits a model."""
import collections
import gc
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/v39_local_r2_20260913'
sys.path.insert(0, str(BASE / 'frozen_training_runtime'))
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from run_v38_train import actual_keys
from run_v39_train import pair_support


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def save(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def count_slice(mask, y, pred, body):
    return {'rows': int(mask.sum()), 'errors': int((mask & (y != pred)).sum()),
            'body_keys': int(len(np.unique(body[mask]))),
            'class_rows': np.bincount(y[mask], minlength=3).tolist(),
            'class_errors': np.bincount(y[mask & (y != pred)], minlength=3).tolist()}


def main():
    out = ROOT / 'evidence/2026-09-13/v40_mechanisms'
    out.mkdir(exist_ok=False)
    prepared = BASE / 'prepared'
    receipt = json.loads((prepared / 'complete.json').read_text(encoding='utf-8'))
    checked = {}
    for name in ['rows.parquet', 'projections.parquet', 'configuration.json']:
        digest = sha(prepared / name)
        assert digest == receipt['files'][name], name
        checked[name] = digest
    rows = pq.read_table(prepared / 'rows.parquet', columns=['row_position','label_index','route','body_group','projection_id','fold']).to_pandas()
    rows = rows[rows.fold >= 0].reset_index(drop=True)
    active, ids = np.unique(rows.projection_id.to_numpy(), return_inverse=True)
    proj = pq.read_table(prepared / 'projections.parquet').to_pandas().iloc[active].reset_index(drop=True)
    facts = [json.loads(f) for f in proj.facts]
    texts = proj.text.tolist()
    y = rows.label_index.to_numpy(dtype=np.int64)
    folds = rows.fold.to_numpy()
    routes = rows.route.to_numpy()
    body = rows.body_group.to_numpy()
    config = json.loads((prepared / 'configuration.json').read_text(encoding='utf-8'))
    summaries = []
    bindings = []
    for fold in range(3):
        folder = BASE / ('primary/fold_%s/SEMANTIC' % fold)
        done = json.loads((folder / 'complete.json').read_text(encoding='utf-8'))
        digest = sha(folder / 'model.joblib')
        assert digest == done['model_sha256']
        bundle = joblib.load(folder / 'model.joblib')
        x = sparse.hstack([bundle['text_encoder'].transform(texts), bundle['fact_encoder'].transform(facts)], format='csr')
        keys = actual_keys(x)
        p = bundle['model'].predict_proba(x)
        pred = p.argmax(axis=1)[ids]
        fit, ev = folds != fold, folds == fold
        original = pq.read_table(folder / 'evaluation.parquet').to_pandas()
        assert np.array_equal(original.row_position, rows.row_position.to_numpy()[ev])
        diff = float(np.abs(original[['p_benign','p_malicious','p_suspicious']].to_numpy() - p[ids[ev]]).max())
        assert diff < 1e-10
        nkeys = int(keys.max() + 1)
        fk = keys[ids]
        fc = np.bincount(fk[fit]*3+y[fit], minlength=nkeys*3).reshape(-1,3)
        ec = np.bincount(fk[ev]*3+y[ev], minlength=nkeys*3).reshape(-1,3)
        seen = fc.sum(axis=1)[fk] > 0
        conflict = ((fc > 0).sum(axis=1) > 1)[fk]
        same = fc.argmax(axis=1)[fk] == y
        parts = {
            'seen_fit_pure_same_label': seen & ~conflict & same,
            'seen_fit_pure_opposite_label': seen & ~conflict & ~same,
            'seen_fit_conflicted': seen & conflict,
            'unseen_fit': ~seen}
        for m in [ev, fit]:
            assert sum(int((v & m).sum()) for v in parts.values()) == int(m.sum())
        partitions = {}
        for name, population in [('all', np.ones(len(rows), dtype=bool)), ('asa', routes == 'asa'), ('auth', routes == 'authentication'), ('flow', routes == 'native_flow')]:
            partitions[name] = {role: {k:count_slice(v & pop & population, y, pred, body) for k,v in parts.items()} for role,pop in [('fit',fit),('evaluation',ev)]}
        asa_counts = np.bincount(fk[fit & (routes == 'asa')]*3+y[fit & (routes == 'asa')], minlength=nkeys*3).reshape(-1,3)
        asa_empirical_floor = int((asa_counts.sum(axis=1)-asa_counts.max(axis=1)).sum())
        all_empirical_floor = int((fc.sum(axis=1)-fc.max(axis=1)).sum())
        wrong = ev & (y != pred) & (routes == 'asa')
        tops = []
        for pid, n in collections.Counter(ids[wrong]).most_common(25):
            m = ev & (ids == pid)
            key = int(keys[pid])
            tops.append({'projection_id': int(active[pid]), 'errors': n, 'eval_counts': np.bincount(y[m], minlength=3).tolist(),
                         'all_fit_key_counts': fc[key].tolist(), 'pred': int(p[pid].argmax()), 'probabilities': p[pid].tolist(),
                         'facts': facts[pid], 'text': texts[pid], 'body_keys': int(len(np.unique(body[m])))})
        # Reproduce original gate verbatim. A relaxed matching diagnostic is
        # not an eligible training pair and does not assign causal labels.
        original_gate = pair_support(rows, facts, texts, ids, keys, fit, config)
        saved_gate = json.loads((folder / 'pair_support.json').read_text(encoding='utf-8'))
        assert original_gate == saved_gate
        uniq = rows.loc[fit, ['projection_id','body_group','label_index','route']].drop_duplicates()
        look = {int(v):i for i,v in enumerate(active)}
        cells = collections.defaultdict(list)
        for item in uniq.itertuples(index=False):
            k = look[item.projection_id]
            if (fc[keys[k]] > 0).sum() > 1:
                continue
            for target in config['pair_gate']['target_fields']:
                if target not in facts[k]:
                    continue
                if target == 'icmp_code' and not (facts[k].get('transport_protocol') == 'icmp' and facts[k].get('icmp_type') == 3):
                    continue
                other = dict(facts[k]); value = other.pop(target)
                if target == 'icmp_code': other.pop('icmp_unreachable', None)
                if target == 'outcome': other.pop('auth_result', None)
                if target == 'response': other.pop('authentication_interaction', None)
                ctx = json.dumps([target, item.route, sorted(facts[k]), other], sort_keys=True)
                cells[ctx].append((int(item.body_group), int(item.label_index), str(value), k))
        relaxed = []
        for ctx, items in cells.items():
            ns = [len({a[0] for a in items if a[1] == c}) for c in range(3)]
            cs = [c for c in range(3) if ns[c] >= 3]
            if len(cs) >= 2 and len({a[2] for a in items if a[1] in cs}) >= 2:
                relaxed.append({'context': json.loads(ctx), 'class_body_support': ns,
                    'examples': [{'label':c,'value':v,'text':texts[k]} for c,v,k in sorted({(a[1],a[2],a[3]) for a in items})][:12]})
        report = {'fold':fold, 'partitions':partitions, 'fit_empirical_minimum_errors_all':all_empirical_floor,
                  'fit_empirical_minimum_errors_asa':asa_empirical_floor, 'actual_fit_errors_all':int((fit & (y!=pred)).sum()),
                  'actual_fit_errors_asa':int((fit & (routes=='asa') & (y!=pred)).sum()),
                  'top_asa_errors':tops, 'original_gate':original_gate,
                  'relaxed_text_removed_cells_diagnostic_only':relaxed, 'new_training_executed':False}
        save(out / ('fold_%s.json' % fold), report)
        summaries.append({'fold':fold,'partitions':partitions,'fit_minimum_errors_all':all_empirical_floor,
            'fit_minimum_errors_asa':asa_empirical_floor,'actual_fit_errors_all':report['actual_fit_errors_all'],
            'actual_fit_errors_asa':report['actual_fit_errors_asa'],'relaxed_cells':len(relaxed)})
        bindings.append({'fold':fold,'model_sha256':digest, 'evaluation_sha256':sha(folder/'evaluation.parquet'), 'replay_max_difference':diff})
        print(json.dumps({'fold':fold,'fit_floor':all_empirical_floor, 'actual_fit_errors':report['actual_fit_errors_all'], 'relaxed_cells':len(relaxed)}), flush=True)
        del x, p, bundle, original
        gc.collect()
    auth = []
    for k in np.unique(ids[routes == 'authentication']):
        m = ids == k
        auth.append({'projection_id':int(active[k]),'text':texts[k],'facts':facts[k], 'counts':np.bincount(y[m],minlength=3).tolist(), 'fold_body_support':[[int(len(np.unique(body[m & (folds==f) & (y==c)]))) for c in range(3)] for f in range(3)]})
    save(out/'auth_examples.json',auth)
    save(out/'summary.json',{'scope':'Inspected v39 development; frozen model replay and fit-support diagnosis only',
         'inputs':checked,'script_sha256':sha(__file__), 'models':bindings,'folds':summaries,
         'new_training_executed':False,'fresh_blind_test':False,
         'caveats':['Fit purity is sample-dependent, not proof of ground truth or future identifiability.',
          'Empirical fit lower bounds are descriptive, not a deployable classifier or attainable held-out score.',
          'Relaxed text-free matches are diagnostic only and do not authorize contrastive training.']})


if __name__ == '__main__':
    main()
