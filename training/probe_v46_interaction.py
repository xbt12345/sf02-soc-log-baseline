"""One preregistered behavior interaction, with frozen v45 reduced-model controls."""
import argparse
import importlib.util
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

GROUPS = {'src': ['src_port_fixed', 'src_port_range'],
          'dst': ['dst_port_fixed', 'dst_port_range']}
VIEWS = {'full': [], 'no_src': GROUPS['src'], 'no_dst': GROUPS['dst'],
         'no_ports': GROUPS['src'] + GROUPS['dst']}
SCENARIOS = {'original': [], 'hide_source': GROUPS['src'],
             'hide_destination': GROUPS['dst'], 'hide_both': VIEWS['no_ports']}
FIELDS = ['action', 'outcome', 'transport_protocol', 'src_role', 'dst_role',
          'src_port_fixed', 'dst_port_fixed', 'src_port_range', 'dst_port_range',
          'icmp_type', 'icmp_code', 'icmp_message', 'icmp_unreachable']
MISSING = '__UNOBSERVED__'
COMBO_FIELDS = ['transport_protocol', 'src_role', 'dst_role']
COMBO = 'protocol_roles'
COVERAGES = [.5, .8, .95]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def helpers(out):
    return module(out/'v45_probe_reference.py', 'v45_reference'), module(out/'v44_probe_reference.py', 'v44_reference')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def records(x, view, interaction=True):
    excluded = set(VIEWS[view])
    result = []
    for row in x.to_dict(orient='records'):
        d = {k: str(row[k]) for k in FIELDS if k not in excluded and row[k] != MISSING}
        if interaction and all(k in d for k in COMBO_FIELDS):
            d[COMBO] = json.dumps([d[k] for k in COMBO_FIELDS], separators=(',', ':'))
        result.append(d)
    return result


def routing(x):
    src = x.src_port_fixed.to_numpy() != MISSING
    dst = x.dst_port_fixed.to_numpy() != MISSING
    return np.where(src, np.where(dst, 'full', 'no_dst'), np.where(dst, 'no_src', 'no_ports'))


def predict(models, x):
    routes = routing(x); q = np.empty(len(x))
    for view in VIEWS:
        use = routes == view
        if use.any():
            q[use] = models[view].predict_proba(records(x.loc[use], view))[:, 1]
    return q


def key_array(x, view):
    return np.array([json.dumps(a, sort_keys=True, separators=(',', ':')) for a in records(x, view, False)])


def risk_coverage(y, q):
    """Expected selective risk under uniform tie breaking; no label-based tie order."""
    confidence = np.maximum(q, 1-q)
    wrong = (np.where(q >= .5, 1, 2) != y).astype(float)
    groups = pd.DataFrame({'c': confidence, 'e': wrong}).groupby('c').e.agg(['sum', 'size']).sort_index(ascending=False)
    risk = np.empty(len(y)); previous_n = 0; previous_e = 0.
    for errors, count in groups[['sum', 'size']].itertuples(index=False, name=None):
        count = int(count); j = np.arange(1, count+1)
        risk[previous_n:previous_n+count] = (previous_e+j*errors/count)/(previous_n+j)
        previous_n += count; previous_e += errors
    return {'tie_expected_AURC': float(risk.mean()),
            'fixed_coverage_risk': {str(c): {'rows': max(1, int(np.floor(c*len(y)))),
                'risk': float(risk[max(1, int(np.floor(c*len(y))))-1])} for c in COVERAGES},
            'tie_handling': 'Expected risk of uniform random ordering within exact confidence ties; no row or label priority.'}


def metrics(v, y, q):
    if not len(y):
        return {'rows': 0, 'supported': False}
    return dict(v.metrics(y, q), **risk_coverage(y, q))


def prepare(root, out):
    assert not out.exists(), 'Never overwrite an experiment.'
    old = root/'artifacts/v45_missingness_20260914'
    v = module(old/'probe_v45_missingness.py', 'v45_reference'); m = v.reference(old)
    c = read(old/'configuration.json'); pre = read(old/'preregistered.json'); rep = read(old/'report.json')
    assert m.sha(old/'configuration.json') == pre['configuration_sha256']
    assert m.sha(old/'probe_v45_missingness.py') == pre['script_sha256']
    assert m.sha(old/'v44_probe_reference.py') == c['reference_sha256']
    bindings = dict(c['source_bindings'])
    for p, h in rep['model_sha256'].items():
        assert m.sha(old/p) == h
        bindings[(old/p).relative_to(root).as_posix()] = h
    for name in ['configuration.json', 'preregistered.json', 'report.json', 'evaluation.parquet',
                 'probe_v45_missingness.py', 'v44_probe_reference.py', 'verification.json',
                 'information_and_calibration_audit.json', 'remaining_destination_errors.json']:
        bindings[(old/name).relative_to(root).as_posix()] = m.sha(old/name)
    for p, h in bindings.items(): assert m.sha(root/p) == h, p
    r, x, _, _ = m.data(root)
    out.mkdir(parents=True)
    shutil.copyfile(__file__, out/Path(__file__).name)
    shutil.copyfile(old/'probe_v45_missingness.py', out/'v45_probe_reference.py')
    shutil.copyfile(old/'v44_probe_reference.py', out/'v44_probe_reference.py')
    config = {'version': 'v46-one-protocol-role-interaction-1', 'lr': c['lr'], 'threshold': .5,
        'fields': FIELDS, 'views': VIEWS, 'scenarios': SCENARIOS, 'interaction': COMBO_FIELDS,
        'interaction_missing': 'Omit if any component unobserved; no dedicated missing tuple.',
        'fit_policy': 'Same original body-isolated folds and rows once. Each view uses ALL other-fold ASA rows. No weighting, imputation, augmentation, thresholds or hyperparameter search.',
        'rows': len(r), 'fits_planned': 12, 'packages': c['packages'],
        'comparison': 'I = R plus one categorical protocol-role interaction; old R models never refit.',
        'quality_gates': {
            'natural': 'Total errors strictly below R; M->S <=600 and S->M <=4835. At least two folds have fewer total errors.',
            'unseen': 'Original unseen-full-key errors <=1070; unknown remaining-view-key errors do not increase in any scenario.',
            'removal': 'Source, destination and both-port-removal errors do not increase vs R; destination-removal strictly improves to substantiate its targeted fix.',
            'probability': 'Log loss and Brier do not increase in any of four scenarios (tolerance 1e-12).',
            'selective': 'Tie-expected AURC does not increase in any scenario (1e-12). Report risk at fixed 50/80/95% coverage; no count-only confidence gate.',
            'scope': 'Implementation/raw-input/normal-boundary checks must pass. Then run old broad-template stress with same roles. No automatic promotion or claim of blind transfer.'},
        'source_bindings': bindings, 'script_sha256': m.sha(__file__),
        'reference_sha256': {n: m.sha(out/n) for n in ['v45_probe_reference.py', 'v44_probe_reference.py']},
        'limitations': ['Previously inspected development data; improvement remains exploratory.',
            'A deterministic interaction adds expressiveness, not new information or attack ground truth.',
            'Port removal is information loss; different outputs need not be wrong.',
            'Observed feature collisions cannot be separated by this interaction. No synthetic labels or data.']}
    m.save(out/'configuration.json', config)
    m.save(out/'preregistered.json', {'configuration_sha256': m.sha(out/'configuration.json'),
        'script_sha256': m.sha(out/Path(__file__).name), 'fits_so_far': 0})
    print(json.dumps({'prepared': str(out), 'new_fits': 0, 'planned': 12}), flush=True)


def fit(root, out):
    v, m = helpers(out); start = time.monotonic(); c = read(out/'configuration.json'); pre = read(out/'preregistered.json')
    assert m.sha(__file__) == pre['script_sha256']
    assert m.sha(out/'configuration.json') == pre['configuration_sha256']
    for name, h in c['reference_sha256'].items(): assert m.sha(out/name) == h
    for p, h in c['source_bindings'].items(): assert m.sha(root/p) == h, p
    assert not (out/'started.json').exists()
    m.save(out/'started.json', {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'config': pre['configuration_sha256']})
    r, x, b, ab = m.data(root); y = r.label_index.to_numpy()
    old = root/'artifacts/v45_missingness_20260914'; old_e = pd.read_parquet(old/'evaluation.parquet')
    v44_e = pd.read_parquet(root/'artifacts/v44_ready_methods_20260914/evaluation.parquet')
    assert np.array_equal(v44_e.row_position, r.row_position)
    fit_log = []; hashes = {}; result = []
    for fold in range(3):
        tr = r.fold.to_numpy() != fold; ev = ~tr; folder = out/('fold_%s'%fold); folder.mkdir()
        models = {}; controls = {}; fit_keys = {}
        for view in VIEWS:
            t = time.monotonic(); rec = records(x.loc[tr], view)
            model = make_pipeline(DictVectorizer(sparse=True), LogisticRegression(**c['lr']))
            model.fit(rec, y[tr] == 1)
            assert model[-1].n_iter_.max() < c['lr']['max_iter'], 'Non-convergence'
            names = model[0].get_feature_names_out()
            assert not any(MISSING in n for n in names)
            assert {n.split('=', 1)[0] for n in names} <= (set(FIELDS)-set(VIEWS[view]))|{COMBO}
            assert model[-1].classes_.tolist() == [False, True]
            p = folder/(view+'.joblib'); joblib.dump(model, p); models[view] = joblib.load(p)
            hashes[p.relative_to(out).as_posix()] = m.sha(p)
            controls[view] = joblib.load(old/('fold_%s/%s.joblib'%(fold, view)))
            assert np.array_equal(model.predict_proba(records(x.loc[ev], view)), models[view].predict_proba(records(x.loc[ev], view)))
            fit_keys[view] = set(key_array(x.loc[tr], view))
            fit_log.append({'fold': fold, 'view': view, 'fit_rows': int(tr.sum()), 'features': len(names),
                'interaction_categories': sum(n.startswith(COMBO+'=') for n in names),
                'iterations': int(model[-1].n_iter_.max()), 'seconds': time.monotonic()-t})
        for scenario, cols in SCENARIOS.items():
            xx = x.loc[ev].copy(); xx[cols] = MISSING
            d = old_e[(old_e.fold == fold)&(old_e.scenario == scenario)].sort_values('row_position').copy()
            assert np.array_equal(d.row_position, r.loc[ev, 'row_position'])
            assert np.array_equal(v.predict_reduced(controls, xx), d.R_qM.to_numpy())
            assert np.array_equal(routing(xx), d.reduced_view.to_numpy())
            d['I_qM'] = predict(models, xx)
            d['seen_original_key'] = v44_e.loc[ev, 'seen_fit_key'].to_numpy()
            known = np.empty(len(xx), dtype=bool)
            for view in VIEWS:
                use = routing(xx) == view
                if use.any(): known[use] = [k in fit_keys[view] for k in key_array(xx.loc[use], view)]
            d['seen_remaining_view_key'] = known
            result.append(d)
        print(json.dumps({'fold': fold, 'fits_done': len(fit_log),
            'original': {n: v.metrics(y[ev], result[-4][n+'_qM'].to_numpy()) for n in ['R', 'I']}}), flush=True)
    e = pd.concat(result).sort_values(['scenario', 'row_position']).reset_index(drop=True)
    e.to_parquet(out/'evaluation.parquet', index=False)
    scenarios = {}; slices = {}; folds = {}
    for scenario in SCENARIOS:
        d = e[(e.scenario == scenario)&e.eligible_stress_row]
        scenarios[scenario] = {n: metrics(v, d.label_index.to_numpy(), d[n+'_qM'].to_numpy()) for n in ['R', 'I']}
        slices[scenario] = {}
        for name, use in [('unseen_original_key', ~d.seen_original_key), ('unseen_remaining_key', ~d.seen_remaining_view_key)]:
            z = d[use]
            slices[scenario][name] = {n: metrics(v, z.label_index.to_numpy(), z[n+'_qM'].to_numpy()) for n in ['R', 'I']}
        for view in VIEWS:
            z = d[d.reduced_view == view]
            slices[scenario]['view_'+view] = {n: metrics(v, z.label_index.to_numpy(), z[n+'_qM'].to_numpy()) for n in ['R', 'I']}
        folds[scenario] = {str(f): {n: metrics(v, z.label_index.to_numpy(), z[n+'_qM'].to_numpy()) for n in ['R', 'I']}
                              for f in range(3) for z in [d[d.fold == f]]}
    gates = {}; base = scenarios['original']['R']; new = scenarios['original']['I']
    gates['original_total_strictly_improves'] = new['errors'] < base['errors']
    gates['original_M_errors_do_not_increase'] = new['confusion_M_S'][0][1] <= base['confusion_M_S'][0][1]
    gates['original_S_errors_do_not_increase'] = new['confusion_M_S'][1][0] <= base['confusion_M_S'][1][0]
    gates['at_least_two_folds_improve'] = sum(d['I']['errors'] < d['R']['errors'] for d in folds['original'].values()) >= 2
    gates['unseen_original_errors_do_not_increase'] = slices['original']['unseen_original_key']['I']['errors'] <= slices['original']['unseen_original_key']['R']['errors']
    for s in SCENARIOS:
        a = scenarios[s]['R']; z = scenarios[s]['I']
        if s != 'original': gates[s+'_errors_do_not_increase'] = z['errors'] <= a['errors']
        if s == 'hide_destination': gates[s+'_errors_strictly_improve'] = z['errors'] < a['errors']
        for k in ['conditional_log_loss', 'brier', 'tie_expected_AURC']:
            gates[s+'_'+k+'_not_increased'] = z[k] <= a[k]+1e-12
        uu = slices[s]['unseen_remaining_key']
        gates[s+'_unseen_remaining_errors_not_increased'] = uu['I'].get('errors', 0) <= uu['R'].get('errors', 0)
    m.save(out/'report.json', {'status': 'development_comparison_not_promoted', 'new_fits': len(fit_log),
        'scenario_metrics': scenarios, 'slices': slices, 'fold_metrics': folds,
        'quality_gates': gates, 'all_primary_quality_gates_pass': all(gates.values()),
        'model_sha256': hashes, 'fit_log': fit_log, 'seconds': time.monotonic()-start,
        'limitations': c['limitations']})
    print(json.dumps({'done': True, 'seconds': time.monotonic()-start, 'primary_quality_pass': all(gates.values()),
        'errors': {s: {n: z[n]['errors'] for n in ['R', 'I']} for s, z in scenarios.items()},
        'failed_gates': [k for k, val in gates.items() if not val]}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['prepare', 'fit']); p.add_argument('--root', required=True); p.add_argument('--out', required=True)
    a = p.parse_args(); (prepare if a.mode == 'prepare' else fit)(Path(a.root).resolve(), Path(a.out).resolve())
