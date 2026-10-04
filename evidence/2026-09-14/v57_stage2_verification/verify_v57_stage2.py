"""Independent fit objective, branch isolation, split and selection verification."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax, logsumexp
from sklearn.model_selection import StratifiedGroupKFold
import run_v54 as v
import run_v56 as t
import run_v57_stage2 as run
from verify_v55 import manual
from verify_v56 import independent_keys
from verify_v57_stage1 import independent_field_keys, codes, add_offset, independent_gate

OUT = t.ROOT / 'evidence/2026-09-14/v57_stage2_verification'

def head_mask(x, field):
    other = 'dst_role' if field == 'src_role' else 'src_role'
    return (x[field].to_numpy() == v.MISSING) & (x[other].to_numpy() != v.MISSING)

def row_objective(bundle, base, x, y, bodies, strength):
    maps, counts, supports, names = {}, [], [], []
    for level, values in independent_field_keys(x).items():
        temp = pd.DataFrame({'key': values, 'body': np.asarray(bodies)}).dropna(subset=['key'])
        g = temp.groupby('key', sort=True).agg(n=('body', 'size'), b=('body', 'nunique'))
        maps[level] = {key: len(names) + i for i, key in enumerate(g.index)}
        names.extend([[level, key] for key in g.index])
        counts.extend(g.n)
        supports.extend(g.b)
    design, w = bundle['design'], bundle['weights']
    assert maps == design['maps'] and names == design['feature_keys']
    np.testing.assert_array_equal(counts, design['rows'])
    np.testing.assert_array_equal(supports, design['bodies'])
    ids = codes(design, x)
    z = add_offset(manual(base, x), ids, w)
    y = np.asarray(y)
    precision = strength * np.asarray(counts) / len(y) / np.sqrt(supports)
    objective = float((logsumexp(z, axis=1) - z[np.arange(len(y)), y]).mean() + .5 * np.sum(precision[:, None] * w * w))
    delta = softmax(z, axis=1)
    delta[np.arange(len(y)), y] -= 1
    delta /= len(y)
    grad = precision[:, None] * w
    for idx in ids.values():
        hit = idx >= 0
        for k in range(3):
            grad[:, k] += np.bincount(idx[hit], weights=delta[hit, k], minlength=len(w))
    gmax = float(abs(grad / np.sqrt(precision)[:, None]).max())
    assert abs(objective - bundle['optimizer']['objective']) < 1e-10
    assert abs(gmax - bundle['optimizer']['scaled_gradient']) < 1e-9
    assert bundle['optimizer']['eligible'] == (bundle['optimizer']['success'] and gmax <= 5e-6)
    return objective, gmax

def replay(basepred, candidate, heads, original, r):
    count, gap, unchanged = 0, 0., 0
    for view, fields in v.VIEWS.items():
        x = original.copy()
        x[fields] = v.MISSING
        a = candidate[candidate.scenario == view].reset_index(drop=True)
        b = basepred[basepred.scenario == view].reset_index(drop=True)
        pd.testing.assert_frame_equal(a[r.columns], r.reset_index(drop=True))
        np.testing.assert_array_equal(a.eligible_stress, b.eligible_stress)
        p = b[['p_0', 'p_1', 'p_2']].to_numpy().astype(float)
        pred = b.pred.to_numpy(copy=True)
        used = np.zeros(len(x), bool)
        for name, bundle in heads.items():
            field = 'src_role' if name == 'no_src_role' else 'dst_role'
            mask = head_mask(x, field)
            assert not (used & mask).any()
            used |= mask
            if mask.any():
                q = softmax(add_offset(p[mask], codes(bundle['design'], x.loc[mask]), bundle['weights']), axis=1)
                p[mask] = q
                pred[mask] = q.argmax(axis=1)
        np.testing.assert_array_equal(p[~used], b.loc[~used, ['p_0', 'p_1', 'p_2']])
        np.testing.assert_allclose(p, a[['p_0', 'p_1', 'p_2']], atol=1e-12, rtol=0)
        np.testing.assert_array_equal(pred, a.pred)
        if view == 'full':
            asa = r.route.eq('asa').to_numpy()
            assert not used[asa].any()
            np.testing.assert_array_equal(p[asa], b.loc[asa, ['p_0', 'p_1', 'p_2']])
            unchanged += int(asa.sum())
        count += len(a)
        gap = max(gap, float(abs(p - a[['p_0', 'p_1', 'p_2']].to_numpy()).max()))
    return count, gap, unchanged

def main():
    assert not OUT.exists()
    c, r, x = run.load()
    OUT.mkdir(parents=True)
    fine = np.asarray(independent_keys(x)['fine'])
    asa = r.route.eq('asa').to_numpy()
    all_records, maxp, maxbase, total, unchanged = [], 0., 0., 0, 0
    for protocol in t.PROTOCOLS:
        folder = run.RUN / protocol
        complete = v.read(folder / 'complete.json')
        for p, h in complete['bindings'].items():
            assert v.sha(folder / p) == h, p
        before = v.read(folder / 'before_evaluation.json')
        assert not before['evaluation_generated'] and not any(p.endswith('evaluation.parquet') for p in before['bindings'])
        previous = pd.read_parquet(t.RUN / (protocol + '_split.parquet')).role.to_numpy(copy=True)
        idx = np.flatnonzero((previous != 'evaluation') & asa)
        aa, bb = next(StratifiedGroupKFold(3, shuffle=True, random_state=c['split_seed']).split(idx, r.label_index.to_numpy()[idx], fine[idx]))
        previous[idx[aa]], previous[idx[bb]] = 'fit', 'calibration'
        roles = pd.read_parquet(run.RUN / (protocol + '_split.parquet')).role.to_numpy()
        np.testing.assert_array_equal(roles, previous)
        assert set(fine[asa & (roles == 'fit')]).isdisjoint(fine[asa & (roles == 'calibration')])
        fit = roles == 'fit'
        base = joblib.load(folder / 'base.joblib')
        assert base['model'].t_ == c['epochs'] * fit.sum()
        for j, field in enumerate(v.FIELDS):
            assert base['input']['seen'][j] == set(x.loc[fit, field])
        selection = v.read(folder / 'selection.json')
        heads, objectives = {}, {}
        for name, field in run.HEADS.items():
            xx = x.loc[fit].copy()
            xx[field] = v.MISSING
            active = head_mask(xx, field)
            bundle = joblib.load(folder / (name + '.joblib'))
            assert bundle['base_sha256'] == v.sha(folder / 'base.joblib')
            objective, gradient = row_objective(bundle, base, xx.loc[active], r.loc[fit, 'label_index'].to_numpy()[active], r.loc[fit, 'body_group'].to_numpy()[active], c['strength'])
            inactive_loss = 0.
            if (~active).any():
                pp = manual(base, xx.loc[~active])
                yy = r.loc[fit, 'label_index'].to_numpy()[~active]
                inactive_loss = float(-np.log(np.maximum(pp[np.arange(len(yy)), yy], 1e-30)).sum())
            objectives[name] = {'active_row_mean_penalized_objective': objective, 'scaled_gradient': gradient, 'active_parents': int(active.sum()), 'inactive_constant_loss_parents': int((~active).sum()), 'original_parent_mean_head_objective': (objective * active.sum() + inactive_loss) / fit.sum()}
            heads[name] = bundle
        basecal = pd.read_parquet(folder / 'base_calibration.parquet')
        accepted = {}
        for name, bundle in heads.items():
            a = pd.read_parquet(folder / (name + '_calibration.parquet'))
            gate = independent_gate(basecal, a)
            assert gate == selection['heads'][name]['gate']
            passed = bundle['optimizer']['eligible'] and gate['quality_passed']
            assert passed == selection['heads'][name]['accepted']
            if passed:
                accepted[name] = bundle
            ct, gp, un = replay(basecal, a, {name: bundle}, x.loc[roles == 'calibration'].reset_index(drop=True), r.loc[roles == 'calibration'].reset_index(drop=True))
            total += ct; maxp = max(maxp, gp); unchanged += un
        assert list(accepted) == selection['accepted']
        for role in ['calibration'] + (['evaluation'] if selection['eligible_for_evaluation'] else []):
            mask = roles == role
            baseline = pd.read_parquet(folder / ('base_' + role + '.parquet'))
            candidate = pd.read_parquet(folder / ('selected_calibration.parquet' if role == 'calibration' else 'evaluation.parquet'))
            for view, fields in v.VIEWS.items():
                xx = x.loc[mask].copy(); xx[fields] = v.MISSING
                p = manual(base, xx)
                a = baseline[baseline.scenario == view].reset_index(drop=True)
                np.testing.assert_allclose(p, a[['p_0', 'p_1', 'p_2']], atol=3e-6, rtol=0)
                np.testing.assert_array_equal(p.argmax(axis=1), a.pred)
                maxbase = max(maxbase, float(abs(p - a[['p_0', 'p_1', 'p_2']].to_numpy()).max()))
            ct, gp, un = replay(baseline, candidate, accepted, x.loc[mask].reset_index(drop=True), r.loc[mask].reset_index(drop=True))
            total += ct; maxp = max(maxp, gp); unchanged += un
            gate = independent_gate(baseline, candidate)
            if role == 'calibration':
                assert gate == selection['combined_gate']
            else:
                assert gate == v.read(folder / 'evaluation_comparison.json')
        assert (folder / 'evaluation.parquet').exists() == selection['eligible_for_evaluation']
        fitp = manual(base, x.loc[fit])
        y = r.loc[fit, 'label_index'].to_numpy()
        base_loss = float(-np.log(np.maximum(fitp[np.arange(len(y)), y], 1e-30)).mean())
        all_records.append({'protocol': protocol, 'accepted': list(accepted), 'objectives': objectives, 'original_parent_mean_total_nominal_objective': .5 * base_loss + .25 * sum(a['original_parent_mean_head_objective'] for a in objectives.values()), 'evaluation_executed': selection['eligible_for_evaluation']})
        print(json.dumps({'verified': protocol}), flush=True)
    result = {'all_checks_passed': True, 'records': all_records, 'new_base_fits': 4, 'new_adapter_fits': 8, 'score_rows_replayed': total, 'max_correction_probability_difference': maxp, 'max_manual_base_probability_difference': maxbase, 'full_ASA_probability_values_exactly_unchanged_replays': unchanged, 'scope': 'Independent original-parent mass and branch objective, active-only fit support, behavior/body isolation, base forward pass, branch activation and calibration selection. Inactive probabilities are numerically exact; dataframe dtype or file bytes need not match. Repeated views/protocols are not independent records.'}
    v.save(OUT / 'verification.json', result)
    (OUT / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    v.save(OUT / 'receipt.json', {'files': {p.name: v.sha(p) for p in OUT.iterdir() if p.is_file()}})
    print(json.dumps({k: a for k, a in result.items() if k != 'records'}), flush=True)

if __name__ == '__main__':
    main()
