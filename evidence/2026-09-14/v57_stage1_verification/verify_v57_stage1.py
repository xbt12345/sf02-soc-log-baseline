"""Independent row-level replay of the fixed applicability experiment."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.special import softmax, logsumexp
import run_v54 as v
import run_v56 as t
import run_v57_stage1 as run
from verify_v55 import manual
from verify_v56 import independent_keys

OUT = t.ROOT / 'evidence/2026-09-14/v57_stage1_verification'
FIELDS = ['transport_protocol', 'src_role', 'dst_role', 'src_port_range', 'dst_port_range', 'icmp_type', 'icmp_code', 'icmp_message', 'icmp_unreachable']

def independent_field_keys(x):
    out = {'field:' + f: [] for f in FIELDS}
    for row in x.to_dict('records'):
        for f in FIELDS:
            valid = row[f] not in [v.MISSING, v.UNKNOWN]
            if f in ['src_port_range', 'dst_port_range']:
                valid = valid and row['transport_protocol'] in ['tcp', 'udp']
            if f.startswith('icmp_'):
                valid = valid and row['transport_protocol'] == 'icmp'
            out['field:' + f].append(row[f] if valid else None)
    return {**out, **independent_keys(x)}

def codes(design, x):
    return {level: np.array([design['maps'][level].get(k, -1) for k in values]) for level, values in independent_field_keys(x).items()}

def add_offset(basep, indices, w):
    z = np.log(np.maximum(basep.astype(float), 1e-30))
    for ids in indices.values():
        hit = ids >= 0
        z[hit] += w[ids[hit]]
    return z

def independent_gate(base, candidate):
    cells, improved, failures = [], False, []
    for view in v.VIEWS:
        b = base[base.scenario == view].reset_index(drop=True)
        a = candidate[candidate.scenario == view].reset_index(drop=True)
        pd.testing.assert_frame_equal(a[['row_position', 'label_index', 'route', 'eligible_stress']], b[['row_position', 'label_index', 'route', 'eligible_stress']])
        for name, klass in [('M', 1), ('S', 2), ('normal', 0), ('ASA_to_normal', None)]:
            mask = a.route.eq('asa_acl') if name == 'normal' else a.route.eq('asa') & a.eligible_stress
            if klass in [1, 2]:
                mask &= a.label_index.eq(klass)
            if name == 'ASA_to_normal':
                before = int(b.loc[mask, 'pred'].eq(0).sum())
                after = int(a.loc[mask, 'pred'].eq(0).sum())
            else:
                before = int(b.loc[mask, 'pred'].ne(b.loc[mask, 'label_index']).sum())
                after = int(a.loc[mask, 'pred'].ne(a.loc[mask, 'label_index']).sum())
            cell = {'view': view, 'class': name, 'support': int(mask.sum()), 'base_errors': before, 'candidate_errors': after, 'delta': after - before}
            cells.append(cell)
            if not mask.any() or after > before:
                failures.append(cell)
            improved |= name in ['M', 'S'] and after < before
    return {'cells': cells, 'failures': failures, 'all_cells_nonincrease': not failures, 'any_M_S_improvement': bool(improved), 'quality_passed': bool(not failures and improved)}

def main():
    assert not OUT.exists()
    c, rows, x = run.load()
    OUT.mkdir(parents=True)
    records, maxp, maxloss, replay = [], 0., 0., 0
    for protocol in t.PROTOCOLS:
        folder = run.RUN / protocol
        complete = v.read(folder / 'complete.json')
        for p, h in complete['bindings'].items():
            assert v.sha(folder / p) == h, p
        roles = pd.read_parquet(t.RUN / (protocol + '_split.parquet')).role
        fit = roles.eq('fit').to_numpy()
        for aa, bb in [('fit', 'calibration'), ('fit', 'evaluation'), ('calibration', 'evaluation')]:
            assert set(rows.loc[roles.eq(aa), 'body_group']).isdisjoint(rows.loc[roles.eq(bb), 'body_group'])
        bundle = joblib.load(folder / 'model.joblib')
        base = joblib.load(t.ROOT / bundle['base_source'])
        assert v.sha(t.ROOT / bundle['base_source']) == bundle['base_sha256']
        maps, counts, supports, names = {}, [], [], []
        for level, values in independent_field_keys(x.loc[fit]).items():
            tmp = pd.DataFrame({'key': values, 'body': rows.loc[fit, 'body_group'].to_numpy()}).dropna(subset=['key'])
            grouped = tmp.groupby('key', sort=True).agg(n=('body', 'size'), b=('body', 'nunique'))
            maps[level] = {value: len(names) + i for i, value in enumerate(grouped.index)}
            names.extend([[level, value] for value in grouped.index])
            counts.extend(grouped.n)
            supports.extend(grouped.b)
        design = bundle['design']
        assert maps == design['maps'] and names == design['feature_keys']
        np.testing.assert_array_equal(counts, design['rows'])
        np.testing.assert_array_equal(supports, design['bodies'])
        indices = codes(design, x.loc[fit])
        z = add_offset(manual(base, x.loc[fit]), indices, bundle['weights'])
        y = rows.loc[fit, 'label_index'].to_numpy()
        precision = c['strengths'][protocol] * np.array(counts) / fit.sum() / np.sqrt(supports)
        w = bundle['weights']
        loss = float((logsumexp(z, axis=1) - z[np.arange(len(y)), y]).mean() + .5 * np.sum(precision[:, None] * w * w))
        error = softmax(z, axis=1)
        error[np.arange(len(y)), y] -= 1
        error /= len(y)
        grad = precision[:, None] * w
        for ids in indices.values():
            hit = ids >= 0
            for klass in range(3):
                grad[:, klass] += np.bincount(ids[hit], weights=error[hit, klass], minlength=len(w))
        gmax = float(abs(grad / np.sqrt(precision)[:, None]).max())
        gap = abs(loss - bundle['optimizer']['objective'])
        maxloss = max(maxloss, gap)
        assert gap < 1e-10 and abs(gmax - bundle['optimizer']['scaled_gradient']) < 1e-9
        assert bundle['optimizer']['eligible'] == (bundle['optimizer']['success'] and gmax <= 5e-6)
        selection = v.read(folder / 'selection.json')
        cal = pd.read_parquet(folder / 'calibration.parquet')
        basecal = pd.read_parquet(t.RUN / protocol / 'base_calibration.parquet')
        gate = independent_gate(basecal, cal)
        assert gate == selection['against_base']
        selected = 'applicable' if bundle['optimizer']['eligible'] and gate['quality_passed'] else 'zero'
        assert selected == selection['selected'] == complete['selected']
        assert (folder / 'evaluation.parquet').exists() == (selected == 'applicable')
        before = v.read(folder / 'before_evaluation.json')
        assert before['evaluation_generated'] is False
        assert all('evaluation.parquet' not in p for p in before['bindings'])
        for role in ['calibration'] + (['evaluation'] if selected == 'applicable' else []):
            mask = roles.eq(role).to_numpy()
            basepred = pd.read_parquet(t.RUN / protocol / ('base_' + role + '.parquet'))
            pred = pd.read_parquet(folder / (role + '.parquet'))
            for view, fields in v.VIEWS.items():
                xx = x.loc[mask].copy()
                xx[fields] = v.MISSING
                b = basepred[basepred.scenario == view].reset_index(drop=True)
                a = pred[pred.scenario == view].reset_index(drop=True)
                pd.testing.assert_frame_equal(a[rows.columns], rows.loc[mask].reset_index(drop=True))
                p = softmax(add_offset(b[['p_0', 'p_1', 'p_2']].to_numpy(), codes(design, xx), w), axis=1)
                maxp = max(maxp, float(abs(p - a[['p_0', 'p_1', 'p_2']].to_numpy()).max()))
                np.testing.assert_allclose(p, a[['p_0', 'p_1', 'p_2']], atol=1e-12, rtol=0)
                np.testing.assert_array_equal(p.argmax(axis=1), a.pred)
                replay += len(a)
        records.append({'protocol': protocol, 'selected': selected, 'original_rows_objective': loss, 'scaled_gradient': gmax, 'calibration_cells': gate['cells'], 'evaluation_executed': selected == 'applicable'})
    result = {'all_checks_passed': True, 'new_fits': 4, 'records': records, 'score_rows_replayed': replay, 'max_probability_difference': maxp, 'max_original_row_objective_difference': maxloss, 'scope': 'Independent original-row objective/gradient, fit support and masks, saved outputs, zero-candidate gate. Rejected candidates have no new evaluation file. Passing numeric checks is not passing quality.'}
    v.save(OUT / 'verification.json', result)
    (OUT / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    v.save(OUT / 'receipt.json', {'files': {p.name: v.sha(p) for p in OUT.iterdir() if p.is_file()}})
    print(json.dumps({k: a for k, a in result.items() if k != 'records'}), flush=True)

if __name__ == '__main__':
    main()
