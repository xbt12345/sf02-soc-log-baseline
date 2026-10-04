"""Post-hoc attribution and support review of frozen v56. No fit or selection."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import joblib
from scipy.special import softmax
import run_v54 as v
import run_v56 as t
import v56_shared_control as s
from verify_v56 import independent_keys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evidence/2026-09-14/v56_followup_review'

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def counts(y):
    return np.bincount(np.asarray(y, dtype=int), minlength=3).tolist()

def errors(y, pred, mask):
    return [int(((y == k) & (pred != k) & mask).sum()) for k in range(3)]

def main():
    assert not OUT.exists(), 'Do not overwrite completed evidence'
    delivery = json.loads((ROOT / 'evidence/2026-09-14/v56_delivery/delivery.json').read_text())
    for name, digest in delivery['bindings'].items():
        assert sha(ROOT / name) == digest, name
    _, r, x = s.load()
    OUT.mkdir(parents=True)
    all_cells, targets, hardest, source_bindings = [], [], [], {}
    max_replay_gap = 0.
    target_pairs = {('body', 'no_src_role'), ('behavior_0', 'full'), ('behavior_1', 'no_src_role'), ('behavior_2', 'full')}
    for protocol in t.PROTOCOLS:
        folder = s.RUN / protocol
        selection = json.loads((folder / 'selection.json').read_text())
        model_path = folder / (selection['selected'] + '.joblib')
        bundle = joblib.load(model_path)
        roles = pd.read_parquet(t.RUN / (protocol + '_split.parquet')).role.to_numpy()
        masks = {role: roles == role for role in ['fit', 'calibration', 'evaluation']}
        w, design = bundle['weights'], bundle['design']
        input_counts = {field: {role: counts(r.loc[m & x[field].eq(v.MISSING).to_numpy(), 'label_index']) for role, m in masks.items()} for field in ['src_role', 'dst_role']}
        for role in ['calibration', 'evaluation']:
            old_path = t.RUN / protocol / ('base_' + role + '.parquet')
            new_path = folder / ((selection['selected'] + '_calibration.parquet') if role == 'calibration' else 'evaluation.parquet')
            old, new = pd.read_parquet(old_path), pd.read_parquet(new_path)
            source_bindings.update({p.relative_to(ROOT).as_posix(): sha(p) for p in [old_path, new_path, model_path]})
            rr, xxfull = r.loc[masks[role]].reset_index(drop=True), x.loc[masks[role]].reset_index(drop=True)
            for view, dropped in v.VIEWS.items():
                b = old[old.scenario == view].reset_index(drop=True)
                a = new[new.scenario == view].reset_index(drop=True)
                pd.testing.assert_frame_equal(a[r.columns], rr)
                pd.testing.assert_frame_equal(b[r.columns], rr)
                y, bp, ap = a.label_index.to_numpy(), b.pred.to_numpy(), a.pred.to_numpy()
                keep = a.route.eq('asa').to_numpy() & a.eligible_stress.to_numpy()
                repair, regress = (bp != y) & (ap == y) & keep, (bp == y) & (ap != y) & keep
                all_cells.append({'protocol': protocol, 'role': role, 'view': view, 'support_B_M_S': counts(y[keep]), 'base_errors_B_M_S': errors(y, bp, keep), 'shared_errors_B_M_S': errors(y, ap, keep), 'repairs_B_M_S': counts(y[repair]), 'regressions_B_M_S': counts(y[regress])})
                if role != 'evaluation' or (protocol, view) not in target_pairs:
                    continue
                xx = xxfull.copy()
                xx[dropped] = v.MISSING
                keys = {**{'field:' + f: xx[f].to_numpy() for f in s.SHARED}, **independent_keys(xx)}
                indices = {level: np.array([design['maps'][level].get(k, -1) for k in values]) for level, values in keys.items()}
                pieces = {}
                z = np.log(np.maximum(b[['p_0', 'p_1', 'p_2']].to_numpy().astype(float), 1e-30))
                for level, ids in indices.items():
                    piece = np.zeros_like(z)
                    hit = ids >= 0
                    piece[hit] = w[ids[hit]]
                    pieces[level] = piece
                    z += piece
                p = softmax(z, axis=1)
                max_replay_gap = max(max_replay_gap, float(abs(p - a[['p_0', 'p_1', 'p_2']].to_numpy()).max()))
                np.testing.assert_allclose(p, a[['p_0', 'p_1', 'p_2']], atol=1e-12, rtol=0)
                np.testing.assert_array_equal(p.argmax(1), ap)
                regress_s = regress & (y == 2)
                factors = []
                for level, piece in pieces.items():
                    knock = (z - piece).argmax(1)
                    delta = piece[:, 1] - piece[:, 2]
                    factors.append({'level': level, 'mean_M_minus_S_on_new_S_errors': float(delta[regress_s].mean()) if regress_s.any() else None, 'S_regressions_reverted_by_removal': int((regress_s & (knock == 2)).sum()), 'all_errors_after_removal_B_M_S': errors(y, knock, keep)})
                fine = np.asarray(independent_keys(xxfull)['fine'])
                detail = pd.DataFrame({'fine': fine, 'y': y, 'base': bp, 'new': ap, 'repair': repair, 'regress': regress, 'body': rr.body_group})
                grouped = []
                for key, g in detail.loc[keep].groupby('fine'):
                    grouped.append({'fine': key, 'rows': len(g), 'bodies': g.body.nunique(), 'labels_B_M_S': counts(g.y), 'base_errors_B_M_S': errors(g.y.to_numpy(), g.base.to_numpy(), np.ones(len(g), bool)), 'new_errors_B_M_S': errors(g.y.to_numpy(), g.new.to_numpy(), np.ones(len(g), bool)), 'repairs_B_M_S': counts(g.loc[g.repair, 'y']), 'regressions_B_M_S': counts(g.loc[g.regress, 'y'])})
                grouped.sort(key=lambda g: (g['regressions_B_M_S'][2], sum(g['new_errors_B_M_S'])), reverse=True)
                parameter_records = []
                for field in s.SHARED:
                    level = 'field:' + field
                    for value in sorted(set(xx.loc[regress_s, field])):
                        idx = design['maps'][level].get(value)
                        fit_sel = masks['fit'] & x[field].eq(value).to_numpy()
                        parameter_records.append({'field': field, 'value': value, 'new_S_errors_with_value': int((regress_s & xx[field].eq(value).to_numpy()).sum()), 'fit_labels_B_M_S': counts(r.loc[fit_sel, 'label_index']), 'fit_body_groups': int(r.loc[fit_sel, 'body_group'].nunique()), 'M_minus_S_effect': float(w[idx, 1] - w[idx, 2]) if idx is not None else 0.})
                targets.append({'protocol': protocol, 'view': view, 'fit_missing_role_counts': input_counts, 'groups': grouped, 'level_removal': factors, 'parameters_on_new_S_errors': parameter_records})
                changed = keep & (bp != ap)
                record = rr.loc[changed, ['row_position', 'label_index', 'body_group']].copy()
                record['base_pred'] = bp[changed]
                record['shared_pred'] = ap[changed]
                record['base_M_minus_S'] = np.log(np.maximum(b.p_1.to_numpy()[changed], 1e-30)) - np.log(np.maximum(b.p_2.to_numpy()[changed], 1e-30))
                for level, piece in pieces.items():
                    record[level] = (piece[:, 1] - piece[:, 2])[changed]
                record.to_parquet(OUT / (protocol + '_' + view + '_changes.parquet'), index=False)
                if protocol == 'behavior_2':
                    fullkeys = np.asarray(independent_keys(xxfull)['exact'])
                    for key in sorted(set(fine[keep & (y == 2)])):
                        m = keep & (fine == key)
                        d = pd.DataFrame({'key': fullkeys[m], 'y': y[m]})
                        tab = pd.crosstab(d.key, d.y).reindex(columns=[0, 1, 2], fill_value=0)
                        majority = tab.to_numpy().argmax(1)
                        oracle_pred = np.array([majority[tab.index.get_loc(k)] for k in d.key])
                        spec = dict(zip(v.FIELDS, xxfull.loc[np.flatnonzero(m)[0], v.FIELDS]))
                        supports = {}
                        for mode, fields in {'same_ICMP': ['transport_protocol', 'icmp_type', 'icmp_code'], 'same_direction': ['transport_protocol', 'src_role', 'dst_role'], 'same_ICMP_without_src_role': ['transport_protocol', 'dst_role', 'icmp_type', 'icmp_code'], 'same_ICMP_without_dst_role': ['transport_protocol', 'src_role', 'icmp_type', 'icmp_code']}.items():
                            match = np.ones(len(x), bool)
                            for f in fields:
                                match &= x[f].eq(spec[f]).to_numpy()
                            supports[mode] = {role2: {'labels_B_M_S': counts(r.loc[mask & match, 'label_index']), 'bodies': int(r.loc[mask & match, 'body_group'].nunique())} for role2, mask in masks.items()}
                        hardest.append({'fine': key, 'labels_B_M_S': counts(y[m]), 'base_errors_B_M_S': errors(y, bp, m), 'shared_errors_B_M_S': errors(y, ap, m), 'bodies': int(rr.loc[m, 'body_group'].nunique()), 'unique_full_inputs': len(tab), 'mixed_full_inputs': int(((tab > 0).sum(axis=1) > 1).sum()), 'empirical_min_total_errors': int((tab.sum(axis=1) - tab.max(axis=1)).sum()), 'total_error_minimizer_errors_B_M_S': errors(d.y.to_numpy(), oracle_pred, np.ones(len(d), bool)), 'first_example_facts': spec, 'fit_and_calibration_support': supports})
        print(json.dumps({'reviewed': protocol}), flush=True)
    for name, digest in delivery['bindings'].items():
        assert sha(ROOT / name) == digest, name
    result = {'scope': 'Post-hoc review, no new fit, no threshold selection, no original pressure evaluation. Level removals are inference interventions, not refits or independent contributions. Marginal label support is not causal evidence. Empirical majority floor is conditional on this finite evaluation set and total-error loss.', 'frozen_files_verified_before_and_after': len(delivery['bindings']), 'max_replayed_probability_difference': max_replay_gap, 'cells': all_cells, 'targets': targets, 'hardest_behavior_2': hardest, 'source_bindings': source_bindings}
    save(OUT / 'diagnosis.json', result)
    (OUT / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    save(OUT / 'receipt.json', {'files': {p.name: sha(p) for p in OUT.iterdir() if p.is_file()}})
    print(json.dumps({'finished': True, 'cells': len(all_cells), 'max_replay_gap': max_replay_gap, 'frozen_files_unchanged': len(delivery['bindings'])}), flush=True)

if __name__ == '__main__':
    main()
