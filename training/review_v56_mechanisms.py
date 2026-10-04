"""Mechanism controls for v56 follow-up; diagnostic interventions, no training."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import run_v54 as v
import run_v56 as t
import v56_shared_control as s
from verify_v56 import independent_keys
from review_v56_followup import sha, save, counts, errors

ROOT = t.ROOT
OUT = ROOT / 'evidence/2026-09-14/v56_followup_mechanisms'

def frontier(table, budgets):
    """Exact 0/1 knapsack over identical-input classes; verify each witness."""
    a = table.reindex(columns=[1, 2], fill_value=0).to_numpy(dtype=int)
    maxbudget = max(budgets)
    pure = (a[:, 0] == 0) & (a[:, 1] > 0)
    items = np.flatnonzero((a[:, 0] > 0) & (a[:, 0] <= maxbudget) & (a[:, 1] > 0))
    dp = np.full(maxbudget + 1, -10**9, dtype=int)
    dp[0] = 0
    take = np.zeros((len(items), maxbudget + 1), bool)
    for j, i in enumerate(items):
        cost, benefit = a[i]
        previous = dp.copy()
        alternative = previous[:-cost] + benefit
        take[j, cost:] = alternative > previous[cost:]
        dp[cost:] = np.maximum(previous[cost:], alternative)
    out = []
    for budget in budgets:
        spent = int(dp[:budget + 1].argmax())
        current = spent
        chosen = list(np.flatnonzero(pure))
        for j in range(len(items) - 1, -1, -1):
            if take[j, current]:
                chosen.append(int(items[j]))
                current -= a[items[j], 0]
        assert current == 0
        achieved_m = int(a[chosen, 0].sum())
        correct_s = int(a[chosen, 1].sum())
        assert achieved_m == spent <= budget
        assert correct_s == int(a[pure, 1].sum() + dp[spent])
        out.append({'maximum_M_errors': budget, 'witness_M_errors': achieved_m, 'minimum_S_errors': int(a[:, 1].sum() - correct_s), 'witness_S_correct': correct_s, 'selected_observation_count': len(chosen)})
    return out

def main():
    assert not OUT.exists()
    _, r, x = s.load()
    receipt = v.read(ROOT / 'evidence/2026-09-14/v56_followup_review/receipt.json')
    for p, h in receipt['files'].items():
        assert sha(ROOT / 'evidence/2026-09-14/v56_followup_review' / p) == h
    OUT.mkdir(parents=True)
    results = {'scope': 'Post-hoc controls only, no fitted candidate or deployable oracle. Removing existing coefficients is not retraining. Oracle frontier uses evaluation labels exclusively to characterize finite-sample distinguishability, never to fit or select a model.', 'duplicate_and_missing_effects': [], 'calibration_cell_gate': [], 'hardest_port_support': {}}
    for protocol in t.PROTOCOLS:
        roles = pd.read_parquet(t.RUN / (protocol + '_split.parquet')).role.to_numpy()
        fit, ev = roles == 'fit', roles == 'evaluation'
        selected = v.read(s.RUN / protocol / 'selection.json')
        bundle = joblib.load(s.RUN / protocol / (selected['selected'] + '.joblib'))
        # Use independent key generation and explicitly reconstruct contributions.
        trainx = x.loc[fit]
        fields = ['icmp_type', 'icmp_code', 'icmp_message', 'icmp_unreachable']
        eq = trainx[fields].eq(v.MISSING).to_numpy()
        identical = bool(np.all(eq == eq[:, :1]))
        param_indices = [bundle['design']['maps']['field:' + f].get(v.MISSING) for f in fields]
        weights = bundle['weights'][param_indices]
        info = {'protocol': protocol, 'four_missing_indicator_columns_identical_on_fit': identical, 'fit_active_counts': eq.sum(0).tolist(), 'body_supports': bundle['design']['bodies'][param_indices].astype(int).tolist(), 'M_minus_S_effects': (weights[:, 1] - weights[:, 2]).tolist(), 'probes': []}
        b = pd.read_parquet(t.RUN / protocol / 'base_evaluation.parquet')
        a = pd.read_parquet(s.RUN / protocol / 'evaluation.parquet')
        for view in ['full', 'no_src_role', 'no_dst_role']:
            xx = x.loc[ev].reset_index(drop=True).copy()
            xx[v.VIEWS[view]] = v.MISSING
            keys = {**{'field:' + f: xx[f].to_numpy() for f in s.SHARED}, **independent_keys(xx)}
            bb, aa = b[b.scenario == view].reset_index(drop=True), a[a.scenario == view].reset_index(drop=True)
            z = np.log(np.maximum(bb[['p_0', 'p_1', 'p_2']].to_numpy().astype(float), 1e-30))
            pieces = {}
            for level, values in keys.items():
                ids = np.array([bundle['design']['maps'][level].get(k, -1) for k in values])
                hit = ids >= 0
                p = np.zeros_like(z)
                p[hit] = bundle['weights'][ids[hit]]
                pieces[level] = p
                z += p
            np.testing.assert_array_equal(z.argmax(1), aa.pred)
            y = aa.label_index.to_numpy()
            keep = aa.route.eq('asa').to_numpy() & aa.eligible_stress.to_numpy()
            duplicate = sum(pieces['field:' + f] * xx[f].eq(v.MISSING).to_numpy()[:, None] for f in fields)
            src_missing = pieces['field:src_role'] * xx.src_role.eq(v.MISSING).to_numpy()[:, None]
            for name, delta in [('remove_four_ICMP_missing_effects', duplicate), ('remove_four_ICMP_missing_and_source_missing_effects', duplicate + src_missing)]:
                pred = (z - delta).argmax(1)
                regression_s = keep & (y == 2) & (bb.pred.to_numpy() == 2) & (aa.pred.to_numpy() != 2)
                info['probes'].append({'view': view, 'probe': name, 'errors_B_M_S': errors(y, pred, keep), 'new_S_regressions_reverted': int((regression_s & (pred == 2)).sum()), 'ASA_to_normal': int((keep & (pred == 0)).sum())})
        results['duplicate_and_missing_effects'].append(info)
        basecal = pd.read_parquet(t.RUN / protocol / 'base_calibration.parquet')
        for name, candidate in selected['candidates'].items():
            cand = pd.read_parquet(s.RUN / protocol / (name + '_calibration.parquet'))
            failures = []
            for view in v.VIEWS:
                bb = basecal[basecal.scenario == view].reset_index(drop=True)
                aa = cand[cand.scenario == view].reset_index(drop=True)
                for klass in [1, 2]:
                    mask = aa.route.eq('asa') & aa.eligible_stress & aa.label_index.eq(klass)
                    olderr = int((bb.loc[mask, 'pred'] != klass).sum())
                    newerr = int((aa.loc[mask, 'pred'] != klass).sum())
                    if newerr > olderr:
                        failures.append({'view': view, 'class': klass, 'increase': newerr - olderr, 'base': olderr, 'new': newerr})
            results['calibration_cell_gate'].append({'protocol': protocol, 'candidate': name, 'numerically_eligible': candidate['eligible'], 'selected_v56': name == selected['selected'], 'all_six_views_class_nonincrease': not failures, 'failures': failures})
        if protocol == 'behavior_2':
            target = (x.transport_protocol.eq('tcp') & x.src_role.eq('outside') & x.dst_role.eq('dmz') & x.src_port_fixed.eq(v.MISSING) & x.dst_port_fixed.ne(v.MISSING)).to_numpy()
            targetev = target & ev & r.route.eq('asa').to_numpy()
            assert targetev.sum() == 17058
            targetx, targetr = x.loc[targetev], r.loc[targetev]
            frame = pd.DataFrame({'key': independent_keys(targetx)['exact'], 'y': targetr.label_index.to_numpy()})
            tab = pd.crosstab(frame.key, frame.y)
            results['hardest_frontier'] = frontier(tab, [0, 48, 216, 328])
            # Can the remaining original fit records provide per-destination-port evidence?
            for scope, available in [('same_TCP_direction', x.transport_protocol.eq('tcp') & x.src_role.eq('outside') & x.dst_role.eq('dmz')), ('all_TCP', x.transport_protocol.eq('tcp')), ('all_ASA', r.route.eq('asa'))]:
                rr = r.loc[fit & available.to_numpy()].copy()
                rr['port'] = x.loc[rr.index, 'dst_port_fixed']
                pt = pd.crosstab(rr.port, rr.label_index).reindex(columns=[0, 1, 2], fill_value=0)
                matched = pt.reindex(targetx.dst_port_fixed, fill_value=0)
                strata = np.where(matched.sum(axis=1).to_numpy() == 0, 'port_unseen', np.where(matched[2].to_numpy() == 0, 'seen_without_S', 'seen_with_S'))
                results['hardest_port_support'][scope] = {key: {'evaluation_labels_B_M_S': counts(targetr.loc[strata == key, 'label_index']), 'unique_destination_ports': int(targetx.loc[strata == key, 'dst_port_fixed'].nunique())} for key in sorted(set(strata))}
            maskable = fit & (x.transport_protocol.eq('tcp') & x.src_role.eq('outside') & x.dst_role.eq('dmz') & x.src_port_fixed.ne(v.MISSING) & x.dst_port_fixed.ne(v.MISSING)).to_numpy()
            results['hardest_masking_analogue'] = {'fit_labels_B_M_S': counts(r.loc[maskable, 'label_index']), 'fit_body_groups': int(r.loc[maskable, 'body_group'].nunique()), 'note': 'Deleting source-port fields makes the coarse observed pattern match. It does not prove the natural missing subset has the same conditional label distribution.'}
    save(OUT / 'controls.json', results)
    (OUT / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    save(OUT / 'receipt.json', {'files': {p.name: sha(p) for p in OUT.iterdir() if p.is_file()}})
    print(json.dumps(results), flush=True)

if __name__ == '__main__':
    main()
