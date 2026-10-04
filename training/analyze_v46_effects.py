"""No fits or threshold search: body-level sensitivity and exact changed cases."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd


def main(root, out):
    assert not (out/'group_sensitivity.json').exists()
    spec = importlib.util.spec_from_file_location('v46', out/'probe_v46_interaction.py')
    v = importlib.util.module_from_spec(spec); spec.loader.exec_module(v); _, m = v.helpers(out)
    r, x, _, _ = m.data(root); e = pd.read_parquet(out/'evaluation.parquet'); row_index = pd.Index(r.row_position)
    summaries = {}; changes = []; fit_support = {}
    for fold in range(3):
        tr = r.fold.to_numpy() != fold
        for view in v.VIEWS:
            fit = pd.DataFrame({'key': v.key_array(x.loc[tr], view),
                'label': r.loc[tr, 'label_index'].to_numpy(), 'body': r.loc[tr, 'body_group'].to_numpy()})
            counts = pd.crosstab(fit.key, fit.label).reindex(columns=[1, 2], fill_value=0)
            bodies = fit.groupby('key').body.nunique()
            fit_support[(fold, view)] = (counts, bodies)
    for scenario, cols in v.SCENARIOS.items():
        d = e[(e.scenario == scenario)&e.eligible_stress_row].copy()
        y = d.label_index.to_numpy(); wr = np.where(d.R_qM >= .5, 1, 2) != y; wi = np.where(d.I_qM >= .5, 1, 2) != y
        d['R_wrong'] = wr; d['I_wrong'] = wi; d['net'] = wr.astype(int)-wi.astype(int)
        g = d.groupby('body_group').agg(rows=('row_position', 'size'), net=('net', 'sum'),
            R_error_rate=('R_wrong', 'mean'), I_error_rate=('I_wrong', 'mean'), fold=('fold', 'first'))
        top = g.nlargest(1, 'net'); net = int(g.net.sum()); largest = int(top.net.sum())
        summaries[scenario] = {'rows': len(d), 'body_groups': len(g), 'row_net_improvement': net,
            'improved_body_groups': int((g.net > 0).sum()), 'worsened_body_groups': int((g.net < 0).sum()),
            'unchanged_net_body_groups': int((g.net == 0).sum()),
            'equal_body_mean_error_R': float(g.R_error_rate.mean()), 'equal_body_mean_error_I': float(g.I_error_rate.mean()),
            'largest_improvement_body': top.reset_index().to_dict(orient='records'),
            'net_without_largest_improvement_body': net-largest,
            'largest_body_gain_divided_by_net_gain': largest/net if net else None}
        xx = x.copy(); xx[cols] = v.MISSING
        for (fold, body), z in d[d.net != 0].groupby(['fold', 'body_group']):
            ix = row_index.get_loc(z.row_position.iloc[0]); view = z.reduced_view.iloc[0]
            key = v.key_array(xx.iloc[[ix]], view)[0]
            table, bodies = fit_support[(fold, view)]
            counts = table.loc[key].astype(int).tolist() if key in table.index else [0, 0]
            changes.append({'scenario': scenario, 'fold': int(fold), 'body_group': int(body), 'rows': len(z), 'net': int(z.net.sum()),
                'evaluation_label_counts_M_S': [int((z.label_index == k).sum()) for k in [1, 2]],
                'view': view, 'R_qM': float(z.R_qM.iloc[0]), 'I_qM': float(z.I_qM.iloc[0]),
                'fit_same_input_counts_M_S': counts, 'fit_same_input_bodies': int(bodies.get(key, 0)),
                'remaining_facts': json.loads(key), 'representative_row_position': int(z.row_position.iloc[0])})
    # Algebraic counterexample to the proposal 'a single temperature fixes decisions
    # and binary confidence ordering': positive T preserves both margin sign and |margin| order.
    margin = np.array([-9., -2., -.01, 0., .03, 1.7, 5.]); tt = []
    for temp in [.5, 2., 10.]:
        assert np.array_equal(margin >= 0, margin/temp >= 0)
        assert np.array_equal(np.argsort(abs(margin)), np.argsort(abs(margin/temp)))
        tt.append({'positive_temperature': temp, 'decision_changed': False, 'absolute_margin_order_changed': False})
    shutil.copyfile(__file__, out/Path(__file__).name)
    m.save(out/'group_sensitivity.json', {'scenarios': summaries, 'changed_groups': changes,
        'temperature_counterexample': tt, 'new_fits': 0, 'thresholds_selected': False,
        'scope': 'Exploratory body-level sensitivity, not resampling confidence intervals, training reweighting or exclusion of hard evaluation rows.'})
    print(json.dumps({'summary': summaries, 'changed_groups': len(changes)}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); p.add_argument('--out', required=True)
    a = p.parse_args(); main(Path(a.root).resolve(), Path(a.out).resolve())
