"""Account for every fixed and regressed OOF decision, including small tails."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from run_v39_prepare import sha, save


def main(a):
    root, run = Path(a.root), Path(a.run)
    out = run / 'primary_review/all_regressions.json'
    assert not out.exists()
    result = {}
    labels = np.array(['benign', 'malicious', 'suspicious'])
    for view in ['R', 'N', 'I']:
        changes = []
        for fold in range(3):
            baseline = pq.read_table(root / ('artifacts/v39_local_r2_20260913/primary/fold_%s/SEMANTIC/evaluation.parquet' % fold),
                columns=['row_position', 'p_benign', 'p_malicious', 'p_suspicious']).to_pandas()
            new = pq.read_table(run / ('primary/fold_%s/%s/evaluation.parquet' % (fold, view))).to_pandas()
            assert np.array_equal(baseline.row_position, new.row_position)
            b = baseline[['p_benign', 'p_malicious', 'p_suspicious']].to_numpy().argmax(1)
            p = new[['p_benign', 'p_malicious', 'p_suspicious']].to_numpy().argmax(1)
            y = new.label_index.to_numpy()
            affected = (b == y) != (p == y)
            changed = new.loc[affected, ['row_position', 'route', 'product', 'body_group', 'projection_id', 'observation_mask']].copy()
            changed['class'] = labels[y[affected]]
            changed['old_prediction'] = labels[b[affected]]
            changed['new_prediction'] = labels[p[affected]]
            changed['change'] = np.where(p[affected] == y[affected], 'fixed', 'regressed')
            changed['fold'] = fold
            changes.append(changed)
        changes = pd.concat(changes, ignore_index=True)
        grouped = changes.groupby(['route', 'product', 'class', 'change'], dropna=False).agg(rows=('row_position', 'size'),
            distinct_body_keys=('body_group', 'nunique')).reset_index().to_dict('records')
        result[view] = {'summary': grouped, 'fixed': int((changes.change == 'fixed').sum()),
            'regressed': int((changes.change == 'regressed').sum()),
            'non_ASA_regressed_records': changes.loc[(changes.change == 'regressed') & (changes.route != 'asa')].to_dict('records')}
        print(json.dumps({'view': view, 'fixed': result[view]['fixed'], 'regressed': result[view]['regressed'],
            'outside_ASA': [r for r in grouped if r['route'] != 'asa' and r['change'] == 'regressed']}), flush=True)
    save(out, {'views': result, 'script_sha256': sha(__file__), 'fitted_models': 0,
        'scope': 'Exhaustive same-row correctness changes against B; different wrong classes are not counted as fixes or regressions'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); p.add_argument('--run', required=True); main(p.parse_args())
