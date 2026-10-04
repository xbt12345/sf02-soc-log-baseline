"""Independent readout of completed phase-A models; no fitting or selection."""
import json
import numpy as np
import pandas as pd

from run_v75 import ROOT, OUT, read, save, sha
from v89_common import LAST
from v104_phase_a import DEST, FOLDS, class_metrics


def main():
    if (DEST / 'phase_A_extended_diagnosis.json').exists():
        raise FileExistsError(DEST / 'phase_A_extended_diagnosis.json')
    a = pd.read_parquet(DEST / 'phase_A_ASA_OOF_ledger.parquet')
    f = pd.read_parquet(FOLDS, columns=['proposed_fold'])
    r = pd.read_parquet(OUT / 'rows.parquet', columns=['route', 'label_index'])
    fid = np.load(LAST / 'row_feature_id.npy', mmap_mode='r')
    folds = f.proposed_fold.to_numpy()
    truth = r.label_index.to_numpy()
    assert len(folds) == len(truth) == len(fid) == 2056871
    all_preds = {}
    for population in ('small', 'full'):
        p = np.empty(len(fid), dtype=np.int8)
        for k in range(3):
            take = folds == k
            score = np.load(DEST / f'fold{k}_{population}' / 'scores_all_input_ids.npy', mmap_mode='r')
            p[take] = score[fid[take]].argmax(1).astype(np.int8)
        assert np.array_equal(p[r.route.eq('asa').to_numpy()], a[f'{population}_pred'].to_numpy())
        all_preds[population] = p
    result = {'status': 'extended_phase_A_readout_no_fit', 'classifier_fits_added': 0,
              'full_task_OOF': {}, 'route_class': [], 'ASA': {},
              'prediction_files': {}}
    for population, p in all_preds.items():
        np.save(DEST / f'{population}_all_original_OOF_pred.npy', p)
        result['prediction_files'][population] = sha(DEST / f'{population}_all_original_OOF_pred.npy')
        result['full_task_OOF'][population] = class_metrics(truth, p)
        for (route, cl), index in r.groupby(['route', 'label_index']).indices.items():
            ii = np.asarray(index)
            result['route_class'].append({'population': population, 'route': route,
                'class': int(cl), 'rows': int(len(ii)),
                'correct': int((p[ii] == cl).sum()),
                'recall': float((p[ii] == cl).mean()),
                'predicted': np.bincount(p[ii], minlength=3).astype(int).tolist()})
    asa_truth = a.truth.to_numpy()
    large = a[a.truth == 2].groupby('root').size().idxmax()
    no_giant = a.root.to_numpy() != large
    result['ASA']['giant_S_root'] = int(large)
    result['ASA']['giant_S_rows'] = int(((a.root == large) & (a.truth == 2)).sum())
    for pop in ('small', 'full'):
        pred = a[f'{pop}_pred'].to_numpy()
        result['ASA'][f'{pop}_excluding_giant_root'] = class_metrics(asa_truth[no_giant], pred[no_giant])
    target = pd.read_parquet(ROOT / 'artifacts/v103_plan_preflight_20260928/113_target_support_after_split.parquet',
                             columns=['row_position'])
    t = a[a.row_position.isin(target.row_position)]
    assert len(t) == 113
    result['ASA']['historic_113_by_fold'] = {
        str(k): {'rows': int(len(g)),
                 'small_correct': int((g.small_pred == g.truth).sum()),
                 'full_correct': int((g.full_pred == g.truth).sum()),
                 'small_S_minus_M_median': float((g.small_S_score - g.small_M_score).median()),
                 'full_S_minus_M_median': float((g.full_S_score - g.full_M_score).median()),
                 'full_positive_S_minus_M': int(((g.full_S_score - g.full_M_score) > 0).sum())}
        for k, g in t.groupby('fold')}
    result['source_sha256'] = sha(__file__)
    result['input_sha256'] = {str(path.relative_to(ROOT)): sha(path) for path in
        [DEST / 'phase_A_evaluation.json', DEST / 'phase_A_ASA_OOF_ledger.parquet',
         FOLDS, OUT / 'rows.parquet', LAST / 'row_feature_id.npy']}
    save(DEST / 'phase_A_extended_diagnosis.json', result)
    print(json.dumps({'stage': 'extended_diagnosis',
        'full_task_errors': {x: result['full_task_OOF'][x]['errors'] for x in ('small', 'full')},
        'without_giant_ASA_errors': {x: result['ASA'][f'{x}_excluding_giant_root']['errors']
                                     for x in ('small', 'full')},
        'historic_113_by_fold': result['ASA']['historic_113_by_fold']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
