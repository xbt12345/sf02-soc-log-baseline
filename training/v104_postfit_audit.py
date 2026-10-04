"""Post-hoc error mechanism audit; never selects or changes a model."""
import json
import numpy as np
import pandas as pd
from run_v75 import ROOT, read, save, sha
from v104_phase_b import DEST


def main():
    target = DEST / 'postfit_error_mechanism.json'
    if target.exists():
        raise FileExistsError(target)
    report = read(DEST / 'phase_B_evaluation.json')
    assert report['selected'] is None and report['model_promoted'] is False
    a = pd.read_parquet(DEST / 'phase_B_ASA_OOF_ledger.parquet')
    y = a.truth.to_numpy()
    old = a.N1_teacher.to_numpy()
    new = a.C_TabM_epoch25.to_numpy()
    root = a.root.to_numpy()
    new_m_wrong = (y == 1) & (old == y) & (new != y)
    fixed_s = (y == 2) & (old != y) & (new == y)
    counts = {}
    for name, mask in [('new_M_errors', new_m_wrong), ('repaired_S_errors', fixed_s)]:
        g = a.loc[mask].groupby('root').size().sort_values(ascending=False)
        counts[name] = {'rows': int(mask.sum()), 'roots': len(g),
                        'top1_rows': int(g.iloc[0]),
                        'top5_rows': int(g.head(5).sum()),
                        'top_root': int(g.index[0]),
                        'top_root_fold': int(a.loc[a.root == g.index[0], 'fold'].iat[0])}
    giant = a[a.truth == 2].groupby('root').size().idxmax()
    counts['repaired_S_errors']['on_largest_S_root'] = int((fixed_s & (root == giant)).sum())
    counts['repaired_S_errors']['outside_largest_S_root'] = int((fixed_s & (root != giant)).sum())
    # An intentionally optimistic *in-sample* boundary diagnostic. Labels
    # on the same OOF evaluation rows determine this threshold, so the
    # resulting rate is not a valid calibrated score or a deployed candidate.
    margin = (a.C_TabM_epoch25_S_prob - a.C_TabM_epoch25_M_prob).to_numpy()
    baseline_m_wrong = int(((y == 1) & (old != y)).sum())
    order = np.argsort(margin, kind='stable')
    values = margin[order]
    labels = y[order]
    distinct, first = np.unique(values, return_index=True)
    # Count all rows sharing the same score as one threshold unit.
    last = np.r_[first[1:] - 1, len(values) - 1]
    m_below = np.cumsum(labels == 1)[last]
    s_below = np.cumsum(labels == 2)[last]
    m_above = int((y == 1).sum()) - m_below
    s_above = int((y == 2).sum()) - s_below
    eligible = np.flatnonzero(m_above <= baseline_m_wrong)
    boundary = None
    if len(eligible):
        j = int(eligible[0])
        boundary = {'threshold_selected_with_evaluation_labels': float(distinct[j]),
                    'M_wrong': int(m_above[j]), 'S_correct': int(s_above[j]),
                    'baseline_M_wrong_budget': baseline_m_wrong,
                    'baseline_S_correct': int(((y == 2) & (old == y)).sum()),
                    'unusable_for_model_selection': True,
                    'next_valid_test': 'Predeclare a separate train-side or nested calibration rule; apply without using target fold labels.'}
    result = {'status': 'descriptive_postfit_audit_no_fits_no_model_selection',
              'candidate': 'C_TabM_epoch25', 'group_concentration': counts,
              'label_assisted_uniform_boundary_upper_bound': boundary,
              'source_sha256': sha(__file__),
              'input_sha256': {str(p.relative_to(ROOT)): sha(p) for p in
                 [DEST / 'phase_B_evaluation.json', DEST / 'phase_B_ASA_OOF_ledger.parquet']}}
    save(target, result)
    print(json.dumps({'stage': result['status'], 'concentration': counts,
                      'optimistic_boundary': boundary}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
