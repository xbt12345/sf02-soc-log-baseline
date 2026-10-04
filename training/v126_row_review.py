"""Zero-fit row-level comparison and support audit for the V126 review."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from v126_frozen_audit import ROOT, PARENT, OUT, sha, save


def main():
    target = OUT / 'row_review.json'
    if target.exists():
        raise FileExistsError(target)
    pred = pd.read_parquet(PARENT / 'expert_ASA_predictions.parquet')
    ladder_path = ROOT / 'artifacts/v123_targeted_plan_20260929/support_ladder.parquet'
    ladder = pd.read_parquet(ladder_path)
    if not pred.row_position.equals(ladder.row_position):
        raise ValueError('Support identities differ')
    if not np.array_equal(pred.truth, ladder.truth) or len(pred) != 112807:
        raise ValueError('Truth or population differs')
    details = pred.copy()
    columns = ['parameter_observed', 'destination_M_rows', 'destination_M_roots',
               'destination_S_rows', 'destination_S_roots', 'behavior_M_rows',
               'behavior_S_rows', 'full_fact_fit_M_rows', 'full_fact_fit_S_rows',
               'exact_M_rows', 'exact_S_rows', 'protocol', 'icmp_type', 'icmp_code']
    for key in columns:
        details[key] = ladder[key]
    for arm in 'BC':
        details[arm + '_repair'] = pred.expert_pred_A.ne(pred.truth) & pred['expert_pred_' + arm].eq(pred.truth)
        details[arm + '_regression'] = pred.expert_pred_A.eq(pred.truth) & pred['expert_pred_' + arm].ne(pred.truth)
    # Recompute this partition for ALL S, not only the historical persistent-S slice.
    multi = details.destination_M_roots.ge(2) & details.destination_S_roots.ge(2)
    details['S_support_bucket'] = 'not_S'
    s = details.truth.eq(2)
    details.loc[s & multi & details.parameter_observed, 'S_support_bucket'] = 'known_parameter_dual_multi_root'
    details.loc[s & multi & ~details.parameter_observed, 'S_support_bucket'] = 'unknown_parameter_pooled'
    details.loc[s & ~multi & details.destination_S_rows.gt(0), 'S_support_bucket'] = 'thin_same_class_support'
    details.loc[s & details.destination_S_rows.eq(0), 'S_support_bucket'] = 'no_destination_same_class_support'
    if (details.loc[s, 'S_support_bucket'] == 'not_S').any():
        raise ValueError('Incomplete S partition')
    rows = []
    for arm in 'ABC':
        p = details['expert_pred_' + arm]
        for c in (1, 2):
            q = details.truth.eq(c)
            rows.append({'arm': arm, 'class': c, 'support': int(q.sum()),
                         'correct': int((q & p.eq(c)).sum()), 'errors': int((q & p.ne(c)).sum()),
                         'repairs_vs_A': int((q & details.expert_pred_A.ne(c) & p.eq(c)).sum()),
                         'regressions_vs_A': int((q & details.expert_pred_A.eq(c) & p.ne(c)).sum())})
    masks = {'root29_new_B_S_errors': s & details.root.eq(29) & details.B_regression,
             'all_B_S_errors': s & details.expert_pred_B.ne(2),
             'all_A_S_errors': s & details.expert_pred_A.ne(2)}
    supports = {}
    for name, q in masks.items():
        supports[name] = {'rows': int(q.sum()),
                          'partition': details.loc[q, 'S_support_bucket'].value_counts().to_dict(),
                          'coarse_behavior_has_S_support': int((q & details.behavior_S_rows.gt(0)).sum()),
                          'coarse_behavior_has_both_class_support': int((q & details.behavior_S_rows.gt(0) & details.behavior_M_rows.gt(0)).sum())}
    trajectory = json.loads((OUT / 'checkpoint_trajectory.json').read_text())
    fold1 = [r for r in trajectory if r['fold'] == 1]
    details.to_parquet(OUT / 'row_decision_review.parquet', index=False)
    out = {'status': 'row_recount_complete_no_fit', 'ASA_rows': len(details),
           'class_changes': rows, 'support_review': supports,
           'fold1_trajectory': fold1, 'classifier_fits': 0, 'optimizer_steps': 0,
           'sources': {str(p.relative_to(ROOT)): sha(p) for p in [
               PARENT / 'expert_ASA_predictions.parquet', ladder_path,
               OUT / 'checkpoint_trajectory.json', Path(__file__)]},
           'row_ledger_sha256': sha(OUT / 'row_decision_review.parquet'),
           'limits': ['Support counts are inherited from verified source-closed V123 behavior definitions; no organizer intent inferred.',
                      'Visible-parameter pooled support is not proof of a transferable M/S label rule.']}
    save(target, out)
    print(json.dumps({k: v for k, v in out.items() if k != 'fold1_trajectory'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
