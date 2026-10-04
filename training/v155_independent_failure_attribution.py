"""Attribute already scored V155 changes and risks; no model execution."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from v155_independent_endpoint_audit_v2 import ROOT, RUN, OUT, REF, read, sha


def main():
    assert (OUT/'audit.json').exists() and not (OUT/'failure_attribution.json').exists()
    original = pd.read_parquet(REF/'all_original_classifier_gap_and_control_ledger.parquet')
    decisions = pd.read_parquet(RUN/'ASA_prediction_ledger.parquet')
    assert np.array_equal(original.row_position, decisions.row_position)
    q = {a: np.empty((len(original), 3)) for a in 'AB'}
    summaries = []
    for f in range(3):
        mask = original.fold.eq(f).to_numpy()
        for a in 'AB':
            folder = RUN/f'fold{f}_{a}'
            qq = np.load(folder/'sealed_all_prob.npy')
            q[a][mask] = qq[original.loc[mask, 'local']]
            history = read(folder/'progress.json')
            trials = [json.loads(z) for z in (folder/'proposals.jsonl').read_text().splitlines()]
            accepted = [z for z in trials if z['accepted']]
            baseline = read(RUN/'preflight.json')['folds'][f]['base']['member_CE']
            proxy_rises = sum(v['frozen_epsilon_proxy_CE'] > u['frozen_epsilon_proxy_CE'] for u, v in zip(history, history[1:]))
            summaries.append(dict(arm=a, fold=f, initial_ordinary_member_CE=baseline,
                endpoint_ordinary_member_CE=history[-1]['ordinary_member_CE'],
                changing_epsilon_proxy_rises=int(proxy_rises),
                classification_guard_rejections=int(sum(not z['classification_guard'] for z in trials)),
                last_proposal_pure_member_CE_contribution=accepted[-1]['ordinary_CE_contributions']['pure_member_CE_contribution'],
                last_proposal_mixed_member_CE_contribution=accepted[-1]['ordinary_CE_contributions']['mixed_member_CE_contribution']))
    joined = original[['row_position','local','root','fold','truth','raw_message','facts_json','canonical_key','known_578_cohort','same_family_and_outer_fold_control_S']].copy()
    for name in ['pred_A0','pred_V146_A','pred_A','pred_B']:
        joined[name] = decisions[name].to_numpy()
    for arm in 'AB':
        for c in range(3):
            joined[f'{arm}_p{c}'] = q[arm][:, c]
    changes = joined[joined.pred_A.ne(joined.pred_B) | joined.pred_V146_A.ne(joined.pred_B)].copy()
    changes['B_new_error_vs_V146_A'] = changes.pred_V146_A.eq(changes.truth) & changes.pred_B.ne(changes.truth)
    changes['B_fix_vs_V146_A'] = changes.pred_V146_A.ne(changes.truth) & changes.pred_B.eq(changes.truth)
    changes['B_fix_vs_matched_A'] = changes.pred_A.ne(changes.truth) & changes.pred_B.eq(changes.truth)
    root_rows = changes.groupby(['root','fold','truth']).agg(rows=('row_position','size'),
        B_new_errors_vs_V146_A=('B_new_error_vs_V146_A','sum'),
        B_fixes_vs_V146_A=('B_fix_vs_V146_A','sum'),B_fixes_vs_matched_A=('B_fix_vs_matched_A','sum')).reset_index()
    hard = original.known_578_cohort.to_numpy()
    hard_summary = dict(original_rows=int(hard.sum()),
        maximum_probability_change_B_vs_V146_A=float(np.max(np.abs(q['B'][hard]-original.loc[hard,['outer_A_p0','outer_A_p1','outer_A_p2']].to_numpy()))),
        hard_B_errors=int((q['B'][hard].argmax(1) != original.truth.to_numpy()[hard]).sum()),
        hard_B_median_S_probability=float(np.median(q['B'][hard,2])))
    assert int(changes.B_new_error_vs_V146_A.sum()) == 16
    assert int(changes.B_fix_vs_V146_A.sum()) == 0
    assert int(changes.B_fix_vs_matched_A.sum()) == 2
    assert sum(z['changing_epsilon_proxy_rises'] for z in summaries if z['arm']=='B') == 298
    report=dict(status='actual_failure_changes_and_risks_recounted',risk=summaries,
        roots=root_rows.to_dict('records'),hard578=hard_summary,own_model_forwards=0,own_gradients=0,
        source_sha256={str(Path(__file__).relative_to(ROOT)).replace('\\','/'):sha(Path(__file__)),
                       'training/v155_independent_endpoint_audit_v2.py':sha(ROOT/'training/v155_independent_endpoint_audit_v2.py')},
        limitation='Changes attribute observed predictions, not causal proof or new label semantics.')
    changes.to_parquet(OUT/'all_changed_original_rows_with_messages_and_probabilities.parquet',index=False)
    (OUT/'failure_attribution.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__ == '__main__':
    main()
