"""All-row post-fit support strata and paired classification; no tuning or fitting."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from v143_pair_geometry_probe import observed_key

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/v146_guarded_pair_training_20261001'


def paired(d, before, after):
    y, a, b = d.truth, d[before], d[after]
    return dict(rows=len(d), sources=int(d.root.nunique()), before_errors=int(a.ne(y).sum()),
        after_errors=int(b.ne(y).sum()), repairs=int((a.ne(y) & b.eq(y)).sum()),
        new_errors=int((a.eq(y) & b.ne(y)).sum()))


def main():
    assert (OUT / 'final_delivery.json').is_file(), 'Do not diagnose HELD while fitting'
    d = pd.read_parquet(OUT / 'ASA_prediction_ledger.parquet')
    trace = pd.read_parquet(ROOT / 'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet', columns=['row_position', 'facts_json'])
    d = d.merge(trace, on='row_position', validate='one_to_one')
    facts = d.facts_json.map(json.loads)
    keys = facts.map(observed_key)
    d['both_body_ports_known'] = facts.map(lambda f: f.get('transport_protocol') in ['tcp', 'udp'] and
        0 <= f.get('src_port_fixed', 65536) <= 65535 and 0 <= f.get('dst_port_fixed', 65536) <= 65535)
    d['registered_auxiliary_key'] = False
    for fold in range(3):
        cells = pd.read_parquet(OUT / f'fold{fold}_pair_reference.parquet')
        q = d.fold.eq(fold)
        d.loc[q, 'registered_auxiliary_key'] = keys[q].isin(set(cells.pair_key))
    truth = pd.read_parquet(ROOT / 'data/official/train.parquet', columns=['label_binary']).label_binary.map({'benign': 0, 'malicious': 1, 'suspicious': 2}).to_numpy(np.int8)
    assert len(d) == 112807 and np.array_equal(d.truth, truth[d.row_position])
    strata = []
    for (cl, known, direct), g in d.groupby(['truth', 'both_body_ports_known', 'registered_auxiliary_key'], dropna=False):
        strata.append(dict(truth=int(cl), both_body_ports_known=bool(known), registered_auxiliary_key=bool(direct),
            B_vs_A=paired(g, 'pred_A', 'pred_B'), B_vs_V142=paired(g, 'pred_V142', 'pred_B'), B_vs_A0=paired(g, 'pred_A0', 'pred_B')))
    hard = d[d.truth.eq(2) & d.both_body_ports_known & d.pred_V142.ne(2)]
    assert len(hard) == 578 and not hard.registered_auxiliary_key.any()
    roots = d.groupby('root').apply(lambda g: pd.Series(dict(fold=int(g.fold.iloc[0]), rows=len(g),
        A_errors=int(g.pred_A.ne(g.truth).sum()), B_errors=int(g.pred_B.ne(g.truth).sum()),
        V142_errors=int(g.pred_V142.ne(g.truth).sum()), A0_errors=int(g.pred_A0.ne(g.truth).sum()))), include_groups=False).reset_index()
    roots.to_parquet(OUT / 'paired_root_effects.parquet', index=False)
    result = dict(status='whole_ASA_post_fit_support_effect_recounted', new_fits=0, new_updates=0,
        official_truth_verified=True, ASA_population=len(d), strata=strata,
        original_578_known_port_S_errors=dict(direct_auxiliary_key_rows=0,
            B_vs_A=paired(hard, 'pred_A', 'pred_B'), B_vs_V142=paired(hard, 'pred_V142', 'pred_B'), B_vs_A0=paired(hard, 'pred_A0', 'pred_B')),
        limits=['Support association and post-fit gains do not establish a causal missing-support explanation.',
                'Previously inspected folds; no HELD labels were used to fit or select V146 endpoints.'])
    (OUT / 'support_effect_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
