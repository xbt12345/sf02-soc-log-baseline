"""Count unique original controls without double-counting TRAIN-role exposures."""
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/v152_independent_leave_root_conditions_20261001'
LEDGER = OUT / 'all_legal_train_leave_root_support.parquet'
TRACE = ROOT / 'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet'
TARGET = OUT / 'original_support_and_domain_qualification.json'


def main():
    assert not TARGET.exists(), 'Do not overwrite observed evidence.'
    q = pd.read_parquet(LEDGER)
    t = pd.read_parquet(TRACE, columns=['row_position', 'root', 'fold', 'truth'])
    assert len(q) == 451228 and len(t) == 112807
    roots = t.groupby('root').agg(rows=('row_position', 'size'), classes=('truth', 'nunique'))
    summaries = []
    for scope, selected in [('all_S', q.truth.eq(2)), ('known_578', q.known_578_cohort)]:
        data = q.loc[selected]
        u = data.groupby(['key_scope', 'row_position']).agg(
            roles=('outer_training_role', 'nunique'), min_same=('same_class_other_root_rows', 'min'),
            max_same=('same_class_other_root_rows', 'max'), min_same_roots=('same_class_other_root_count', 'min'),
            max_same_roots=('same_class_other_root_count', 'max'), max_opposite=('opposite_class_other_root_rows', 'max'),
            actual_outer_correct=('actual_V146_B_correct', 'first'))
        assert u.roles.eq(2).all()
        for key, part in u.groupby(level=0):
            no_same = part.max_same.eq(0)
            qualified_both = part.min_same_roots.ge(2) & part.max_opposite.eq(0)
            summaries.append(dict(scope=scope, key_scope=key, unique_original_rows=len(part),
                any_same_support=int(part.max_same.gt(0).sum()),
                same_support_both_roles=int(part.min_same.gt(0).sum()),
                no_same_support_either_role=int(no_same.sum()),
                no_same_support_but_historical_outer_correct=int((no_same & part.actual_outer_correct).sum()),
                multi_root_same_only_both_roles=int(qualified_both.sum()),
                any_opposite_support=int(part.max_opposite.gt(0).sum()),
                any_multi_root_same_support=int(part.max_same_roots.ge(2).sum())))
    result = dict(status='unique_original_support_controls_and_root_label_coverage_counted',
        latest_actual_classifier='V146', new_model_forwards=0, new_fits=0, new_gradients=0, new_updates=0,
        unique_original_ASA_rows=len(t), unique_original_S_rows=int(t.truth.eq(2).sum()),
        root_label_coverage=dict(roots=len(roots), single_class_roots=int(roots.classes.eq(1).sum()),
            single_class_original_rows=int(roots.loc[roots.classes.eq(1), 'rows'].sum()),
            mixed_class_roots=int(roots.classes.gt(1).sum()),
            mixed_class_original_rows=int(roots.loc[roots.classes.gt(1), 'rows'].sum())),
        summaries=summaries,
        scope='Original-row diagnostics; root is a leakage group, not a verified physical domain. '
              'Historical outer predictions are controls, not these roles\' training predictions. '
              'No support-based threat classifier, reweighting or noisy-label inference.',
        source_sha256={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [Path(__file__), LEDGER, TRACE]})
    TARGET.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
