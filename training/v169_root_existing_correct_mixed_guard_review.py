"""Saved original-row audit of an existing-correct protection wiring gap."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_review import ROOT, sha, check_bindings

OUT = ROOT / 'artifacts/v169_root_existing_correct_mixed_guard_review_20261002'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    files = [Path(__file__).resolve(), ROOT/'training/v169_prior_pair_training_entry_v7.py',
             ROOT/'training/v165_fixed_endpoint_decision_floor_diagnostic.py',
             ROOT/'training/v160_fixed_endpoint_diagnostic_v3.py']
    cohorts, roles = [], []
    OUT.mkdir()
    for role, expected in enumerate([106, 6, 94]):
        source = ROOT/f'artifacts/v164_short_supervised_trajectory_20261002/role{role}/endpoint/OOF_original_rows.parquet'
        rows = pd.read_parquet(source)
        correct = rows.pred.eq(rows.truth).to_numpy()
        current = rows.protected_correct.to_numpy(bool)
        assert not np.any(current & ~correct)
        gap = rows.loc[correct & ~current].copy()
        assert len(gap) == expected and not gap.pure_current_input.any()
        gap['role'] = role
        target = OUT/f'role{role}_current_correct_unprotected_original_rows.parquet'
        gap.to_parquet(target, index=False)
        np.save(OUT/f'role{role}_required_all_current_correct_mask.npy', current | correct)
        files.extend([source, target, OUT/f'role{role}_required_all_current_correct_mask.npy'])
        cohorts.append(gap)
        roles.append(dict(role=role, original_population=len(rows),
            previous_protection=int(current.sum()), all_current_correct=int(correct.sum()),
            omitted_current_correct_original_rows=expected,
            omitted_all_mixed=True,
            explicit_protection_must_include_all_V164_current_correct=True))
    combined = pd.concat(cohorts, ignore_index=True)
    bindings = {p.relative_to(ROOT).as_posix():sha(p) for p in files}
    check_bindings(bindings)
    # Equal total class errors do not imply preservation of old correct rows.
    before = np.array([1, 2]); after = np.array([2, 1]); truth = np.array([1, 1])
    assert np.count_nonzero(before != truth) == np.count_nonzero(after != truth) == 1
    assert np.count_nonzero((before == truth) & (after != truth)) == 1
    report = dict(status='existing_correct_mixed_original_rows_missing_from_runtime_guard',
        all_checks_passed=False, wiring_gap_confirmed=True, roles=roles,
        total_role_observations=206,
        unique_original_row_positions=int(combined.row_position.nunique()),
        overlapping_roles_must_not_be_added_as_independent_quality_gain=True,
        equal_error_total_synthetic_counterexample_old_correct_regression=1,
        required_action='Both arms must explicitly protect every current V164-correct original OOF row, including the fixed omitted mixed cohorts, then cumulatively protect all actual repairs.',
        class_total_nonincrease_is_not_row_retention=True,
        official_heads=0, official_features=0, official_derivatives=0,
        fits=0, permanent_updates=0, supports_physical_seal=False,
        source_sha256=bindings,
        scope='Original saved row masks and a two-row logical guard example only. Does not assert the example is attainable by the official model, and does not change official labels or scores.')
    (OUT/'review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['source_sha256','roles']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
