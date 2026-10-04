"""Replay actual V160 outputs against fixed-error risk, without model calls."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/v161_independent_actual_objective_counterexample_replay_20261002'
OLD = ROOT / 'artifacts/v160_fixed_endpoint_diagnostic_20261002'
CACHED = ROOT / 'artifacts/v160_cached_polished_direction_finite_probe_20261002'
COHORT = ROOT / 'artifacts/v161_independent_frozen_error_cohort_review_20261002'
EPS = float(np.finfo(np.float64).eps)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def risk(frame, target):
    denominator = np.bincount(frame.truth, minlength=3)
    whole = np.array([math.fsum(frame.loc[frame.truth.eq(cls), 'stable_CE']) / int(denominator[cls])
                      for cls in [1, 2]])
    indexed = frame.set_index('row_position').loc[target.row_position]
    assert np.array_equal(indexed.truth, target.truth)
    assert np.array_equal(indexed.local, target.local)
    fixed = np.array([math.fsum(indexed.loc[indexed.truth.eq(cls), 'stable_CE']) / int(denominator[cls])
                      for cls in [1, 2]])
    return whole, fixed


def main():
    assert not OUT.exists()
    OUT.mkdir()
    cases, bindings = [], {}
    for role, parent in [(0, OLD / 'role0/round0/probe8'),
                         (1, CACHED / 'role1/probe13'),
                         (2, CACHED / 'role2/probe4')]:
        baseline_path = OLD / f'role{role}/baseline_class1/OOF_original_rows.parquet'
        candidate_path = parent / 'OOF_original_rows.parquet'
        target_path = COHORT / f'role{role}/fixed_pure_error_targets.parquet'
        before, after, target = [pd.read_parquet(p) for p in [baseline_path, candidate_path, target_path]]
        for name in ['row_position', 'local', 'truth', 'pure_current_input']:
            assert np.array_equal(before[name], after[name])
        assert np.array_equal(before.pred, after.pred)
        bw, bf = risk(before, target)
        aw, af = risk(after, target)
        limit = 16 * EPS * np.maximum(1., np.maximum(np.abs(bf), np.abs(af)))
        whole_limit = 16 * EPS * np.maximum(1., np.maximum(np.abs(bw), np.abs(aw)))
        preserved = before.pure_current_input & before.pred.eq(before.truth)
        protected_regressions = int((preserved & after.pred.ne(after.truth)).sum())
        assert protected_regressions == 0
        whole_component_pass = bool(np.all(bw - aw > whole_limit))
        focused_component_pass = bool(np.all(bf - af > limit))
        assert whole_component_pass
        assert focused_component_pass == (role != 2)
        cases.append(dict(role=role, fixed_original_targets=len(target), whole_risk_before=bw.tolist(),
                          whole_risk_after=aw.tolist(), whole_risk_drop=(bw - aw).tolist(),
                          focused_risk_before=bf.tolist(), focused_risk_after=af.tolist(),
                          focused_risk_drop=(bf - af).tolist(), focused_actual_resolution=limit.tolist(),
                          whole_resolved_decrease_component_pass=whole_component_pass,
                          focused_resolved_decrease_component_pass=focused_component_pass,
                          protected_regressions=protected_regressions, classification_repairs=0))
        for path in [baseline_path, candidate_path, target_path]:
            bindings[str(path.relative_to(ROOT))] = sha(path)
    report = dict(status='actual_V160_aggregate_CE_counterexample_rejected_by_fixed_error_risk_component',
                  cases=cases, input_sha256=bindings, source_sha256=sha(Path(__file__)),
                  official_heads=0, official_features=0, official_gradients=0, official_fits=0,
                  permanent_updates=0, quality_acceptance=False,
                  scope='Replays only the actual risk-decrease component. New focused-objective gradients and Armijo slopes were not measured; role0/1 component pass is not new finite-step acceptance.')
    (OUT / 'review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=report['status'], cases=cases), ensure_ascii=False))


if __name__ == '__main__':
    main()
